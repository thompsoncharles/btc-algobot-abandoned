"""
BTCUSD data ingestion.

Godel Terminal is temporarily unavailable, so this is a self-contained data
path that does not depend on it: a bundled seed CSV (12h bars, courtesy of
the qt4 research repo) gets the model runnable offline immediately, and
Binance's public klines REST API extends/refreshes it going forward - no
API key required for public market data. If Binance is geo-blocked, Kraken's
public BTC/USD OHLC endpoint is used as a fallback for recent closed bars.

Crypto trades 24/7; unlike quant/data/market_data.py there is no session
concept here, so bars are plain UTC-indexed OHLCV with no market-hours
filtering anywhere downstream.
"""
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from loguru import logger

from quant.data.store import DataStore

SEED_CSV = Path(__file__).parent / "seed" / "BTCUSDT_12h_ohlc.csv"
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
KRAKEN_OHLC_URL = "https://api.kraken.com/0/public/OHLC"

# Binance kline interval strings: https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data
_INTERVAL_MAP = {"1d": "1d", "12h": "12h", "4h": "4h", "1h": "1h", "5m": "5m", "1m": "1m"}

_INTERVAL_TIMEDELTA = {
    "1d": timedelta(days=1), "12h": timedelta(hours=12), "4h": timedelta(hours=4),
    "1h": timedelta(hours=1), "5m": timedelta(minutes=5), "1m": timedelta(minutes=1),
}

_KRAKEN_INTERVAL_MINUTES = {"1d": 1440, "12h": 720, "4h": 240,
                            "1h": 60, "5m": 5, "1m": 1}


def _cache_is_fresh(bars: pd.DataFrame, interval: str) -> bool:
    """
    True when no newer *closed* bar can exist yet, so ingest can skip the
    Binance round-trip. The newest stored bar opened at index.max() and
    closed one interval later; the next bar doesn't close until two
    intervals after that open. So we're fresh while age(from open) < 2x
    interval - checking sooner just refetches the same closed bar.
    """
    if bars is None or bars.empty:
        return False
    delta = _INTERVAL_TIMEDELTA.get(interval, timedelta(days=1))
    expected = pd.date_range(bars.index.min(), bars.index.max(), freq=pd.Timedelta(delta))
    if len(expected.difference(bars.index)) > 0:
        return False
    age = datetime.utcnow() - bars.index.max().to_pydatetime()
    return age < 2 * delta


def load_seed(ticker: str = "BTCUSD") -> pd.DataFrame:
    """Bundled 12h OHLC history (no network) - the offline fallback."""
    if not SEED_CSV.exists():
        raise FileNotFoundError(f"seed CSV missing at {SEED_CSV}")
    df = pd.read_csv(SEED_CSV, parse_dates=["datetime"]).set_index("datetime")
    df.columns = [c.capitalize() for c in df.columns]
    if "Volume" not in df.columns:
        df["Volume"] = float("nan")  # seed CSV has no volume column
    return df[["Open", "High", "Low", "Close", "Volume"]].sort_index()


def fetch_binance_klines(symbol: str = "BTCUSDT", interval: str = "12h",
                         limit: int = 1000,
                         end_time_ms: Optional[int] = None) -> pd.DataFrame:
    """One page (<=1000 bars) of Binance spot klines, most-recent-first paging."""
    params = {"symbol": symbol, "interval": _INTERVAL_MAP[interval], "limit": limit}
    if end_time_ms is not None:
        params["endTime"] = end_time_ms
    r = requests.get(BINANCE_KLINES_URL, params=params, timeout=30)
    r.raise_for_status()
    rows = r.json()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=[
        "open_time", "Open", "High", "Low", "Close", "Volume", "close_time",
        "quote_volume", "trades", "taker_base", "taker_quote", "ignore",
    ])
    df = drop_unclosed(df)
    if df.empty:
        return pd.DataFrame()
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    df = df.set_index("datetime")[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    return df.sort_index()


def drop_unclosed(df: pd.DataFrame, now_ms: Optional[int] = None) -> pd.DataFrame:
    """
    Drop the in-progress candle. Binance streams the current (unclosed) bar
    with a close_time in the future; training or backtesting on a half-formed
    12h bar is look-ahead garbage, so keep only bars whose close_time has
    already passed. Returns a copy with the raw kline columns intact.
    """
    if df.empty or "close_time" not in df.columns:
        return df
    if now_ms is None:
        now_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    return df[df["close_time"].astype("int64") <= now_ms]


def fetch_binance_history(symbol: str = "BTCUSDT", interval: str = "12h",
                          bars: int = 3000) -> pd.DataFrame:
    """Page backwards through Binance klines to build a longer history."""
    frames = []
    end_time_ms = None
    remaining = bars
    while remaining > 0:
        page = fetch_binance_klines(symbol, interval, limit=min(1000, remaining),
                                    end_time_ms=end_time_ms)
        if page.empty:
            break
        frames.append(page)
        remaining -= len(page)
        end_time_ms = int(page.index[0].timestamp() * 1000) - 1
        if len(page) < min(1000, remaining + len(page)):
            break  # hit the start of available history
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated(keep="last")]


def fetch_kraken_ohlc(pair: str = "XBTUSD", interval: str = "4h",
                      since: Optional[int] = None) -> tuple[pd.DataFrame, Optional[int]]:
    """One page of recent closed BTC/USD OHLC bars from Kraken's public API."""
    params = {"pair": pair, "interval": _KRAKEN_INTERVAL_MINUTES[interval]}
    if since is not None:
        params["since"] = since
    r = requests.get(KRAKEN_OHLC_URL, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    errors = payload.get("error") or []
    if errors:
        raise RuntimeError(f"Kraken OHLC error: {errors}")
    result = payload.get("result", {})
    key = next((k for k in result.keys() if k != "last"), None)
    rows = result.get(key, []) if key else []
    if not rows:
        return pd.DataFrame(), result.get("last")
    df = pd.DataFrame(rows, columns=[
        "time", "Open", "High", "Low", "Close", "vwap", "Volume", "count",
    ])
    df["datetime"] = pd.to_datetime(df["time"].astype("int64"), unit="s", utc=True).dt.tz_localize(None)
    delta = _INTERVAL_TIMEDELTA.get(interval, timedelta(days=1))
    now = pd.Timestamp.utcnow()
    if now.tzinfo is not None:
        now = now.tz_convert("UTC").tz_localize(None)
    df = df[df["datetime"] + pd.Timedelta(delta) <= now]
    if df.empty:
        return pd.DataFrame(), result.get("last")
    df = df.set_index("datetime")[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    return df.sort_index(), result.get("last")


def fetch_kraken_history(pair: str = "XBTUSD", interval: str = "12h",
                         since: Optional[int] = None,
                         max_pages: int = 10) -> pd.DataFrame:
    """
    Kraken does not offer 12h directly, so fetch 4h and resample to 12h.
    Other supported intervals are returned directly.
    """
    request_interval = "4h" if interval == "12h" else interval
    frames = []
    cursor = since
    seen = set()
    for _ in range(max_pages):
        page, last = fetch_kraken_ohlc(pair, request_interval, since=cursor)
        if page.empty:
            break
        frames.append(page)
        if last is None or last in seen:
            break
        seen.add(last)
        cursor = int(last)
        now = pd.Timestamp.utcnow()
        if now.tzinfo is not None:
            now = now.tz_convert("UTC").tz_localize(None)
        if page.index.max() + pd.Timedelta(_INTERVAL_TIMEDELTA[request_interval]) >= now:
            break
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames).sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]
    if interval != "12h":
        return raw
    resampled = pd.DataFrame({
        "Open": raw["Open"].resample("12h").first(),
        "High": raw["High"].resample("12h").max(),
        "Low": raw["Low"].resample("12h").min(),
        "Close": raw["Close"].resample("12h").last(),
        "Volume": raw["Volume"].resample("12h").sum(),
        "_count": raw["Close"].resample("12h").count(),
    })
    resampled = resampled[resampled["_count"] >= 3].drop(columns="_count")
    now = pd.Timestamp.utcnow()
    if now.tzinfo is not None:
        now = now.tz_convert("UTC").tz_localize(None)
    resampled = resampled[resampled.index + pd.Timedelta(_INTERVAL_TIMEDELTA["12h"]) <= now]
    return resampled.dropna()


def fetch_yahoo_resampled(interval: str = "12h") -> pd.DataFrame:
    """Yahoo BTC-USD hourly fallback resampled to the configured BTC interval."""
    if interval != "12h":
        return pd.DataFrame()
    try:
        import yfinance as yf
        raw = yf.download("BTC-USD", period="730d", interval="1h",
                          auto_adjust=True, progress=False)
    except Exception as e:
        raise RuntimeError(f"Yahoo BTC-USD hourly fallback failed: {e}") from e
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns=str.capitalize)
    raw = raw[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
    raw.index = pd.to_datetime(raw.index)
    if getattr(raw.index, "tz", None) is not None:
        raw.index = raw.index.tz_convert("UTC").tz_localize(None)
    resampled = pd.DataFrame({
        "Open": raw["Open"].resample("12h").first(),
        "High": raw["High"].resample("12h").max(),
        "Low": raw["Low"].resample("12h").min(),
        "Close": raw["Close"].resample("12h").last(),
        "Volume": raw["Volume"].resample("12h").sum(),
        "_count": raw["Close"].resample("12h").count(),
    })
    resampled = resampled[resampled["_count"] >= 10].drop(columns="_count")
    now = pd.Timestamp.utcnow()
    if now.tzinfo is not None:
        now = now.tz_convert("UTC").tz_localize(None)
    resampled = resampled[resampled.index + pd.Timedelta(_INTERVAL_TIMEDELTA["12h"]) <= now]
    return resampled.dropna()


def fetch_fallback_history(bars: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Fallback source for environments where Binance public API is blocked."""
    since = None
    if bars is not None and not bars.empty:
        # Ask for a small overlap so duplicate replacement keeps the latest print.
        since = int((bars.index.max() - pd.Timedelta(_INTERVAL_TIMEDELTA[interval])).timestamp())
    frames = []
    try:
        kraken = fetch_kraken_history("XBTUSD", interval, since=since)
        if not kraken.empty:
            frames.append(kraken)
    except Exception as e:
        logger.warning("Kraken fallback failed: {}", e)
    try:
        yahoo = fetch_yahoo_resampled(interval)
        if not yahoo.empty:
            frames.append(yahoo)
    except Exception as e:
        logger.warning("Yahoo fallback failed: {}", e)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated(keep="last")]


def get_btc_bars(store: Optional[DataStore] = None, ticker: str = "BTCUSD",
                 interval: str = "12h", refresh: bool = True,
                 target_bars: int = 3000) -> pd.DataFrame:
    """
    Seed + extend + cache BTCUSD OHLCV.

    First call with no cache: loads the bundled seed, tries to extend it
    from Binance or Kraken fallback (best-effort - offline environments still get the seed).
    Subsequent calls: cached parquet, optionally refreshed with the latest
    bars from Binance.
    """
    store = store or DataStore()
    cached = store.load_bars(ticker, interval)

    if cached is None or cached.empty:
        bars = load_seed(ticker)
        logger.info("BTC seed loaded: {} bars ({} to {})", len(bars),
                   bars.index.min().date(), bars.index.max().date())
    else:
        bars = cached

    if refresh and _cache_is_fresh(bars, interval):
        logger.info("BTC cache is fresh (last bar within one {} interval); "
                    "skipping Binance refresh", interval)
        refresh = False

    if refresh:
        try:
            fresh = fetch_binance_history("BTCUSDT", interval,
                                          bars=max(target_bars - len(bars), 200))
            if not fresh.empty:
                bars = pd.concat([bars, fresh]).sort_index()
                bars = bars[~bars.index.duplicated(keep="last")]
                logger.info("BTC extended via Binance: {} bars total", len(bars))
        except Exception as e:
            logger.warning("Binance refresh failed ({}); trying Kraken fallback", e)
            fresh = fetch_fallback_history(bars, interval)
            if not fresh.empty:
                bars = pd.concat([bars, fresh]).sort_index()
                bars = bars[~bars.index.duplicated(keep="last")]
                logger.info("BTC extended via fallback sources: {} bars total", len(bars))
            else:
                logger.warning("fallback sources returned no closed bars; using {} cached/seed bars",
                               len(bars))

    store.save_bars(ticker, interval, bars)
    return bars


def fetch_yahoo_daily_max() -> pd.DataFrame:
    """Full daily BTC-USD history from Yahoo (2014+) - the multi-regime set."""
    import yfinance as yf
    raw = yf.download("BTC-USD", period="max", interval="1d",
                      auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns=str.capitalize)
    raw = raw[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
    raw.index = pd.to_datetime(raw.index)
    if getattr(raw.index, "tz", None) is not None:
        raw.index = raw.index.tz_convert("UTC").tz_localize(None)
    # Yahoo's most recent daily row is the live (unclosed) UTC day - drop it.
    now = pd.Timestamp.utcnow()
    if now.tzinfo is not None:
        now = now.tz_convert("UTC").tz_localize(None)
    raw = raw[raw.index + pd.Timedelta(days=1) <= now]
    return raw.sort_index()


def get_btc_daily_history(store: Optional[DataStore] = None,
                          refresh: bool = True) -> pd.DataFrame:
    """
    Multi-regime daily BTCUSD history (2014+), cached as ("BTCUSD", "1d").

    Yahoo BTC-USD daily is the primary source (11+ years: bull, bear,
    sideways). Kraken daily is the fallback for recent bars if Yahoo is
    unavailable. This is the robustness dataset - the 12h cache stays the
    trading-resolution dataset and is not touched here.
    """
    store = store or DataStore()
    cached = store.load_bars("BTCUSD", "1d")

    if refresh and cached is not None and not cached.empty and _cache_is_fresh(cached, "1d"):
        return cached

    if refresh:
        try:
            fresh = fetch_yahoo_daily_max()
            if not fresh.empty:
                if cached is not None and not cached.empty:
                    fresh = pd.concat([cached, fresh]).sort_index()
                    fresh = fresh[~fresh.index.duplicated(keep="last")]
                store.save_bars("BTCUSD", "1d", fresh)
                logger.info("BTC daily history: {} bars ({} to {})", len(fresh),
                            fresh.index.min().date(), fresh.index.max().date())
                return fresh
        except Exception as e:
            logger.warning("Yahoo daily history failed ({}); trying Kraken", e)
        try:
            kraken = fetch_kraken_history("XBTUSD", "1d")
            if not kraken.empty:
                merged = kraken
                if cached is not None and not cached.empty:
                    merged = pd.concat([cached, kraken]).sort_index()
                    merged = merged[~merged.index.duplicated(keep="last")]
                store.save_bars("BTCUSD", "1d", merged)
                return merged
        except Exception as e:
            logger.warning("Kraken daily fallback failed: {}", e)

    if cached is None or cached.empty:
        raise RuntimeError("no daily BTC history available (Yahoo and Kraken failed, no cache)")
    return cached
