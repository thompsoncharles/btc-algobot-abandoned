"""
BTC data-source audit.

The research symbol BTCUSD is currently a local alias for Binance BTCUSDT
klines. This module reports what is actually cached, whether it is stale or
malformed, and optionally compares recent daily bars against Yahoo BTC-USD
and IBKR crypto historical bars when those sources are available.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Optional

import pandas as pd
from loguru import logger

from quant import config
from quant.data.store import DataStore

INTERVAL_DELTAS = {
    "1d": timedelta(days=1),
    "12h": timedelta(hours=12),
    "8h": timedelta(hours=8),
    "4h": timedelta(hours=4),
    "1h": timedelta(hours=1),
    "5m": timedelta(minutes=5),
    "1m": timedelta(minutes=1),
}


def _to_utc_naive(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts
    return ts.tz_convert("UTC").tz_localize(None)


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_convert("UTC").tz_localize(None)
    return out.sort_index()


def _expected_latest_closed_open(interval: str,
                                 now: Optional[pd.Timestamp] = None) -> Optional[pd.Timestamp]:
    delta = INTERVAL_DELTAS.get(interval)
    if delta is None:
        return None
    now = _to_utc_naive(now or pd.Timestamp.utcnow())
    if interval.endswith("m"):
        minutes = int(delta.total_seconds() // 60)
        floored = now.floor(f"{minutes}min")
    elif interval.endswith("h"):
        hours = int(delta.total_seconds() // 3600)
        floored = now.floor(f"{hours}h")
    elif interval == "1d":
        floored = now.floor("1d")
    else:
        return None
    return floored - pd.Timedelta(delta)


def _missing_bars(index: pd.Index, interval: str) -> dict:
    delta = INTERVAL_DELTAS.get(interval)
    if delta is None or len(index) < 2:
        return {"expected": None, "missing_count": None, "first_missing": None}
    expected = pd.date_range(index.min(), index.max(), freq=pd.Timedelta(delta))
    missing = expected.difference(index)
    return {
        "expected": int(len(expected)),
        "missing_count": int(len(missing)),
        "first_missing": str(missing[0]) if len(missing) else None,
    }


def _basic_report(df: Optional[pd.DataFrame], interval: str, source: str,
                  symbol: str) -> dict:
    if df is None or df.empty:
        return {"source": source, "symbol": symbol, "available": False}
    bars = _normalize_index(df)
    latest = pd.Timestamp(bars.index.max())
    expected_latest = _expected_latest_closed_open(interval)
    delta = INTERVAL_DELTAS.get(interval)
    duplicate_count = int(bars.index.duplicated().sum())
    missing = _missing_bars(bars.index, interval)
    volume = bars["Volume"] if "Volume" in bars.columns else pd.Series(dtype=float)
    staleness_hours = None
    bars_behind = None
    if expected_latest is not None:
        staleness_hours = max(0.0, (expected_latest - latest).total_seconds() / 3600)
        if delta is not None:
            bars_behind = max(0, int((expected_latest - latest) / pd.Timedelta(delta)))
    unfinished = False
    if delta is not None:
        unfinished = latest + pd.Timedelta(delta) > _to_utc_naive(pd.Timestamp.utcnow())
    return {
        "source": source,
        "symbol": symbol,
        "available": True,
        "interval": interval,
        "bars": int(len(bars)),
        "from": str(bars.index.min()),
        "to": str(latest),
        "expected_latest_closed_bar_open": str(expected_latest) if expected_latest is not None else None,
        "staleness_hours_vs_expected_closed_bar": staleness_hours,
        "bars_behind_expected": bars_behind,
        "duplicate_bars": duplicate_count,
        "missing_bars": missing,
        "volume_non_null_pct": float(volume.notna().mean() * 100) if len(volume) else 0.0,
        "volume_positive_pct": float((volume.fillna(0) > 0).mean() * 100) if len(volume) else 0.0,
        "possible_unfinished_latest_candle": bool(unfinished),
    }


def _fetch_yahoo_daily() -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
        df = yf.download("BTC-USD", period="max", interval="1d",
                         auto_adjust=True, progress=False)
    except Exception as e:
        logger.warning("Yahoo BTC-USD fetch failed: {}", e)
        return None
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.capitalize)
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[cols].dropna(how="all")


def _fetch_ibkr_daily() -> Optional[pd.DataFrame]:
    """
    Best-effort IBKR crypto daily bars. Requires TWS/Gateway and crypto market
    data permissions. If unavailable, the audit still succeeds without it.
    """
    try:
        from ib_async import Contract, util
        from ibkr.connection import IBConnection
    except Exception as e:
        logger.warning("IBKR imports unavailable: {}", e)
        return None

    conn = IBConnection()
    try:
        conn.connect()
        ib = conn.ib
        last_error = None
        for exchange in ("PAXOS", "ZEROHASH"):
            try:
                contract = Contract(symbol="BTC", secType="CRYPTO",
                                    exchange=exchange, currency="USD")
                ib.qualifyContracts(contract)
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime="",
                    durationStr="1 Y",
                    barSizeSetting="1 day",
                    whatToShow="AGGTRADES",
                    useRTH=False,
                    formatDate=1,
                    keepUpToDate=False,
                )
                if bars:
                    df = util.df(bars).set_index("date")
                    df = df.rename(columns={"open": "Open", "high": "High",
                                            "low": "Low", "close": "Close",
                                            "volume": "Volume"})
                    out = df[["Open", "High", "Low", "Close", "Volume"]]
                    out.attrs["ibkr_exchange"] = exchange
                    return out
            except Exception as e:
                last_error = e
        logger.warning("IBKR BTC historical bars unavailable: {}", last_error)
        return None
    except Exception as e:
        logger.warning("IBKR connection/data audit skipped: {}", e)
        return None
    finally:
        try:
            conn.disconnect()
        except Exception:
            pass


def _daily_from_intraday(df: pd.DataFrame) -> pd.DataFrame:
    bars = _normalize_index(df)
    daily = pd.DataFrame({
        "Open": bars["Open"].resample("1D").first(),
        "High": bars["High"].resample("1D").max(),
        "Low": bars["Low"].resample("1D").min(),
        "Close": bars["Close"].resample("1D").last(),
        "Volume": bars["Volume"].resample("1D").sum() if "Volume" in bars.columns else None,
    })
    return daily.dropna(subset=["Open", "High", "Low", "Close"])


def _compare_daily(left: Optional[pd.DataFrame], right: Optional[pd.DataFrame],
                   left_name: str, right_name: str) -> dict:
    if left is None or right is None or left.empty or right.empty:
        return {"left": left_name, "right": right_name, "available": False}
    ldf = _normalize_index(left)
    rdf = _normalize_index(right)
    idx = ldf.index.normalize().intersection(rdf.index.normalize())
    if len(idx) == 0:
        return {"left": left_name, "right": right_name, "available": False,
                "reason": "no overlapping dates"}
    ldf = ldf.copy()
    rdf = rdf.copy()
    ldf.index = ldf.index.normalize()
    rdf.index = rdf.index.normalize()
    common = ldf.loc[idx, ["Open", "High", "Low", "Close"]].join(
        rdf.loc[idx, ["Open", "High", "Low", "Close"]],
        lsuffix=f"_{left_name}", rsuffix=f"_{right_name}",
    ).dropna()
    rows = {"left": left_name, "right": right_name, "available": True,
            "overlap_days": int(len(common))}
    for col in ("Open", "High", "Low", "Close"):
        a = common[f"{col}_{left_name}"].astype(float)
        b = common[f"{col}_{right_name}"].astype(float)
        pct = ((a - b).abs() / b.abs().replace(0, pd.NA) * 100).dropna()
        rows[f"{col.lower()}_median_abs_diff_pct"] = float(pct.median()) if len(pct) else None
        rows[f"{col.lower()}_max_abs_diff_pct"] = float(pct.max()) if len(pct) else None
    rows["latest_left"] = str(ldf.index.max())
    rows["latest_right"] = str(rdf.index.max())
    return rows


def _quality_gate(report: dict) -> tuple[bool, str]:
    """
    Strict gate: blocks btc-eval if data is unsafe for research.

    Returns (passes, verdict_text)
    """
    if not report.get("cached_primary", {}).get("available"):
        return False, "UNSAFE - no cached BTC data"

    cached = report["cached_primary"]

    # Gate 1: too many duplicates
    if cached.get("duplicate_bars", 0) > 0:
        return False, "UNSAFE - duplicate bars found (data corruption)"

    # Gate 2: too many missing bars (>1%)
    missing = cached.get("missing_bars", {})
    expected = missing.get("expected", 0)
    missing_count = missing.get("missing_count", 0)
    if expected and missing_count / expected > 0.01:
        return False, f"UNSAFE - {missing_count}/{expected} bars missing ({100*missing_count/expected:.2f}%, >1% threshold)"

    # Gate 3: stale data (>6 hours behind expected)
    staleness = cached.get("staleness_hours_vs_expected_closed_bar", 0)
    if staleness and staleness > 6:
        return False, f"UNSAFE - data is {staleness:.1f} hours stale (>6h threshold)"

    # Gate 4: unfinished candle in latest bar (training/backtest look-ahead)
    if cached.get("possible_unfinished_latest_candle"):
        return False, "UNSAFE - latest bar appears to be unfinished (look-ahead risk)"

    # Gate 5: volume is critically sparse (may indicate data source issues)
    vol_pct = cached.get("volume_positive_pct", 0)
    if vol_pct < 50:
        return False, f"UNSAFE - volume sparse ({vol_pct:.1f}% positive, <50% threshold)"

    # Passed strict gates
    return True, "SAFE FOR RESEARCH - cached data passes quality gates"


def build_audit(include_yahoo: bool = True, include_ibkr: bool = False) -> dict:
    store = DataStore()
    cached = store.load_bars("BTCUSD", config.BTC_INTERVAL)
    cached_report = _basic_report(
        cached,
        config.BTC_INTERVAL,
        source="local cache: bundled BTCUSDT seed + Binance BTCUSDT + Kraken/Yahoo fallbacks",
        symbol="BTCUSD alias for BTC spot USD/USDT research bars",
    )

    yahoo = _fetch_yahoo_daily() if include_yahoo else None
    yahoo_report = _basic_report(yahoo, "1d", "Yahoo Finance", "BTC-USD")

    ibkr = _fetch_ibkr_daily() if include_ibkr else None
    ibkr_source = "IBKR crypto historical AGGTRADES"
    if ibkr is not None and ibkr.attrs.get("ibkr_exchange"):
        ibkr_source += f" ({ibkr.attrs['ibkr_exchange']})"
    ibkr_report = _basic_report(ibkr, "1d", ibkr_source, "BTC.USD")

    cached_daily = _daily_from_intraday(cached) if cached is not None and not cached.empty else None
    comparisons = [
        _compare_daily(cached_daily, yahoo, "binance", "yahoo"),
    ]
    if include_ibkr:
        comparisons.append(_compare_daily(cached_daily, ibkr, "binance", "ibkr"))
        comparisons.append(_compare_daily(yahoo, ibkr, "yahoo", "ibkr"))

    # Build complete report with quality gate
    full_report = {
        "symbol_mapping": {
            "research_symbol": "BTCUSD",
            "actual_primary_source_symbol": "Bundled/Binance BTCUSDT, repaired by Kraken XBTUSD and Yahoo BTC-USD when needed",
            "daily_robustness_symbol": "Yahoo BTC-USD",
            "execution_reference": "IBKR BTC/USD crypto when TWS/Gateway is available",
        },
        "cached_primary": cached_report,
        "yahoo_daily": yahoo_report,
        "ibkr_daily": ibkr_report,
        "comparisons": comparisons,
    }

    # Apply quality gate
    passes, verdict = _quality_gate(full_report)
    full_report["quality_gate"] = {
        "passes": passes,
        "verdict": verdict,
    }

    # Recommendation based on quality gate and audit findings
    if passes:
        recommendation = (
            "Data PASSES quality gate - safe for research and paper trading. "
            "Binance/Yahoo hybrid source is acceptable. "
            "For live execution, compare against IBKR BTC/USD daily when available."
        )
    else:
        recommendation = f"Data FAILS quality gate: {verdict}. Do not use for model evaluation."

    full_report["recommendation"] = recommendation

    return full_report
