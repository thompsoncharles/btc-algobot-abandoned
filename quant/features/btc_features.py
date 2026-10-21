"""
BTC feature set for the linear model - richer than lagged returns alone.

All features are causal (rolling/shifted, no look-ahead) and reuse the
existing indicator library (quant.features.technical / .volatility) rather
than reimplementing. Volume features are skipped automatically when the
source has no volume (the bundled seed CSV has none).

Feature groups (the review's "add stronger features" list):
  - momentum:   lagged log returns, RSI, MACD histogram
  - trend:      distance from SMA(fast) and SMA(slow), fast/slow MA spread
  - volatility: realized vol, vol percentile (also drives the no-trade filter)
  - flow:       volume shock (only when volume is present)
  - external:   prior daily SPY/QQQ/VIX/DXY/rates context when cached
"""
from typing import List

import numpy as np
import pandas as pd

from quant.features import technical, volatility
from quant.models.linear_ar import lagged_log_return_features


FEATURE_GROUPS = ("momentum", "trend", "volatility", "flow")
OPTIONAL_FEATURE_GROUPS = FEATURE_GROUPS + ("external", "funding")
CONTEXT_TICKERS = ("SPY", "QQQ", "^VIX", "DX-Y.NYB", "^TNX")


def assemble_btc_features(bars: pd.DataFrame, n_lags: int = 5,
                          forecast_horizon: int = 1,
                          ma_window: int = 20, groups=None,
                          include_lag0: bool = False) -> pd.DataFrame:
    """
    Causal feature matrix aligned to bars.index.

    The AR lag block is always present (the base). `groups` selects which
    extra feature groups to add on top - None = all, a set/list = a subset
    (used by the ablation harness). None = current default groups
    (momentum, trend, volatility, flow). Add "external" explicitly to use
    shifted SPY/QQQ/VIX/DXY/rates context. `include_lag0` adds the
    just-completed bar's return.
    """
    want = set(FEATURE_GROUPS) if groups is None else set(groups)
    close = bars["Close"].astype(float)
    f = lagged_log_return_features(close, n_lags, forecast_horizon,
                                   include_lag0=include_lag0)

    if "momentum" in want:
        f["rsi14"] = technical.rsi(close, 14) / 100.0
        _, _, macd_hist = technical.macd(close)
        f["macd_hist_norm"] = macd_hist / close

    if "trend" in want:
        slow = ma_window * 3
        f["dist_ma_fast"] = close / technical.sma(close, ma_window) - 1
        f["dist_ma_slow"] = close / technical.sma(close, slow) - 1
        f["ma_spread"] = technical.sma(close, ma_window) / technical.sma(close, slow) - 1

    if "volatility" in want:
        # 120-bar percentile lookback (~60 days on 12h bars) so it's valid
        # inside a walk-forward warmup window rather than needing a full year.
        f["rvol20"] = volatility.historical_vol(close, 20, annualize=False)
        f["vol_pctile"] = volatility.vol_percentile(close, 20, lookback=120)

    if "flow" in want:
        vol = bars["Volume"].astype(float)
        if vol.notna().any() and (vol.fillna(0) > 0).any():
            f["vol_shock"] = vol / vol.rolling(20).mean() - 1

    if "external" in want:
        ext = external_context_features(bars.index)
        if not ext.empty:
            f = f.join(ext, how="left")

    if "funding" in want:
        fund = funding_features(bars.index)
        if not fund.empty:
            f = f.join(fund, how="left")

    return f


def funding_features(target_index: pd.Index) -> pd.DataFrame:
    """
    Perp funding-rate context aligned to BTC bars (OKX BTC-USDT-SWAP, 8h
    settlements, cached by quant.data.btc_derivs).

    Causality: a settled funding print is public at its funding time; we
    additionally shift by one settlement before aligning, so a bar can only
    see funding that settled strictly before it. Features:
      funding_last   last settled rate (typically +-0.01%=1bp scale, x1e4)
      funding_ma21   ~1-week mean (21 settlements)
      funding_z90    z-score vs ~1-month window (90 settlements)
    Persistent positive funding = crowded longs paying shorts (euphoria);
    negative = crowded shorts (capitulation). A regime/sentiment input, not
    an entry signal by itself.
    """
    from quant.data.btc_derivs import get_funding_series

    fund = get_funding_series(refresh=False)
    if fund is None or fund.empty:
        return pd.DataFrame(index=target_index)
    rate = fund["funding_rate"].astype(float).sort_index() * 1e4  # bps units
    rate = rate.shift(1)  # one-settlement safety margin
    df = pd.DataFrame({
        "funding_last": rate,
        "funding_ma21": rate.rolling(21).mean(),
        "funding_z90": (rate - rate.rolling(90).mean()) / rate.rolling(90).std(),
    })

    target = pd.DatetimeIndex(pd.to_datetime(target_index))
    if getattr(target, "tz", None) is not None:
        target = target.tz_convert("UTC").tz_localize(None)
    aligned = df.reindex(target, method="ffill")
    aligned.index = target_index
    return aligned


def external_context_features(target_index: pd.Index) -> pd.DataFrame:
    """
    Prior daily cross-asset context aligned to BTC bars.

    All context features are shifted by one daily bar before forward-filling to
    BTC timestamps. This is intentionally conservative: a BTC 12h signal cannot
    accidentally see a same-day equity/macro close that happened later.
    """
    from quant.data.store import DataStore

    store = DataStore()
    target = pd.DatetimeIndex(pd.to_datetime(target_index))
    if getattr(target, "tz", None) is not None:
        target = target.tz_convert("UTC").tz_localize(None)
    frames = []

    for ticker in CONTEXT_TICKERS:
        bars = store.load_bars(ticker, "1d")
        if bars is None or bars.empty or "Close" not in bars.columns:
            continue
        b = bars.copy()
        b.index = pd.DatetimeIndex(pd.to_datetime(b.index))
        if getattr(b.index, "tz", None) is not None:
            b.index = b.index.tz_convert("UTC").tz_localize(None)
        close = b["Close"].astype(float).sort_index()
        key = ticker.lower().replace("^", "").replace("-", "_").replace(".", "_")
        df = pd.DataFrame(index=close.index)
        if ticker in ("SPY", "QQQ"):
            df[f"{key}_ret1"] = np.log(close / close.shift(1))
            df[f"{key}_ret5"] = np.log(close / close.shift(5))
            df[f"{key}_trend20"] = close / close.rolling(20).mean() - 1
        elif ticker == "^VIX":
            df["vix_level"] = close / 100.0
            df["vix_chg5"] = close.diff(5) / 100.0
        else:
            df[f"{key}_ret5"] = np.log(close / close.shift(5))
            df[f"{key}_trend20"] = close / close.rolling(20).mean() - 1
        frames.append(df.shift(1))

    if not frames:
        return pd.DataFrame(index=target)
    daily = pd.concat(frames, axis=1).sort_index()
    aligned = daily.reindex(target, method="ffill")
    aligned.index = target_index
    return aligned


def forward_return_target(close: pd.Series, forecast_horizon: int = 1) -> pd.Series:
    """Forward log-return over the horizon; positive = up (label = 1)."""
    return np.log(close.shift(-forecast_horizon) / close)


def latest_vol_pctile(bars: pd.DataFrame) -> float:
    """Most recent realized-vol percentile - the no-trade regime gate input."""
    p = volatility.vol_percentile(bars["Close"].astype(float), 20, lookback=120)
    p = p.dropna()
    return float(p.iloc[-1]) if len(p) else 0.5
