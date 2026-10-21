"""
Causal BTC regime labels and diagnostics.

Regime labels use only information available before the bar starts: prior
close vs prior moving average, prior trailing return, and prior realized-vol
percentile. This keeps regime reporting explanatory without sneaking future
data into the decision process.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.features.volatility import vol_percentile
from quant.metrics.performance import max_drawdown


def classify_btc_regimes(bars: pd.DataFrame,
                         trend_window: int = 120,
                         return_window: int = 60,
                         vol_window: int = 20,
                         vol_lookback: int = 120) -> pd.DataFrame:
    close = bars["Close"].astype(float)
    prior_close = close.shift(1)
    prior_sma = close.rolling(trend_window, min_periods=max(20, trend_window // 3)).mean().shift(1)
    prior_ret = np.log(close.shift(1) / close.shift(return_window + 1))
    prior_vol_pct = vol_percentile(close, vol_window, vol_lookback).shift(1)

    trend = pd.Series("sideways", index=bars.index, dtype="object")
    trend[(prior_close > prior_sma) & (prior_ret > 0)] = "bull"
    trend[(prior_close < prior_sma) & (prior_ret < 0)] = "bear"

    vol = pd.Series("mid_vol", index=bars.index, dtype="object")
    vol[prior_vol_pct >= 0.67] = "high_vol"
    vol[prior_vol_pct <= 0.33] = "low_vol"

    return pd.DataFrame({
        "trend_regime": trend,
        "vol_regime": vol,
        "prior_return": prior_ret,
        "prior_vol_pctile": prior_vol_pct,
    }, index=bars.index)


def _trade_exposure_bars(trades: pd.DataFrame, regime_index: pd.Index,
                         mask: pd.Series) -> int:
    if trades is None or trades.empty:
        return 0
    exposure = 0
    active_dates = set(regime_index[mask.reindex(regime_index).fillna(False)])
    for _, row in trades.iterrows():
        dates = regime_index[(regime_index >= row["entry_date"]) &
                             (regime_index <= row["exit_date"])]
        exposure += sum(d in active_dates for d in dates)
    return int(exposure)


def _summarize_bucket(name: str, mask: pd.Series, bars: pd.DataFrame,
                      regimes: pd.DataFrame, trades: pd.DataFrame,
                      equity: pd.DataFrame) -> dict:
    mask = mask.reindex(bars.index).fillna(False)
    bucket_trades = pd.DataFrame()
    if trades is not None and not trades.empty:
        entry_regime = regimes.reindex(pd.to_datetime(trades["entry_date"]))
        if name in ("bull", "bear", "sideways"):
            keep = entry_regime["trend_regime"].values == name
        else:
            keep = entry_regime["vol_regime"].values == name
        bucket_trades = trades.loc[keep].copy()

    eq_dd = 0.0
    if equity is not None and not equity.empty:
        eq = equity.copy()
        eq["date"] = pd.to_datetime(eq["date"])
        eq = eq[eq["date"].isin(set(bars.index[mask]))]
        if not eq.empty:
            eq_dd = float(max_drawdown(pd.Series(eq["equity"].values, index=eq["date"])) * 100)

    exposure_bars = _trade_exposure_bars(trades, bars.index, mask)
    return {
        "bars": int(mask.sum()),
        "trades": int(len(bucket_trades)),
        "net_expectancy_usd": float(bucket_trades["pnl"].mean()) if not bucket_trades.empty else 0.0,
        "win_rate_pct": float((bucket_trades["pnl"] > 0).mean() * 100) if not bucket_trades.empty else 0.0,
        "net_pnl_usd": float(bucket_trades["pnl"].sum()) if not bucket_trades.empty else 0.0,
        "max_drawdown_pct": eq_dd,
        "exposure_pct": float(exposure_bars / mask.sum() * 100) if mask.sum() else 0.0,
        "auc": None,
        "lift": None,
    }


def regime_breakdown(bars: pd.DataFrame, trades_records: list,
                     equity_records: list) -> dict:
    bars = bars.sort_index()
    regimes = classify_btc_regimes(bars)
    trades = pd.DataFrame(trades_records)
    equity = pd.DataFrame(equity_records)
    if not trades.empty:
        trades["entry_date"] = pd.to_datetime(trades["entry_date"])
        trades["exit_date"] = pd.to_datetime(trades["exit_date"])

    out = {}
    for name in ("bull", "bear", "sideways"):
        out[name] = _summarize_bucket(
            name, regimes["trend_regime"] == name, bars, regimes, trades, equity)
    for name in ("high_vol", "low_vol"):
        out[name] = _summarize_bucket(
            name, regimes["vol_regime"] == name, bars, regimes, trades, equity)
    out["definition"] = (
        "Causal labels: prior close vs prior 120-bar SMA plus prior 60-bar "
        "return for bull/bear/sideways; prior 20-bar realized-vol percentile "
        "over 120 bars for high_vol/low_vol. Trades are attributed by entry bar."
    )
    return out
