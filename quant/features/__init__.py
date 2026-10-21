"""Section 3 — Feature engineering.

All functions take an OHLCV DataFrame (columns Open/High/Low/Close/Volume,
DatetimeIndex) and return Series/DataFrames aligned to it. Everything is
vectorized NumPy/pandas - this is the "fast path" (no native code needed
at this data scale).
"""
import pandas as pd

from quant.features import technical, price_action, volatility


def assemble_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Standard single-name feature matrix from OHLCV bars."""
    f = pd.DataFrame(index=bars.index)
    close = bars["Close"]

    f["sma20_ratio"] = close / technical.sma(close, 20)
    f["ema9_ratio"] = close / technical.ema(close, 9)
    f["rsi14"] = technical.rsi(close, 14)
    macd_line, signal_line, hist = technical.macd(close)
    f["macd_hist"] = hist
    f["atr14_pct"] = technical.atr(bars, 14) / close
    upper, mid, lower = technical.bollinger(close)
    f["bb_pos"] = (close - lower) / (upper - lower)
    f["vwap_ratio"] = close / technical.vwap(bars)
    f["rvol"] = technical.relative_volume(bars["Volume"], 14)

    f["gap_pct"] = price_action.gap_size(bars)
    f["mom_5"] = price_action.momentum(close, 5)
    f["mom_20"] = price_action.momentum(close, 20)
    f["intraday_range"] = price_action.intraday_range(bars)
    f["overnight_ret"] = price_action.overnight_return(bars)
    f["zscore_20"] = price_action.mean_reversion_zscore(close, 20)

    f["hvol_20"] = volatility.historical_vol(close, 20)
    f["parkinson_20"] = volatility.parkinson_vol(bars, 20)
    f["vol_pctile"] = volatility.vol_percentile(close, 20)

    return f


__all__ = ["assemble_features", "technical", "price_action", "volatility"]
