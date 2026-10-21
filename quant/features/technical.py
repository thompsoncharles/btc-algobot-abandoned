"""Technical indicators: SMA, EMA, RSI, MACD, ATR, Bollinger, VWAP, RVOL."""
from typing import Tuple

import numpy as np
import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window).mean()


def ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast: int = 12, slow: int = 26,
         signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def true_range(bars: pd.DataFrame) -> pd.Series:
    prev_close = bars["Close"].shift(1)
    return pd.concat([
        bars["High"] - bars["Low"],
        (bars["High"] - prev_close).abs(),
        (bars["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)


def atr(bars: pd.DataFrame, window: int = 14) -> pd.Series:
    return true_range(bars).ewm(alpha=1 / window, adjust=False).mean()


def bollinger(close: pd.Series, window: int = 20,
              num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(close, window)
    std = close.rolling(window).std()
    return mid + num_std * std, mid, mid - num_std * std


def vwap(bars: pd.DataFrame) -> pd.Series:
    """Session-anchored for intraday bars (resets each day); cumulative for daily."""
    typical = (bars["High"] + bars["Low"] + bars["Close"]) / 3
    pv = typical * bars["Volume"]
    if isinstance(bars.index, pd.DatetimeIndex) and bars.index.normalize().nunique() > 1 \
            and (bars.index.time != bars.index[0].time()).any():
        day = bars.index.normalize()
        return pv.groupby(day).cumsum() / bars["Volume"].groupby(day).cumsum()
    return pv.cumsum() / bars["Volume"].cumsum()


def relative_volume(volume: pd.Series, lookback: int = 14) -> pd.Series:
    """Today's volume vs trailing average (excludes current bar)."""
    avg = volume.shift(1).rolling(lookback).mean()
    return volume / avg
