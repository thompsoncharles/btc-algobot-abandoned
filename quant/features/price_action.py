"""Price action: gap, momentum, intraday range, overnight return, mean reversion."""
import pandas as pd


def gap_size(bars: pd.DataFrame) -> pd.Series:
    """Open vs prior close, fractional."""
    return bars["Open"] / bars["Close"].shift(1) - 1


def momentum(close: pd.Series, window: int) -> pd.Series:
    return close / close.shift(window) - 1


def intraday_range(bars: pd.DataFrame) -> pd.Series:
    """(High - Low) / Open."""
    return (bars["High"] - bars["Low"]) / bars["Open"]


def overnight_return(bars: pd.DataFrame) -> pd.Series:
    """Prior close -> open."""
    return bars["Open"] / bars["Close"].shift(1) - 1


def close_position_in_range(bars: pd.DataFrame) -> pd.Series:
    """Where the close sits in the bar's range: 1 = at high, 0 = at low."""
    rng = (bars["High"] - bars["Low"]).replace(0, float("nan"))
    return (bars["Close"] - bars["Low"]) / rng


def mean_reversion_zscore(close: pd.Series, window: int = 20) -> pd.Series:
    """Z-score of price vs its rolling mean; extreme values = stretched."""
    mean = close.rolling(window).mean()
    std = close.rolling(window).std()
    return (close - mean) / std
