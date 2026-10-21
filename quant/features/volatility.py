"""Volatility features: historical, realized (Parkinson / Garman-Klass),
volatility percentile. Implied vol is stubbed (needs an options feed)."""
import numpy as np
import pandas as pd

from quant.config import TRADING_DAYS_PER_YEAR


def historical_vol(close: pd.Series, window: int = 20,
                   annualize: bool = True) -> pd.Series:
    """Close-to-close log-return volatility."""
    ret = np.log(close / close.shift(1))
    vol = ret.rolling(window).std()
    return vol * np.sqrt(TRADING_DAYS_PER_YEAR) if annualize else vol


def parkinson_vol(bars: pd.DataFrame, window: int = 20,
                  annualize: bool = True) -> pd.Series:
    """High-low range estimator - more efficient than close-to-close."""
    hl = np.log(bars["High"] / bars["Low"]) ** 2
    var = hl.rolling(window).mean() / (4 * np.log(2))
    vol = np.sqrt(var)
    return vol * np.sqrt(TRADING_DAYS_PER_YEAR) if annualize else vol


def garman_klass_vol(bars: pd.DataFrame, window: int = 20,
                     annualize: bool = True) -> pd.Series:
    hl = 0.5 * np.log(bars["High"] / bars["Low"]) ** 2
    co = (2 * np.log(2) - 1) * np.log(bars["Close"] / bars["Open"]) ** 2
    var = (hl - co).rolling(window).mean()
    vol = np.sqrt(var.clip(lower=0))
    return vol * np.sqrt(TRADING_DAYS_PER_YEAR) if annualize else vol


def vol_percentile(close: pd.Series, window: int = 20,
                   lookback: int = 252) -> pd.Series:
    """Rank of current vol within its trailing distribution, in [0, 1]."""
    vol = historical_vol(close, window, annualize=False)
    return vol.rolling(lookback).rank(pct=True)


def implied_vol(ticker: str):
    """IV / IV percentile. Needs an options chain feed (IBKR reqSecDefOptParams
    + option market data subscription, or a vendor). Stubbed."""
    raise NotImplementedError("Implied vol requires an options data feed; stubbed.")
