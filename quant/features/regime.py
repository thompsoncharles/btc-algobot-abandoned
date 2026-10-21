"""
Market regime features: VIX, trend regime, sector strength, breadth,
correlations. Also exposes the SPY PRT snapshot history from the Godel
database as a regime input.
"""
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from quant.data.market_data import MarketData
from quant.data.store import DataStore

SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]


class RegimeFeatures:
    def __init__(self, store: Optional[DataStore] = None):
        self.store = store or DataStore()
        self.md = MarketData(self.store)

    def refresh(self) -> None:
        """Pull VIX, SPY and sector ETF daily history into the store."""
        self.md.fetch_history(["^VIX", "SPY"] + SECTOR_ETFS, interval="1d")

    def vix(self) -> Optional[pd.DataFrame]:
        bars = self.store.load_bars("^VIX", "1d")
        if bars is None:
            return None
        out = pd.DataFrame(index=bars.index)
        out["vix"] = bars["Close"]
        out["vix_percentile"] = bars["Close"].rolling(252).rank(pct=True)
        return out

    def trend_regime(self, ticker: str = "SPY") -> Optional[pd.Series]:
        """+1 above rising SMA200, -1 below falling SMA200, 0 otherwise."""
        bars = self.store.load_bars(ticker, "1d")
        if bars is None or len(bars) < 210:
            return None
        close = bars["Close"]
        sma200 = close.rolling(200).mean()
        slope = sma200.diff(20)
        regime = pd.Series(0, index=close.index)
        regime[(close > sma200) & (slope > 0)] = 1
        regime[(close < sma200) & (slope < 0)] = -1
        return regime

    def sector_strength(self, window: int = 20) -> pd.DataFrame:
        """Sector ETF returns relative to SPY over `window` days."""
        closes = self.store.load_universe_bars(SECTOR_ETFS + ["SPY"], "1d")
        if closes.empty:
            return pd.DataFrame()
        rets = closes.pct_change(window)
        rel = rets[SECTOR_ETFS].sub(rets["SPY"], axis=0)
        return rel

    def breadth(self, tickers: List[str]) -> pd.DataFrame:
        """% of universe above its 50d and 200d SMA."""
        closes = self.store.load_universe_bars(tickers, "1d")
        if closes.empty:
            return pd.DataFrame()
        out = pd.DataFrame(index=closes.index)
        out["pct_above_50d"] = (closes > closes.rolling(50).mean()).mean(axis=1)
        out["pct_above_200d"] = (closes > closes.rolling(200).mean()).mean(axis=1)
        return out

    def correlation_matrix(self, tickers: List[str], window: int = 60) -> pd.DataFrame:
        closes = self.store.load_universe_bars(tickers, "1d")
        if closes.empty:
            return pd.DataFrame()
        return closes.pct_change().tail(window).corr()

    def godel_spy_regime(self, database_path: Path) -> Optional[pd.DataFrame]:
        """SPY prob_up history from Godel PRT as a regime signal."""
        from quant.data.godel_signals import spy_snapshots
        return spy_snapshots(database_path)
