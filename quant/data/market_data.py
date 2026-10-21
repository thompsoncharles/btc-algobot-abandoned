"""
Market data ingestion: daily / 5-minute / 1-minute OHLCV.

Sources:
  - yfinance for bulk screening history (free, no request cap)
  - IBKR historical bars for execution-grade data on the tickers we trade

Tick data and Level II order book require paid IBKR market-data
subscriptions and are exposed as explicit stubs (see quotes.py for L2).
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from quant import config
from quant.data.store import DataStore

_YF_INTERVAL_LIMITS = {
    # yfinance lookback caps per interval
    "1m": timedelta(days=7),
    "5m": timedelta(days=59),
    "1d": None,
}


def to_yahoo_symbol(ticker: str) -> str:
    """IBKR class shares use a space ('BRK B'); Yahoo uses a hyphen ('BRK-B')."""
    return ticker.strip().upper().replace(" ", "-").replace(".", "-")


def to_ibkr_symbol(ticker: str) -> str:
    return ticker.strip().upper().replace("-", " ")


class MarketData:
    """Fetch + cache OHLCV bars."""

    def __init__(self, store: Optional[DataStore] = None):
        self.store = store or DataStore()

    # -------------------------------------------------------- yfinance
    def fetch_history(self, tickers: List[str], interval: str = "1d",
                      years: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """Batch-download OHLCV via yfinance and cache to parquet."""
        import yfinance as yf

        limit = _YF_INTERVAL_LIMITS.get(interval)
        if limit is not None:
            start = datetime.now() - limit + timedelta(days=1)
        else:
            start = datetime.now() - timedelta(days=365 * (years or config.HISTORY_YEARS))

        yahoo_map = {to_yahoo_symbol(t): t for t in tickers}
        raw = yf.download(
            tickers=" ".join(yahoo_map),
            start=start.strftime("%Y-%m-%d"),
            interval=interval,
            group_by="ticker",
            auto_adjust=True,
            threads=5,
            progress=False,
        )
        out: Dict[str, pd.DataFrame] = {}
        for ysym, ticker in yahoo_map.items():
            try:
                bars = raw[ysym].dropna(how="all") if len(yahoo_map) > 1 else raw.dropna(how="all")
                if bars.empty:
                    continue
                bars = bars[["Open", "High", "Low", "Close", "Volume"]]
                self.store.save_bars(ticker, interval, bars)
                out[ticker] = bars
            except (KeyError, IndexError, ValueError) as e:
                logger.debug("no {} bars for {}: {}", interval, ticker, e)
        logger.info("fetched {} bars for {}/{} tickers", interval, len(out), len(tickers))
        return out

    # ------------------------------------------------------------ IBKR
    def fetch_ibkr_bars(self, ib, ticker: str, duration: str = "2 D",
                        bar_size: str = "5 mins", what: str = "TRADES") -> Optional[pd.DataFrame]:
        """
        Execution-grade bars from IBKR (requires live TWS connection).

        Args:
            ib: connected ib_async.IB instance (ibkr.connection.IBConnection.ib)
        """
        from ib_async import Stock, util

        contract = Stock(to_ibkr_symbol(ticker), "SMART", "USD")
        ib.qualifyContracts(contract)
        bars = ib.reqHistoricalData(
            contract, endDateTime="", durationStr=duration,
            barSizeSetting=bar_size, whatToShow=what, useRTH=True,
        )
        if not bars:
            return None
        df = util.df(bars).set_index("date")
        df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                                "close": "Close", "volume": "Volume"})
        return df[["Open", "High", "Low", "Close", "Volume"]]

    # ------------------------------------------------------------ stubs
    def fetch_ticks(self, ib, ticker: str):
        """Tick-by-tick data (TAQ). Requires IBKR tick-data subscription.

        Wire via ib.reqTickByTickData(contract, 'AllLast') when subscribed.
        """
        raise NotImplementedError(
            "Tick/TAQ data needs an IBKR tick-data subscription; "
            "enable it in Account Management, then implement via reqTickByTickData."
        )
