"""
Quotes: bid/ask snapshots via IBKR; Level II order book stub.

Uses the existing ibkr.market_data.MarketDataClient where a shared
connection is available, or raw ib_async when handed an IB instance.
"""
from dataclasses import dataclass
from typing import Optional

from loguru import logger


@dataclass
class Quote:
    ticker: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    last: float

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.last

    @property
    def spread_bps(self) -> float:
        m = self.mid
        if m <= 0 or self.bid <= 0 or self.ask <= 0:
            return float("inf")
        return (self.ask - self.bid) / m * 10_000

    @property
    def order_imbalance(self) -> float:
        """(bid_size - ask_size) / (bid_size + ask_size), in [-1, 1]."""
        total = self.bid_size + self.ask_size
        if total <= 0:
            return 0.0
        return (self.bid_size - self.ask_size) / total


def snapshot(ib, ticker: str, timeout: float = 5.0) -> Optional[Quote]:
    """Snapshot NBBO from a connected ib_async.IB instance."""
    from ib_async import Stock

    contract = Stock(ticker.upper().replace("-", " "), "SMART", "USD")
    ib.qualifyContracts(contract)
    t = ib.reqMktData(contract, "", snapshot=True)
    ib.sleep(timeout)
    try:
        def _f(x):
            return float(x) if x == x and x is not None else 0.0  # NaN-safe
        q = Quote(
            ticker=ticker.upper(),
            bid=_f(t.bid), ask=_f(t.ask),
            bid_size=_f(t.bidSize), ask_size=_f(t.askSize),
            last=_f(t.last) or _f(t.close),
        )
        if q.mid <= 0:
            logger.warning("empty quote for {}", ticker)
            return None
        return q
    finally:
        ib.cancelMktData(contract)


def level2_book(ib, ticker: str, depth: int = 10):
    """Level II order book. Requires an IBKR depth-of-book subscription.

    Wire via ib.reqMktDepth(contract, numRows=depth) once subscribed.
    """
    raise NotImplementedError(
        "Level II depth needs an IBKR market-depth subscription; "
        "implement via reqMktDepth when enabled."
    )
