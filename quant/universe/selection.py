"""
Universe selection: liquidity, price, market cap, exchange, average daily
dollar volume, spread, borrow availability (stub - no free locate feed).

Candidate tickers come from anywhere (S&P 500 list, Godel MOST output);
this module decides which are tradeable.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from quant import config
from quant.data.fundamentals import Fundamentals
from quant.data.store import DataStore
from constants.blacklist import filter_blacklisted_tickers


@dataclass
class UniverseReport:
    survivors: List[str] = field(default_factory=list)
    rejected: Dict[str, str] = field(default_factory=dict)  # ticker -> reason

    def summary(self) -> str:
        return f"{len(self.survivors)} survivors, {len(self.rejected)} rejected"


class UniverseSelector:
    def __init__(self, store: Optional[DataStore] = None):
        self.store = store or DataStore()
        self.fundamentals = Fundamentals(self.store)

    def select(self, candidates: List[str], lookback_days: int = 20,
               check_market_cap: bool = False,
               quotes: Optional[Dict[str, "object"]] = None) -> UniverseReport:
        """
        Filter candidates using cached daily bars (run pipeline ingest first).

        Args:
            candidates: raw ticker list
            lookback_days: window for ADDV / liquidity averages
            check_market_cap: hit yfinance per-ticker for market cap (slow)
            quotes: optional {ticker: quant.data.quotes.Quote} for spread filter
        """
        report = UniverseReport()
        candidates, blacklisted = filter_blacklisted_tickers([c.upper() for c in candidates])
        for ticker in blacklisted:
            report.rejected[ticker] = "commission blacklist"

        for ticker in candidates:
            bars = self.store.load_bars(ticker, "1d")
            if bars is None or len(bars) < lookback_days:
                report.rejected[ticker] = "insufficient history"
                continue
            recent = bars.tail(lookback_days)
            price = float(recent["Close"].iloc[-1])
            addv = float((recent["Close"] * recent["Volume"]).mean())

            if price < config.UNIVERSE_MIN_PRICE:
                report.rejected[ticker] = f"price {price:.2f} < {config.UNIVERSE_MIN_PRICE}"
                continue
            if addv < config.UNIVERSE_MIN_ADDV_USD:
                report.rejected[ticker] = f"ADDV ${addv:,.0f} too low"
                continue
            if (recent["Volume"] <= 0).mean() > 0.2:
                report.rejected[ticker] = "illiquid (frequent zero-volume days)"
                continue

            if quotes and ticker in quotes:
                spread = quotes[ticker].spread_bps
                if spread > config.UNIVERSE_MAX_SPREAD_BPS:
                    report.rejected[ticker] = f"spread {spread:.1f}bps too wide"
                    continue

            if check_market_cap:
                info = self.fundamentals.snapshot_info(ticker)
                mcap = info.get("market_cap")
                if mcap and mcap < config.UNIVERSE_MIN_MARKET_CAP:
                    report.rejected[ticker] = f"mcap ${mcap:,.0f} too small"
                    continue
                exch = info.get("exchange")
                if exch and exch not in config.UNIVERSE_ALLOWED_EXCHANGES:
                    report.rejected[ticker] = f"exchange {exch} not allowed"
                    continue

            report.survivors.append(ticker)

        logger.info("universe: {}", report.summary())
        return report

    @staticmethod
    def borrow_availability(ticker: str):
        """Shortable shares / borrow rate. IBKR exposes this via the FTP
        shortable-stocks file or reqSecDefOptParams; stubbed until shorting
        with size is actually needed."""
        raise NotImplementedError("Borrow/locate feed not wired; stubbed.")
