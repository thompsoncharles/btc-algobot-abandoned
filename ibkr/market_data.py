"""
IBKR market data: batch quotes and market-hours checks.

Uses ib.reqTickers() for batch snapshots - one round trip for many symbols,
which auto-cancels the market data line afterwards - instead of the old
Schwab client's one-REST-call-per-symbol approach. This matters because the
rolling strategy can price up to PRT_MAX_TICKERS symbols per cycle; a
snapshot batch avoids exhausting the account's concurrent streaming-line
limit the way persistent per-symbol subscriptions would.
"""
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pytz
from loguru import logger
from ib_async import Stock, Ticker

from ibkr.connection import IBConnection

REFERENCE_SYMBOL = "SPY"  # used for market-hours lookups


class MarketDataClient:
    """Reads IBKR market data."""

    def __init__(self, conn: IBConnection):
        self.conn = conn
        self.ib = conn.ib

    def get_quotes_batch(self, symbols: List[str]) -> Dict[str, Ticker]:
        """
        One-shot snapshot for many symbols in a single round trip.
        """
        if not symbols:
            return {}
        contracts = [Stock(s.upper(), "SMART", "USD") for s in symbols]
        self.ib.qualifyContracts(*contracts)
        tickers = self.ib.reqTickers(*contracts)
        return {t.contract.symbol: t for t in tickers}

    def get_quote(self, symbol: str) -> Optional[float]:
        """Current price for a single symbol (last, falling back to close/bid/ask)."""
        quotes = self.get_quotes_batch([symbol])
        ticker = quotes.get(symbol.upper())
        if ticker is None:
            logger.warning(f"Could not get quote for {symbol}")
            return None
        price = self._price_from_ticker(ticker)
        if price is None:
            logger.warning(f"No price data available for {symbol}")
        return price

    @staticmethod
    def _price_from_ticker(ticker: Ticker) -> Optional[float]:
        for price in (ticker.last, ticker.close, ticker.bid, ticker.ask):
            if price is not None and not math.isnan(price) and price > 0:
                return float(price)
        return None

    def is_market_open(self, delay_minutes: int = 0) -> Tuple[bool, str]:
        """
        Uses the SPY contract's liquidHours (IBKR's own contract details,
        which already account for holidays/half-days) instead of a separate
        market-hours API call the way Schwab's client did.
        """
        try:
            contract = Stock(REFERENCE_SYMBOL, "SMART", "USD")
            self.ib.qualifyContracts(contract)
            details_list = self.ib.reqContractDetails(contract)
            if not details_list:
                return False, "Could not retrieve contract details for market hours"
            details = details_list[0]

            try:
                tz = pytz.timezone(details.timeZoneId)
            except Exception:
                tz = pytz.timezone("US/Eastern")
            now = datetime.now(tz)
            today_str = now.strftime("%Y%m%d")

            # liquidHours format: "20240102:0930-20240102:1600;20240103:CLOSED;..."
            for day_segment in details.liquidHours.split(";"):
                day_segment = day_segment.strip()
                if not day_segment or "CLOSED" in day_segment:
                    continue

                start_part, sep, end_part = day_segment.partition("-")
                if not sep:
                    continue

                start_date, _, start_time = start_part.partition(":")
                end_date, _, end_time = end_part.partition(":")
                if start_date != today_str:
                    continue

                open_time = tz.localize(datetime.strptime(f"{start_date}{start_time}", "%Y%m%d%H%M"))
                close_time = tz.localize(datetime.strptime(f"{end_date}{end_time}", "%Y%m%d%H%M"))

                if now < open_time:
                    minutes_until = (open_time - now).total_seconds() / 60
                    return False, f"Market opens at {open_time.strftime('%H:%M')} ET (opens in {int(minutes_until)} minutes)"
                if now >= close_time:
                    return False, "Market is currently closed"

                minutes_open = (now - open_time).total_seconds() / 60
                if minutes_open >= delay_minutes:
                    return True, f"Market is open (open for {int(minutes_open)} minutes)"
                return False, f"Market is open but has only been open for {int(minutes_open)} minutes (need {delay_minutes})"

            return False, "Market is closed today (no session found)"

        except Exception as e:
            logger.error(f"Error checking market hours: {e}")
            return False, f"Error checking market hours: {str(e)}"
