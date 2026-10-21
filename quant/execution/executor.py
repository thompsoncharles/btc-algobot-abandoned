"""
Smart execution on top of ibkr/orders.py.

House rule #8: limit orders first, always. This executor reprices a limit
toward the market a bounded number of times and only then escalates to a
market order - logging every escalation. Adds TWAP slicing for larger
orders, partial-fill accounting, and order-vs-position reconciliation.

SMART routing is already IBKR's smart order router (contract exchange
"SMART" in ibkr/orders.py); we don't route manually.
"""
import time
from dataclasses import dataclass, field
from typing import List, Optional

from loguru import logger

from constants.parameters import (
    LIMIT_ORDER_TIMEOUT_SECONDS,
    LIMIT_ORDER_MAX_ATTEMPTS,
    LIMIT_ORDER_PRICE_OFFSET_PERCENT,
    LIMIT_ORDER_ADJUSTMENT_PERCENT,
)
from quant import config
from quant.data.quotes import snapshot

_TERMINAL = {"Filled", "Cancelled", "ApiCancelled", "Inactive"}


@dataclass
class ExecutionReport:
    ticker: str
    side: str
    requested: int
    filled: int = 0
    avg_price: float = 0.0
    attempts: int = 0
    escalated_to_market: bool = False
    child_fills: List[dict] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return self.filled >= self.requested


class SmartExecutor:
    def __init__(self, conn, order_manager, journal=None):
        """
        Args:
            conn: ibkr.connection.IBConnection
            order_manager: ibkr.orders.OrderManager
            journal: optional quant.journal.JournalLogger for audit rows
        """
        self.conn = conn
        self.ib = conn.ib
        self.om = order_manager
        self.journal = journal

    # ------------------------------------------------------ primitives
    def _wait_terminal(self, trade, timeout: float) -> str:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.ib.sleep(0.25)
            if trade.orderStatus.status in _TERMINAL:
                break
        return trade.orderStatus.status

    def _log(self, event: str, **fields) -> None:
        if self.journal is not None:
            self.journal.log_order(event=event, **fields)

    # -------------------------------------------------- limit-then-market
    def execute(self, ticker: str, side: str, quantity: int) -> ExecutionReport:
        """
        Limit order walked toward the market up to LIMIT_ORDER_MAX_ATTEMPTS
        times, then (and only then) a market order for the unfilled balance.
        """
        side = side.upper()
        report = ExecutionReport(ticker=ticker.upper(), side=side, requested=quantity)
        remaining = quantity

        for attempt in range(1, LIMIT_ORDER_MAX_ATTEMPTS + 1):
            q = snapshot(self.ib, ticker)
            if q is None:
                logger.warning("no quote for {}; cannot price a limit", ticker)
                break
            offset = (LIMIT_ORDER_PRICE_OFFSET_PERCENT +
                      LIMIT_ORDER_ADJUSTMENT_PERCENT * (attempt - 1)) / 100
            price = q.mid * (1 + offset) if side == "BUY" else q.mid * (1 - offset)

            trade = self.om.create_limit_order(ticker, side, remaining, price)
            report.attempts = attempt
            status = self._wait_terminal(trade, LIMIT_ORDER_TIMEOUT_SECONDS)
            filled_now = int(trade.orderStatus.filled or 0)
            if filled_now:
                report.child_fills.append({
                    "type": "limit", "qty": filled_now,
                    "price": float(trade.orderStatus.avgFillPrice or price),
                })
            remaining -= filled_now

            if status != "Filled" and status not in _TERMINAL:
                self.ib.cancelOrder(trade.order)
                self._wait_terminal(trade, 5)
                # count late fills that landed during cancellation
                late = int(trade.orderStatus.filled or 0) - filled_now
                if late > 0:
                    remaining -= late
                    report.child_fills.append({
                        "type": "limit_late", "qty": late,
                        "price": float(trade.orderStatus.avgFillPrice or price),
                    })
            self._log("limit_attempt", ticker=ticker, side=side, attempt=attempt,
                      status=status, remaining=remaining)
            if remaining <= 0:
                break

        if remaining > 0:
            logger.warning("{} {} limit attempts exhausted; market order for {} "
                           "(logged per house rules)", ticker, side, remaining)
            trade = self.om.create_market_order(ticker, side, remaining)
            status = self._wait_terminal(trade, 30)
            filled_now = int(trade.orderStatus.filled or 0)
            if filled_now:
                report.child_fills.append({
                    "type": "market", "qty": filled_now,
                    "price": float(trade.orderStatus.avgFillPrice or 0.0),
                })
            remaining -= filled_now
            report.escalated_to_market = True
            self._log("market_escalation", ticker=ticker, side=side,
                      status=status, remaining=remaining)

        report.filled = quantity - max(remaining, 0)
        if report.child_fills:
            notional = sum(f["qty"] * f["price"] for f in report.child_fills)
            report.avg_price = notional / report.filled if report.filled else 0.0
        return report

    # ------------------------------------------------------------ TWAP
    def execute_twap(self, ticker: str, side: str, quantity: int,
                     slices: Optional[int] = None,
                     interval_s: Optional[int] = None) -> ExecutionReport:
        """Time-sliced execution: quantity split into equal child orders."""
        n = slices or config.TWAP_SLICES
        wait = interval_s or config.TWAP_INTERVAL_SECONDS
        child = max(1, quantity // n)
        total = ExecutionReport(ticker=ticker.upper(), side=side.upper(), requested=quantity)
        remaining = quantity
        while remaining > 0:
            qty = min(child, remaining)
            rep = self.execute(ticker, side, qty)
            total.child_fills.extend(rep.child_fills)
            total.filled += rep.filled
            total.attempts += rep.attempts
            total.escalated_to_market |= rep.escalated_to_market
            remaining -= qty
            if remaining > 0:
                self.ib.sleep(wait)
        if total.child_fills and total.filled:
            notional = sum(f["qty"] * f["price"] for f in total.child_fills)
            total.avg_price = notional / total.filled
        return total

    # -------------------------------------------------- reconciliation
    def reconcile(self) -> List[str]:
        """
        Compare IBKR positions against open orders; report orphans -
        positions with no protective stop working (violates house rule #8).
        """
        problems: List[str] = []
        positions = {p.contract.symbol: p.position for p in self.ib.positions()
                     if p.position != 0}
        open_orders = self.ib.openTrades()
        protected = {t.contract.symbol for t in open_orders
                     if t.order.orderType in ("STP", "STP LMT")}
        for sym, qty in positions.items():
            if sym not in protected:
                problems.append(f"{sym}: position {qty} has NO working stop")
        for t in open_orders:
            if t.contract.symbol not in positions and \
                    t.order.orderType in ("STP", "STP LMT"):
                problems.append(f"{t.contract.symbol}: stop order with no position")
        for p in problems:
            logger.error("RECONCILE: {}", p)
        self._log("reconcile", problems=problems)
        return problems


def vwap_execution(*args, **kwargs):
    """True VWAP execution needs intraday volume-curve data and child-order
    scheduling against it; IBKR also offers native VWAP algo orders
    (algoStrategy='Vwap') which is the practical route. Stubbed."""
    raise NotImplementedError(
        "Use IBKR native VWAP algo (Order.algoStrategy='Vwap') when needed; stubbed."
    )
