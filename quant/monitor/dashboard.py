"""
Monitoring: real-time console dashboard + persisted snapshots.

Shows positions, P&L, exposure, model confidence, working orders, loop
latency, and recent errors. Complements monitoring/account_size.py (the
live bot's account logger); snapshots append to DuckDB for history.
"""
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from quant.data.store import DataStore


class Dashboard:
    def __init__(self, store: Optional[DataStore] = None):
        self.store = store or DataStore()
        self._loop_times: List[float] = []
        self._errors: List[str] = []
        self._last_tick: Optional[float] = None

    # ----------------------------------------------------- heartbeats
    def loop_start(self) -> None:
        self._last_tick = time.monotonic()

    def loop_end(self) -> None:
        if self._last_tick is not None:
            self._loop_times.append(time.monotonic() - self._last_tick)
            self._loop_times = self._loop_times[-100:]

    def record_error(self, msg: str) -> None:
        self._errors.append(f"{datetime.now():%H:%M:%S} {msg}")
        self._errors = self._errors[-20:]

    @property
    def avg_loop_latency_s(self) -> float:
        return sum(self._loop_times) / len(self._loop_times) if self._loop_times else 0.0

    # ------------------------------------------------------- rendering
    def render(self, ib, confidences: Optional[Dict[str, float]] = None,
               equity: Optional[float] = None) -> str:
        """Build the dashboard text from a live IB connection."""
        lines = ["=" * 72,
                 f" QUANT DASHBOARD  {datetime.now():%Y-%m-%d %H:%M:%S}",
                 "=" * 72]

        positions = [p for p in ib.positions() if p.position != 0]
        gross = 0.0
        lines.append(f"{'TICKER':<8}{'QTY':>8}{'AVG COST':>12}{'CONF':>8}")
        for p in positions:
            sym = p.contract.symbol
            notional = abs(p.position * p.avgCost)
            gross += notional
            conf = confidences.get(sym) if confidences else None
            lines.append(f"{sym:<8}{p.position:>8.0f}{p.avgCost:>12.2f}"
                         f"{conf if conf is not None else '-':>8}")
        if not positions:
            lines.append("(no open positions)")

        if equity:
            lines.append(f"\nEquity: ${equity:,.0f}   Gross exposure: ${gross:,.0f} "
                         f"({gross / equity * 100 if equity else 0:.1f}%)")

        open_trades = ib.openTrades()
        lines.append(f"\nWorking orders: {len(open_trades)}")
        for t in open_trades[:10]:
            lines.append(f"  {t.contract.symbol} {t.order.action} {t.order.totalQuantity}"
                         f" {t.order.orderType} [{t.orderStatus.status}]")

        lines.append(f"\nLoop latency (avg of {len(self._loop_times)}): "
                     f"{self.avg_loop_latency_s:.2f}s")
        if self._errors:
            lines.append("\nRecent errors:")
            lines.extend(f"  {e}" for e in self._errors[-5:])
        lines.append("=" * 72)
        return "\n".join(lines)

    def print(self, ib, **kwargs) -> None:
        print(self.render(ib, **kwargs))

    # ------------------------------------------------------- snapshots
    def snapshot(self, equity: float, gross_exposure: float,
                 n_positions: int, n_orders: int) -> None:
        """Persist one dashboard snapshot row for history."""
        row = pd.DataFrame([{
            "ts": datetime.now(timezone.utc).isoformat(),
            "equity": equity,
            "gross_exposure": gross_exposure,
            "n_positions": n_positions,
            "n_orders": n_orders,
            "avg_loop_latency_s": self.avg_loop_latency_s,
            "recent_errors": len(self._errors),
        }])
        try:
            self.store.append_rows("dashboard_snapshots", row)
        except Exception as e:
            logger.warning("snapshot persist failed: {}", e)
