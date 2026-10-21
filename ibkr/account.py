"""
IBKR account operations: positions, account values, open orders.

All reads here are local memory access - ib_async keeps this state live via
a background reader thread, so there is no network round trip per call the
way there was with the old Schwab REST client.
"""
from typing import Dict, List

from ib_async import Position, Trade

from ibkr.connection import IBConnection


class AccountClient:
    """Reads account state that ib_async keeps synced in memory."""

    def __init__(self, conn: IBConnection):
        self.conn = conn
        self.ib = conn.ib

    def get_positions(self) -> List[Position]:
        """Current non-zero positions."""
        return self.ib.positions()

    def get_account_info(self) -> Dict[str, float]:
        """
        Flat dict of account value tags (NetLiquidation, BuyingPower,
        TotalCashValue, EquityWithLoanValue, ...) pulled from
        ib.accountSummary(). Unlike Schwab there is no nested
        securitiesAccount.currentBalances shape to unwrap.
        """
        info: Dict[str, float] = {}
        for av in self.ib.accountSummary():
            try:
                info[av.tag] = float(av.value)
            except (TypeError, ValueError):
                continue
        return info

    def get_all_open_orders(self) -> List[Trade]:
        """All open/working trades on the account."""
        return self.ib.openTrades()
