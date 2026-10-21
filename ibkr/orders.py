"""
IBKR order management: placing, cancelling, and reading status of orders.

Order status is read directly off the Trade object's orderStatus - ib_async
keeps this live via the event loop, so there is no REST poll here the way
there was with the old Schwab OrderManager.get_order_status().
"""
from loguru import logger
from ib_async import Stock, MarketOrder, LimitOrder, StopOrder, Trade

from ibkr.connection import IBConnection


class OrderManager:
    """Places and manages IBKR orders."""

    def __init__(self, conn: IBConnection):
        self.conn = conn
        self.ib = conn.ib

    def _qualified_stock(self, symbol: str) -> Stock:
        contract = Stock(symbol.upper(), "SMART", "USD")
        self.ib.qualifyContracts(contract)
        return contract

    @staticmethod
    def _round_price(price: float) -> float:
        return round(price, 4) if price < 1.0 else round(price, 2)

    def create_market_order(self, symbol: str, action: str, quantity: int) -> Trade:
        """action: 'BUY' or 'SELL'."""
        contract = self._qualified_stock(symbol)
        order = MarketOrder(action.upper(), int(quantity))
        trade = self.ib.placeOrder(contract, order)
        logger.info(f"Market order placed: {action.upper()} {quantity} {symbol}")
        return trade

    def create_limit_order(self, symbol: str, action: str, quantity: int, price: float) -> Trade:
        """action: 'BUY' or 'SELL'."""
        contract = self._qualified_stock(symbol)
        limit_price = self._round_price(price)
        order = LimitOrder(action.upper(), int(quantity), limit_price)
        trade = self.ib.placeOrder(contract, order)
        logger.info(f"Limit order placed: {action.upper()} {quantity} {symbol} @ ${limit_price:.4f}")
        return trade

    def create_stop_loss_order(self, symbol: str, action: str, quantity: int,
                                entry_price: float, stop_loss_percent: float) -> Trade:
        """
        Protective stop. action: 'SELL' to protect a LONG position,
        'BUY' to protect a SHORT position.
        """
        action_upper = action.upper()
        if action_upper == "SELL":
            stop_price = entry_price * (1 - stop_loss_percent / 100)
        elif action_upper == "BUY":
            stop_price = entry_price * (1 + stop_loss_percent / 100)
        else:
            raise ValueError(f"Invalid action for stop loss: {action}")

        stop_price = self._round_price(stop_price)

        contract = self._qualified_stock(symbol)
        order = StopOrder(action_upper, int(quantity), stop_price)
        trade = self.ib.placeOrder(contract, order)
        logger.info(
            f"Stop loss order placed: {action_upper} {quantity} {symbol} @ ${stop_price:.4f} stop "
            f"(entry: ${entry_price:.4f}, {stop_loss_percent}% loss)"
        )
        return trade

    def cancel_order(self, trade: Trade) -> bool:
        try:
            self.ib.cancelOrder(trade.order)
            logger.info(f"Cancel requested for order {trade.order.orderId} ({trade.contract.symbol})")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order {trade.order.orderId}: {e}")
            return False

    def get_order_status(self, trade: Trade) -> str:
        """Live status - no network call, ib_async keeps trade.orderStatus current."""
        return trade.orderStatus.status
