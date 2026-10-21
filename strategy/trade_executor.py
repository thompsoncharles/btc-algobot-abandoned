"""
Trade execution module for algobot_v2.

Places trades and confirms fills against IBKR's live-updating Trade /
orderStatus objects - ib_async keeps these current via a background event
loop, so confirming a fill is a local attribute read, not a REST poll like
the old Schwab client used. Waiting between fill checks uses
IBConnection.sleep() (which pumps that event loop) instead of a plain
time.sleep() that would starve it.
"""
import math
from typing import List, Dict, Optional

from loguru import logger
from ib_async import Trade as IBTrade

from ibkr.connection import IBConnection
from ibkr.account import AccountClient
from ibkr.market_data import MarketDataClient
from ibkr.orders import OrderManager
from constants.parameters import USE_LIMIT_ORDERS

DONE_STATUSES = {'Filled', 'Cancelled', 'ApiCancelled', 'Inactive'}


def _valid_price(price: Optional[float]) -> bool:
    return price is not None and not math.isnan(price) and price > 0


class TradeExecutor:
    """
    Executes trades using limit orders with optimal pricing.
    Uses bid/ask (or last as a fallback) for limit order pricing.
    """

    FILL_CHECK_INTERVAL_SECONDS = 2
    MAX_FILL_CHECKS = 5

    def __init__(self, conn: IBConnection, account_client: AccountClient,
                 market_data_client: MarketDataClient, order_manager: OrderManager):
        """
        Initialize trade executor.

        Args:
            conn: Shared IBKR connection (used to pump the event loop while waiting on fills)
            account_client: IBKR account client
            market_data_client: IBKR market data client
            order_manager: Order manager instance
        """
        self.conn = conn
        self.account_client = account_client
        self.market_data_client = market_data_client
        self.order_manager = order_manager
        logger.info("TradeExecutor initialized")

    def get_bid_ask_prices(self, ticker: str) -> Optional[Dict[str, float]]:
        """
        Get bid and ask prices for a ticker, falling back to last/close if
        the market is thin.

        Returns:
            dict: {'bid': float, 'ask': float} or None if unavailable
        """
        quotes = self.market_data_client.get_quotes_batch([ticker])
        t = quotes.get(ticker.upper())
        if t is None:
            logger.warning(f"Could not get quote for {ticker}")
            return None

        fallback = t.last if _valid_price(t.last) else (t.close if _valid_price(t.close) else None)
        bid = t.bid if _valid_price(t.bid) else fallback
        ask = t.ask if _valid_price(t.ask) else fallback

        if not _valid_price(bid) or not _valid_price(ask):
            logger.warning(f"No price data available for {ticker}")
            return None

        return {'bid': float(bid), 'ask': float(ask)}

    def get_limit_price(self, ticker: str, direction: str) -> Optional[float]:
        """
        Get optimal limit price for a trade.

        Args:
            ticker: Stock ticker symbol
            direction: 'LONG' or 'SHORT'

        Returns:
            float: Limit price, or None if unavailable
        """
        prices = self.get_bid_ask_prices(ticker)
        if not prices:
            return None

        direction_upper = direction.upper()
        if direction_upper == 'LONG':
            # Buying: use bid (or last/close fallback already baked into get_bid_ask_prices)
            limit_price = prices['bid']
        elif direction_upper == 'SHORT':
            # Selling short: use ask
            limit_price = prices['ask']
        else:
            logger.error(f"Invalid direction: {direction}")
            return None

        logger.debug(f"{direction_upper} {ticker}: using limit price ${limit_price:.2f}")
        return limit_price

    def execute_trade(self, sized_trade: Dict) -> Dict:
        """
        Place a single trade. Uses a limit order at the optimal price unless
        USE_LIMIT_ORDERS is disabled or no price is available, in which case
        it falls back to a market order.

        Args:
            sized_trade: Dict with 'ticker', 'direction', 'shares', 'price' keys

        Returns:
            dict: Execution result with 'success', 'ib_trade', 'ticker', 'direction', 'shares', 'price'
        """
        ticker = sized_trade.get('ticker', '').upper()
        direction = sized_trade.get('direction', '').upper()
        shares = sized_trade.get('shares', 0)
        expected_price = sized_trade.get('price', 0.0)

        if not ticker:
            logger.error("Trade missing ticker")
            return {'success': False, 'error': 'Missing ticker'}

        if direction not in ['LONG', 'SHORT']:
            logger.error(f"Invalid direction: {direction}")
            return {'success': False, 'error': f'Invalid direction: {direction}'}

        if shares <= 0:
            logger.error(f"Invalid share count: {shares}")
            return {'success': False, 'error': f'Invalid shares: {shares}'}

        # IBKR has no separate SELL_SHORT/BUY_TO_COVER instruction - a plain
        # SELL on a symbol with no existing long position opens a short.
        action = 'BUY' if direction == 'LONG' else 'SELL'
        limit_price = self.get_limit_price(ticker, direction)

        logger.info(f"Executing {direction} {ticker}: {shares} shares (action={action})")

        try:
            if USE_LIMIT_ORDERS and limit_price is not None:
                ib_trade = self.order_manager.create_limit_order(ticker, action, shares, limit_price)
            else:
                if limit_price is None:
                    logger.warning(f"Could not get limit price for {ticker}, using market order")
                ib_trade = self.order_manager.create_market_order(ticker, action, shares)

            logger.info(f"✓ Order placed: {direction} {ticker} {shares} shares (Order ID: {ib_trade.order.orderId})")
            return {
                'success': True,
                'ib_trade': ib_trade,
                'ticker': ticker,
                'direction': direction,
                'shares': shares,
                'price': limit_price if limit_price is not None else expected_price,
                'expected_price': expected_price,
            }

        except Exception as e:
            logger.error(f"Error executing trade for {ticker}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'ticker': ticker,
                'direction': direction
            }

    def execute_trades(self, sized_trades: List[Dict]) -> List[Dict]:
        """
        Execute multiple trades, confirm fills, and retry unfilled limit
        orders at a fresh price, falling back to a market order after
        MAX_FILL_CHECKS attempts.

        Args:
            sized_trades: List of sized trade dicts with 'ticker', 'direction', 'shares', 'price'

        Returns:
            list: List of execution results
        """
        logger.info("=" * 60)
        logger.info("Trade Execution")
        logger.info("=" * 60)
        logger.info(f"Executing {len(sized_trades)} trade(s)")

        execution_results = []
        pending = []

        # Phase 1: place all orders
        for i, sized_trade in enumerate(sized_trades, 1):
            ticker = sized_trade.get('ticker', 'Unknown')
            logger.info(f"[{i}/{len(sized_trades)}] Executing trade for {ticker}...")
            result = self.execute_trade(sized_trade)
            execution_results.append(result)
            if result.get('success') and result.get('ib_trade') is not None:
                pending.append(result)

        # Phase 2: confirm fills off live orderStatus, retry at a fresh price,
        # fall back to a market order on the last attempt
        for attempt in range(1, self.MAX_FILL_CHECKS + 1):
            if not pending:
                break

            self.conn.sleep(self.FILL_CHECK_INTERVAL_SECONDS)

            still_pending = []
            for result in pending:
                ib_trade: IBTrade = result['ib_trade']
                status = self.order_manager.get_order_status(ib_trade)

                if status == 'Filled':
                    result['filled'] = True
                    fill_price = ib_trade.orderStatus.avgFillPrice
                    if fill_price:
                        result['price'] = float(fill_price)
                    logger.info(f"✓ Order filled for {result['ticker']} (Order ID: {ib_trade.order.orderId})")
                else:
                    if status in ('Cancelled', 'ApiCancelled', 'Inactive'):
                        logger.warning(f"Order {status} for {result['ticker']}, will retry")
                    still_pending.append(result)

            if not still_pending:
                pending = []
                break

            if attempt == self.MAX_FILL_CHECKS:
                # Last attempt: cancel whatever is left and use a market order
                for result in still_pending:
                    ib_trade = result['ib_trade']
                    status = self.order_manager.get_order_status(ib_trade)
                    if status == 'Filled':
                        result['filled'] = True
                        continue
                    ticker = result['ticker']
                    logger.warning(f"Order for {ticker} not filled after {attempt} checks (status: {status}), using market order")
                    self.order_manager.cancel_order(ib_trade)
                    action = 'BUY' if result['direction'] == 'LONG' else 'SELL'
                    result['ib_trade'] = self.order_manager.create_market_order(ticker, action, result['shares'])

                self.conn.sleep(self.FILL_CHECK_INTERVAL_SECONDS)
                for result in still_pending:
                    if result.get('filled'):
                        continue
                    ib_trade = result['ib_trade']
                    status = self.order_manager.get_order_status(ib_trade)
                    if status == 'Filled':
                        result['filled'] = True
                        fill_price = ib_trade.orderStatus.avgFillPrice
                        if fill_price:
                            result['price'] = float(fill_price)
                        logger.info(f"✓ Market order filled for {result['ticker']}")
                    else:
                        logger.error(f"✗ Market order for {result['ticker']} still not filled (status: {status})")
            else:
                # Retry unfilled/cancelled limit orders at a fresh price
                for result in still_pending:
                    ib_trade = result['ib_trade']
                    ticker = result['ticker']
                    direction = result['direction']
                    status = self.order_manager.get_order_status(ib_trade)
                    if status not in DONE_STATUSES:
                        self.order_manager.cancel_order(ib_trade)

                    new_price = self.get_limit_price(ticker, direction)
                    action = 'BUY' if direction == 'LONG' else 'SELL'
                    if new_price is None:
                        logger.warning(f"Could not get new price for {ticker}, using market order")
                        result['ib_trade'] = self.order_manager.create_market_order(ticker, action, result['shares'])
                    else:
                        logger.info(f"Retrying {ticker} (attempt {attempt + 1}) @ ${new_price:.2f}...")
                        result['ib_trade'] = self.order_manager.create_limit_order(ticker, action, result['shares'], new_price)

                pending = still_pending

        filled_count = sum(1 for r in execution_results if r.get('filled'))
        failed_count = len(execution_results) - filled_count

        logger.info("=" * 60)
        logger.info("Trade Execution Complete")
        logger.info(f"Filled: {filled_count}, Failed: {failed_count}")
        logger.info("=" * 60)

        return execution_results
