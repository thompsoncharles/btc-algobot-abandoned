"""
Emergency Position Closer - Closes all open positions immediately.
Run this script if something goes wrong with the main algo.

This script:
1. Cancels all open orders (stop losses, etc.)
2. Closes all positions with market orders (emergency use only)

Follows RULES.md: Use limit orders when possible, but this is emergency use.
"""
import sys
from pathlib import Path
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ibkr.connection import IBConnection
from ibkr.account import AccountClient
from ibkr.orders import OrderManager


def get_open_orders(account_client: AccountClient) -> list:
    """
    Get all open/pending orders.

    Args:
        account_client: Account client instance

    Returns:
        list: List of open Trade objects
    """
    return account_client.get_all_open_orders()


def get_positions(account_client: AccountClient) -> list:
    """
    Get all open positions.

    Args:
        account_client: Account client instance

    Returns:
        list: List of Position objects
    """
    return account_client.get_positions()


def close_position(order_manager: OrderManager, symbol: str, quantity: float, is_long: bool = True) -> dict:
    """
    Close a position with a market order (emergency use only).

    Args:
        order_manager: Order manager instance
        symbol: Stock ticker symbol
        quantity: Number of shares
        is_long: True if long position, False if short

    Returns:
        dict: Order result
    """
    action = "SELL" if is_long else "BUY"

    try:
        trade = order_manager.create_market_order(symbol, action, int(abs(quantity)))
        logger.info(f"✓ Closed {symbol}: {action} {int(abs(quantity))} shares (Order ID: {trade.order.orderId})")
        return {'success': True, 'order_id': trade.order.orderId}
    except Exception as e:
        logger.error(f"✗ Failed to close {symbol}: {e}")
        return {'success': False, 'error': str(e)}


def main():
    """Main function for emergency position closer."""
    print("\n" + "="*60)
    print("  EMERGENCY POSITION CLOSER")
    print("  This will cancel ALL orders and close ALL positions")
    print("="*60 + "\n")

    # Confirm before proceeding
    confirm = input("Are you sure? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("Aborted.")
        return

    conn = None
    try:
        # Initialize clients
        logger.info("Connecting to IBKR...")
        conn = IBConnection()
        conn.connect()
        account_client = AccountClient(conn)
        order_manager = OrderManager(conn)

        logger.info("Connected to IBKR")

        # ===== STEP 1: Cancel all open orders =====
        logger.info("\n--- STEP 1: Cancelling all open orders ---")
        open_orders = get_open_orders(account_client)

        if open_orders:
            logger.info(f"Found {len(open_orders)} open order(s) to cancel")
            for trade in open_orders:
                symbol = trade.contract.symbol
                order_type = trade.order.orderType
                status = trade.orderStatus.status
                order_id = trade.order.orderId

                logger.info(f"  Cancelling {order_type} order for {symbol} (ID: {order_id}, Status: {status})")
                if not order_manager.cancel_order(trade):
                    logger.warning(f"    Could not cancel order {order_id}")

            logger.info("Waiting 1 second for cancellations to process...")
            conn.sleep(1)
        else:
            logger.info("No open orders found")

        # ===== STEP 2: Get and close all positions =====
        logger.info("\n--- STEP 2: Closing all positions ---")
        positions = [p for p in get_positions(account_client) if p.position != 0]

        if not positions:
            logger.info("No open positions found. Done!")
            return

        logger.info(f"Found {len(positions)} position(s)")
        print("-" * 40)

        # Display positions
        for pos in positions:
            symbol = pos.contract.symbol
            quantity = pos.position
            avg_cost = pos.avgCost
            cost_basis = abs(quantity) * avg_cost
            position_type = "LONG" if quantity > 0 else "SHORT"
            print(f"  {symbol}: {abs(quantity):.0f} shares ({position_type}) @ ${avg_cost:.2f} avg cost = ${cost_basis:.2f}")

        print("-" * 40)

        # Confirm closing positions
        confirm2 = input(f"\nClose all {len(positions)} position(s) with MARKET orders? (yes/no): ").strip().lower()
        if confirm2 != "yes":
            print("Aborted.")
            return

        # Close each position
        print("\nClosing positions...")
        success_count = 0
        fail_count = 0

        for pos in positions:
            symbol = pos.contract.symbol
            quantity = pos.position
            is_long = quantity > 0

            result = close_position(order_manager, symbol, quantity, is_long=is_long)
            if result.get('success'):
                success_count += 1
            else:
                fail_count += 1

        print("\n" + "="*40)
        print(f"  COMPLETE: {success_count} closed, {fail_count} failed")
        print("="*40 + "\n")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        if conn is not None:
            conn.disconnect()


if __name__ == "__main__":
    main()
