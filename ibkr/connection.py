"""
Interactive Brokers connection management.
Owns the single shared IB() instance used by the rest of the ibkr package.
"""
import time

from ib_async import IB
from loguru import logger

from constants.parameters import IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID, PAPER_TRADING

LIVE_PORTS = {7496, 4001}
PAPER_PORTS = {7497, 4002}


class IBConnection:
    """
    Owns the single shared IB() connection used by AccountClient, OrderManager,
    and MarketDataClient. All three take an IBConnection and read `.ib` off it.
    """

    def __init__(self):
        self._validate_paper_live_guard()
        self.ib = IB()

    def _validate_paper_live_guard(self) -> None:
        if PAPER_TRADING and IBKR_PORT in LIVE_PORTS:
            raise RuntimeError(f"ABORT: PAPER_TRADING=True but IBKR_PORT={IBKR_PORT} is a live port")
        if not PAPER_TRADING and IBKR_PORT in PAPER_PORTS:
            raise RuntimeError(f"ABORT: PAPER_TRADING=False but IBKR_PORT={IBKR_PORT} is a paper port")

    def connect(self) -> None:
        """Connect to TWS/IB Gateway, retrying once after 5 seconds on failure."""
        try:
            self.ib.connect(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID)
        except Exception as e:
            logger.warning(f"IBKR connect failed ({e}), retrying once in 5s...")
            time.sleep(5)
            self.ib.connect(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID)
        logger.info(
            f"Connected to IBKR at {IBKR_HOST}:{IBKR_PORT} "
            f"(clientId={IBKR_CLIENT_ID}, paper={PAPER_TRADING})"
        )

    def ensure_connected(self) -> None:
        """Reconnect if the socket has dropped."""
        if not self.ib.isConnected():
            logger.warning("IBKR connection lost, reconnecting...")
            self.connect()

    def disconnect(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from IBKR")

    def sleep(self, seconds: float) -> None:
        """
        Sleep while pumping the ib_async event loop so positions, fills, and
        account values keep updating in the background. A plain time.sleep()
        would starve the socket reader and freeze all live state for the
        duration of the sleep.
        """
        self.ib.sleep(seconds)
