"""
Account size monitoring module.
Periodically checks and logs account size to CSV file for graphing.
"""
import csv
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
from loguru import logger
from ibkr.account import AccountClient


class AccountSizeMonitor:
    """
    Monitors account size and logs to CSV file for graphing.
    """
    
    def __init__(self, account_client: AccountClient, log_file: Optional[str] = None):
        """
        Initialize account size monitor.
        
        Args:
            account_client: IBKR account client
            log_file: Path to CSV log file (default: account_size_history.csv in project root)
        """
        self.account_client = account_client
        
        if log_file is None:
            project_root = Path(__file__).parent.parent
            log_file = project_root / "account_size_history.csv"
        
        self.log_file = Path(log_file)
        
        # Initialize CSV file with headers if it doesn't exist
        if not self.log_file.exists():
            self._initialize_csv()
        
        logger.info(f"AccountSizeMonitor initialized - logging to {self.log_file}")
    
    def _initialize_csv(self):
        """Initialize CSV file with headers."""
        try:
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'account_value',
                    'buying_power',
                    'intraday_buying_power',
                    'cash_balance',
                    'equity',
                    'liquidation_value'
                ])
            logger.info(f"Initialized account size log file: {self.log_file}")
        except Exception as e:
            logger.error(f"Error initializing CSV file: {e}")
    
    def get_account_metrics(self) -> Dict:
        """
        Get current account metrics.

        Returns:
            dict: Account metrics including value, buying power, etc.
        """
        try:
            account_info = self.account_client.get_account_info()

            if not account_info:
                logger.error("Failed to retrieve account info")
                return {}

            net_liquidation = account_info.get('NetLiquidation')
            buying_power = account_info.get('BuyingPower')

            metrics = {
                'account_value': net_liquidation,
                'buying_power': buying_power,
                # IBKR's BuyingPower tag already reflects day-trading buying
                # power for margin accounts - there's no separate
                # intraday/regular split the way Schwab exposed it.
                'intraday_buying_power': buying_power,
                'cash_balance': account_info.get('TotalCashValue'),
                'equity': account_info.get('EquityWithLoanValue'),
                'liquidation_value': net_liquidation,
            }

            # Convert None to empty string for CSV
            return {k: (v if v is not None else '') for k, v in metrics.items()}

        except Exception as e:
            logger.error(f"Error getting account metrics: {e}")
            return {}
    
    def log_account_size(self) -> bool:
        """
        Log current account size to CSV file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            metrics = self.get_account_metrics()
            
            if not metrics:
                logger.warning("No account metrics retrieved, skipping log")
                return False
            
            # Get current timestamp
            timestamp = datetime.now().isoformat()
            
            # Append to CSV file
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp,
                    metrics.get('account_value', ''),
                    metrics.get('buying_power', ''),
                    metrics.get('intraday_buying_power', ''),
                    metrics.get('cash_balance', ''),
                    metrics.get('equity', ''),
                    metrics.get('liquidation_value', '')
                ])
            
            account_value = metrics.get('account_value', 'N/A')
            logger.info(f"Account size logged: ${account_value:,.2f}" if isinstance(account_value, (int, float)) else f"Account size logged: {account_value}")
            return True
            
        except Exception as e:
            logger.error(f"Error logging account size: {e}")
            return False

