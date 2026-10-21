"""
Fundamental data: financial statements, earnings, analyst estimates,
insider transactions, institutional ownership.

Primary free source is yfinance; Godel Terminal DES snapshots stored in
the bot's SQLite database are used as a secondary source when present.
"""
import sqlite3
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from loguru import logger

from quant.data.market_data import to_yahoo_symbol
from quant.data.store import DataStore


class Fundamentals:
    def __init__(self, store: Optional[DataStore] = None):
        self.store = store or DataStore()

    def fetch(self, ticker: str) -> Dict[str, pd.DataFrame]:
        """Pull the full free fundamental set for one ticker and cache it."""
        import yfinance as yf

        t = yf.Ticker(to_yahoo_symbol(ticker))
        out: Dict[str, pd.DataFrame] = {}
        grabs = {
            "income_statement": lambda: t.income_stmt,
            "balance_sheet": lambda: t.balance_sheet,
            "cash_flow": lambda: t.cash_flow,
            "earnings_dates": lambda: t.get_earnings_dates(limit=12),
            "analyst_estimates": lambda: t.earnings_estimate,
            "recommendations": lambda: t.recommendations,
            "insider_transactions": lambda: t.insider_transactions,
            "institutional_holders": lambda: t.institutional_holders,
        }
        for name, fn in grabs.items():
            try:
                df = fn()
                if isinstance(df, pd.DataFrame) and not df.empty:
                    out[name] = df
                    self.store.save_table(f"{ticker.upper()}_{name}", df)
            except Exception as e:
                logger.debug("fundamentals {} {} unavailable: {}", ticker, name, e)
        return out

    def snapshot_info(self, ticker: str) -> dict:
        """Fast scalar snapshot: market cap, sector, exchange, float, beta."""
        import yfinance as yf

        t = yf.Ticker(to_yahoo_symbol(ticker))
        try:
            fi = t.fast_info
            info = {
                "ticker": ticker.upper(),
                "market_cap": getattr(fi, "market_cap", None),
                "exchange": getattr(fi, "exchange", None),
                "last_price": getattr(fi, "last_price", None),
                "year_high": getattr(fi, "year_high", None),
                "year_low": getattr(fi, "year_low", None),
            }
        except Exception as e:
            logger.debug("fast_info failed for {}: {}", ticker, e)
            info = {"ticker": ticker.upper()}
        try:
            meta = t.get_info()
            info.setdefault("market_cap", meta.get("marketCap"))
            info["sector"] = meta.get("sector")
            info["beta"] = meta.get("beta")
            info["float_shares"] = meta.get("floatShares")
        except Exception:
            pass
        return info

    # ------------------------------------------------ Godel DES source
    def godel_des(self, database_path: Path) -> Optional[pd.DataFrame]:
        """Read Godel Terminal DES snapshots from the bot's SQLite database."""
        if not Path(database_path).exists():
            return None
        con = sqlite3.connect(database_path)
        try:
            return pd.read_sql_query("SELECT * FROM des_data", con)
        except Exception as e:
            logger.debug("no des_data table: {}", e)
            return None
        finally:
            con.close()
