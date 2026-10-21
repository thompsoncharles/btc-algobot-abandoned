"""
Godel Terminal PRT signals as a quant data source.

The live bot (strategy/rolling_strategy.py) already drives Godel via
Selenium and stores PRT output. This module exposes that output to the
quant layer two ways:

  1. from_database(): historical prt_data / spy_prt_snapshots rows out of
     the bot's SQLite DB - training data for blending and calibration.
  2. from_controller(): a live PRT run through the existing PRTCommand,
     returning prob_up / edge per ticker right now.
"""
import sqlite3
from pathlib import Path
from typing import List, Optional

import pandas as pd
from loguru import logger


def from_database(database_path: Path) -> Optional[pd.DataFrame]:
    """Historical PRT rows joined to their trades (adds realized outcome)."""
    if not Path(database_path).exists():
        logger.warning("bot database not found at {}", database_path)
        return None
    con = sqlite3.connect(database_path)
    try:
        return pd.read_sql_query(
            """
            SELECT t.ticker, t.action, t.entry_price, t.exit_price,
                   t.profit_loss_percent, t.is_winner, t.time_placed,
                   p.prob_up, p.edge, p.mean, p.p10, p.p90, p.n, p.timestamp
            FROM prt_data p
            JOIN trades t ON t.id = p.trade_id
            """,
            con,
        )
    except Exception as e:
        logger.debug("prt_data unavailable: {}", e)
        return None
    finally:
        con.close()


def spy_snapshots(database_path: Path) -> Optional[pd.DataFrame]:
    """SPY PRT time series (market-regime signal)."""
    if not Path(database_path).exists():
        return None
    con = sqlite3.connect(database_path)
    try:
        return pd.read_sql_query("SELECT * FROM spy_prt_snapshots ORDER BY timestamp", con)
    except Exception:
        return None
    finally:
        con.close()


def from_controller(controller, tickers: List[str]) -> Optional[pd.DataFrame]:
    """
    Run PRT live through the existing Godel controller.

    Args:
        controller: connected, logged-in core.controller.GodelTerminalController
        tickers: symbols to analyze (<= PRT_MAX_TICKERS)

    Returns:
        DataFrame with at least [ticker, prob_up, edge] or None.
    """
    from commands.prt_command import PRTCommand
    from constants.parameters import PRT_MAX_TICKERS

    batch = tickers[:PRT_MAX_TICKERS]
    cmd = PRTCommand(controller, tickers=batch)
    try:
        result = cmd.execute()
    except Exception as e:
        logger.error("live PRT run failed: {}", e)
        return None
    df = getattr(cmd, "df", None)
    if df is None and isinstance(result, pd.DataFrame):
        df = result
    if df is None or (hasattr(df, "empty") and df.empty):
        logger.warning("PRT returned no rows")
        return None
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df
