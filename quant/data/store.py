"""
Storage layer: Parquet files for history, DuckDB for queries.

Layout:
    quant_data/bars/{interval}/{TICKER}.parquet   OHLCV per ticker/interval
    quant_data/fundamentals/{TICKER}.parquet
    quant_data/macro/{SERIES}.parquet
    quant_data/quant.duckdb                       query engine over the parquet lake
"""
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from quant import config

try:
    import duckdb
    _HAS_DUCKDB = True
except ImportError:  # pragma: no cover
    duckdb = None
    _HAS_DUCKDB = False


class DataStore:
    """Parquet-backed store with optional DuckDB SQL access."""

    def __init__(self):
        config.ensure_dirs()

    # ------------------------------------------------------------- bars
    def bar_path(self, ticker: str, interval: str) -> Path:
        d = config.BARS_DIR / interval
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{ticker.upper().replace(' ', '_')}.parquet"

    def save_bars(self, ticker: str, interval: str, bars: pd.DataFrame) -> None:
        """Persist OHLCV bars (DatetimeIndex, columns Open/High/Low/Close/Volume)."""
        if bars is None or bars.empty:
            return
        path = self.bar_path(ticker, interval)
        if path.exists():
            old = pd.read_parquet(path)
            bars = pd.concat([old, bars])
            bars = bars[~bars.index.duplicated(keep="last")].sort_index()
        bars.to_parquet(path)

    def load_bars(self, ticker: str, interval: str = "1d") -> Optional[pd.DataFrame]:
        path = self.bar_path(ticker, interval)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def load_universe_bars(self, tickers: List[str], interval: str = "1d",
                           column: str = "Close") -> pd.DataFrame:
        """Wide frame of one bar column across tickers (aligned on index)."""
        frames: Dict[str, pd.Series] = {}
        for t in tickers:
            bars = self.load_bars(t, interval)
            if bars is not None and column in bars.columns:
                frames[t] = bars[column]
        if not frames:
            return pd.DataFrame()
        return pd.DataFrame(frames).sort_index()

    def available_tickers(self, interval: str = "1d") -> List[str]:
        d = config.BARS_DIR / interval
        if not d.exists():
            return []
        return sorted(p.stem.replace("_", " ") for p in d.glob("*.parquet"))

    # ----------------------------------------------------- fundamentals
    def save_table(self, name: str, df: pd.DataFrame, subdir: str = "fundamentals") -> None:
        if df is None or df.empty:
            return
        d = config.DATA_DIR / subdir
        d.mkdir(parents=True, exist_ok=True)
        df.to_parquet(d / f"{name}.parquet")

    def load_table(self, name: str, subdir: str = "fundamentals") -> Optional[pd.DataFrame]:
        path = config.DATA_DIR / subdir / f"{name}.parquet"
        if not path.exists():
            return None
        return pd.read_parquet(path)

    # ------------------------------------------------------------- SQL
    def sql(self, query: str) -> pd.DataFrame:
        """
        Run SQL over the parquet lake via DuckDB, e.g.:
            store.sql("SELECT * FROM read_parquet('quant_data/bars/1d/*.parquet')")
        """
        if not _HAS_DUCKDB:
            raise RuntimeError("duckdb not installed - pip install duckdb")
        con = duckdb.connect(str(config.DUCKDB_PATH))
        try:
            return con.execute(query).df()
        finally:
            con.close()

    def append_rows(self, table: str, df: pd.DataFrame) -> None:
        """Append rows to a persistent DuckDB table (created on first write)."""
        if df is None or df.empty:
            return
        if not _HAS_DUCKDB:
            # Degrade to parquet append so audit data is never dropped.
            logger.warning("duckdb missing; appending {} rows to parquet instead", len(df))
            path = config.DATA_DIR / f"{table}.parquet"
            if path.exists():
                df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
            df.to_parquet(path)
            return
        con = duckdb.connect(str(config.DUCKDB_PATH))
        try:
            con.register("_incoming", df)
            con.execute(f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM _incoming LIMIT 0")
            con.execute(f"INSERT INTO {table} SELECT * FROM _incoming")
        finally:
            con.close()
