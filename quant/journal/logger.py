"""
Audit journal: every prediction, order, fill, error, model version, and
parameter change - one JSONL line each, mirrored into DuckDB for queries.

House rule #8 requires full audit trails including decision rationale;
this is that trail. JSONL is the source of truth (append-only, crash-safe);
DuckDB rows are a queryable convenience.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from quant import config
from quant.data.store import DataStore


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JournalLogger:
    def __init__(self, store: Optional[DataStore] = None):
        config.ensure_dirs()
        self.store = store or DataStore()
        self.dir = config.JOURNAL_DIR

    # ------------------------------------------------------------ core
    def _write(self, stream: str, record: dict) -> None:
        record = {"ts": _now(), **record}
        path = self.dir / f"{stream}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
        try:
            self.store.append_rows(f"journal_{stream}", pd.DataFrame([record]))
        except Exception as e:
            logger.debug("duckdb mirror failed for {}: {}", stream, e)

    # --------------------------------------------------------- streams
    def log_prediction(self, ticker: str, prob_up: float, source: str,
                       model_version: Optional[str] = None, **extra) -> None:
        self._write("predictions", {
            "ticker": ticker.upper(), "prob_up": float(prob_up),
            "source": source, "model_version": model_version, **extra,
        })

    def log_order(self, event: str, **fields) -> None:
        self._write("orders", {"event": event, **fields})

    def log_fill(self, ticker: str, side: str, qty: int, price: float,
                 order_id, rationale: str = "", **extra) -> None:
        self._write("fills", {
            "ticker": ticker.upper(), "side": side, "qty": int(qty),
            "price": float(price), "order_id": str(order_id),
            "rationale": rationale, **extra,
        })

    def log_error(self, where: str, error: str, **extra) -> None:
        self._write("errors", {"where": where, "error": error, **extra})

    def log_model_event(self, name: str, version: int, event: str,
                        metrics: Optional[dict] = None) -> None:
        self._write("models", {
            "model": name, "version": version, "event": event,
            "metrics": metrics or {},
        })

    def log_parameter_change(self) -> str:
        """Hash constants/parameters.py so any config change is on record."""
        params_file = config.PROJECT_ROOT / "constants" / "parameters.py"
        digest = ""
        if params_file.exists():
            digest = hashlib.sha256(params_file.read_bytes()).hexdigest()[:16]
        self._write("parameters", {"sha256_16": digest})
        return digest

    # ---------------------------------------------------------- readers
    def read(self, stream: str) -> pd.DataFrame:
        path = self.dir / f"{stream}.jsonl"
        if not path.exists():
            return pd.DataFrame()
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        return pd.DataFrame(rows)
