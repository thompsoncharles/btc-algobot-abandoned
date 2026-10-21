"""
SQLite operational log for the BTC paper-observation workflow.

Database: quant_data/ops_monitor.db (stdlib sqlite3, no dependencies).
Tables:
  routine_runs   one row per btc-daily-routine run (status, gates, verdict,
                 alert outcome, error text on failure)
  observations   one row per logged daily observation
  safety_checks  one row per paper-safety verification
  data_quality   one row per data-gate evaluation

Monitoring only - append-style inserts, never consulted by strategy code,
imports nothing that can place orders. Included in the safety scan list.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Optional

from quant import config

DB_PATH = config.DATA_DIR / "ops_monitor.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS routine_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    started TEXT,
    status TEXT NOT NULL,             -- SUCCESS | FAILURE
    safety_passed INTEGER,
    data_gate TEXT,
    observation_logged INTEGER,
    would_enter INTEGER,
    paper_verdict TEXT,
    alert_level TEXT,
    alert_sent INTEGER,
    alert_detail TEXT,
    error TEXT
);
CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    signal_date TEXT,
    close REAL,
    prob_up REAL,
    would_enter INTEGER,
    gates_failed TEXT,
    expected_edge_usd REAL,
    round_trip_fees_usd REAL
);
CREATE TABLE IF NOT EXISTS safety_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    passed INTEGER NOT NULL,
    files_scanned INTEGER,
    tokens_checked INTEGER,
    detail TEXT
);
CREATE TABLE IF NOT EXISTS data_quality (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    gate TEXT,
    bars INTEGER,
    last_bar TEXT,
    bars_behind INTEGER,
    missing_bars INTEGER,
    duplicate_bars INTEGER
);
"""


def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=15)
    conn.executescript(_SCHEMA)
    return conn


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def log_routine_run(report: dict, alert: Optional[dict] = None,
                    db_path: Optional[Path] = None) -> None:
    rec = report.get("latest_observation") or {}
    alert = alert or {}
    email = alert.get("email", {})
    with closing(_connect(db_path)) as conn, conn:
        conn.execute(
            "INSERT INTO routine_runs (ts, started, status, safety_passed, data_gate,"
            " observation_logged, would_enter, paper_verdict, alert_level, alert_sent,"
            " alert_detail, error) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (_now(), report.get("started"),
             report.get("status", "SUCCESS"),
             int(bool(report.get("safety", {}).get("passed"))),
             report.get("data_gate", {}).get("gate"),
             int(bool(report.get("observation_logged"))),
             int(bool(rec.get("would_enter"))),
             report.get("paper_verdict"),
             alert.get("worst_level"),
             int(bool(email.get("sent"))),
             email.get("reason") or email.get("to"),
             report.get("error")))


def log_observation(rec: dict, db_path: Optional[Path] = None) -> None:
    with closing(_connect(db_path)) as conn, conn:
        conn.execute(
            "INSERT INTO observations (ts, signal_date, close, prob_up, would_enter,"
            " gates_failed, expected_edge_usd, round_trip_fees_usd)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (_now(), rec.get("signal_date"), rec.get("close"),
             rec.get("predicted_prob_up"), int(bool(rec.get("would_enter"))),
             json.dumps(rec.get("gates_failed", [])),
             rec.get("expected_edge_usd"), rec.get("round_trip_fees_usd")))


def log_safety_check(safety: dict, db_path: Optional[Path] = None) -> None:
    with closing(_connect(db_path)) as conn, conn:
        conn.execute(
            "INSERT INTO safety_checks (ts, passed, files_scanned, tokens_checked, detail)"
            " VALUES (?,?,?,?,?)",
            (_now(), int(bool(safety.get("passed"))),
             len(safety.get("files_scanned", [])),
             safety.get("tokens_checked"),
             safety.get("error")))


def log_data_quality(gate: dict, db_path: Optional[Path] = None) -> None:
    missing = (gate.get("missing_bars") or {})
    with closing(_connect(db_path)) as conn, conn:
        conn.execute(
            "INSERT INTO data_quality (ts, gate, bars, last_bar, bars_behind,"
            " missing_bars, duplicate_bars) VALUES (?,?,?,?,?,?,?)",
            (_now(), gate.get("gate"), gate.get("bars"), gate.get("to"),
             gate.get("bars_behind_expected"),
             missing.get("missing_count") if isinstance(missing, dict) else None,
             gate.get("duplicate_bars")))
