"""
One-command daily routine for the BTC paper-observation workflow.

    python -m quant.pipeline btc-daily-routine [--offline]

Order of operations:
  0. safety verification  - proves the observation path cannot place orders
  1. data refresh + gate  - daily bars refreshed once (unless --offline),
                            then quality-checked; observer runs offline
                            against this just-refreshed cache
  2. btc-observe-daily    - logs today's prediction/gates/regime to journal
  3. btc-paper-report     - scoreboard + markdown + CSV artifacts
  4. daily status file    - concise quant_data/daily_status/YYYY-MM-DD.md

This module is paper-only BY CONSTRUCTION and verify_paper_safety() enforces
it: the observation-path sources are scanned for order-placement tokens on
every run, and the routine aborts if any appear.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from quant import config

# Order-placement tokens that must NEVER appear in the observation path.
FORBIDDEN_TOKENS = (
    "ibkr.orders", "OrderManager", "placeOrder", "place_order",
    "create_market_order", "create_limit_order", "create_stop_loss_order",
    "quant.execution.executor", "SmartExecutor",
)

# Files that make up the paper-observation path (monitoring modules included:
# alerts/ops_db run inside the routine, so they get the same no-order-code scan).
_OBSERVATION_PATH_FILES = (
    "quant/monitor/btc_daily_observer.py",
    "quant/monitor/btc_paper_report.py",
    "quant/monitor/btc_daily_routine.py",
    "quant/monitor/alerts.py",
    "quant/monitor/ops_db.py",
)

CANDIDATE_FILE = Path(__file__).resolve().parents[1] / "candidates" / \
    "btc_daily_cost5bar_full_trend200vol.json"

STATUS_DIR = config.DATA_DIR / "daily_status"


def _imported_modules(source: str) -> set:
    """All module paths imported by a source file (AST - ignores strings/comments)."""
    import ast
    mods = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    return mods


def verify_paper_safety() -> dict:
    """
    Assert the paper-observation path cannot place orders.

    1. AST import check on every observation-path file: no `ibkr.*` and no
       execution/order module may be imported (catches real imports, immune
       to string literals like this docstring).
    2. Token scan of the observer/report sources for order-API names
       (catches getattr-style or commented-back-in usage).
    3. The frozen candidate metadata still says live_trading=disabled and
       PAPER_OBSERVATION_ONLY, and matches the runtime CANDIDATE dict.
    """
    from quant.monitor.btc_daily_observer import CANDIDATE

    root = Path(__file__).resolve().parents[2]
    violations = []
    for rel in _OBSERVATION_PATH_FILES:
        src = (root / rel).read_text(encoding="utf-8")
        for mod in _imported_modules(src):
            if mod.split(".")[0] == "ibkr" or mod.startswith("quant.execution.executor"):
                violations.append(f"{rel}: imports order-capable module '{mod}'")
        if not rel.endswith("btc_daily_routine.py"):  # this file defines the token list
            for token in FORBIDDEN_TOKENS:
                if token in src:
                    violations.append(f"{rel}: contains '{token}'")
    if violations:
        raise AssertionError("PAPER SAFETY VIOLATION: " + "; ".join(violations))

    if CANDIDATE["live_trading"] != "disabled":
        raise AssertionError("PAPER SAFETY VIOLATION: candidate live_trading != disabled")
    if CANDIDATE["status"] != "PAPER_OBSERVATION_ONLY":
        raise AssertionError("PAPER SAFETY VIOLATION: candidate status != PAPER_OBSERVATION_ONLY")

    frozen = json.loads(CANDIDATE_FILE.read_text(encoding="utf-8"))
    if frozen["live_trading"] != "disabled" or frozen["status"] != "PAPER_OBSERVATION_ONLY":
        raise AssertionError("PAPER SAFETY VIOLATION: frozen candidate file altered")
    for key in ("name", "status", "live_trading"):
        if frozen[key] != CANDIDATE[key]:
            raise AssertionError(f"PAPER SAFETY VIOLATION: frozen file and runtime "
                                 f"CANDIDATE disagree on '{key}'")

    return {"passed": True, "files_scanned": list(_OBSERVATION_PATH_FILES),
            "tokens_checked": len(FORBIDDEN_TOKENS),
            "candidate_file": str(CANDIDATE_FILE)}


def _daily_data_gate(bars: pd.DataFrame) -> dict:
    """Quality gate for the DAILY dataset (the one the observer trades on paper)."""
    from quant.data.btc_audit import _basic_report, _normalize_index

    report = _basic_report(_normalize_index(bars), "1d",
                           "daily multi-regime cache (Yahoo primary)", "BTCUSD 1d")
    missing = (report.get("missing_bars") or {}).get("missing_count") or 0
    behind = report.get("bars_behind_expected")
    ok = bool(
        report.get("available")
        and report.get("duplicate_bars", 1) == 0
        and missing <= 10
        # Yahoo publishes the completed UTC day with lag; 2 bars behind is tolerable
        and (behind is None or behind <= 2)
        and not report.get("possible_unfinished_latest_candle", False)
    )
    report["gate"] = "PASS" if ok else "FAIL"
    return report


def run_daily_routine(offline: bool = False) -> dict:
    """
    Daily routine with operational monitoring: every run (success or
    failure) is emailed (if alerts_config.json is configured) and logged to
    quant_data/ops_monitor.db. Monitoring is best-effort - a broken mailbox
    or locked database never blocks the observation itself.
    """
    started = datetime.now().isoformat(timespec="seconds")
    try:
        report = _run_daily_routine_core(started, offline)
    except Exception as e:
        # FAILURE path: alert + log, then re-raise so the runner records exit 1
        failure = {"places_orders": False, "started": started,
                   "status": "FAILURE", "error": f"{type(e).__name__}: {e}"}
        _monitor(failure)
        raise
    _monitor(report)
    return report


def _monitor(report: dict) -> None:
    """Best-effort alerting + SQLite logging; never raises."""
    alert = None
    try:
        from quant.monitor.alerts import alert_for_routine
        alert = alert_for_routine(report)
        report["alert"] = {"worst_level": alert["worst_level"],
                           "email": alert["email"],
                           "events": [e["event"] for e in alert["events"]]}
    except Exception as e:
        logger.warning("alerting failed (routine unaffected): {}", e)
    try:
        from quant.monitor import ops_db
        ops_db.log_routine_run(report, alert)
        if report.get("safety") is not None:
            ops_db.log_safety_check(report["safety"])
        if report.get("_data_gate_full") is not None:
            ops_db.log_data_quality(report.pop("_data_gate_full"))
        if report.get("latest_observation"):
            ops_db.log_observation(report["latest_observation"])
    except Exception as e:
        logger.warning("ops_db logging failed (routine unaffected): {}", e)
    report.pop("_data_gate_full", None)


def _run_daily_routine_core(started: str, offline: bool) -> dict:
    from quant.data.btc_data import get_btc_daily_history
    from quant.monitor.btc_daily_observer import log_daily_observations
    from quant.monitor.btc_paper_report import build_paper_report, write_artifacts
    from quant.research.btc_label_eval import _position_notional

    # 0. safety first - abort before touching anything if violated
    safety = verify_paper_safety()

    # 1. refresh once, then everything downstream runs against the cache
    bars = get_btc_daily_history(refresh=not offline)
    data_gate = _daily_data_gate(bars)

    observation = None
    paper = None
    if data_gate["gate"] == "PASS":
        # 2. observe today's bar (offline: data was just refreshed above)
        observation = log_daily_observations(bars, _position_notional(), count=1)
        # 3. scoreboard + artifacts
        paper = build_paper_report()
        write_artifacts(paper)
    else:
        logger.error("daily data gate FAILED - skipping observation so a bad "
                     "bar never enters the journal: {}", data_gate)

    # 4. concise daily status file
    status = _write_status_file(started, safety, data_gate, observation, paper)

    return {
        "places_orders": False,
        "started": started,
        "status": "SUCCESS",
        "safety": safety,
        "data_gate": {"gate": data_gate["gate"], "bars": data_gate.get("bars"),
                      "to": data_gate.get("to"),
                      "bars_behind": data_gate.get("bars_behind_expected")},
        "_data_gate_full": data_gate,
        "observation_logged": observation is not None,
        "latest_observation": (observation or {}).get("latest_record"),
        "paper_verdict": (paper or {}).get("verdict", "SKIPPED - data gate failed"),
        "status_file": status,
    }


def _write_status_file(started: str, safety: dict, data_gate: dict,
                       observation: dict | None, paper: dict | None) -> str:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    path = STATUS_DIR / f"{today}.md"

    lines = [
        f"# BTC daily paper status - {today}",
        "",
        f"- run started: {started}",
        f"- safety: {'PASS' if safety.get('passed') else 'FAIL'} "
        f"({safety.get('tokens_checked')} forbidden tokens scanned, 0 found)",
        f"- data gate: {data_gate['gate']} "
        f"({data_gate.get('bars')} bars through {data_gate.get('to')})",
    ]
    if observation and observation.get("latest_record"):
        rec = observation["latest_record"]
        reasons = ", ".join(rec.get("gates_failed", [])) or "none"
        lines += [
            f"- signal date: {rec.get('signal_date')}",
            f"- prob(up): {rec.get('predicted_prob_up')}",
            f"- would enter: {rec.get('would_enter')}",
            f"- gates failed: {reasons}",
            f"- expected edge: ${rec.get('expected_edge_usd')} "
            f"vs fees ${rec.get('round_trip_fees_usd')}",
        ]
    else:
        lines.append("- observation: SKIPPED (data gate failed)")
    if paper:
        gate = paper["reconsideration_gate"]
        lines += [
            f"- paper progress: {gate['signals_so_far']}/{gate['min_paper_signals']} signals, "
            f"{gate['days_so_far']}/{gate['min_calendar_days']} days",
            f"- verdict: {paper['verdict']}",
        ]
    lines += ["", "live trading: DISABLED (paper observation only)"]
    path.write_text("\n".join(lines), encoding="utf-8")
    (STATUS_DIR / "latest.md").write_text("\n".join(lines), encoding="utf-8")
    return str(path)
