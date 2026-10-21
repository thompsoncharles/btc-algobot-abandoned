"""
Email alerts for the BTC daily paper-observation routine.

Configuration lives in alerts_config.json at the project root (gitignored -
it holds SMTP credentials; copy alerts_config.json.template to start).
Missing/incomplete config = alerts gracefully disabled: the routine runs
normally and every alert call returns {"sent": false, "reason": ...}.

Alert-worthy events (built from the routine report):
  - routine FAILURE (exception anywhere in the run)
  - safety gate failure
  - data gate failure
  - would_enter = true  (a paper SIGNAL day - the one you must not miss)
  - routine success digest (one line, so silence never means "broken")

Monitoring only: this module imports smtplib/ssl/json and nothing that can
place orders. It is included in verify_paper_safety()'s scan list.
"""
from __future__ import annotations

import json
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path
from typing import List, Optional

from loguru import logger

CONFIG_PATH = Path(__file__).resolve().parents[2] / "alerts_config.json"
REQUIRED_KEYS = ("smtp_host", "smtp_port", "smtp_user", "smtp_password",
                 "from_addr", "to_addr")


def load_alert_config(path: Optional[Path] = None) -> Optional[dict]:
    """Parsed config, or None when alerts are (or should be) disabled."""
    path = path or CONFIG_PATH
    if not path.exists():
        return None
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        logger.warning("alerts_config.json is invalid JSON ({}); alerts disabled", e)
        return None
    if not cfg.get("enabled", False):
        return None
    missing = [k for k in REQUIRED_KEYS if not cfg.get(k)]
    if missing:
        logger.warning("alerts_config.json missing {}; alerts disabled", missing)
        return None
    return cfg


def send_email(subject: str, body: str, cfg: Optional[dict] = None) -> dict:
    """Best-effort email. Never raises - a broken mailbox must not kill the routine."""
    cfg = cfg if cfg is not None else load_alert_config()
    if cfg is None:
        return {"sent": False, "reason": "alerts not configured "
                "(alerts_config.json missing/disabled/incomplete)"}
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = cfg["from_addr"]
        msg["To"] = cfg["to_addr"]
        msg.set_content(body)
        port = int(cfg["smtp_port"])
        if cfg.get("use_ssl", False):
            with smtplib.SMTP_SSL(cfg["smtp_host"], port,
                                  context=ssl.create_default_context(),
                                  timeout=30) as s:
                s.login(cfg["smtp_user"], cfg["smtp_password"])
                s.send_message(msg)
        else:
            with smtplib.SMTP(cfg["smtp_host"], port, timeout=30) as s:
                s.starttls(context=ssl.create_default_context())
                s.login(cfg["smtp_user"], cfg["smtp_password"])
                s.send_message(msg)
        return {"sent": True, "to": cfg["to_addr"]}
    except Exception as e:
        logger.warning("alert email failed: {}", e)
        return {"sent": False, "reason": f"send failed: {e}"}


def build_alert_events(routine_report: dict) -> List[dict]:
    """Alert-worthy events from a routine report (or a failure stub)."""
    events = []
    if routine_report.get("status") == "FAILURE":
        events.append({"level": "FAILURE",
                       "event": "routine_failure",
                       "detail": routine_report.get("error", "unknown error")})
        return events  # a failed run has nothing else trustworthy to say

    safety = routine_report.get("safety", {})
    if not safety.get("passed", False):
        events.append({"level": "FAILURE", "event": "safety_gate_failure",
                       "detail": "paper-safety verification did not pass"})
    gate = routine_report.get("data_gate", {})
    if gate.get("gate") != "PASS":
        events.append({"level": "WARNING", "event": "data_gate_failure",
                       "detail": f"data gate={gate.get('gate')} bars={gate.get('bars')} "
                                 f"behind={gate.get('bars_behind')}"})
    rec = routine_report.get("latest_observation") or {}
    if rec.get("would_enter"):
        events.append({
            "level": "SIGNAL", "event": "would_enter_true",
            "detail": (f"PAPER SIGNAL {rec.get('signal_date')}: prob={rec.get('predicted_prob_up')} "
                       f"edge=${rec.get('expected_edge_usd')} fees=${rec.get('round_trip_fees_usd')} "
                       "- run the manual checklist in btc_paper_report.md before mirroring."),
        })
    if not events:
        events.append({"level": "OK", "event": "routine_success",
                       "detail": (f"observation logged for {rec.get('signal_date', '?')}, "
                                  f"no signal (reasons: {', '.join(rec.get('gates_failed', [])) or 'n/a'}). "
                                  f"{routine_report.get('paper_verdict', '')}")})
    return events


def alert_for_routine(routine_report: dict) -> dict:
    """One email per routine run, subject graded by the worst event level."""
    events = build_alert_events(routine_report)
    levels = [e["level"] for e in events]
    worst = ("FAILURE" if "FAILURE" in levels else
             "SIGNAL" if "SIGNAL" in levels else
             "WARNING" if "WARNING" in levels else "OK")
    date = (routine_report.get("latest_observation") or {}).get("signal_date") \
        or routine_report.get("started", "")[:10]
    subject = f"[BTC-PAPER] {worst} - {date}"
    body_lines = [f"{e['level']}: {e['event']}\n  {e['detail']}" for e in events]
    body_lines.append("\nlive trading: DISABLED (paper observation only)")
    result = send_email(subject, "\n\n".join(body_lines))
    return {"events": events, "worst_level": worst, "email": result}
