"""
Paper-observation scoreboard for the promoted daily BTC candidate.

    python -m quant.pipeline btc-paper-report

Reads the btc_daily_observations journal, recomputes realized outcomes
against the freshest daily bars (journal records are never mutated), and
issues a paper verdict. Reconsideration is gated: at least
MIN_PAPER_SIGNALS would-enter signals OR MIN_PAPER_DAYS calendar days of
observation - before that the only verdict is KEEP OBSERVING.

Read-only + simulation-only: no orders, no default changes.
"""
from __future__ import annotations

import json
from typing import List, Optional

import numpy as np
import pandas as pd

from quant import config
from quant.execution.fees import round_trip_cost
from quant.monitor.btc_daily_observer import (
    CANDIDATE,
    JOURNAL_TABLE,
    MIN_PAPER_DAYS,
    MIN_PAPER_SIGNALS,
)


def load_observations() -> List[dict]:
    path = config.JOURNAL_DIR / f"{JOURNAL_TABLE}.jsonl"
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("mode") == "paper_observation_daily":
            records.append(rec)
    # dedupe by signal date, keep the most recently written record
    by_date = {}
    for rec in records:
        key = rec.get("signal_date") or rec.get("signal_ts")
        by_date[key] = rec
    return [by_date[k] for k in sorted(by_date)]


def _resolve_outcome(bars: pd.DataFrame, signal_ts: pd.Timestamp,
                     notional: float) -> Optional[dict]:
    """Re-simulate the hypothetical trade against the freshest bars."""
    idx = bars.index.get_indexer([signal_ts])[0]
    if idx < 0 or idx + 1 >= len(bars):
        return None
    entry_pos = idx + 1
    entry = float(bars["Open"].iloc[entry_pos])
    stop = entry * (1 - config.BTC_STOP_PCT / 100)
    target = entry * (1 + config.BTC_TARGET_PCT / 100)
    max_exit = min(len(bars) - 1, entry_pos + config.BTC_MAX_HOLD_BARS)
    result = None
    for pos in range(entry_pos, max_exit + 1):
        row = bars.iloc[pos]
        if float(row["Low"]) <= stop:
            result = ("stop", -config.BTC_STOP_PCT, pos)
            break
        if float(row["High"]) >= target:
            result = ("target", config.BTC_TARGET_PCT, pos)
            break
    if result is None:
        if max_exit == entry_pos + config.BTC_MAX_HOLD_BARS:
            exit_price = float(bars["Close"].iloc[max_exit])
            result = ("time", (exit_price / entry - 1) * 100, max_exit)
        else:
            return {"status": "open"}
    reason, gross_pct, pos = result
    net = gross_pct / 100 * notional - round_trip_cost(notional)
    exit_price = entry * (1 + gross_pct / 100)
    return {"status": "closed", "reason": reason,
            "gross_return_pct": round(gross_pct, 4),
            "net_pnl_usd": round(net, 2),
            "entry_ts": str(bars.index[entry_pos]),
            "entry_price": round(entry, 2),
            "exit_price": round(exit_price, 2),
            "exit_ts": str(bars.index[pos])}


def build_paper_report() -> dict:
    from quant.data.store import DataStore
    from quant.research.btc_label_eval import _position_notional

    obs = load_observations()
    bars = DataStore().load_bars("BTCUSD", "1d")
    bars = bars.sort_index() if bars is not None else None
    notional = _position_notional()

    signals = [r for r in obs if r.get("would_enter")]
    n_signals = len(signals)
    if obs:
        first = pd.Timestamp(obs[0]["signal_date"])
        last = pd.Timestamp(obs[-1]["signal_date"])
        days_elapsed = int((last - first).days)
    else:
        days_elapsed = 0

    # re-resolve every would-enter signal against the freshest bars
    closed, open_, wins = [], 0, 0
    for rec in signals:
        if bars is None:
            break
        out = _resolve_outcome(bars, pd.Timestamp(rec["signal_ts"]), notional)
        if out is None:
            continue
        if out["status"] == "open":
            open_ += 1
            continue
        closed.append(out)
        if out["net_pnl_usd"] > 0:
            wins += 1

    pnl = [c["net_pnl_usd"] for c in closed]
    gross_win = sum(p for p in pnl if p > 0)
    gross_loss = -sum(p for p in pnl if p < 0)
    stats = {
        "observations_logged": len(obs),
        "would_enter_signals": n_signals,
        "calendar_days_observed": days_elapsed,
        "closed_paper_trades": len(closed),
        "open_paper_trades": open_,
        "paper_win_rate_pct": round(wins / len(closed) * 100, 2) if closed else None,
        "paper_net_expectancy_usd": round(float(np.mean(pnl)), 2) if pnl else None,
        "paper_total_net_pnl_usd": round(float(np.sum(pnl)), 2) if pnl else None,
        "paper_profit_factor": round(gross_win / gross_loss, 3) if gross_loss > 0 else None,
        "oos_reference": CANDIDATE["oos_reference"],
    }

    # ---- verdict, gated by the reconsideration rule
    enough = n_signals >= MIN_PAPER_SIGNALS or days_elapsed >= MIN_PAPER_DAYS
    if not enough:
        verdict = (
            f"KEEP OBSERVING - {n_signals}/{MIN_PAPER_SIGNALS} paper signals and "
            f"{days_elapsed}/{MIN_PAPER_DAYS} days observed; no reconsideration "
            "until one threshold is met. Live trading remains disabled."
        )
    elif not closed:
        verdict = ("KEEP OBSERVING - threshold met but no closed paper trades "
                   "to judge yet; keep observing until signals resolve.")
    elif stats["paper_net_expectancy_usd"] > 0 and (stats["paper_profit_factor"] or 0) > 1.2:
        verdict = (
            f"PAPER-READY - {len(closed)} closed paper trades, "
            f"${stats['paper_net_expectancy_usd']}/trade, PF {stats['paper_profit_factor']} "
            f"(OOS reference: +$15.50, PF 1.58). Eligible for reconsideration: "
            "re-run btc-signal-search to confirm before any step beyond paper. "
            "Live trading remains disabled."
        )
    else:
        verdict = (
            f"REJECT - paper contradicts the backtest: {len(closed)} closed paper "
            f"trades, ${stats['paper_net_expectancy_usd']}/trade net "
            "(OOS reference: +$15.50, PF 1.58). Send the candidate back to "
            "research; do not tune it to rescue the result. Live trading remains disabled."
        )

    return {
        "candidate": CANDIDATE,
        "reconsideration_gate": {
            "min_paper_signals": MIN_PAPER_SIGNALS,
            "min_calendar_days": MIN_PAPER_DAYS,
            "signals_so_far": n_signals,
            "days_so_far": days_elapsed,
            "met": bool(enough),
        },
        "paper_stats": stats,
        "verdict": verdict,
        "journal": str(config.JOURNAL_DIR / f"{JOURNAL_TABLE}.jsonl"),
        "_observations": obs,
    }


# ------------------------------------------------------------------ artifacts
REASON_CODES = {
    "regime_trend200_vol": "REGIME_FILTER",
    "confidence": "LOW_CONFIDENCE",
    "entry_probability": "PROB_BELOW_ENTRY",
    "volatility_floor": "VOL_TOO_LOW",
    "edge_to_cost": "EDGE_BELOW_COST",
    "features_available": "NO_FEATURES",
}


def reason_codes(rec: dict) -> str:
    """Short no-trade reason codes for one observation ('' if it would enter)."""
    if rec.get("would_enter"):
        return ""
    failed = rec.get("gates_failed")
    if failed is None:
        failed = [g["name"] for g in rec.get("gates", []) if not g.get("passed")]
    return "+".join(REASON_CODES.get(name, name.upper()) for name in failed)


def observations_csv(obs: List[dict]) -> pd.DataFrame:
    rows = []
    for rec in obs:
        realized = rec.get("realized") or {}
        hyp = (rec.get("hypothetical") or {})
        outcome = hyp.get("outcome") or {}
        rows.append({
            "date": rec.get("signal_date"),
            "close": rec.get("close"),
            "prob_up": rec.get("predicted_prob_up"),
            "would_enter": rec.get("would_enter"),
            "no_trade_reasons": reason_codes(rec),
            "regime_trend200": (rec.get("regime_filter") or {}).get("trend200_above"),
            "vol_pctile": (rec.get("regime_filter") or {}).get("vol_pctile"),
            "expected_edge_usd": rec.get("expected_edge_usd"),
            "fees_usd": rec.get("round_trip_fees_usd"),
            "fwd_1bar_pct": realized.get("fwd_1bar_pct"),
            "fwd_3bar_pct": realized.get("fwd_3bar_pct"),
            "fwd_5bar_pct": realized.get("fwd_5bar_pct"),
            "hypo_entry": hyp.get("entry_price"),
            "hypo_exit_reason": outcome.get("reason"),
            "hypo_net_pnl_usd": outcome.get("net_pnl_usd"),
        })
    return pd.DataFrame(rows)


def paper_trades_tax_rows(signals: List[dict], bars: pd.DataFrame,
                          notional: float) -> pd.DataFrame:
    """
    Tax-ready ledger of resolved hypothetical paper trades. Columns follow
    the fields a 1099-B/Form-8949 style export needs, so if paper ever
    graduates to real trades the bookkeeping schema already exists.
    Paper trades themselves have no tax consequence - this is practice.
    """
    from quant.execution.fees import ibkr_crypto_fee

    rows = []
    for rec in signals:
        out = _resolve_outcome(bars, pd.Timestamp(rec["signal_ts"]), notional)
        if out is None or out.get("status") != "closed":
            continue
        entry_price = out["entry_price"]
        exit_price = out["exit_price"]
        qty = notional / entry_price
        entry_fee = ibkr_crypto_fee(qty * entry_price)
        exit_fee = ibkr_crypto_fee(qty * exit_price)
        cost_basis = qty * entry_price + entry_fee
        proceeds = qty * exit_price - exit_fee
        entry_ts = pd.Timestamp(out["entry_ts"])
        exit_ts = pd.Timestamp(out["exit_ts"])
        holding_days = int((exit_ts - entry_ts).days)
        rows.append({
            "entry_time": str(entry_ts),
            "exit_time": str(exit_ts),
            "asset": "BTC",
            "quantity": round(qty, 8),
            "entry_price_usd": round(entry_price, 2),
            "exit_price_usd": round(exit_price, 2),
            "entry_fee_usd": round(entry_fee, 2),
            "exit_fee_usd": round(exit_fee, 2),
            "cost_basis_usd": round(cost_basis, 2),
            "proceeds_usd": round(proceeds, 2),
            "realized_gain_loss_usd": round(proceeds - cost_basis, 2),
            "holding_period_days": holding_days,
            "short_or_long_term": "long_term" if holding_days > 365 else "short_term",
            "strategy_id": CANDIDATE["name"],
            "candidate_version": f"promoted_{CANDIDATE['promoted_at']}",
            "exit_reason": out["reason"],
            "paper_only": True,
        })
    return pd.DataFrame(rows)


def reason_code_counts(obs: List[dict]) -> dict:
    """How often each no-trade reason fired across all observations."""
    counts: dict = {}
    for rec in obs:
        codes = reason_codes(rec)
        if not codes:
            continue
        for code in codes.split("+"):
            counts[code] = counts.get(code, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def manual_checklist(report: dict) -> List[dict]:
    """
    The human pre-paper-trade checklist: every box must be checked before
    manually mirroring a signal in the paper account. Computed from the
    live system state, not hand-maintained.
    """
    from quant.monitor.btc_daily_routine import verify_paper_safety, _daily_data_gate
    from quant.data.store import DataStore

    obs = report.get("_observations", [])
    latest = obs[-1] if obs else {}
    gates = {g["name"]: bool(g.get("passed")) for g in latest.get("gates", [])}

    try:
        safety_ok = bool(verify_paper_safety().get("passed"))
        candidate_match = True  # verify_paper_safety asserts frozen==runtime
    except AssertionError:
        safety_ok = False
        candidate_match = False
    try:
        bars = DataStore().load_bars("BTCUSD", "1d")
        data_ok = _daily_data_gate(bars.sort_index())["gate"] == "PASS" if bars is not None else False
    except Exception:
        data_ok = False

    return [
        {"item": "data gate passed", "ok": data_ok},
        {"item": "safety gate passed (no order code in observation path)", "ok": safety_ok},
        {"item": "candidate file matches runtime", "ok": candidate_match},
        {"item": "would_enter is true for the latest observation",
         "ok": bool(latest.get("would_enter"))},
        {"item": "regime filter passed (trend200+vol)",
         "ok": gates.get("regime_trend200_vol", False)},
        {"item": "probability gate passed (prob >= 0.58)",
         "ok": gates.get("entry_probability", False)},
        {"item": "edge-to-cost passed (edge >= 2x fees)",
         "ok": gates.get("edge_to_cost", False)},
        {"item": "fees included in expectations (~$6 round trip)",
         "ok": latest.get("round_trip_fees_usd") is not None},
        {"item": "live trading disabled",
         "ok": CANDIDATE["live_trading"] == "disabled"},
    ]


def render_markdown(report: dict) -> str:
    gate = report["reconsideration_gate"]
    stats = report["paper_stats"]
    ref = CANDIDATE["oos_reference"]
    sig_pct = min(100, round(gate["signals_so_far"] / gate["min_paper_signals"] * 100))
    day_pct = min(100, round(gate["days_so_far"] / gate["min_calendar_days"] * 100))

    def fmt(v, prefix="", suffix=""):
        return f"{prefix}{v}{suffix}" if v is not None else "n/a"

    lines = [
        "# BTC paper observation report",
        "",
        f"**Candidate:** `{CANDIDATE['name']}` - {CANDIDATE['dataset']} / "
        f"{CANDIDATE['label']} / {CANDIDATE['features']} / {CANDIDATE['regime_filter']}",
        f"**Status:** {CANDIDATE['status']} | live trading: {CANDIDATE['live_trading']}",
        "",
        "## Progress toward reconsideration",
        "",
        f"- signals: {gate['signals_so_far']}/{gate['min_paper_signals']} ({sig_pct}%)",
        f"- days: {gate['days_so_far']}/{gate['min_calendar_days']} ({day_pct}%)",
        f"- gate met: {'YES' if gate['met'] else 'no'}",
        "",
        "## Paper vs out-of-sample reference",
        "",
        "| metric | paper (so far) | OOS reference |",
        "|--------|---------------|---------------|",
        f"| trades | {stats['closed_paper_trades']} closed, {stats['open_paper_trades']} open | {ref['trades']} |",
        f"| net expectancy/trade | {fmt(stats['paper_net_expectancy_usd'], '$')} | ${ref['net_expectancy_usd']} |",
        f"| profit factor | {fmt(stats['paper_profit_factor'])} | {ref['profit_factor']} |",
        f"| win rate | {fmt(stats['paper_win_rate_pct'], suffix='%')} | 49.4% |",
        "",
        "## Verdict",
        "",
        report["verdict"],
        "",
    ]

    obs = report.get("_observations", [])
    if obs:
        latest = obs[-1]
        lines += [
            "## Latest observation",
            "",
            f"- date: {latest.get('signal_date')} | close: ${latest.get('close'):,.2f}"
            if latest.get("close") else f"- date: {latest.get('signal_date')}",
            f"- prob(up): {latest.get('predicted_prob_up')} | would enter: "
            f"{'**YES**' if latest.get('would_enter') else 'no'}",
            f"- no-trade reasons: {reason_codes(latest) or 'none - signal!'}",
            "",
        ]

    counts = reason_code_counts(obs)
    if counts:
        lines += ["## No-trade reason counts (all observations)", ""]
        lines += [f"- {code}: {n}" for code, n in counts.items()]
        lines += [""]

    lines += ["## Manual paper-trade checklist", "",
              "_Before manually mirroring a signal in the paper account, every box "
              "must be checked (computed live, not hand-maintained):_", ""]
    for item in manual_checklist(report):
        lines.append(f"- [{'x' if item['ok'] else ' '}] {item['item']}")
    lines += [
        "",
        f"_observations logged: {stats['observations_logged']} | journal: `{report['journal']}`_",
    ]
    return "\n".join(lines)


def write_artifacts(report: dict) -> dict:
    """Markdown summary + observations CSV + tax-ready paper-trade ledger."""
    from quant.data.store import DataStore
    from quant.research.btc_label_eval import _position_notional

    md_path = config.DATA_DIR / "btc_paper_report.md"
    csv_path = config.DATA_DIR / "btc_observations.csv"
    tax_path = config.DATA_DIR / "btc_paper_trades.csv"

    md_path.write_text(render_markdown(report), encoding="utf-8")
    obs = report.get("_observations", [])
    obs_df = observations_csv(obs)
    obs_df.to_csv(csv_path, index=False)

    bars = DataStore().load_bars("BTCUSD", "1d")
    signals = [r for r in obs if r.get("would_enter")]
    tax_df = (paper_trades_tax_rows(signals, bars.sort_index(), _position_notional())
              if bars is not None and signals else pd.DataFrame(columns=TAX_COLUMNS))
    tax_df.to_csv(tax_path, index=False)

    out = {"markdown": str(md_path), "csv": str(csv_path),
           "csv_rows": len(obs), "tax_ledger": str(tax_path),
           "tax_ledger_rows": len(tax_df)}
    out["excel"] = write_excel(report, obs_df)
    return out


def write_excel(report: dict, obs_df: Optional[pd.DataFrame] = None) -> Optional[str]:
    """
    Excel workbook mirror of the paper report:
    quant_data/reports/btc_paper_report.xlsx
    (Summary / Observations / Signals Only / Reason Codes / Forward Returns /
    Candidate Config / Safety Status). Falls back gracefully to the existing
    CSV/Markdown outputs when openpyxl is not installed.
    """
    try:
        import openpyxl  # noqa: F401 - pandas ExcelWriter engine
    except ImportError:
        print("EXCEL EXPORT SKIPPED: openpyxl is not installed - CSV/Markdown "
              "outputs are unaffected. Install with: venv\\Scripts\\python.exe "
              "-m pip install openpyxl")
        return None

    obs = report.get("_observations", [])
    if obs_df is None:
        obs_df = observations_csv(obs)

    gate = report["reconsideration_gate"]
    stats = report["paper_stats"]
    ref = CANDIDATE["oos_reference"]
    summary_rows = [
        ("candidate", CANDIDATE["name"]),
        ("status", CANDIDATE["status"]),
        ("live_trading", CANDIDATE["live_trading"]),
        ("verdict", report["verdict"]),
        ("signals_progress", f"{gate['signals_so_far']}/{gate['min_paper_signals']}"),
        ("days_progress", f"{gate['days_so_far']}/{gate['min_calendar_days']}"),
        ("reconsideration_gate_met", gate["met"]),
        ("observations_logged", stats["observations_logged"]),
        ("closed_paper_trades", stats["closed_paper_trades"]),
        ("open_paper_trades", stats["open_paper_trades"]),
        ("paper_win_rate_pct", stats["paper_win_rate_pct"]),
        ("paper_net_expectancy_usd", stats["paper_net_expectancy_usd"]),
        ("paper_profit_factor", stats["paper_profit_factor"]),
        ("oos_ref_trades", ref["trades"]),
        ("oos_ref_net_expectancy_usd", ref["net_expectancy_usd"]),
        ("oos_ref_profit_factor", ref["profit_factor"]),
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["metric", "value"])

    signals_df = obs_df[obs_df["would_enter"] == True] if not obs_df.empty \
        else obs_df  # noqa: E712 - pandas boolean column comparison
    counts = reason_code_counts(obs)
    reasons_df = pd.DataFrame(
        [{"reason_code": k, "days_fired": v} for k, v in counts.items()]
    ) if counts else pd.DataFrame(columns=["reason_code", "days_fired"])
    fwd_cols = ["date", "prob_up", "would_enter",
                "fwd_1bar_pct", "fwd_3bar_pct", "fwd_5bar_pct"]
    fwd_df = obs_df[fwd_cols] if not obs_df.empty else pd.DataFrame(columns=fwd_cols)

    def _flatten(d: dict, prefix: str = "") -> list:
        rows = []
        for k, v in d.items():
            if isinstance(v, dict):
                rows.extend(_flatten(v, f"{prefix}{k}."))
            else:
                rows.append((f"{prefix}{k}", str(v)))
        return rows

    config_df = pd.DataFrame(_flatten(CANDIDATE), columns=["setting", "value"])

    try:
        from quant.monitor.btc_daily_routine import verify_paper_safety
        safety = verify_paper_safety()
        safety_rows = [("safety_scan", "PASS"),
                       ("files_scanned", "; ".join(safety["files_scanned"])),
                       ("forbidden_tokens_checked", safety["tokens_checked"])]
    except Exception as e:
        safety_rows = [("safety_scan", f"FAIL: {e}")]
    safety_rows += [("live_trading", CANDIDATE["live_trading"]),
                    ("places_orders", "false - observation path imports no order code")]
    for item in manual_checklist(report):
        safety_rows.append((f"checklist: {item['item']}",
                            "PASS" if item["ok"] else "not met"))
    safety_df = pd.DataFrame(safety_rows, columns=["check", "state"])

    reports_dir = config.DATA_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = reports_dir / "btc_paper_report.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
        summary_df.to_excel(xw, sheet_name="Summary", index=False)
        obs_df.to_excel(xw, sheet_name="Observations", index=False)
        signals_df.to_excel(xw, sheet_name="Signals Only", index=False)
        reasons_df.to_excel(xw, sheet_name="Reason Codes", index=False)
        fwd_df.to_excel(xw, sheet_name="Forward Returns", index=False)
        config_df.to_excel(xw, sheet_name="Candidate Config", index=False)
        safety_df.to_excel(xw, sheet_name="Safety Status", index=False)
    return str(xlsx_path)


TAX_COLUMNS = [
    "entry_time", "exit_time", "asset", "quantity", "entry_price_usd",
    "exit_price_usd", "entry_fee_usd", "exit_fee_usd", "cost_basis_usd",
    "proceeds_usd", "realized_gain_loss_usd", "holding_period_days",
    "short_or_long_term", "strategy_id", "candidate_version",
    "exit_reason", "paper_only",
]
