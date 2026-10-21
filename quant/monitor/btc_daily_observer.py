"""
Daily BTC paper observer for the promoted btc-signal-search candidate.

Candidate (promoted 2026-07-04, PAPER OBSERVATION ONLY):
    dataset=1d, label=cost_5bar, features=full, regime filter=trend200+vol

This module NEVER places orders: it imports no IBKR/order/execution code,
every record carries places_orders=False, and the 12h btc-eval default path
is untouched. Each run logs, for the latest daily bar (or a backfill window):

    date, model probability, every gate with pass/fail, regime filter state,
    hypothetical entry/exit, expected edge, fees, and realized forward
    returns after 1/3/5 bars (when those bars exist yet).

Journal: quant_data/journal/btc_daily_observations.jsonl
Report:  python -m quant.pipeline btc-paper-report
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from quant import config
from quant.execution.fees import round_trip_cost, round_trip_cost_pct
from quant.journal import JournalLogger
from quant.probability.engine import expected_return, prob_target_before_stop

CANDIDATE = {
    "name": "btc_daily_cost5bar_full_trend200vol",
    "promoted_at": "2026-07-04",
    "promoted_by": "btc-signal-search (60-config walk-forward OOS grid)",
    "dataset": "1d",
    "label": "cost_5bar",
    "features": "full",
    "regime_filter": "trend200+vol",
    "status": "PAPER_OBSERVATION_ONLY",
    "live_trading": "disabled",
    "oos_reference": {"trades": 85, "net_expectancy_usd": 15.50,
                      "profit_factor": 1.58, "frac_folds_positive": 0.8},
}

JOURNAL_TABLE = "btc_daily_observations"

# Paper-verdict thresholds (reconsideration gate)
MIN_PAPER_SIGNALS = 20
MIN_PAPER_DAYS = 60


def fit_candidate_model(train_bars: pd.DataFrame):
    """Fit the promoted candidate exactly as the search judged it."""
    from quant.research.btc_label_eval import LABEL_SPECS, _fit_model, _position_notional

    spec = next(s for s in LABEL_SPECS if s.name == CANDIDATE["label"])
    return _fit_model(train_bars, spec, _position_notional(),
                      groups=None, C=1.0, lag0=False, class_weight=None)


def regime_state(history: pd.DataFrame) -> dict:
    """trend200+vol filter state, itemized so the journal shows why."""
    from quant.features.btc_features import latest_vol_pctile

    close = history["Close"].astype(float)
    trend_ok = len(close) >= 200 and float(close.iloc[-1]) > float(close.tail(200).mean())
    vol_pct = latest_vol_pctile(history)
    vol_ok = 0.20 <= vol_pct <= 0.90
    return {
        "trend200_above": bool(trend_ok),
        "vol_pctile": float(vol_pct),
        "vol_in_band_20_90": bool(vol_ok),
        "passed": bool(trend_ok and vol_ok),
    }


def _forward_returns(bars: pd.DataFrame, signal_pos: int) -> dict:
    """Realized close-to-close forward returns after 1/3/5 bars (None if future missing)."""
    close = bars["Close"].astype(float)
    base = float(close.iloc[signal_pos])
    out = {}
    for h in (1, 3, 5):
        pos = signal_pos + h
        out[f"fwd_{h}bar_pct"] = (
            round((float(close.iloc[pos]) / base - 1) * 100, 4)
            if pos < len(bars) else None
        )
    return out


def _hypothetical_trade(bars: pd.DataFrame, signal_pos: int,
                        notional: float) -> Optional[dict]:
    """Next-open entry, stop/target/time exit, net of fees - simulation only."""
    if signal_pos + 1 >= len(bars):
        return {"entry_ts": None, "entry_price": None, "outcome": None,
                "note": "entry bar not formed yet - resolves on next run"}
    entry_pos = signal_pos + 1
    entry = float(bars["Open"].iloc[entry_pos])
    stop = entry * (1 - config.BTC_STOP_PCT / 100)
    target = entry * (1 + config.BTC_TARGET_PCT / 100)
    max_exit = min(len(bars) - 1, entry_pos + config.BTC_MAX_HOLD_BARS)
    outcome = None
    for pos in range(entry_pos, max_exit + 1):
        row = bars.iloc[pos]
        if float(row["Low"]) <= stop:
            outcome = {"exit_ts": str(bars.index[pos]), "reason": "stop",
                       "exit_price": round(stop, 2), "gross_return_pct": -config.BTC_STOP_PCT}
            break
        if float(row["High"]) >= target:
            outcome = {"exit_ts": str(bars.index[pos]), "reason": "target",
                       "exit_price": round(target, 2), "gross_return_pct": config.BTC_TARGET_PCT}
            break
    if outcome is None:
        if max_exit == entry_pos + config.BTC_MAX_HOLD_BARS:
            exit_price = float(bars["Close"].iloc[max_exit])
            outcome = {"exit_ts": str(bars.index[max_exit]), "reason": "time",
                       "exit_price": round(exit_price, 2),
                       "gross_return_pct": round((exit_price / entry - 1) * 100, 4)}
        else:
            outcome = {"exit_ts": None, "reason": "open",
                       "exit_price": None, "gross_return_pct": None,
                       "note": "holding window not complete yet"}
    if outcome.get("gross_return_pct") is not None:
        fees = round_trip_cost(notional)
        net = outcome["gross_return_pct"] / 100 * notional - fees
        outcome["net_pnl_usd"] = round(net, 2)
    return {"entry_ts": str(bars.index[entry_pos]), "entry_price": round(entry, 2),
            "outcome": outcome}


def build_daily_observation(model, bars: pd.DataFrame, notional: float,
                            signal_pos: Optional[int] = None) -> dict:
    bars = bars.sort_index()
    signal_pos = len(bars) - 1 if signal_pos is None else signal_pos
    history = bars.iloc[:signal_pos + 1]
    signal_ts = history.index[-1]

    rec = {
        "mode": "paper_observation_daily",
        "places_orders": False,
        "candidate": CANDIDATE["name"],
        "ticker": "BTCUSD",
        "cadence": "1d",
        "signal_ts": str(signal_ts),
        "signal_date": str(pd.Timestamp(signal_ts).date()),
        "close": float(history["Close"].iloc[-1]),
    }

    regime = regime_state(history)
    rec["regime_filter"] = regime

    feat = model.latest_features(history)
    if feat.empty:
        rec["gates"] = [{"name": "features_available", "passed": False}]
        rec["would_enter"] = False
        return rec

    prob = float(model.predict_proba(feat)[0])
    confidence = abs(prob - 0.5)
    hit = prob_target_before_stop(prob, config.BTC_TARGET_PCT, config.BTC_STOP_PCT)
    edge_pct = expected_return(hit, config.BTC_TARGET_PCT, config.BTC_STOP_PCT)
    edge_usd = edge_pct / 100 * notional
    fees_usd = round_trip_cost(notional)

    gates = [
        {"name": "regime_trend200_vol", "passed": regime["passed"],
         "value": {"trend200": regime["trend200_above"],
                   "vol_pctile": regime["vol_pctile"]}},
        {"name": "confidence", "passed": confidence >= config.BTC_MIN_CONFIDENCE,
         "value": round(confidence, 4), "threshold": config.BTC_MIN_CONFIDENCE},
        {"name": "entry_probability", "passed": prob >= config.BTC_PROB_ENTRY,
         "value": round(prob, 4), "threshold": config.BTC_PROB_ENTRY},
        {"name": "volatility_floor",
         "passed": regime["vol_pctile"] >= config.BTC_MIN_VOL_PCTILE,
         "value": regime["vol_pctile"], "threshold": config.BTC_MIN_VOL_PCTILE},
        {"name": "edge_to_cost",
         "passed": edge_usd >= config.BTC_MIN_EDGE_TO_COST * fees_usd,
         "value": round(edge_usd / fees_usd, 3) if fees_usd else None,
         "threshold": config.BTC_MIN_EDGE_TO_COST},
    ]
    would_enter = all(g["passed"] for g in gates)

    rec.update({
        "predicted_prob_up": round(prob, 4),
        "gates": gates,
        "gates_failed": [g["name"] for g in gates if not g["passed"]],
        "would_enter": bool(would_enter),
        "expected_edge_pct": round(edge_pct, 4),
        "expected_edge_usd": round(edge_usd, 2),
        "round_trip_fees_usd": round(fees_usd, 2),
        "round_trip_fees_pct": round(round_trip_cost_pct(notional), 4),
        "notional_usd": notional if would_enter else 0.0,
        "realized": _forward_returns(bars, signal_pos),
        "hypothetical": _hypothetical_trade(bars, signal_pos, notional) if would_enter else None,
    })
    return rec


def log_daily_observations(bars: pd.DataFrame, notional: float,
                           count: int = 1) -> dict:
    """
    Log paper observations for the last `count` daily bars. The model is
    fit only on bars BEFORE the earliest logged position (no look-ahead;
    for backfills later rows use a slightly stale model, which is the
    conservative direction).
    """
    count = max(1, int(count))
    start = max(210, len(bars) - count)  # need 200-bar SMA + slack to signal at all
    train_bars = bars.iloc[:start]
    model = fit_candidate_model(train_bars)

    journal = JournalLogger()
    records = []
    for pos in range(start, len(bars)):
        rec = build_daily_observation(model, bars, notional, signal_pos=pos)
        rec["model_source"] = f"fit_on_{len(train_bars)}_daily_bars_pre_observation"
        journal.log_prediction("BTCUSD", rec.get("predicted_prob_up", 0.5),
                               "paper_observation_daily",
                               model_version=CANDIDATE["name"])
        journal._write(JOURNAL_TABLE, rec)
        records.append(rec)
    return {
        "places_orders": False,
        "candidate": CANDIDATE,
        "records_logged": len(records),
        "journal": str(config.JOURNAL_DIR / f"{JOURNAL_TABLE}.jsonl"),
        "latest_record": records[-1] if records else None,
    }
