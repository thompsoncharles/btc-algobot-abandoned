"""
BTC paper-observation logger.

This module records what the BTC model would have done on the latest cached
bar without placing orders. It is intentionally one-way: prediction and gate
diagnostics go to the journal, and no IBKR order/execution code is imported.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from quant import config
from quant.execution.fees import round_trip_cost, round_trip_cost_pct
from quant.features.btc_features import latest_vol_pctile
from quant.journal import JournalLogger
from quant.probability.engine import expected_return, prob_target_before_stop


def _float_dict(row: pd.Series, max_items: int = 40) -> dict:
    out = {}
    for key, val in row.iloc[:max_items].items():
        try:
            out[str(key)] = float(val)
        except Exception:
            out[str(key)] = None
    return out


def _hypothetical_outcome(bars: pd.DataFrame, signal_pos: int,
                          entry_price: Optional[float]) -> Optional[dict]:
    """
    Outcome for historical observation rows only. For the latest live row this
    returns None because no future bars exist yet.
    """
    if entry_price is None or signal_pos + 1 >= len(bars):
        return None
    entry_pos = signal_pos + 1
    stop = entry_price * (1 - config.BTC_STOP_PCT / 100)
    target = entry_price * (1 + config.BTC_TARGET_PCT / 100)
    max_exit = min(len(bars) - 1, entry_pos + config.BTC_MAX_HOLD_BARS)
    for pos in range(entry_pos, max_exit + 1):
        row = bars.iloc[pos]
        if float(row["Low"]) <= stop:
            return {"exit_ts": str(bars.index[pos]), "reason": "stop",
                    "exit_price": stop, "return_pct": -config.BTC_STOP_PCT}
        if float(row["High"]) >= target:
            return {"exit_ts": str(bars.index[pos]), "reason": "target",
                    "exit_price": target, "return_pct": config.BTC_TARGET_PCT}
    exit_price = float(bars["Close"].iloc[max_exit])
    return {"exit_ts": str(bars.index[max_exit]), "reason": "time",
            "exit_price": exit_price,
            "return_pct": (exit_price / entry_price - 1) * 100}


def build_observation(model, bars: pd.DataFrame, notional: float,
                      signal_pos: Optional[int] = None,
                      data_source: str = "local BTCUSD cache") -> dict:
    bars = bars.sort_index()
    signal_pos = len(bars) - 1 if signal_pos is None else signal_pos
    history = bars.iloc[:signal_pos + 1]
    signal_ts = history.index[-1]
    latest_bar = history.iloc[-1]

    feat = model.latest_features(history)
    gates = []
    if feat.empty:
        return {
            "mode": "paper_observation",
            "places_orders": False,
            "signal_ts": str(signal_ts),
            "data_source": data_source,
            "gates": [{"name": "features_available", "passed": False}],
        }

    prob = float(model.predict_proba(feat)[0])
    confidence = abs(prob - 0.5)
    vol_pctile = latest_vol_pctile(history)
    hit = prob_target_before_stop(prob, config.BTC_TARGET_PCT, config.BTC_STOP_PCT)
    edge_pct = expected_return(hit, config.BTC_TARGET_PCT, config.BTC_STOP_PCT)
    edge_dollars = edge_pct / 100 * notional
    cost_dollars = round_trip_cost(notional)

    gates.append({"name": "confidence", "passed": confidence >= config.BTC_MIN_CONFIDENCE,
                  "value": confidence, "threshold": config.BTC_MIN_CONFIDENCE})
    gates.append({"name": "entry_probability", "passed": prob >= config.BTC_PROB_ENTRY,
                  "value": prob, "threshold": config.BTC_PROB_ENTRY})
    gates.append({"name": "volatility_floor", "passed": vol_pctile >= config.BTC_MIN_VOL_PCTILE,
                  "value": vol_pctile, "threshold": config.BTC_MIN_VOL_PCTILE})
    gates.append({"name": "edge_to_cost", "passed": edge_dollars >= config.BTC_MIN_EDGE_TO_COST * cost_dollars,
                  "value": edge_dollars / cost_dollars if cost_dollars else None,
                  "threshold": config.BTC_MIN_EDGE_TO_COST})
    would_enter = all(g["passed"] for g in gates)

    next_open = None
    if signal_pos + 1 < len(bars):
        next_open = float(bars["Open"].iloc[signal_pos + 1])

    return {
        "mode": "paper_observation",
        "places_orders": False,
        "ticker": "BTCUSD",
        "data_source": data_source,
        "signal_ts": str(signal_ts),
        "latest_bar": {
            "Open": float(latest_bar["Open"]),
            "High": float(latest_bar["High"]),
            "Low": float(latest_bar["Low"]),
            "Close": float(latest_bar["Close"]),
            "Volume": float(latest_bar["Volume"]) if pd.notna(latest_bar.get("Volume")) else None,
        },
        "features": _float_dict(feat.iloc[-1]),
        "predicted_prob_up": prob,
        "gates": gates,
        "would_enter": would_enter,
        "expected_edge_pct": edge_pct,
        "expected_edge_usd": edge_dollars,
        "estimated_round_trip_fees_usd": cost_dollars,
        "estimated_round_trip_cost_pct": round_trip_cost_pct(notional),
        "hypothetical": {
            "entry_ts": str(bars.index[signal_pos + 1]) if signal_pos + 1 < len(bars) else None,
            "entry_price": next_open,
            "notional_usd": notional if would_enter else 0.0,
            "outcome": _hypothetical_outcome(bars, signal_pos, next_open) if would_enter else None,
        },
    }


def load_or_fit_model(bars: pd.DataFrame):
    from quant.models.registry import ModelRegistry
    from quant.pipeline import BTC_MODEL_NAME, _btc_model

    try:
        model, meta = ModelRegistry().load(BTC_MODEL_NAME)
        if not hasattr(model, "include_lag0"):
            model.include_lag0 = False
        if model.__class__.__name__ == "LinearBTCModel" and not hasattr(model, "groups"):
            model.groups = None
        return model, f"registry:{BTC_MODEL_NAME}:v{meta.get('version')}"
    except FileNotFoundError:
        model = _btc_model().fit_from_bars(bars)
        return model, "fit_from_cached_bars"


def log_observations(bars: pd.DataFrame, notional: float, count: int = 1) -> dict:
    model, model_source = load_or_fit_model(bars)
    journal = JournalLogger()
    count = max(1, int(count))
    start = max(0, len(bars) - count)
    records = []
    for pos in range(start, len(bars)):
        rec = build_observation(model, bars, notional, signal_pos=pos)
        rec["model_source"] = model_source
        journal.log_prediction(
            "BTCUSD",
            rec.get("predicted_prob_up", 0.5),
            "paper_observation",
            model_version=model_source,
        )
        journal._write("btc_observations", rec)
        records.append(rec)
    return {
        "places_orders": False,
        "records_logged": len(records),
        "model_source": model_source,
        "journal": str(config.JOURNAL_DIR / "btc_observations.jsonl"),
        "latest_record": records[-1] if records else None,
    }
