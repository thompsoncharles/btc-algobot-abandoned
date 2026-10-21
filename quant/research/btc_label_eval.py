"""
BTC label-definition experiment harness.

The current 28% OOS win rate is a symptom, not a knob. This module tests
whether the supervised target is wrong for the trade the engine is actually
trying to make: a fee-aware, long/flat BTC trade with stops, targets, and a
maximum holding period.

Run through the pipeline command:

    python -m quant.pipeline btc-label-eval

It is research-only. It writes a JSON report but does not change btc-eval
defaults or promote a model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from quant import config
from quant.backtest.validation import walk_forward_splits


BTC_TICKER = "BTCUSD"
FEATURE_SETS = {
    "ar_only": set(),
    "ar+momentum": {"momentum"},
    "ar+trend": {"trend"},
    "ar+volatility": {"volatility"},
    "ar+flow": {"flow"},
    "ar+external": {"external"},
    "full": None,
    "full+external": {"momentum", "trend", "volatility", "flow", "external"},
}
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0]


@dataclass(frozen=True)
class LabelSpec:
    name: str
    horizon: int
    description: str


LABEL_SPECS = [
    LabelSpec(
        "next_bar_up",
        1,
        "legacy target: next bar close higher than current close",
    ),
    LabelSpec(
        "cost_3bar",
        3,
        "3-bar forward return clears round-trip commission plus slippage",
    ),
    LabelSpec(
        "cost_5bar",
        5,
        "5-bar forward return clears round-trip commission plus slippage",
    ),
    LabelSpec(
        "trade_horizon_net",
        config.BTC_MAX_HOLD_BARS,
        "max-hold forward return clears the 2x edge-to-cost gate",
    ),
    LabelSpec(
        "target_before_stop",
        config.BTC_MAX_HOLD_BARS,
        "target is hit before stop inside the max holding window",
    ),
]


def _position_notional() -> float:
    equity = config.ACCOUNT_EQUITY_USD
    risk_dollars = equity * config.BTC_RISK_PER_TRADE_PCT / 100
    raw = risk_dollars / (config.BTC_STOP_PCT / 100)
    cap = equity * config.BTC_MAX_POSITION_PCT / 100
    return min(max(raw, config.BTC_MIN_TRADE_NOTIONAL_USD), cap)


def _engine_kwargs(notional: float) -> dict:
    from quant.execution.fees import ibkr_crypto_fee

    return dict(
        initial_equity=config.ACCOUNT_EQUITY_USD,
        max_positions=1,
        prob_entry=config.BTC_PROB_ENTRY,
        stop_pct=config.BTC_STOP_PCT,
        target_pct=config.BTC_TARGET_PCT,
        max_hold_bars=config.BTC_MAX_HOLD_BARS,
        commission_fn=ibkr_crypto_fee,
        slippage_bps=config.BTC_SLIPPAGE_BPS,
        allow_fractional=True,
        position_notional=notional,
        min_trade_notional=config.BTC_MIN_TRADE_NOTIONAL_USD,
    )


def _all_in_trade_cost_pct(notional: float) -> float:
    from quant.execution.fees import round_trip_cost_pct

    # slippage_bps is one side; 1 bp = 0.01%.
    round_trip_slippage_pct = 2 * config.BTC_SLIPPAGE_BPS / 100
    return round_trip_cost_pct(notional) + round_trip_slippage_pct


def build_label(bars: pd.DataFrame, spec: LabelSpec, notional: float) -> pd.DataFrame:
    """Return columns y and fwd_return for the requested label."""
    close = bars["Close"].astype(float)
    fwd = close.shift(-spec.horizon) / close - 1
    all_in_cost = _all_in_trade_cost_pct(notional) / 100

    if spec.name == "next_bar_up":
        y = (fwd > 0).astype(float)
    elif spec.name in {"cost_3bar", "cost_5bar"}:
        y = (fwd > all_in_cost).astype(float)
    elif spec.name == "trade_horizon_net":
        hurdle = all_in_cost * config.BTC_MIN_EDGE_TO_COST
        y = (fwd > hurdle).astype(float)
    elif spec.name == "target_before_stop":
        y = _target_before_stop_label(bars)
    else:
        raise ValueError(f"unknown label: {spec.name}")

    out = pd.DataFrame({"y": y, "fwd_return": fwd}, index=bars.index)
    out.loc[fwd.isna(), ["y", "fwd_return"]] = np.nan
    return out


def _target_before_stop_label(bars: pd.DataFrame) -> pd.Series:
    high = bars["High"].astype(float).to_numpy()
    low = bars["Low"].astype(float).to_numpy()
    close = bars["Close"].astype(float).to_numpy()
    n = len(bars)
    y = np.full(n, np.nan)
    target_mult = 1 + config.BTC_TARGET_PCT / 100
    stop_mult = 1 - config.BTC_STOP_PCT / 100

    for i in range(n):
        end = min(n, i + 1 + config.BTC_MAX_HOLD_BARS)
        if end <= i + 1:
            continue
        target = close[i] * target_mult
        stop = close[i] * stop_mult
        label = 0.0
        for j in range(i + 1, end):
            stopped = low[j] <= stop
            targeted = high[j] >= target
            if stopped and targeted:
                # Intrabar order is unknown. Count the mixed bar as failure so
                # the label does not get optimistic.
                label = 0.0
                break
            if stopped:
                label = 0.0
                break
            if targeted:
                label = 1.0
                break
        y[i] = label
    y[-config.BTC_MAX_HOLD_BARS:] = np.nan
    return pd.Series(y, index=bars.index)


def _make_model(groups, C: float, lag0: bool, class_weight):
    from quant.models.linear_ar import LinearBTCModel

    return LinearBTCModel(
        n_lags=config.BTC_AR_LAGS,
        forecast_horizon=config.BTC_FORECAST_HORIZON_BARS,
        ma_window=config.BTC_MA_WINDOW,
        C=C,
        include_lag0=lag0,
        groups=groups,
        class_weight=class_weight,
    )


def _fit_model(train_bars: pd.DataFrame, spec: LabelSpec, notional: float,
               groups, C: float, lag0: bool, class_weight):
    model = _make_model(groups, C, lag0, class_weight)
    X = model.build_features(train_bars)
    labels = build_label(train_bars, spec, notional)
    df = X.join(labels["y"]).replace([np.inf, -np.inf], np.nan).dropna()
    if df["y"].nunique() < 2:
        raise ValueError("training label has only one class")
    model.fit(df.drop(columns="y"), df["y"].astype(int))
    return model


def _gated_signal_fn(model, notional: float, gates: Dict[str, int],
                     feature_cache: Optional[pd.DataFrame] = None,
                     regime_filter=None):
    from quant.execution.fees import edge_clears_costs
    from quant.features.btc_features import latest_vol_pctile
    from quant.probability.engine import expected_return, prob_target_before_stop

    def signal_fn(day, history: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        gates["evaluated"] += 1
        h = history.get(BTC_TICKER)
        if h is None or len(h) < 60:
            gates["no_history"] += 1
            return {}
        if regime_filter is not None and not regime_filter(h):
            gates["regime"] = gates.get("regime", 0) + 1
            return {}
        if feature_cache is not None:
            feat = feature_cache.reindex([h.index[-1]]).dropna()
        else:
            feat = model.latest_features(h)
        if feat.empty:
            gates["no_features"] += 1
            return {}
        prob = float(model.predict_proba(feat)[0])
        if abs(prob - 0.5) < config.BTC_MIN_CONFIDENCE:
            gates["confidence"] += 1
            return {}
        if prob < config.BTC_PROB_ENTRY:
            gates["entry_probability"] += 1
            return {}
        if latest_vol_pctile(h) < config.BTC_MIN_VOL_PCTILE:
            gates["volatility"] += 1
            return {}
        hit = prob_target_before_stop(prob, config.BTC_TARGET_PCT, config.BTC_STOP_PCT)
        edge_pct = expected_return(hit, config.BTC_TARGET_PCT, config.BTC_STOP_PCT)
        if not edge_clears_costs(edge_pct, notional):
            gates["edge_to_cost"] += 1
            return {}
        gates["passed"] += 1
        return {BTC_TICKER: prob}

    return signal_fn


def run_candidate(bars: pd.DataFrame, spec: LabelSpec, splits, notional: float,
                  groups=None, C: float = 1.0, lag0: bool = False,
                  class_weight=None, tag: str = "",
                  regime_filter=None, keep_arrays: bool = False) -> dict:
    from sklearn.metrics import roc_auc_score

    from quant.backtest.engine import BacktestEngine
    from quant.metrics.performance import max_drawdown, profit_factor

    engine_kwargs = _engine_kwargs(notional)
    labels_full = build_label(bars, spec, notional)
    fold_rows = []
    all_trades: List[pd.DataFrame] = []
    equity_rows: List[pd.DataFrame] = []
    all_prob: List[float] = []
    all_y: List[int] = []
    all_fwd: List[float] = []
    gates = {
        "evaluated": 0,
        "no_history": 0,
        "no_features": 0,
        "confidence": 0,
        "entry_probability": 0,
        "volatility": 0,
        "edge_to_cost": 0,
        "passed": 0,
    }

    for split in splits:
        test_start = int(split.test_idx[0])
        test_end = int(split.test_idx[-1])
        safe_train_end = max(0, test_start - spec.horizon)
        if safe_train_end < 120:
            continue
        train_bars = bars.iloc[:safe_train_end]

        try:
            model = _fit_model(train_bars, spec, notional, groups, C, lag0, class_weight)
        except Exception as exc:
            logger.warning("{} fold {} fit failed: {}", spec.name, split.label, exc)
            continue
        Xfull = model.build_features(bars).replace([np.inf, -np.inf], np.nan)

        warmup_start = max(0, test_start - 252)
        sub_bars = bars.iloc[warmup_start:test_end + 1]
        signal_fn = _gated_signal_fn(model, notional, gates, Xfull,
                                     regime_filter=regime_filter)
        result = BacktestEngine(
            {BTC_TICKER: sub_bars},
            trade_start=bars.index[test_start],
            **engine_kwargs,
        ).run(signal_fn)

        eq = result.equity_curve
        trades = result.trades if result.trades is not None else pd.DataFrame()
        if not trades.empty:
            trades = trades.copy()
            trades["fold"] = split.label
            all_trades.append(trades)
        if eq is not None and len(eq):
            equity_rows.append(pd.DataFrame({
                "fold": split.label,
                "date": eq.index,
                "equity": eq.values,
            }))
        fold_rows.append({
            "fold": split.label,
            "train_bars": int(len(train_bars)),
            "oos_bars": int(len(split.test_idx)),
            "n_trades": int(len(trades)),
            "net_return_pct": float((eq.iloc[-1] / eq.iloc[0] - 1) * 100) if len(eq) > 1 else 0.0,
            "max_drawdown_pct": float(max_drawdown(eq) * 100) if len(eq) else 0.0,
            "net_expectancy_usd": float(trades["pnl"].mean()) if not trades.empty else 0.0,
        })

        test_ts = bars.index[split.test_idx]
        rows = Xfull.reindex(test_ts).replace([np.inf, -np.inf], np.nan).dropna()
        valid = rows.index.intersection(labels_full.dropna().index)
        if len(valid):
            prob = model.predict_proba(rows.loc[valid])
            y = labels_full.loc[valid, "y"].astype(int)
            fwd = labels_full.loc[valid, "fwd_return"].astype(float)
            all_prob.extend(map(float, prob))
            all_y.extend(map(int, y))
            all_fwd.extend(map(float, fwd))

    folds = pd.DataFrame(fold_rows)
    pooled = pd.concat(all_trades) if all_trades else pd.DataFrame()
    equity = pd.concat(equity_rows) if equity_rows else pd.DataFrame()
    fold_ret = folds["net_return_pct"].to_numpy(dtype=float) if not folds.empty else np.array([])
    auc = float("nan")
    if len(set(all_y)) == 2:
        auc = float(roc_auc_score(np.array(all_y), np.array(all_prob)))

    return {
        "tag": tag,
        "label": spec.name,
        "label_description": spec.description,
        "features": _feature_label(groups),
        "C": C,
        "lag0": lag0,
        "class_weight": "balanced" if class_weight == "balanced" else "none",
        "n_folds": int(len(fold_rows)),
        "total_oos_trades": int(len(pooled)),
        "mean_fold_net_return_pct": float(fold_ret.mean()) if len(fold_ret) else 0.0,
        "median_fold_net_return_pct": float(np.median(fold_ret)) if len(fold_ret) else 0.0,
        "min_fold_net_return_pct": float(fold_ret.min()) if len(fold_ret) else 0.0,
        "frac_folds_positive": float((fold_ret > 0).mean()) if len(fold_ret) else 0.0,
        "worst_fold_drawdown_pct": float(folds["max_drawdown_pct"].min()) if not folds.empty else 0.0,
        "pooled_net_expectancy_usd": float(pooled["pnl"].mean()) if not pooled.empty else 0.0,
        "pooled_net_win_rate_pct": float((pooled["pnl"] > 0).mean() * 100) if not pooled.empty else 0.0,
        "pooled_profit_factor": float(profit_factor(pooled)) if not pooled.empty else 0.0,
        "pooled_total_fees_usd": float(pooled["commission"].sum()) if "commission" in pooled.columns else 0.0,
        "exit_reasons": pooled["exit_reason"].value_counts().to_dict() if "exit_reason" in pooled.columns else {},
        "auc": auc,
        "positive_label_rate_pct": float(np.mean(all_y) * 100) if all_y else 0.0,
        "gate_counts": gates,
        "gate_pass_rate_pct": float(gates["passed"] / gates["evaluated"] * 100) if gates["evaluated"] else 0.0,
        "calibration": calibration_table(all_prob, all_y, all_fwd),
        "folds": fold_rows,
        "_trades": pooled.to_dict("records") if not pooled.empty else [],
        "_equity": equity.to_dict("records") if not equity.empty else [],
        **({"_prob": all_prob, "_y": all_y, "_fwd": all_fwd} if keep_arrays else {}),
    }


def calibration_table(probs: Iterable[float], labels: Iterable[int],
                      fwd_returns: Iterable[float]) -> List[dict]:
    p = np.asarray(list(probs), dtype=float)
    y = np.asarray(list(labels), dtype=float)
    r = np.asarray(list(fwd_returns), dtype=float)
    if len(p) == 0:
        return []
    edges = [0.0, 0.45, 0.50, 0.55, 0.58, 0.62, 0.66, 0.70, 1.01]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi)
        if not mask.any():
            continue
        rows.append({
            "prob_bucket": f"[{lo:.2f},{hi:.2f})",
            "n": int(mask.sum()),
            "label_hit_rate_pct": float(y[mask].mean() * 100),
            "avg_fwd_return_pct": float(r[mask].mean() * 100),
        })
    return rows


def _feature_label(groups) -> str:
    if groups is None:
        return "full"
    if not groups:
        return "ar_only"
    return "ar+" + "+".join(sorted(groups))


def _candidate_key(label: str, features: str, C: float, lag0: bool, class_weight) -> tuple:
    return (label, features, float(C), bool(lag0), "balanced" if class_weight == "balanced" else "none")


def _strip_raw(result: dict) -> dict:
    return {k: v for k, v in result.items() if not k.startswith("_")}


def _rank_key(result: dict) -> tuple:
    return (
        result["pooled_net_expectancy_usd"],
        result["frac_folds_positive"],
        result["pooled_profit_factor"],
        result["mean_fold_net_return_pct"],
    )


def _promotion_verdict(result: dict, baselines: list) -> str:
    hold = next((b for b in baselines if b["strategy"] == "buy_and_hold"), {})
    never = next((b for b in baselines if b["strategy"] == "never_trade"), {})
    hold_ret = hold.get("net_return_pct", 0.0)
    never_ret = never.get("net_return_pct", 0.0)
    trades = result["total_oos_trades"]
    exp = result["pooled_net_expectancy_usd"]
    folds = result["frac_folds_positive"]
    pf = result["pooled_profit_factor"]
    ret = result["mean_fold_net_return_pct"]

    if trades < 10:
        return "REJECT: too few trades to trust."
    if exp <= 0:
        return f"REJECT: negative net expectancy (${exp:.2f}/trade)."
    if pf < 1.2:
        return f"REJECT: profit factor {pf:.2f} is too weak after fees."
    if folds < 0.6:
        return f"REJECT: only {folds:.0%} of folds positive."
    if ret <= never_ret:
        return "REJECT: does not beat never-trade."
    if ret <= hold_ret:
        return f"REJECT: does not beat buy-and-hold ({ret:.2f}% vs {hold_ret:.2f}%)."
    return "PAPER ONLY: label candidate clears numeric gates; still needs live observation."


def _data_quality_gate(bars: pd.DataFrame) -> dict:
    from quant.data.btc_audit import (
        _basic_report,
        _normalize_index,
    )

    normalized = _normalize_index(bars)
    report = _basic_report(normalized, config.BTC_INTERVAL,
                           "local research cache", BTC_TICKER)
    missing = report.get("missing_bars", {}).get("missing_count")
    bars_behind = report.get("bars_behind_expected")
    ready = bool(
        report.get("available")
        and report.get("duplicate_bars", 1) == 0
        and (missing is None or missing <= 5)
        and (bars_behind is None or bars_behind <= 1)
        and not report.get("possible_unfinished_latest_candle", False)
    )
    report["research_ready"] = ready
    report["gate"] = (
        "PASS"
        if ready
        else "WARN: data has freshness/gap issues; use this as research, not promotion."
    )
    return report


def run_label_eval(bars: Optional[pd.DataFrame] = None) -> dict:
    from quant.backtest.baselines import run_all
    from quant.backtest.regimes import regime_breakdown
    from quant.data.store import DataStore
    from quant.execution.fees import round_trip_cost_pct

    if bars is None:
        bars = DataStore().load_bars(BTC_TICKER, config.BTC_INTERVAL)
    if bars is None or len(bars) < 300:
        raise SystemExit("no usable BTC history - run `python -m quant.pipeline btc-ingest` first")

    bars = bars.sort_index()
    notional = _position_notional()
    splits = walk_forward_splits(bars.index, n_folds=5, min_train_frac=0.4, embargo=5)
    if not splits:
        raise SystemExit("not enough BTC history for walk-forward splits")

    data_gate = _data_quality_gate(bars)
    oos_start = int(splits[0].test_idx[0])
    oos_bars = bars.iloc[oos_start:]
    baselines = run_all(oos_bars, config.ACCOUNT_EQUITY_USD, ma_window=config.BTC_MA_WINDOW)

    logger.info("label sweep: {} labels x {} folds", len(LABEL_SPECS), len(splits))
    label_results = [
        run_candidate(
            bars, spec, splits, notional,
            groups=None, C=1.0, lag0=False, class_weight=None,
            tag="label_sweep",
        )
        for spec in LABEL_SPECS
    ]
    best_label = max(label_results, key=_rank_key)
    best_spec = next(s for s in LABEL_SPECS if s.name == best_label["label"])

    variants = []
    seen = set()

    def add_variant(features_name, groups, C, lag0, class_weight, tag):
        key = _candidate_key(best_spec.name, features_name, C, lag0, class_weight)
        if key in seen:
            return
        seen.add(key)
        variants.append(run_candidate(
            bars, best_spec, splits, notional,
            groups=groups, C=C, lag0=lag0, class_weight=class_weight, tag=tag,
        ))

    for name, groups in FEATURE_SETS.items():
        add_variant(name, groups, 1.0, False, None, "feature_sweep")
    for C in C_GRID:
        add_variant("full", None, C, False, None, "C_sweep")
        add_variant("full", None, C, False, "balanced", "C_class_weight_sweep")
    add_variant("full", None, 1.0, True, None, "lag0_ab")

    all_results = label_results + variants
    best_overall = max(all_results, key=_rank_key)
    regimes = regime_breakdown(oos_bars, best_overall.get("_trades", []),
                               best_overall.get("_equity", []))
    verdict = _promotion_verdict(best_overall, baselines)

    report = {
        "oos_span": {
            "from": str(oos_bars.index.min()),
            "to": str(oos_bars.index.max()),
            "bars": int(len(oos_bars)),
        },
        "data_quality_gate": data_gate,
        "position_notional_usd": notional,
        "round_trip_commission_pct": round(round_trip_cost_pct(notional), 4),
        "all_in_round_trip_cost_pct": round(_all_in_trade_cost_pct(notional), 4),
        "current_gates": {
            "prob_entry": config.BTC_PROB_ENTRY,
            "min_confidence": config.BTC_MIN_CONFIDENCE,
            "min_vol_pctile": config.BTC_MIN_VOL_PCTILE,
            "min_edge_to_cost": config.BTC_MIN_EDGE_TO_COST,
            "stop_pct": config.BTC_STOP_PCT,
            "target_pct": config.BTC_TARGET_PCT,
            "max_hold_bars": config.BTC_MAX_HOLD_BARS,
        },
        "label_sweep": [_strip_raw(r) for r in label_results],
        "best_label": _strip_raw(best_label),
        "variant_sweep_for_best_label": [_strip_raw(r) for r in variants],
        "best_overall": _strip_raw(best_overall),
        "baselines": baselines,
        "regime_breakdown_best_overall": regimes,
        "verdict": verdict,
        "note": (
            "Research-only. Do not change live/default btc-eval settings unless "
            "a candidate clears the verdict gates and survives a separate robustness pass."
        ),
    }

    out = config.DATA_DIR / "btc_label_eval_result.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    return report


def print_report(report: dict) -> None:
    rows = []
    for row in report["label_sweep"]:
        rows.append({
            "label": row["label"],
            "trades": row["total_oos_trades"],
            "win%": round(row["pooled_net_win_rate_pct"], 2),
            "exp$/trade": round(row["pooled_net_expectancy_usd"], 2),
            "PF": round(row["pooled_profit_factor"], 3),
            "folds+": round(row["frac_folds_positive"], 2),
            "AUC": round(row["auc"], 3) if not np.isnan(row["auc"]) else None,
        })
    variants = pd.DataFrame(report["variant_sweep_for_best_label"])
    if not variants.empty:
        variants = variants.sort_values(
            ["pooled_net_expectancy_usd", "frac_folds_positive", "pooled_profit_factor"],
            ascending=False,
        )

    print(json.dumps({
        "oos_span": report["oos_span"],
        "data_quality_gate": {
            "gate": report["data_quality_gate"]["gate"],
            "bars": report["data_quality_gate"].get("bars"),
            "to": report["data_quality_gate"].get("to"),
            "missing_bars": report["data_quality_gate"].get("missing_bars"),
            "bars_behind_expected": report["data_quality_gate"].get("bars_behind_expected"),
        },
        "label_sweep": rows,
        "best_label": {
            "label": report["best_label"]["label"],
            "exp_per_trade": round(report["best_label"]["pooled_net_expectancy_usd"], 2),
            "win_rate_pct": round(report["best_label"]["pooled_net_win_rate_pct"], 2),
            "trades": report["best_label"]["total_oos_trades"],
        },
        "top_variants": (
            variants[[
                "label", "features", "C", "lag0", "class_weight",
                "total_oos_trades", "pooled_net_win_rate_pct",
                "pooled_net_expectancy_usd", "pooled_profit_factor",
                "frac_folds_positive", "auc",
            ]].head(10).to_dict("records")
            if not variants.empty else []
        ),
        "best_overall": {
            k: report["best_overall"][k]
            for k in [
                "label", "features", "C", "lag0", "class_weight",
                "total_oos_trades", "pooled_net_win_rate_pct",
                "pooled_net_expectancy_usd", "pooled_profit_factor",
                "mean_fold_net_return_pct", "frac_folds_positive", "auc",
                "gate_counts",
            ]
        },
        "baselines": report["baselines"],
        "regime_breakdown_best_overall": report["regime_breakdown_best_overall"],
        "verdict": report["verdict"],
        "full_result": str(config.DATA_DIR / "btc_label_eval_result.json"),
    }, indent=2, default=str))


def main() -> None:
    config.ensure_dirs()
    print_report(run_label_eval())


if __name__ == "__main__":
    main()
