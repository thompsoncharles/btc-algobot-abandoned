"""
Online probability calibration experiment (LAB, research-only).

    python -m quant.research.prob_calibration

Question: the frozen daily candidate's probabilities are usable but coarse.
Does per-fold calibration (Platt / isotonic / rolling-window Platt) improve
(A) probability quality and (B) actual OOS trading through the existing
confidence / entry / edge-to-cost gates - WITHOUT touching the model,
features, filter, or thresholds?

Leakage protocol (per walk-forward fold):
  1. Split the fold's TRAINING window 75/25 chronologically.
  2. Fit the base model on the first 75%, predict the held-out 25%.
  3. Fit each calibrator on those held-out (prediction, label) pairs only.
  4. Refit the base model on the FULL training window (exactly what the
     frozen candidate does), chain the calibrator onto its OOS outputs.
  5. OOS labels are never seen by any calibrator; no full-sample fitting;
     the calibrator choice is fixed up front, not picked per fold.

Judge injection: run_candidate() from btc_label_eval is used UNCHANGED (the
same OOS judge as btc-eval/btc-signal-search). The only difference in a
calibrated run is a wrapper whose predict_proba applies the fold's
calibrator - injected by temporarily patching btc_label_eval._fit_model and
restored in a finally block. Gates, sizing, fees, filter: all identical.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from quant import config
from quant.backtest.validation import walk_forward_splits
from quant.research import btc_label_eval as ble
from quant.research.btc_label_eval import (
    LABEL_SPECS,
    _position_notional,
    build_label,
    run_candidate,
)
from quant.research.btc_signal_search import REGIME_FILTERS

BTC_TICKER = "BTCUSD"
INNER_TRAIN_FRAC = 0.75      # train-window split for calibrator fitting
ROLLING_WINDOW = 250         # most recent held-out rows for rolling Platt
EPS = 1e-6

REJECT_RULES = dict(min_trades=20, min_pf=1.2, min_folds_positive=0.6)


# ------------------------------------------------------------------ calibrators
class PlattCalibrator:
    """Sigmoid recalibration on the logit of the raw probability."""

    name = "platt"

    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        self.lr = LogisticRegression(C=1e6)  # near-unregularized a,b fit

    def fit(self, p: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        z = np.log(np.clip(p, EPS, 1 - EPS) / np.clip(1 - p, EPS, 1 - EPS))
        self.lr.fit(z.reshape(-1, 1), y)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        z = np.log(np.clip(p, EPS, 1 - EPS) / np.clip(1 - p, EPS, 1 - EPS))
        return self.lr.predict_proba(z.reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    name = "isotonic"

    def __init__(self):
        from sklearn.isotonic import IsotonicRegression
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")

    def fit(self, p: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        self.iso.fit(p, y)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return self.iso.predict(p)


class RollingPlattCalibrator(PlattCalibrator):
    """Platt fit only on the most recent ROLLING_WINDOW held-out rows."""

    name = "rolling_platt"

    def fit(self, p: np.ndarray, y: np.ndarray) -> "RollingPlattCalibrator":
        if len(p) > ROLLING_WINDOW:
            p, y = p[-ROLLING_WINDOW:], y[-ROLLING_WINDOW:]
        return super().fit(p, y)


CALIBRATOR_FACTORIES = {
    "platt": PlattCalibrator,
    "isotonic": IsotonicCalibrator,
    "rolling_platt": RollingPlattCalibrator,
}


# ------------------------------------------------------------- fold calibration
def fit_fold_calibrators(bars: pd.DataFrame, train_end: int, spec,
                         notional: float) -> Optional[Dict[str, object]]:
    """Fit every calibrator on the fold's train-internal 25% holdout."""
    train_bars = bars.iloc[:train_end]
    inner_end = int(len(train_bars) * INNER_TRAIN_FRAC)
    if inner_end < 150 or len(train_bars) - inner_end < 60:
        return None
    inner_bars = train_bars.iloc[:inner_end]
    try:
        model75 = ble._fit_model(inner_bars, spec, notional,
                                 groups=None, C=1.0, lag0=False, class_weight=None)
    except Exception as e:
        logger.warning("inner fit failed at train_end={}: {}", train_end, e)
        return None

    X = model75.build_features(train_bars).replace([np.inf, -np.inf], np.nan)
    labels = build_label(train_bars, spec, notional)
    hold_ts = train_bars.index[inner_end:]
    rows = X.reindex(hold_ts).dropna()
    valid = rows.index.intersection(labels.dropna().index)
    if len(valid) < 60:
        return None
    p = model75.predict_proba(rows.loc[valid])
    y = labels.loc[valid, "y"].astype(int).values
    if len(set(y.tolist())) < 2:
        return None

    fitted = {}
    for name, factory in CALIBRATOR_FACTORIES.items():
        try:
            fitted[name] = factory().fit(np.asarray(p, dtype=float), y)
        except Exception as e:
            logger.warning("calibrator {} failed: {}", name, e)
    return fitted or None


class CalibratedModel:
    """Delegates everything to the base model; only predict_proba is reshaped."""

    def __init__(self, base, calibrator):
        self._base = base
        self._cal = calibrator

    def predict_proba(self, X) -> np.ndarray:
        raw = np.asarray(self._base.predict_proba(X), dtype=float)
        return np.clip(self._cal.transform(raw), 0.0, 1.0)

    def __getattr__(self, item):
        return getattr(self._base, item)


# ------------------------------------------------------------------ mode A
def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def prob_quality(p: List[float], y: List[int], label: str) -> dict:
    p_arr, y_arr = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    if len(p_arr) < 50:
        return {"variant": label, "available": False}
    edges = [0.0, 0.45, 0.5, 0.55, 0.6, 0.65, 1.01]
    buckets = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p_arr >= lo) & (p_arr < hi)
        if m.sum() == 0:
            continue
        buckets.append({"bucket": f"[{lo:.2f},{hi:.2f})", "n": int(m.sum()),
                        "mean_predicted_pct": round(float(p_arr[m].mean()) * 100, 2),
                        "hit_rate_pct": round(float(y_arr[m].mean()) * 100, 2)})
    cut = np.quantile(p_arr, 0.9)
    top = y_arr[p_arr >= cut]
    base_rate = float(y_arr.mean())
    return {
        "variant": label,
        "n": int(len(p_arr)),
        "brier": round(brier(p_arr, y_arr), 5),
        "base_hit_rate_pct": round(base_rate * 100, 2),
        "decile_lift": round(float(top.mean()) / base_rate, 3) if base_rate > 0 and len(top) else None,
        "calibration_curve": buckets,
    }


def run_diagnostics(bars, splits, spec, notional,
                    fold_cals: Dict[int, Dict[str, object]]) -> dict:
    """Mode A: pooled OOS probability quality, raw vs calibrated. No trading."""
    pooled: Dict[str, List[float]] = {"raw": []}
    for name in CALIBRATOR_FACTORIES:
        pooled[name] = []
    pooled_y: List[int] = []

    labels_full = build_label(bars, spec, notional)
    for split in splits:
        train_end = int(split.train_idx[-1]) + 1
        cals = fold_cals.get(train_end)
        try:
            model = ble._fit_model(bars.iloc[:train_end], spec, notional,
                                   groups=None, C=1.0, lag0=False, class_weight=None)
        except Exception:
            continue
        X = model.build_features(bars).replace([np.inf, -np.inf], np.nan)
        test_ts = bars.index[split.test_idx]
        rows = X.reindex(test_ts).dropna()
        valid = rows.index.intersection(labels_full.dropna().index)
        if not len(valid):
            continue
        p_raw = np.asarray(model.predict_proba(rows.loc[valid]), dtype=float)
        y = labels_full.loc[valid, "y"].astype(int).values
        pooled["raw"].extend(map(float, p_raw))
        pooled_y.extend(map(int, y))
        for name in CALIBRATOR_FACTORIES:
            if cals and name in cals:
                pooled[name].extend(map(float, cals[name].transform(p_raw)))
            else:
                pooled[name].extend(map(float, p_raw))  # fold without calibrator: pass-through

    return {label: prob_quality(p, pooled_y, label) for label, p in pooled.items()}


# ------------------------------------------------------------------ mode B
def run_gated_backtest(bars, splits, spec, notional, variant: str,
                       fold_cals: Dict[int, Dict[str, object]]) -> dict:
    """
    Mode B: the full OOS judge with gates fed by calibrated probabilities.
    Injected by patching ble._fit_model to wrap the (identically fitted)
    base model; restored in finally. variant="raw" runs unpatched.
    """
    regime_filter = REGIME_FILTERS["trend200+vol"]
    if variant == "raw":
        res = run_candidate(bars, spec, splits, notional, groups=None,
                            C=1.0, lag0=False, class_weight=None,
                            tag="cal_race|raw", regime_filter=regime_filter)
        res["filter"] = "raw"
        return res

    original_fit = ble._fit_model

    def patched_fit(train_bars, spec_, notional_, groups, C, lag0, class_weight):
        base = original_fit(train_bars, spec_, notional_, groups, C, lag0, class_weight)
        cals = fold_cals.get(len(train_bars))
        if cals and variant in cals:
            return CalibratedModel(base, cals[variant])
        return base

    ble._fit_model = patched_fit
    try:
        res = run_candidate(bars, spec, splits, notional, groups=None,
                            C=1.0, lag0=False, class_weight=None,
                            tag=f"cal_race|{variant}", regime_filter=regime_filter)
    finally:
        ble._fit_model = original_fit
    res["filter"] = variant
    return res


# ------------------------------------------------------------------ evaluation
def _row(res: dict) -> dict:
    return {
        "variant": res["filter"],
        "trades": res["total_oos_trades"],
        "net_exp_usd": round(res["pooled_net_expectancy_usd"], 2),
        "pf": round(res["pooled_profit_factor"], 3),
        "win_pct": round(res["pooled_net_win_rate_pct"], 2),
        "mean_fold_ret_pct": round(res["mean_fold_net_return_pct"], 3),
        "frac_folds_positive": res["frac_folds_positive"],
        "worst_fold_dd_pct": round(res["worst_fold_drawdown_pct"], 2),
        "fold_returns_pct": [round(f["net_return_pct"], 3) for f in res["folds"]],
        "last_fold_ret_pct": round(res["folds"][-1]["net_return_pct"], 3) if res["folds"] else None,
        "gate_counts": res["gate_counts"],
        "entry_dates": sorted({str(t["entry_date"]) for t in res.get("_trades", [])}),
    }


def _reject_reasons(row: dict, baseline: dict, diag_better: bool) -> List[str]:
    reasons = []
    if row["trades"] < REJECT_RULES["min_trades"]:
        reasons.append(f"trades {row['trades']} < 20")
    if row["pf"] <= REJECT_RULES["min_pf"]:
        reasons.append(f"PF {row['pf']} <= 1.2")
    if row["net_exp_usd"] <= 0:
        reasons.append(f"expectancy ${row['net_exp_usd']} <= 0")
    if row["frac_folds_positive"] < REJECT_RULES["min_folds_positive"]:
        reasons.append(f"only {row['frac_folds_positive']:.0%} folds positive")
    if (row["last_fold_ret_pct"] is not None and baseline["last_fold_ret_pct"] is not None
            and row["last_fold_ret_pct"] < baseline["last_fold_ret_pct"] - 0.5):
        reasons.append(f"2025-2026 fold worse ({row['last_fold_ret_pct']}% vs "
                       f"{baseline['last_fold_ret_pct']}%)")
    hr, br = row["fold_returns_pct"], baseline["fold_returns_pct"]
    if hr and br and len(hr) == len(br) and row["mean_fold_ret_pct"] > baseline["mean_fold_ret_pct"]:
        h_trim = (sum(hr) - max(hr)) / max(len(hr) - 1, 1)
        b_trim = (sum(br) - max(br)) / max(len(br) - 1, 1)
        if h_trim <= b_trim:
            reasons.append("improvement disappears without its best fold (one lucky fold)")
    if diag_better and row["net_exp_usd"] < baseline["net_exp_usd"]:
        reasons.append("Brier improves but trading performance worsens")
    return reasons


def run_experiment() -> dict:
    from quant.data.store import DataStore

    config.ensure_dirs()
    bars = DataStore().load_bars(BTC_TICKER, "1d")
    if bars is None or len(bars) < 1000:
        raise SystemExit("daily BTC cache required")
    bars = bars.sort_index()
    splits = walk_forward_splits(bars.index, n_folds=5, min_train_frac=0.4, embargo=5)
    spec = next(s for s in LABEL_SPECS if s.name == "cost_5bar")
    notional = _position_notional()

    logger.info("fitting per-fold calibrators (train-internal 75/25 holdout) ...")
    fold_cals: Dict[int, Dict[str, object]] = {}
    for split in splits:
        train_end = int(split.train_idx[-1]) + 1
        cals = fit_fold_calibrators(bars, train_end, spec, notional)
        if cals:
            fold_cals[train_end] = cals

    logger.info("mode A: probability-quality diagnostics ...")
    diagnostics = run_diagnostics(bars, splits, spec, notional, fold_cals)

    logger.info("mode B: gate-using race on the OOS judge ...")
    results = {}
    for variant in ["raw"] + list(CALIBRATOR_FACTORIES):
        logger.info("racing variant: {}", variant)
        results[variant] = run_gated_backtest(bars, splits, spec, notional,
                                              variant, fold_cals)

    rows = {name: _row(res) for name, res in results.items()}
    baseline = rows["raw"]
    raw_brier = diagnostics["raw"].get("brier")

    verdicts = {}
    for name, row in rows.items():
        if name == "raw":
            continue
        v_brier = diagnostics.get(name, {}).get("brier")
        diag_better = (raw_brier is not None and v_brier is not None
                       and v_brier < raw_brier - 1e-5)
        reasons = _reject_reasons(row, baseline, diag_better)
        beats = (not reasons
                 and row["net_exp_usd"] > baseline["net_exp_usd"]
                 and row["mean_fold_ret_pct"] > baseline["mean_fold_ret_pct"]
                 and row["frac_folds_positive"] >= baseline["frac_folds_positive"])
        base_set, var_set = set(baseline["entry_dates"]), set(row["entry_dates"])
        verdicts[name] = {
            "beats_frozen": beats,
            "brier_improved": diag_better,
            "reject_reasons": reasons,
            "trades_added": len(var_set - base_set),
            "trades_removed": len(base_set - var_set),
        }

    winners = [n for n, v in verdicts.items() if v["beats_frozen"]]
    any_pass = [n for n, v in verdicts.items() if not v["reject_reasons"]]
    any_brier = [n for n, v in verdicts.items() if v["brier_improved"]]
    if winners:
        verdict = ("PROPOSE FUTURE PAPER CANDIDATE - " + ", ".join(winners) +
                   " beat the frozen candidate; proposal written, nothing promoted.")
    elif any_pass:
        verdict = ("WATCH CALIBRATION IN LAB - " + ", ".join(any_pass) +
                   " pass absolute gates without beating the frozen candidate.")
    elif any_brier:
        verdict = ("DIAGNOSTICS ONLY - calibration improves probability quality "
                   f"({', '.join(any_brier)}) but not trading through the gates; "
                   "useful as a reporting layer, not for gating.")
    else:
        verdict = "REJECT CALIBRATION - no quality nor trading improvement."

    for row in rows.values():
        row.pop("entry_dates", None)   # keep the JSON readable
        row.pop("gate_counts", None)

    report = {
        "mode": "lab_research_only",
        "places_orders": False,
        "leakage_protocol": "per-fold train-internal 75/25; calibrators never see OOS",
        "judge": "run_candidate (unchanged) - walk-forward OOS net backtest, "
                 "cost_5bar/full/1d/trend200+vol, real fees+sizing+gates",
        "mode_a_diagnostics": diagnostics,
        "mode_b_table": list(rows.values()),
        "verdicts": verdicts,
        "verdict": verdict,
    }
    out = config.DATA_DIR / "prob_calibration_result.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    return report


def print_report(report: dict) -> None:
    pd.set_option("display.width", 220)
    print("\n" + "=" * 100)
    print("MODE A - PROBABILITY QUALITY (pooled OOS, raw vs calibrated):")
    qual = []
    for name, q in report["mode_a_diagnostics"].items():
        if q.get("available") is False:
            continue
        qual.append({"variant": name, "n": q["n"], "brier": q["brier"],
                     "decile_lift": q["decile_lift"],
                     "base_hit_rate_pct": q["base_hit_rate_pct"]})
    print(pd.DataFrame(qual).to_string(index=False))
    print("\ncalibration curve (raw):")
    print(pd.DataFrame(report["mode_a_diagnostics"]["raw"]["calibration_curve"]).to_string(index=False))
    for name in CALIBRATOR_FACTORIES:
        print(f"\ncalibration curve ({name}):")
        print(pd.DataFrame(report["mode_a_diagnostics"][name]["calibration_curve"]).to_string(index=False))

    print("\n" + "=" * 100)
    print("MODE B - GATE-USING RACE (same judge, gates fed by calibrated probs):")
    df = pd.DataFrame(report["mode_b_table"])
    print(df.drop(columns=["fold_returns_pct"]).to_string(index=False))
    print("\nfold-by-fold net returns (%):")
    for row in report["mode_b_table"]:
        print(f"  {row['variant']:<15} {row['fold_returns_pct']}")
    print("\nper-variant verdicts:")
    for name, v in report["verdicts"].items():
        status = ("BEATS frozen candidate" if v["beats_frozen"] else
                  ("rejected: " + "; ".join(v["reject_reasons"])) if v["reject_reasons"]
                  else "passes gates, does not beat frozen")
        print(f"  {name:<15} {status}  (+{v['trades_added']}/-{v['trades_removed']} trades, "
              f"brier_improved={v['brier_improved']})")
    print("\nVERDICT:", report["verdict"])
    print("full result:", str(config.DATA_DIR / "prob_calibration_result.json"))


def main() -> None:
    print_report(run_experiment())


if __name__ == "__main__":
    main()
