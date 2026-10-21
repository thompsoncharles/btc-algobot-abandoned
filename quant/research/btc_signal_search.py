"""
BTC signal search: one command that hunts for real predictive signal across
labels x features x regime filters x datasets, judged only by the existing
no-leakage walk-forward OOS net backtest (same fees, sizing, and gates as
btc-eval).

    python -m quant.pipeline btc-signal-search [--offline]

Datasets:
  12h  trading-resolution cache (2024+, one-ish regime)
  1d   multi-regime history 2014+ (bull/bear/sideways, from Yahoo daily)

Promotion rules (ALL must hold, on the candidate's own dataset):
  - >= 20 OOS trades
  - positive pooled net expectancy after real IBKR fees + slippage
  - profit factor > 1.2
  - >= 60% of folds positive
  - beats never-trade (mean fold net return > 0)
  - survives multi-regime history: the same (label, features, filter) config
    must show positive expectancy and >= 60% positive folds on the 1d
    multi-regime dataset (with >= 10 trades) - one-window artifacts die here.

Research-only: writes a report, changes no btc-eval defaults.
"""
from __future__ import annotations

import json
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from quant import config
from quant.backtest.validation import walk_forward_splits
from quant.research.btc_label_eval import (
    LABEL_SPECS,
    LabelSpec,
    _engine_kwargs,
    _fit_model,
    _position_notional,
    _rank_key,
    _strip_raw,
    run_candidate,
)

BTC_TICKER = "BTCUSD"

FEATURE_SETS = {
    "ar_only": set(),
    "full": None,
    "full+funding": {"momentum", "trend", "volatility", "flow", "funding"},
}


# ---------------------------------------------------------------- regime filters
def _no_filter(h: pd.DataFrame) -> bool:
    return True


def _trend200(h: pd.DataFrame) -> bool:
    """Long-only sanity: only enter when close is above its 200-bar SMA."""
    close = h["Close"].astype(float)
    if len(close) < 200:
        return False
    return float(close.iloc[-1]) > float(close.tail(200).mean())


def _trend200_vol(h: pd.DataFrame) -> bool:
    """Trend filter + skip dead/berserk tape (vol percentile 20-90%)."""
    if not _trend200(h):
        return False
    from quant.features.btc_features import latest_vol_pctile
    p = latest_vol_pctile(h)
    return 0.20 <= p <= 0.90


REGIME_FILTERS: Dict[str, Optional[Callable]] = {
    "none": None,
    "trend200": _trend200,
    "trend200+vol": _trend200_vol,
}


# ---------------------------------------------------------------- diagnostics
def decile_lift(prob: List[float], y: List[int]) -> dict:
    """Hit rate of the top-decile predictions vs base rate (pooled OOS)."""
    p = np.asarray(prob, dtype=float)
    yy = np.asarray(y, dtype=float)
    if len(p) < 50:
        return {"available": False, "reason": "too few OOS predictions"}
    base = float(yy.mean())
    cut = np.quantile(p, 0.9)
    top = yy[p >= cut]
    return {
        "available": True,
        "n_pooled": int(len(p)),
        "base_hit_rate_pct": round(base * 100, 2),
        "top_decile_hit_rate_pct": round(float(top.mean()) * 100, 2) if len(top) else None,
        "top_decile_n": int(len(top)),
        "lift": round(float(top.mean()) / base, 3) if len(top) and base > 0 else None,
    }


def coefficient_report(bars: pd.DataFrame, spec: LabelSpec, notional: float,
                       groups, train_frac: float = 0.4) -> List[dict]:
    """Standardized logistic coefficients fit on the first training window."""
    try:
        train_bars = bars.iloc[:int(len(bars) * train_frac)]
        model = _fit_model(train_bars, spec, notional, groups, 1.0, False, None)
        coefs = model.coefficients().sort_values(key=abs, ascending=False)
        return [{"feature": k, "coef": round(float(v), 4)} for k, v in coefs.items()]
    except Exception as e:
        return [{"error": str(e)}]


# ---------------------------------------------------------------- search
def _run_dataset_grid(bars: pd.DataFrame, dataset: str, notional: float,
                      funding_available: bool) -> List[dict]:
    splits = walk_forward_splits(bars.index, n_folds=5, min_train_frac=0.4, embargo=5)
    if not splits:
        return []
    results = []
    feature_sets = dict(FEATURE_SETS)
    if not funding_available:
        feature_sets.pop("full+funding", None)
    n_cfg = len(LABEL_SPECS) * len(feature_sets) * len(REGIME_FILTERS)
    logger.info("[{}] {} bars, {} configs x {} folds",
                dataset, len(bars), n_cfg, len(splits))
    for spec in LABEL_SPECS:
        for feat_name, groups in feature_sets.items():
            for filt_name, filt in REGIME_FILTERS.items():
                res = run_candidate(
                    bars, spec, splits, notional,
                    groups=groups, C=1.0, lag0=False, class_weight=None,
                    tag=f"{dataset}|{feat_name}|{filt_name}",
                    regime_filter=filt, keep_arrays=True,
                )
                res["dataset"] = dataset
                res["regime_filter"] = filt_name
                res["features"] = feat_name  # keep grid key exact (incl. funding)
                results.append(res)
    return results


def _passes_core_gates(r: dict, min_trades: int = 20) -> tuple[bool, str]:
    if r["total_oos_trades"] < min_trades:
        return False, f"trades {r['total_oos_trades']} < {min_trades}"
    if r["pooled_net_expectancy_usd"] <= 0:
        return False, f"net exp ${r['pooled_net_expectancy_usd']:.2f} <= 0"
    if r["pooled_profit_factor"] <= 1.2:
        return False, f"PF {r['pooled_profit_factor']:.2f} <= 1.2"
    if r["frac_folds_positive"] < 0.6:
        return False, f"folds+ {r['frac_folds_positive']:.0%} < 60%"
    if r["mean_fold_net_return_pct"] <= 0:
        return False, "does not beat never-trade"
    return True, "ok"


def _survives_multi_regime(r: dict, daily_by_key: dict) -> tuple[bool, str]:
    if r["dataset"] == "1d":
        return True, "candidate is on multi-regime data already"
    twin = daily_by_key.get((r["label"], r["features"], r["regime_filter"]))
    if twin is None:
        return False, "no multi-regime twin result"
    if twin["total_oos_trades"] < 10:
        return False, f"multi-regime twin has only {twin['total_oos_trades']} trades"
    if twin["pooled_net_expectancy_usd"] <= 0:
        return False, f"multi-regime twin exp ${twin['pooled_net_expectancy_usd']:.2f} <= 0"
    if twin["frac_folds_positive"] < 0.6:
        return False, f"multi-regime twin folds+ {twin['frac_folds_positive']:.0%} < 60%"
    return True, "twin holds up on 2014+ daily"


def run_signal_search(offline: bool = False) -> dict:
    from quant.backtest.baselines import run_all
    from quant.data.btc_audit import _basic_report, _normalize_index
    from quant.data.btc_data import get_btc_daily_history
    from quant.data.btc_derivs import get_funding_series
    from quant.data.store import DataStore

    config.ensure_dirs()
    store = DataStore()

    # ---- datasets
    bars_12h = store.load_bars(BTC_TICKER, config.BTC_INTERVAL)
    if bars_12h is None or len(bars_12h) < 300:
        raise SystemExit("no 12h BTC cache - run `python -m quant.pipeline btc-ingest`")
    bars_12h = bars_12h.sort_index()

    bars_1d = get_btc_daily_history(store, refresh=not offline)
    bars_1d = bars_1d.sort_index()

    # ---- funding (the one reachable derivatives source, but OKX only serves
    #      ~3 months of funding history via API - require >= 1 year of
    #      accumulated cache before using it as a research feature, otherwise
    #      95% of training rows are NaN and the comparison is meaningless.
    funding = get_funding_series(store, refresh=not offline)
    funding_span_days = (
        (funding.index.max() - funding.index.min()).days
        if funding is not None and not funding.empty else 0
    )
    funding_available = funding_span_days >= 365
    funding_note = (
        f"OKX funding cache: {len(funding)} records, {funding_span_days} days "
        f"({funding.index.min()} to {funding.index.max()})"
        + ("" if funding_available else
           " - SKIPPED as feature: OKX API serves only ~3 months of history; "
           "cache accumulates via incremental refresh and the full+funding set "
           "auto-enables once >= 365 days are stored")
        if funding is not None and not funding.empty
        else "funding unavailable - full+funding feature set skipped"
    )

    # ---- data gates (audit both datasets before trusting results)
    gate_12h = _basic_report(_normalize_index(bars_12h), config.BTC_INTERVAL,
                             "12h research cache", BTC_TICKER)
    gate_1d = _basic_report(_normalize_index(bars_1d), "1d",
                            "daily multi-regime (Yahoo)", "BTC-USD")

    notional = _position_notional()

    # ---- the grid
    results_1d = _run_dataset_grid(bars_1d, "1d", notional, funding_available)
    results_12h = _run_dataset_grid(bars_12h, "12h", notional, funding_available)
    all_results = results_1d + results_12h

    daily_by_key = {(r["label"], r["features"], r["regime_filter"]): r
                    for r in results_1d}

    # ---- promotion evaluation
    for r in all_results:
        core_ok, core_why = _passes_core_gates(r)
        regime_ok, regime_why = _survives_multi_regime(r, daily_by_key)
        r["promotable"] = bool(core_ok and regime_ok)
        r["gate_detail"] = {"core": core_why, "multi_regime": regime_why}

    ranked = sorted(all_results, key=_rank_key, reverse=True)
    promotable = [r for r in ranked if r["promotable"]]

    # ---- baselines on each dataset's OOS span (for context in the report)
    def _oos_baselines(bars):
        splits = walk_forward_splits(bars.index, n_folds=5, min_train_frac=0.4, embargo=5)
        oos = bars.iloc[int(splits[0].test_idx[0]):]
        return run_all(oos, config.ACCOUNT_EQUITY_USD, ma_window=config.BTC_MA_WINDOW)

    baselines = {"12h": _oos_baselines(bars_12h), "1d": _oos_baselines(bars_1d)}

    # ---- diagnostics on the best candidate (promotable if any, else top-ranked)
    best = promotable[0] if promotable else ranked[0]
    best_bars = bars_1d if best["dataset"] == "1d" else bars_12h
    best_spec = next(s for s in LABEL_SPECS if s.name == best["label"])
    feature_sets = dict(FEATURE_SETS)
    best_groups = feature_sets.get(best["features"])
    diagnostics = {
        "decile_lift": decile_lift(best.get("_prob", []), best.get("_y", [])),
        "coefficients": coefficient_report(best_bars, best_spec, notional, best_groups),
        "calibration": best.get("calibration", []),
    }

    # ---- verdict
    if promotable:
        b = promotable[0]
        verdict = (
            f"PROMOTE TO PAPER OBSERVATION: {b['dataset']} {b['label']} / "
            f"{b['features']} / filter={b['regime_filter']} - "
            f"${b['pooled_net_expectancy_usd']:.2f}/trade net, PF "
            f"{b['pooled_profit_factor']:.2f}, {b['frac_folds_positive']:.0%} folds+, "
            f"{b['total_oos_trades']} trades, survives multi-regime. "
            "Paper-observe via btc-observe before any capital decision."
        )
    else:
        verdict = (
            "REJECT ALL: no configuration cleared promotion "
            f"({len(all_results)} configs tested across 12h and multi-regime daily). "
            "No tradable edge found in this search space - do not paper trade, "
            "do not tune thresholds to force a pass."
        )

    report = {
        "notional_usd": notional,
        "datasets": {
            "12h": {"bars": len(bars_12h), "from": str(bars_12h.index.min()),
                    "to": str(bars_12h.index.max()), "gate": gate_12h},
            "1d": {"bars": len(bars_1d), "from": str(bars_1d.index.min()),
                   "to": str(bars_1d.index.max()), "gate": gate_1d},
        },
        "derivatives_availability": {
            "funding": funding_note,
            "skipped": {
                "binance_derivatives": "geo-blocked (HTTP 451)",
                "bybit_open_interest": "unparseable response from this location",
                "okx_long_short_ratio": "only ~30d history - too short for walk-forward",
                "liquidations": "no clean free source with history",
                "fred_macro": "no API key configured - trend/vol regime filters used instead",
            },
        },
        "grid": {
            "labels": [s.name for s in LABEL_SPECS],
            "features": list(FEATURE_SETS.keys() if funding_available
                             else [k for k in FEATURE_SETS if k != "full+funding"]),
            "regime_filters": list(REGIME_FILTERS.keys()),
            "configs_tested": len(all_results),
        },
        "ranked_results": [_strip_search_raw(r) for r in ranked],
        "promotable": [_strip_search_raw(r) for r in promotable],
        "baselines": baselines,
        "best_candidate_diagnostics": diagnostics,
        "verdict": verdict,
    }
    out = config.DATA_DIR / "btc_signal_search_result.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    return report


def _strip_search_raw(r: dict) -> dict:
    keep = ["dataset", "label", "features", "regime_filter", "n_folds",
            "total_oos_trades", "pooled_net_expectancy_usd",
            "pooled_net_win_rate_pct", "pooled_profit_factor",
            "mean_fold_net_return_pct", "min_fold_net_return_pct",
            "frac_folds_positive", "worst_fold_drawdown_pct", "auc",
            "gate_pass_rate_pct", "promotable", "gate_detail"]
    return {k: r.get(k) for k in keep}


def print_report(report: dict) -> None:
    rows = report["ranked_results"]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.rename(columns={
            "total_oos_trades": "trades",
            "pooled_net_expectancy_usd": "exp$",
            "pooled_net_win_rate_pct": "win%",
            "pooled_profit_factor": "PF",
            "mean_fold_net_return_pct": "fold_ret%",
            "frac_folds_positive": "folds+",
            "regime_filter": "filter",
        })
        for c in ("exp$", "win%", "PF", "fold_ret%", "auc"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").round(2)
        df["folds+"] = pd.to_numeric(df["folds+"], errors="coerce").round(2)
        cols = ["dataset", "label", "features", "filter", "trades", "exp$",
                "win%", "PF", "fold_ret%", "folds+", "auc", "promotable"]
        pd.set_option("display.width", 220, "display.max_rows", 100)
        print("\n" + "=" * 110)
        print("RANKED SIGNAL SEARCH RESULTS (top 25 of {}):".format(len(df)))
        print(df[cols].head(25).to_string(index=False))
        print("\nBOTTOM 5:")
        print(df[cols].tail(5).to_string(index=False))

    print("\n" + "=" * 110)
    print("DERIVATIVES/MACRO AVAILABILITY:")
    print(json.dumps(report["derivatives_availability"], indent=2))

    print("\n" + "=" * 110)
    print("BEST-CANDIDATE DIAGNOSTICS:")
    diag = report["best_candidate_diagnostics"]
    print("decile lift:", json.dumps(diag["decile_lift"], indent=2))
    coefs = diag["coefficients"][:10]
    print("top coefficients:", json.dumps(coefs, indent=2))
    if diag.get("calibration"):
        print("calibration:")
        print(pd.DataFrame(diag["calibration"]).to_string(index=False))

    print("\n" + "=" * 110)
    print("VERDICT:", report["verdict"])
    print("full result:", str(config.DATA_DIR / "btc_signal_search_result.json"))


def main() -> None:
    print_report(run_signal_search())


if __name__ == "__main__":
    main()
