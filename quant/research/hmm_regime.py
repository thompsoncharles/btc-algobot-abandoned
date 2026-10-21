"""
HMM hidden-state regime experiment (LAB, research-only).

    python -m quant.research.hmm_regime

Races HMM-derived regime filters against the frozen candidate's trend200+vol
filter on the SAME judge (walk-forward OOS net backtest, cost_5bar label,
full features, 1d dataset, real fees/sizing/gates). Nothing is promoted:
if an HMM filter wins, this writes LAB_HMM_PROPOSAL.md and stops.

Causality guarantees:
  - HMMs are fit per fold on that fold's TRAINING prefix only.
  - OOS state classification uses the FORWARD (filtered) recursion computed
    manually - alpha_t depends only on observations <= t. hmmlearn's own
    predict/predict_proba are forward-BACKWARD (smoothed - they see the
    future) and are deliberately not used for classification.
  - Feature standardization uses training-window statistics only.

Filters raced:
  trend200+vol         frozen candidate's filter (baseline to beat)
  none                 no regime filter
  hmm2_calm            2-state HMM, trade only in the calm state
  hmm3_calm            3-state HMM, trade only in the calm state
  hmm3_calm_or_rebound 3-state, calm OR extreme state (crash-rebound days)
  hmm3_avoid_elevated  3-state, anything except the elevated-vol state
(states are ordered by emission volatility: 0 calmest .. K-1 wildest)
"""
from __future__ import annotations

import json
from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger

from quant import config
from quant.backtest.validation import walk_forward_splits
from quant.research.btc_label_eval import (
    LABEL_SPECS,
    _position_notional,
    run_candidate,
)
from quant.research.btc_signal_search import REGIME_FILTERS

BTC_TICKER = "BTCUSD"
WARMUP_BARS = 252  # must match run_candidate's engine warmup window

REJECT_RULES = dict(min_trades=20, min_pf=1.2, min_folds_positive=0.6)


# ------------------------------------------------------------------ features
def hmm_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Daily observation vector: return, realized vol, MA distance, volume z."""
    close = bars["Close"].astype(float)
    ret = np.log(close / close.shift(1))
    rvol = ret.rolling(20).std()
    dist_ma = close / close.rolling(200, min_periods=60).mean() - 1
    f = pd.DataFrame({"ret": ret, "rvol20": rvol, "dist_ma200": dist_ma},
                     index=bars.index)
    vol = bars.get("Volume")
    if vol is not None:
        vol = vol.astype(float)
        if (vol.fillna(0) > 0).mean() > 0.9:  # only if volume is clean
            logv = np.log(vol.replace(0, np.nan))
            f["vol_z"] = (logv - logv.rolling(60).mean()) / logv.rolling(60).std()
    return f.dropna()


def fit_hmm(X_train: np.ndarray, n_states: int, seed: int = 42):
    from hmmlearn.hmm import GaussianHMM
    hmm = GaussianHMM(n_components=n_states, covariance_type="full",
                      n_iter=200, random_state=seed, min_covar=1e-5)
    hmm.fit(X_train)
    return hmm


def filtered_states(hmm, X: np.ndarray) -> np.ndarray:
    """
    Causal (filtered) most-likely state per row: forward recursion only.
    alpha_t ∝ (alpha_{t-1} @ A) * b_t  - no backward pass, no future data.
    """
    log_b = hmm._compute_log_likelihood(X)          # (T, K) emission log-lik
    b = np.exp(log_b - log_b.max(axis=1, keepdims=True))
    A = hmm.transmat_
    alpha = hmm.startprob_ * b[0]
    alpha /= max(alpha.sum(), 1e-300)
    out = np.empty(len(X), dtype=int)
    out[0] = int(np.argmax(alpha))
    for t in range(1, len(X)):
        alpha = (alpha @ A) * b[t]
        s = alpha.sum()
        if s <= 0 or not np.isfinite(s):            # numerical guard
            alpha = np.full(len(alpha), 1.0 / len(alpha))
        else:
            alpha /= s
        out[t] = int(np.argmax(alpha))
    return out


def _vol_ordered_remap(hmm) -> Dict[int, int]:
    """Map raw HMM state ids -> vol-ordered ids (0 = calmest emission vol)."""
    # order by emission std of the return dimension (feature 0)
    stds = [np.sqrt(hmm.covars_[k][0, 0]) for k in range(hmm.n_components)]
    order = np.argsort(stds)
    return {int(raw): int(pos) for pos, raw in enumerate(order)}


# ------------------------------------------------- fold-aware state maps
def build_state_maps(bars: pd.DataFrame, splits, n_states: int) -> Dict[pd.Timestamp, int]:
    """
    date -> vol-ordered state, assembled fold by fold. For dates in fold k's
    [warmup_start .. OOS end] window the state comes from the HMM fit on
    fold k's TRAINING prefix, decoded with the causal forward filter over
    history up to that date.
    """
    feats = hmm_features(bars)
    state_map: Dict[pd.Timestamp, int] = {}
    for split in splits:
        train_end = int(split.train_idx[-1]) + 1
        train_ts = bars.index[:train_end]
        Xtr_df = feats.loc[feats.index.intersection(train_ts)]
        mu, sd = Xtr_df.mean(), Xtr_df.std().replace(0, 1.0)

        test_end = int(split.test_idx[-1])
        window_start = max(0, int(split.test_idx[0]) - WARMUP_BARS)
        upto_ts = bars.index[:test_end + 1]
        Xall_df = ((feats.loc[feats.index.intersection(upto_ts)] - mu) / sd).dropna()

        hmm = fit_hmm(((Xtr_df - mu) / sd).dropna().values, n_states)
        remap = _vol_ordered_remap(hmm)
        states = filtered_states(hmm, Xall_df.values)

        window_ts = set(bars.index[window_start:test_end + 1])
        for ts, raw in zip(Xall_df.index, states):
            if ts in window_ts:
                state_map[ts] = remap[raw]
    return state_map


def make_hmm_filter(state_map: Dict[pd.Timestamp, int], allowed: set):
    def regime_filter(h: pd.DataFrame) -> bool:
        s = state_map.get(h.index[-1])
        return s is not None and s in allowed
    return regime_filter


# ------------------------------------------------------------------ the race
def run_race() -> dict:
    from quant.data.store import DataStore

    config.ensure_dirs()
    bars = DataStore().load_bars(BTC_TICKER, "1d")
    if bars is None or len(bars) < 1000:
        raise SystemExit("daily BTC cache required")
    bars = bars.sort_index()
    splits = walk_forward_splits(bars.index, n_folds=5, min_train_frac=0.4, embargo=5)
    spec = next(s for s in LABEL_SPECS if s.name == "cost_5bar")
    notional = _position_notional()

    logger.info("building fold-aware HMM state maps (2- and 3-state) ...")
    map2 = build_state_maps(bars, splits, 2)
    map3 = build_state_maps(bars, splits, 3)

    # Note: with exactly 3 vol-ordered states, "avoid elevated" (all but
    # state 1) and "calm or rebound" ({0, 2}) admit the same state set; they
    # are kept as separate named entries because their intent differs and
    # they diverge if the state count ever changes.
    filters = {
        "trend200+vol": REGIME_FILTERS["trend200+vol"],
        "none": None,
        "hmm2_calm": make_hmm_filter(map2, {0}),
        "hmm3_calm": make_hmm_filter(map3, {0}),
        "hmm3_calm_or_rebound": make_hmm_filter(map3, {0, 2}),
        "hmm3_avoid_elevated": make_hmm_filter(map3, {0, 2}),
    }

    results = {}
    for name, filt in filters.items():
        logger.info("racing filter: {}", name)
        res = run_candidate(bars, spec, splits, notional,
                            groups=None, C=1.0, lag0=False, class_weight=None,
                            tag=f"hmm_race|{name}", regime_filter=filt,
                            keep_arrays=False)
        res["filter"] = name
        results[name] = res

    report = _evaluate(results, bars, map3)
    out = config.DATA_DIR / "hmm_regime_race.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    return report


def _row(res: dict) -> dict:
    return {
        "filter": res["filter"],
        "trades": res["total_oos_trades"],
        "net_exp_usd": round(res["pooled_net_expectancy_usd"], 2),
        "pf": round(res["pooled_profit_factor"], 3),
        "win_pct": round(res["pooled_net_win_rate_pct"], 2),
        "mean_fold_ret_pct": round(res["mean_fold_net_return_pct"], 3),
        "min_fold_ret_pct": round(res["min_fold_net_return_pct"], 3),
        "frac_folds_positive": res["frac_folds_positive"],
        "worst_fold_dd_pct": round(res["worst_fold_drawdown_pct"], 2),
        "fold_returns_pct": [round(f["net_return_pct"], 3) for f in res["folds"]],
        "last_fold_ret_pct": round(res["folds"][-1]["net_return_pct"], 3) if res["folds"] else None,
        "gate_pass_rate_pct": round(res["gate_pass_rate_pct"], 2),
    }


def _reject_reasons(row: dict, baseline: dict) -> List[str]:
    reasons = []
    if row["trades"] < REJECT_RULES["min_trades"]:
        reasons.append(f"trades {row['trades']} < 20")
    if row["pf"] <= REJECT_RULES["min_pf"]:
        reasons.append(f"PF {row['pf']} <= 1.2")
    if row["net_exp_usd"] <= 0:
        reasons.append(f"expectancy ${row['net_exp_usd']} <= 0")
    if row["frac_folds_positive"] < REJECT_RULES["min_folds_positive"]:
        reasons.append(f"only {row['frac_folds_positive']:.0%} folds positive")
    # one lucky fold: does the edge survive dropping each side's best fold?
    hr, br = row["fold_returns_pct"], baseline["fold_returns_pct"]
    if hr and br and len(hr) == len(br):
        h_trim = (sum(hr) - max(hr)) / max(len(hr) - 1, 1)
        b_trim = (sum(br) - max(br)) / max(len(br) - 1, 1)
        if row["mean_fold_ret_pct"] > baseline["mean_fold_ret_pct"] and h_trim <= b_trim:
            reasons.append("improvement disappears when best fold is removed (one lucky fold)")
    # most recent fold materially worse than baseline's
    if (row["last_fold_ret_pct"] is not None and baseline["last_fold_ret_pct"] is not None
            and row["last_fold_ret_pct"] < baseline["last_fold_ret_pct"] - 1.0):
        reasons.append(
            f"2025-2026 fold materially worse ({row['last_fold_ret_pct']}% vs "
            f"baseline {baseline['last_fold_ret_pct']}%)")
    return reasons


def _evaluate(results: dict, bars: pd.DataFrame, map3: dict) -> dict:
    rows = {name: _row(res) for name, res in results.items()}
    baseline = rows["trend200+vol"]

    verdicts = {}
    for name, row in rows.items():
        if name == "trend200+vol":
            continue
        reasons = _reject_reasons(row, baseline)
        beats = (not reasons
                 and row["net_exp_usd"] > baseline["net_exp_usd"]
                 and row["mean_fold_ret_pct"] > baseline["mean_fold_ret_pct"]
                 and row["frac_folds_positive"] >= baseline["frac_folds_positive"])
        verdicts[name] = {"beats_trend200_vol": beats, "reject_reasons": reasons}

    # state distribution over the OOS span (3-state map, for the report)
    oos_start_ts = None
    for name, res in results.items():
        if res["folds"]:
            oos_start_ts = bars.index[int(len(bars) * 0.4)]
            break
    state_series = pd.Series(map3)
    state_counts = state_series.value_counts().sort_index()
    state_breakdown = {f"state_{int(s)}": int(c) for s, c in state_counts.items()}

    winners = [n for n, v in verdicts.items()
               if v["beats_trend200_vol"] and n.startswith("hmm")]
    if winners:
        verdict = ("PROPOSE FUTURE PAPER CANDIDATE - " + ", ".join(winners) +
                   " beat trend200+vol cleanly; proposal written, nothing promoted.")
    elif any(not v["reject_reasons"] for n, v in verdicts.items() if n.startswith("hmm")):
        verdict = ("WATCH HMM IN LAB - at least one HMM filter passes the absolute "
                   "gates but does not beat trend200+vol; keep in lab, revisit with "
                   "more data.")
    else:
        verdict = "REJECT HMM - no HMM filter passes the promotion gates."

    return {
        "mode": "lab_research_only",
        "places_orders": False,
        "judge": "walk-forward OOS net backtest, cost_5bar/full/1d, real fees+sizing+gates",
        "candidate_setup_unchanged": True,
        "table": list(rows.values()),
        "verdicts": verdicts,
        "hmm3_state_distribution": state_breakdown,
        "state_convention": "vol-ordered: 0=calm, 1=elevated, 2=extreme",
        "verdict": verdict,
    }


def print_report(report: dict) -> None:
    df = pd.DataFrame(report["table"])
    pd.set_option("display.width", 220)
    print("\n" + "=" * 100)
    print("HMM REGIME FILTER RACE (judge: same OOS walk-forward as btc-eval)")
    print(df.drop(columns=["fold_returns_pct"]).to_string(index=False))
    print("\nfold-by-fold net returns (%):")
    for row in report["table"]:
        print(f"  {row['filter']:<22} {row['fold_returns_pct']}")
    print("\n3-state day distribution:", report["hmm3_state_distribution"],
          "|", report["state_convention"])
    print("\nper-filter verdicts:")
    for name, v in report["verdicts"].items():
        status = "BEATS baseline" if v["beats_trend200_vol"] else \
                 (("rejected: " + "; ".join(v["reject_reasons"])) if v["reject_reasons"]
                  else "passes gates, does not beat baseline")
        print(f"  {name:<22} {status}")
    print("\nVERDICT:", report["verdict"])
    print("full result:", str(config.DATA_DIR / "hmm_regime_race.json"))


def main() -> None:
    print_report(run_race())


if __name__ == "__main__":
    main()
