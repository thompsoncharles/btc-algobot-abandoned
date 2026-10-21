"""
Hidden-state regime diagnostics (LAB, research-only).

    python -m quant.research.regime_diagnostics [--states 2|3]

Priority-1 groundwork from LAB_RESEARCH_ROADMAP.md: before fitting a real
HMM, check whether BTC daily returns even cluster into latent states with
different forward-return behavior. Uses a Gaussian mixture over
(log-return, rolling-vol) - sklearn only, no new dependencies.

Strictly diagnostic: fits on the FIRST 60% of history only (the same
in-sample region the walk-forward judge trains on), reports posterior state
sequences, transition matrix, durations, and per-state NEXT-bar returns on
the held-out 40%. No trading-path changes, no thresholds, no promotion -
output is a JSON report + console table to decide whether a proper
walk-forward HMM experiment is worth building.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from loguru import logger

from quant import config

TRAIN_FRAC = 0.6  # match the walk-forward judge's earliest OOS boundary


def _features(bars: pd.DataFrame) -> pd.DataFrame:
    close = bars["Close"].astype(float)
    ret = np.log(close / close.shift(1))
    vol = ret.rolling(20).std()
    return pd.DataFrame({"ret": ret, "vol": vol}, index=bars.index).dropna()


def fit_mixture(X: np.ndarray, n_states: int, seed: int = 42):
    from sklearn.mixture import GaussianMixture
    gm = GaussianMixture(n_components=n_states, covariance_type="full",
                         n_init=5, random_state=seed)
    gm.fit(X)
    return gm


def transition_matrix(states: np.ndarray, n_states: int) -> list:
    m = np.zeros((n_states, n_states))
    for a, b in zip(states[:-1], states[1:]):
        m[a, b] += 1
    row_sums = m.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return (m / row_sums).round(4).tolist()


def state_durations(states: np.ndarray) -> dict:
    runs, cur, n = [], states[0], 1
    for s in states[1:]:
        if s == cur:
            n += 1
        else:
            runs.append((cur, n))
            cur, n = s, 1
    runs.append((cur, n))
    out = {}
    for s in set(states.tolist()):
        lens = [n for st, n in runs if st == s]
        out[int(s)] = {"mean_bars": round(float(np.mean(lens)), 1),
                       "max_bars": int(max(lens)), "episodes": len(lens)}
    return out


def run_diagnostics(n_states: int = 2) -> dict:
    from quant.data.store import DataStore

    bars = DataStore().load_bars("BTCUSD", "1d")
    if bars is None or len(bars) < 500:
        raise SystemExit("need the daily BTC cache - run btc-ingest / btc-observe-daily first")
    bars = bars.sort_index()
    feats = _features(bars)
    close = bars["Close"].astype(float)
    fwd1 = (np.log(close.shift(-1) / close)).reindex(feats.index)

    split = int(len(feats) * TRAIN_FRAC)
    train, test = feats.iloc[:split], feats.iloc[split:]
    gm = fit_mixture(train.values, n_states)

    # order states by volatility so state 0 = calmest (stable labels)
    order = np.argsort(gm.means_[:, 1])
    remap = {int(old): int(new) for new, old in enumerate(order)}

    def predict(X):
        return np.array([remap[s] for s in gm.predict(X)])

    states_train = predict(train.values)
    states_test = predict(test.values)

    def per_state(feats_idx, states, fwd):
        rows = []
        for s in range(n_states):
            mask = states == s
            f = fwd.loc[feats_idx[mask]].dropna()
            rows.append({
                "state": s,
                "bars": int(mask.sum()),
                "share_pct": round(float(mask.mean() * 100), 1),
                "mean_ret_pct": round(float(feats.loc[feats_idx[mask], "ret"].mean() * 100), 4),
                "mean_vol_pct": round(float(feats.loc[feats_idx[mask], "vol"].mean() * 100), 4),
                "fwd1_mean_pct": round(float(f.mean() * 100), 4) if len(f) else None,
                "fwd1_hit_rate_pct": round(float((f > 0).mean() * 100), 2) if len(f) else None,
            })
        return rows

    report = {
        "mode": "research_diagnostics_only",
        "places_orders": False,
        "n_states": n_states,
        "train_bars": len(train), "test_bars": len(test),
        "train_span": f"{train.index.min().date()} .. {train.index.max().date()}",
        "test_span": f"{test.index.min().date()} .. {test.index.max().date()}",
        "state_convention": "states ordered by volatility: 0 = calmest",
        "train_states": per_state(train.index, states_train, fwd1),
        "test_states_HELD_OUT": per_state(test.index, states_test, fwd1),
        "transition_matrix_train": transition_matrix(states_train, n_states),
        "state_durations_train": state_durations(states_train),
        "interpretation_note": (
            "If held-out per-state fwd1 stats do not separate meaningfully, "
            "a full HMM experiment is unlikely to beat the frozen trend200+vol "
            "filter - record the result in LAB_RESEARCH_ROADMAP.md either way."
        ),
    }
    out = config.DATA_DIR / "regime_diagnostics.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=int, default=2, choices=[2, 3])
    args = ap.parse_args()
    report = run_diagnostics(args.states)
    print(json.dumps(report, indent=2, default=str))
    print(f"\nfull result: {config.DATA_DIR / 'regime_diagnostics.json'}")


if __name__ == "__main__":
    main()
