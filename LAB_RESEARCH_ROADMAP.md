# LAB RESEARCH ROADMAP

**Folder:** `V4.2algobot-master-lab` — research only. Nothing here touches the
frozen paper-observation project (`V4.1algobot-master`), its scheduler, its
journal, or its frozen candidate.

**Standing rules for every experiment below:**
- Judge ONLY with the existing no-leakage walk-forward OOS net backtest
  (same fees, sizing, gates as `btc-eval`/`btc-signal-search`).
- Promotion requires a fresh OOS judge run + the stable project's promotion
  gates (>=20 trades, exp>0, PF>1.2, >=60% folds+, beats never-trade,
  survives multi-regime) — and even then, promotion means *proposing* a
  change to the stable project, never editing it from the lab.
- No live trading. No threshold tuning against old backtests.

**Data provenance caution (found 2026-07-04 during lab baseline):** the 12h
research cache is source-mixed — when Binance is geo-blocked, refresh pulls
Kraken 4h + Yahoo hourly resamples and `keep="last"` REVISES overlapping
historical bars. A btc-eval run before vs after one refresh moved from
-$19.14/57 trades to -$17.09/51 trades (same NOT VIABLE verdict, same code).
The daily (1d) cache is pure Yahoo and stable. Treat 12h backtest numbers as
source-dependent; pin a snapshot before any 12h experiment (copy the parquet
aside) so results are reproducible.

---

## Priority 1 — Hidden-state regime model experiment

**Question:** do BTC returns cluster into latent states that predict the
candidate's per-regime performance better than the causal SMA/vol labels in
`quant/backtest/regimes.py`?

- Start: `python -m quant.research.regime_diagnostics` (built 2026-07-04,
  research-only) — Gaussian-mixture posteriors over daily returns/vol,
  transition matrix, state durations, per-state forward returns.
- Next: fit 2-3 state Gaussian HMM (needs `hmmlearn` — install in lab venv
  only) fitted walk-forward (no full-sample fitting = no leakage), compare
  state-conditioned entry filtering vs the frozen trend200+vol filter on the
  OOS judge.
- Success bar: state filter beats trend200+vol on OOS net expectancy with
  stable folds — then propose (not apply) as a candidate variant.

## Priority 2 — Online probability calibration experiment

**Question:** the promoted candidate's probabilities are usable but coarse
(decile lift 1.17). Does post-hoc calibration (Platt / isotonic, fit on
train folds only) improve edge-gating without changing the model?

- Calibrate per walk-forward fold on train data; apply to OOS predictions;
  re-judge. Strictly additive layer - the underlying logistic stays frozen.
- Success bar: better OOS net expectancy at the SAME entry threshold (no
  threshold tuning), calibration curve closer to diagonal.

## Priority 3 — Strategy/bandit selection across frozen candidates

**Question:** once >1 candidate exists (e.g. the current one + a future
regime-filtered variant), how to allocate paper attention between them?

- Implement as *allocation of observation*, not capital: a simple
  Thompson-sampling / UCB layer over frozen candidates' rolling paper stats.
- Hard rule: bandits select among FROZEN candidates; they never tune inside
  one. Requires priorities 1-2 to have produced at least one extra candidate.

## Priority 4 — Derivatives features (blocked on data depth)

- OKX funding cache accumulates in `quant_data/macro/BTC_FUNDING.parquet`
  (~3 months served per API call; started 2026-04). The `funding` feature
  group auto-enables at >=365 days — revisit ~mid-2027, or find a bulk
  historical funding/OI source before then.
- When available: rerun `btc-signal-search` with `full+funding` on the
  same judge. No special treatment - it competes like any feature group.

## Priority 5 — Tax-ready exports and reporting improvements

- Largely DONE in the stable project (2026-07-04): `btc_paper_trades.csv`
  has entry/exit/fees/cost-basis/proceeds/gain/holding-period/term fields.
- Remaining lab ideas: monthly rollups, wash-sale awareness (n/a for crypto
  today but rules may change), CSV import format matching the paper
  account's actual statements for reconciliation.

## Priority 6 — Packaging/executable experiment

- Only after the observation workflow has run stably for weeks. PyInstaller
  or a pip-installable package of the OBSERVATION path only (never order
  code). Blocked until the 20-signal/60-day paper gate has been exercised.

## Priority 7 — RL / entropy-regularized policy gradients (far future)

- Explicitly parked. Prerequisites before this is even considered: a signal
  with OOS AUC meaningfully > 0.5, years of clean data, a working simpler
  stack, and a cost model validated against real paper fills. RL on 4k bars
  of noisy financial data is an overfitting machine - do not start here.

---

## Experiment log

| date | experiment | verdict |
|------|-----------|---------|
| 2026-07-04 | lab baseline (smoke 36/36, audit SAFE, btc-eval reproduces current stable) | lab isolated + operational |
| 2026-07-04 | regime diagnostics (Priority 1 groundwork) | PROMISING: held-out 3-state mixture separates cleanly - calm state (74% of bars) +0.03%/day fwd, elevated-vol state (25%) -0.15%/day fwd, extreme state (0.4%, n=7) crash-rebound +3.5%/day. Vol-state filtering has real held-out signal; a walk-forward HMM entry-filter experiment is justified. `quant_data/regime_diagnostics.json` |
| 2026-07-04 | Priority 1: HMM regime filter race (quant/research/hmm_regime.py; fold-fit 2/3-state GaussianHMMs, causal forward-filtered states, 6 filters on the OOS judge) | **REJECT HMM.** trend200+vol stays champion ($16.09/trade, PF 1.60, 4/5 folds+). Best HMM (hmm3_calm: $7.72, PF 1.25, 4/5 folds+) fails the recent-fold rule (-5.2% vs baseline -1.2% in 2025-26); calm_or_rebound/avoid_elevated are net-negative; no-filter (-$2.37) confirms the frozen filter is load-bearing. The diagnostics' promise did not survive causal fold-fit filtering. No proposal written. `quant_data/hmm_regime_race.json` |
| 2026-07-04 | Priority 2: online probability calibration (quant/research/prob_calibration.py; Platt/isotonic/rolling-Platt fit per fold on train-internal 75/25 holdout, judged raw-vs-calibrated on same OOS judge) | **REJECT CALIBRATION.** Mode A: NO calibrator improved Brier (raw 0.2510 best; platt 0.2527, isotonic 0.2538, rolling 0.2528) - the raw top-end is already honest ([0.65,1.0) bucket hits 76%). Mode B: calibrators compress probs toward 0.5, so the fixed 0.58 gate starves - trades drop 85->36-42, 3 of 5 folds go silent, folds+ falls to 40%/20%. Platt's better per-trade stats ($21.12, PF 1.87) are survivorship of its 2 hottest folds. Root cause: calibration cannot help under a fixed entry threshold without re-deriving the threshold = forbidden tuning. Calibration is a dead end here; the raw model's probability ranking is not the binding constraint. No proposal written. `quant_data/prob_calibration_result.json` |
