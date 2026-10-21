> **LAB COPY NOTICE:** You are reading the lab copy. Every path and
> instruction below refers to the ORIGINAL frozen project
> (`V4.1algobot-master`, no `-lab`). Do not run the observation workflow
> or scheduler from this lab folder.

# FINAL HANDOFF — BTC Quant Paper-Observation Workflow

**Date:** 2026-07-04
**State:** One candidate strategy promoted to PAPER OBSERVATION ONLY. Live trading disabled everywhere. Daily routine automated and smoke-tested (36/36 passing). Final hardening pass completed 2026-07-04 ahead of Monday paper-account open.

---

## 1. Active project path

```
C:\Users\14ml3\OneDrive\Documents\Desktop\V4.1algobot-master
```

This is the ONLY active folder. `V4algobot-master` (no `.1`), `1algobot_v2-master`, `algobot_v2.3-CopyBfrIBKR`, and the `ibkr` folder are older iterations — do not work in them. All commands below run from the project root using the project venv (`venv\Scripts\python.exe`, or activate the venv first).

## 2. Daily commands

```
# The one command that does everything (safety -> data gate -> observe -> report -> status):
python -m quant.pipeline btc-daily-routine

# Check progress / scoreboard anytime (no new observation logged):
python -m quant.pipeline btc-paper-report

# Individual pieces, if ever needed:
python -m quant.pipeline btc-observe-daily        # log today's observation only
python -m quant.pipeline btc-data-audit --offline # data health check only
```

Run `btc-daily-routine` once per day, **after 20:35 local time** (the BTC daily bar closes at 00:00 UTC; 20:35 local covers EST and EDT plus Yahoo's publication lag). Missing a day is not fatal — the report re-resolves outcomes from bar history — but gaps reduce observation quality.

## 3. Scheduler setup (one time, no admin)

```
powershell -ExecutionPolicy Bypass -File scripts\setup_btc_daily_task.ps1
```

Registers per-user task **"BTC Paper Daily Routine"**, daily at 20:35 local, running `scripts\run_btc_daily_routine.bat` (which logs to `quant_data\daily_status\runner.log`). Per-user tasks only run while you are logged in.

Manage it:
```
schtasks /Query  /TN "BTC Paper Daily Routine"
schtasks /Run    /TN "BTC Paper Daily Routine"      # manual test run
schtasks /Delete /TN "BTC Paper Daily Routine" /F   # remove
```

## 4. How to check latest status

```
type quant_data\daily_status\latest.md          # today's concise status
type quant_data\btc_paper_report.md             # scoreboard vs OOS reference + checklist
quant_data\btc_observations.csv                 # every observation + reason codes
quant_data\btc_paper_trades.csv                 # tax-ready ledger of resolved paper trades
quant_data\daily_status\runner.log              # scheduled-run history
```

The paper-report markdown includes a **manual paper-trade checklist** (computed
live each run): data gate, safety gate, candidate-file match, would_enter,
regime filter, probability gate, edge-to-cost, fees included, live disabled.
On a signal day, mirror the trade in the paper account ONLY if every box is
checked. The tax ledger (`btc_paper_trades.csv`) carries entry/exit times and
prices, per-side fees, cost basis, proceeds, realized gain/loss, holding
period, and short/long-term flags - paper trades have no tax consequence, but
the schema is ready if real minimum-size trades ever begin.

## 5. The frozen candidate

Frozen (and smoke-test-enforced) in `quant/candidates/btc_daily_cost5bar_full_trend200vol.json`:

| Field | Value |
|---|---|
| Dataset / cadence | **Daily bars (UTC)** — signal on close, hypothetical fill next open |
| Label | `cost_5bar` — 5-bar forward return must clear all-in round-trip costs |
| Features | full (AR lags + momentum + trend + volatility + flow) |
| Regime filter | `trend200+vol` — close > 200-day SMA AND vol percentile in [0.20, 0.90] |
| Model | Logistic regression (LinearBTCModel), C=1.0 |
| Exits | 3% stop / 6% target / 8-bar max hold |
| Sizing | $1,666.67 notional (1% risk of $5k equity / 3% stop; floor $500, cap 50%) |
| Fees | IBKR spot crypto: max($1.75, 0.18%)/side, capped 1% → ~$6.00 round trip |
| OOS reference | 85 trades, **+$15.50/trade net, PF 1.58, 49.4% win, 4/5 folds positive** (2019–2026 walk-forward) |

Selected by `btc-signal-search` (60-config grid) as the only config family passing every promotion rule including multi-regime survival.

## 6. Why live trading is disabled

- The edge is **backtest-only evidence** so far; paper observation exists to test it forward before real dollars.
- The most recent walk-forward fold (2025-02 → 2026-07) was **negative** — the strategy may have decayed in current conditions.
- Enforcement is structural, not a promise: the observation path (`quant/monitor/btc_daily_observer.py`, `btc_paper_report.py`, `btc_daily_routine.py`) imports no order code, and `verify_paper_safety()` re-scans those sources (AST imports + order-API tokens) on **every** routine run and aborts on violation. Every record carries `places_orders: false`.

## 7. Paper-observation rules

- **No reconsideration until ≥ 20 would-enter paper signals OR ≥ 60 calendar days of observation.** Until then the report's only verdict is KEEP OBSERVING.
- Do not tune thresholds, retrain differently, or "fix" the strategy mid-observation — that invalidates the sample.
- Progress is printed by `btc-paper-report` (e.g. `signals: 3/20, days: 14/60`).

## 8. Files added/changed (this workflow)

**Added:**
- `quant/monitor/btc_daily_observer.py` — daily observer for the candidate
- `quant/monitor/btc_paper_report.py` — scoreboard, markdown/CSV artifacts, verdict gate
- `quant/monitor/btc_daily_routine.py` — one-command routine + safety verification
- `quant/candidates/btc_daily_cost5bar_full_trend200vol.json` — frozen candidate
- `quant/research/btc_signal_search.py` — the search that promoted it (`btc-signal-search`)
- `quant/data/btc_derivs.py` — OKX funding cache (accumulating; not yet a feature)
- `scripts/run_btc_daily_routine.bat`, `scripts/setup_btc_daily_task.ps1`
- `quant/data/btc_data.py`: `get_btc_daily_history()` / `fetch_yahoo_daily_max()` (multi-regime daily cache)

**Changed:**
- `quant/pipeline.py` — commands: `btc-signal-search`, `btc-observe-daily`, `btc-paper-report`, `btc-daily-routine`
- `quant/features/btc_features.py` — optional `funding` group (auto-disabled until ≥365d cache)
- `quant/research/btc_label_eval.py` — additive `regime_filter`/`keep_arrays` params on `run_candidate`
- `tests/test_quant_smoke.py` — 36 tests total
- `quant/README.md` — findings + workflow docs

**Not changed:** `btc-eval` defaults, 12h model path, all thresholds in `constants/parameters.py`.

## 9. What NOT to touch during observation

- `constants/parameters.py` BTC_* block (gates, stops, sizing) — frozen with the candidate
- `quant/candidates/*.json` — the frozen record; smoke tests fail if it drifts
- The journal `quant_data/journal/btc_daily_observations.jsonl` — append-only evidence; never edit/delete
- `btc-eval` defaults and the 12h path — unchanged by design
- Model training code paths the candidate uses (`btc_label_eval._fit_model`, `LinearBTCModel`, `btc_features`)
- Do not run `btc-signal-search` repeatedly and act on it mid-observation — that's threshold-shopping

## 10. Troubleshooting the scheduled task

1. **Did it run?** `schtasks /Query /TN "BTC Paper Daily Routine" /V /FO LIST` → check "Last Run Time" / "Last Result" (0 = ok).
2. **Check the log:** `type quant_data\daily_status\runner.log` — each run appends a timestamp and exit code.
3. **Exit code 1 with "PAPER SAFETY VIOLATION":** someone edited the observation path to import order code. That is the guard working — revert the edit.
4. **"data gate: FAIL" in status:** usually a stale Yahoo daily cache. Run `python -m quant.pipeline btc-data-audit --offline` to see staleness/gaps; retry later (Yahoo lags right after the UTC close) or run `python -m quant.pipeline btc-daily-routine` manually.
5. **Task never fires:** per-user tasks require you to be logged in at 20:35; run the routine manually or move the trigger time.
6. **Wrong Python:** the .bat uses `venv\Scripts\python.exe` relative to the project root — if the venv was rebuilt/moved, re-run `pip install -r requirements-quant.txt` inside it.
7. **OneDrive sync conflicts** (files locked): re-run the routine manually; it's idempotent per day (journal dedupes by signal date at report time).

## 11. Smoke tests

```
cd C:\Users\14ml3\OneDrive\Documents\Desktop\V4.1algobot-master
venv\Scripts\python.exe -m tests.test_quant_smoke
```

Expected: `ALL SMOKE TESTS PASSED` (36 checks; needs a fresh 12h cache — if the
"BTC data quality gate" check reports stale data, run
`python -m quant.pipeline btc-ingest` first; that is the gate working, not a bug).

## 12. Current known weaknesses

1. **Most recent fold negative:** the 2025-02 → 2026-07 OOS fold lost −1.19% (−$6.59/trade). The edge may be fading; this is precisely what paper observation must settle.
2. **Bull-regime concentration:** ~all profit came from bull regimes (76/85 trades, +$1,356); bear/sideways are near-flat by filter design. If the market stays below the 200-day SMA (as it is today), expect **long stretches of zero signals** — that is correct behavior, not a bug.
3. **12h strategy not viable:** every 12h config failed the search (0–2 trades or losses). Do not run this candidate at 12h cadence; do not resurrect the 12h path without a new search.
4. **Derivatives features unavailable/insufficient:** Binance geo-blocked; OKX funding API serves only ~3 months (cache accumulating in `quant_data/macro/BTC_FUNDING.parquet`; the `funding` feature group auto-enables at ≥365 days — around mid-2027). Bybit OI unparseable, OKX long/short ~30 days, no liquidation source, no FRED key.

## 13. Next decision rules (after the paper gate is met)

When `btc-paper-report` shows the gate met (≥20 signals or ≥60 days):

- **Paper supports** (net expectancy > 0 AND PF > 1.2, in line with the OOS reference): re-run `btc-signal-search` on the refreshed data to confirm the config still wins, then *consider* the next step — which is still not live: it's larger-sample paper or minimum-size real trades ($500 floor) as a deliberate, separate decision.
- **Paper contradicts** (negative expectancy or PF ≤ 1.2): reject the candidate. Return to research (`btc-signal-search` on refreshed data) — do not tune the frozen config to rescue it.
- **Too few closed trades despite gate met** (e.g. 60 days but 3 signals in a bear tape): extend observation; a bull-filtered strategy cannot be judged on a bear sample.
- **Either way:** the reconsideration is a fresh decision with fresh evidence — nothing auto-promotes, and live trading stays disabled until you explicitly build and review an execution path (which does not currently exist in the observation code, by design).

---


