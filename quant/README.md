# Quant Research Framework

Institutional-style research/trading stack living alongside the live Godel+IBKR bot.
The live bot (`scripts/algo_loop.py`) keeps trading PRT signals; this package is where
data, features, models, validation, and risk live. The two meet in the probability
engine, which blends model P(up) with Godel PRT `prob_up`.

## Component map (the 15-part stack)

| # | Component          | Where                                            | Status |
|---|--------------------|--------------------------------------------------|--------|
| 1 | Data layer         | `quant/data/`                                    | OHLCV via yfinance + IBKR; fundamentals via yfinance + Godel DES; macro via FRED; ticks/L2/alt-data stubbed |
| 2 | Universe selection | `quant/universe/selection.py`                    | price, ADDV, liquidity, mcap, exchange, spread; borrow stub |
| 3 | Features           | `quant/features/`                                | technical, price action, volatility, regime, microstructure |
| 4 | Models             | `quant/models/`                                  | linear/logistic/AR, boosting (xgb/lgbm/sklearn), HMM regimes, Kalman, registry |
| 5 | Probability engine | `quant/probability/engine.py`                    | P(target-before-stop), expected return/drawdown, Wilson CI, Godel blend |
| 6 | Position sizing    | `quant/sizing/position_sizing.py`                | fixed fractional, vol targeting, fractional Kelly, exposure caps |
| 7 | Risk management    | `quant/risk/manager.py`                          | daily loss, max DD, correlation/sector limits, circuit breakers |
| 8 | Backtesting        | `quant/backtest/`                                | event-driven, costs+slippage, next-open fills, walk-forward + embargo |
| 9 | Metrics            | `quant/metrics/performance.py`                   | CAGR, Sharpe, Sortino, maxDD, Calmar, PF, expectancy, turnover, alpha/beta, IR |
|10 | Execution          | `quant/execution/executor.py`                    | limit-first w/ repricing, market escalation (logged), TWAP, reconciliation |
|11 | Monitoring         | `quant/monitor/dashboard.py`                     | positions, P&L, exposure, confidence, orders, latency, errors |
|12 | Logging            | `quant/journal/logger.py`                        | predictions/orders/fills/errors/model versions/param changes (JSONL + DuckDB) |
|13 | Retraining         | `quant/retrain/scheduler.py`                     | scheduled challenger-vs-champion, drift detection, auto-rollback |
|14 | Tech stack         | `requirements-quant.txt`                         | numpy/pandas/sklearn/statsmodels/duckdb/parquet/ib_async |
|15 | Dev workflow       | `quant/pipeline.py`                              | ingest -> universe -> train -> backtest -> report CLI |

## Quick start

```bash
venv\Scripts\activate
pip install -r requirements.txt -r requirements-quant.txt

python -m quant.pipeline ingest              # cache daily bars (default ticker set)
python -m quant.pipeline universe            # who's tradeable
python -m quant.pipeline train               # features -> boosted model -> walk-forward
python -m quant.pipeline backtest            # simulate with costs
python -m quant.pipeline report              # scorecard

# your own symbols
python -m quant.pipeline all --tickers NVDA,AMD,SMCI
python -m quant.pipeline all --watchlist my_list.txt
```

All tunables live in `constants/parameters.py` (QUANT_* section).
Storage lands in `quant_data/` (parquet + DuckDB + model registry + journal).

## BTCUSD without Godel

Godel Terminal is temporarily offline. The BTCUSD path does not depend on it
at all - it's a standalone linear autoregressive model (`quant/models/linear_ar.py`,
the sklearn-native equivalent of the single-`nn.Linear`-layer approach in the
`qt4/build-a-quant-trading-strategy-main` reference repo) trained on lagged
log-returns, with its own data source, cost model, and pipeline commands:

```bash
python -m quant.pipeline btc-ingest      # bundled seed CSV + Binance klines extension
python -m quant.pipeline btc-train       # lagged log-returns -> logistic AR -> walk-forward
python -m quant.pipeline btc-backtest    # 24/7 sim, Binance-style taker fee + wider slippage
python -m quant.pipeline btc-all
python -m quant.pipeline btc-ingest --offline   # skip Binance, use only the bundled seed
```

Notes:
- `quant/data/seed/BTCUSDT_12h_ohlc.csv` bootstraps history with no network
  call required; `quant/data/btc_data.py` extends it from Binance's public
  klines REST endpoint (no API key) when reachable. Binance's `api.binance.com`
  geo-blocks some regions (HTTP 451) - `get_btc_bars()` catches that and falls
  back to the cached/seed bars, logged as a warning, not a crash.
- `BacktestEngine` gained `commission_pct` (percent-of-notional, vs. the
  equity default of per-share commission), `slippage_bps`, and
  `allow_fractional` (BTC position sizes are not integer "shares"). All BTC_*
  tunables live in `constants/parameters.py` alongside everything else.
- This is purely additive: `quant/data/godel_signals.py` and
  `QUANT_GODEL_BLEND_WEIGHT` are untouched. `quant.probability.engine`
  already degrades to model-only when `godel_prob_up` is omitted, so nothing
  needed to change there for Godel to go unused - it'll blend back in
  automatically once PRT is reconnected and `godel_prob_up` is passed again.
- Walk-forward AUC on the seed data is ~0.49 (no edge) - expected for a
  first-pass linear model on crypto returns; treat this as the harness to
  iterate feature/model choices on, not as a strategy to trade.

### $5k retail account configuration (IBKR spot crypto)

The BTC path is costed and sized for a real $5,000 IBKR account buying spot
Bitcoin (Paxos custody), not the earlier Binance placeholder:

- **Fees** (`quant/execution/fees.py`): `max($1.75, 0.18% of notional)`,
  capped at 1% - validated against IBKR's published examples ($100->$1.00,
  $500->$1.75, $1,000->$1.80, $10,000->$18.00). The 0.15%/0.12% tiers need
  $100k+/mo volume and are ignored.
- **Sizing** (`ACCOUNT_EQUITY_USD`, `BTC_RISK_PER_TRADE_PCT`, etc. in
  constants/parameters.py): risk 1% of equity per trade with a 3% stop
  -> ~$1,667 position; floored at $500 (below ~$972 the $1.75 minimum
  dominates the rate) and capped at 50% of equity.
- **Edge gate**: signals only trade when implied expected edge >= 2x the
  round-trip fee (`edge_clears_costs`).
- `python -m quant.pipeline btc-fees` prints the full fee-drag table.

### Scoreboard: judge only out-of-sample, net of fees

`btc-backtest` is in-sample and flattering - ignore it. The real scoreboard:

```bash
python -m quant.pipeline btc-walkforward   # OOS net backtest (per-fold + pooled)
python -m quant.pipeline btc-eval          # OOS strategy vs dumb baselines + verdict
```

`btc-eval` refits the model on each walk-forward train slice, backtests on the
out-of-sample slice with real IBKR fees (warmup bars feed features but can't
be traded - `BacktestEngine(trade_start=...)` prevents leakage), then compares
against buy-and-hold, MA-trend, and never-trade over the same OOS span.

Per-trade stats are **net** of commissions (`pnl = gross_pnl - commission`;
`gross_pnl` retained for transparency), so a positive net expectancy can no
longer hide a losing equity curve.

**Current verdict (OOS, seed data, Mar–Oct 2025 span)**: the strategy is now
net-positive per trade (~+$6/trade, 58% win rate, PF 1.36 after fees - the
fee-aware sizing + no-trade filters fixed the old -8% in-sample result). But
**buy-and-hold BTC returned ~+47%** over the same window vs the strategy's
~+0.8%, so `btc-eval` prints *"DOESN'T BEAT BUY-AND-HOLD"*. On a trending
asset in a bull run, a long/flat timing model has to clear that bar and this
one doesn't. Positive net expectancy is real progress; beating buy-and-hold
is the next milestone, and needs a genuinely predictive signal - not more
tuning.

### Signal knobs (all in constants/parameters.py, BTC_* block)

- `BTC_USE_FEATURES` - rich feature set (momentum/trend/vol/flow) vs AR baseline
- `BTC_PROB_ENTRY` (0.58) - raise to trade fewer, stronger setups
- `BTC_MIN_CONFIDENCE` (0.05) - skip near-coin-flip predictions
- `BTC_MIN_VOL_PCTILE` (0.10) - no-trade filter for dead/choppy tape
- `BTC_MIN_EDGE_TO_COST` (2.0) - required edge as a multiple of round-trip fees

### Modeling-path experiment findings (2026-07-03)

`python -m quant.research.btc_experiments [--daily]` runs the AR/modeling-path
experiments on the exact same walk-forward splits as `btc-eval`: lag-0 A/B,
feature ablation, logistic-C sweep, probability calibration, AUC, exposure.

**No change was promoted** - none cleared "improves OOS net expectancy with
stable folds and holds across regimes". What the experiments showed:

- **AUC < 0.5 on every config** (12h *and* daily), and **calibration is
  inverted**: the baseline's highest-confidence bucket [0.70,1.0) has the
  *lowest* hit rate (39.6%) and negative avg forward return. The model has no
  genuine directional skill - where it's most sure, it's most wrong.
- **`ar_lag_0` (include the just-completed return): rejected.** Non-leaking and
  standard AR theory, but it did not help OOS (mostly hurt: full 6.17->5.43).
- **Richer features: rejected.** `full` ($6.17) underperformed `ar_only`
  ($14.85) on the seed; the volatility group was outright negative.
- **C-tuning / the "best" gated config (ar_only, C=0.03, +$34.8 on seed):
  rejected.** On daily BTC-USD (4,308 bars, 2014-2026, bull+bear) it flips to
  **-$10.46 net, 1/5 folds positive, AUC 0.489** - a pure one-bull-window
  artifact.

**Only correctness change kept:** `evaluate_walk_forward` now defaults
`min_train_frac=0.4` so classification AUC and the net backtest use identical
splits (they didn't before - AUC silently used 0.3). Model parametrization
(`include_lag0`, feature `groups`) was added but left at defaults, so the
trading path is byte-for-byte unchanged (`btc-eval` still 31 trades / +$6.17).

**The binding constraint is data, not the model:** ~692 one-regime 12h bars
can't support feature/hyperparameter selection without curve-fitting. The
real unlock is more, multi-regime data (and a signal with OOS AUC > 0.5) -
not more tuning of a linear model on one bull run.

## Using Godel PRT inside the quant layer

```python
from quant.data import godel_signals
from quant.probability import ProbabilityEngine

# historical PRT rows (from the live bot's SQLite DB) for calibration
hist = godel_signals.from_database(Path("algobot_trades.db"))

# blend live model output with Godel prob_up
engine = ProbabilityEngine()
est = engine.estimate("NVDA", target_pct=4.0, stop_pct=2.0,
                      model_prob_up=0.58, godel_prob_up=0.61)
print(est.prob_target_first, est.expected_return_pct)
```

## What is deliberately stubbed (needs paid data / a decision)

- Tick/TAQ data, Level II depth, queue position (IBKR add-on subscriptions)
- Implied volatility (options feed)
- Options flow, ETF flows, satellite, card spending, social sentiment (vendors)
- Economic calendar (Trading Economics or similar)
- Borrow availability (IBKR shortable file - wire when shorting with size)

Each stub raises `NotImplementedError` naming the provider it expects, so
nothing silently pretends to have data it doesn't.

## A caution (kept from the spec, because it's the part that matters)

Promising backtests usually die from overfitting, look-ahead bias,
survivorship bias, ignored costs, or over-eager retraining. The framework
defends where it can - embargoed walk-forward splits, next-open fills,
commissions+slippage in every simulated fill, challenger-vs-champion
promotion - but yfinance history is survivorship-biased (delisted names
are missing). Treat results as upper bounds; paper trade before believing.

### Signal search findings (2026-07-04, `btc-signal-search`)

60 configs (5 labels x 2 feature sets x 3 regime filters x {12h, multi-regime
daily 2014-2026}), judged by the same walk-forward OOS net backtest as
btc-eval. **First candidate to clear every promotion gate:**

- **1d / cost_5bar / full features / trend200+vol filter**: 85 OOS trades,
  +$15.50/trade net, PF 1.58, 4/5 folds positive, worst fold DD -5.5%,
  9.7% exposure. Calibration is finally right-way-up (highest-prob buckets
  have the highest hit rates; top-decile lift 1.17).
- The edge is bull-regime timing by construction: the trend filter keeps it
  ~flat in bear (6 trades, -$121 total). Most recent fold (2025-2026) is
  slightly negative - watch this in paper observation.
- Funding/derivatives features: SKIPPED (OKX API only serves ~3 months);
  cache accumulates incrementally and auto-enables at >= 365 days.
- Status: **paper observation only.** btc-eval defaults unchanged; the
  candidate is a daily-cadence config and must be observed at that cadence.

### Paper observation of the promoted candidate (2026-07-04)

The winning search config is promoted to **PAPER OBSERVATION ONLY** - live
trading stays disabled everywhere, and the 12h btc-eval default is unchanged.

    python -m quant.pipeline btc-observe-daily [--offline] [--count N]
        Logs today's (or the last N) daily prediction(s) for the candidate:
        prob, every gate pass/fail, regime filter state, hypothetical
        entry/exit, expected edge, fees, realized 1/3/5-bar forward returns.
        Journal: quant_data/journal/btc_daily_observations.jsonl
        Never places orders (imports no IBKR/order code).

    python -m quant.pipeline btc-paper-report
        Scoreboard over the observation journal: re-resolves every
        would-enter signal against the freshest bars, reports paper win
        rate / net expectancy / PF vs the OOS reference (+$15.50, PF 1.58).
        Verdict is gated: reconsideration requires >= 20 paper signals OR
        >= 60 calendar days - before that the only verdict is KEEP OBSERVING.

Run btc-observe-daily once per day (after 00:00 UTC when the daily bar
closes); a Task Scheduler entry is the natural next step once observations
accumulate.

### Hardened daily routine (2026-07-04)

    python -m quant.pipeline btc-daily-routine [--offline]

One command that runs, in order: paper-safety verification (AST import scan
+ order-token scan of the observation path - aborts on violation), daily
data refresh + quality gate (observation is SKIPPED if the gate fails, so a
bad bar never enters the journal), btc-observe-daily, btc-paper-report with
artifacts, and a concise status file.

Artifacts per run:
  quant_data/daily_status/YYYY-MM-DD.md   concise daily status (+ latest.md)
  quant_data/btc_paper_report.md          markdown scoreboard vs OOS reference
  quant_data/btc_observations.csv         every observation w/ no-trade reason codes
  quant_data/btc_daily_routine_result.json

The promoted candidate is frozen (human-readable, smoke-test-enforced) in
  quant/candidates/btc_daily_cost5bar_full_trend200vol.json

Task Scheduler (no admin): `powershell -ExecutionPolicy Bypass -File
scripts\setup_btc_daily_task.ps1` registers a per-user task at 20:35 local
(after the 00:00 UTC daily close) running scripts\run_btc_daily_routine.bat.
