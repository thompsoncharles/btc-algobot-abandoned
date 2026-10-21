"""
Section 15 — Development workflow orchestrator.

    python -m quant.pipeline ingest     # 1. gather clean historical data
    python -m quant.pipeline universe   # 2. define the tradeable universe
    python -m quant.pipeline train      # 3-5. features -> model -> walk-forward
    python -m quant.pipeline backtest   # 6. simulate with costs + slippage
    python -m quant.pipeline report     # metrics scorecard of the backtest
    python -m quant.pipeline all        # everything above in order

Default symbol source: sector ETFs + SPY + user tickers via --tickers, or
--watchlist FILE (one symbol per line, '#' comments ok). Paper trading of
the result remains the live bot's job (scripts/algo_loop.py).

BTCUSD path (Godel PRT is temporarily offline; this runs standalone with a
linear autoregressive model - no Godel dependency, see quant/models/linear_ar.py):

    python -m quant.pipeline btc-ingest       # seed CSV + extend via Binance klines
    python -m quant.pipeline btc-train        # fit model -> walk-forward AUC -> registry
    python -m quant.pipeline btc-backtest     # IN-SAMPLE sim (quick sanity read only)
    python -m quant.pipeline btc-walkforward  # OUT-OF-SAMPLE net backtest (the honest number)
    python -m quant.pipeline btc-eval         # OOS strategy vs dumb baselines (the scoreboard)
    python -m quant.pipeline btc-data-audit   # verify BTC sources/staleness/overlap
    python -m quant.pipeline btc-label-eval   # test trade-aware BTC label definitions
    python -m quant.pipeline btc-context      # fetch SPY/QQQ/VIX/DXY/rates context
    python -m quant.pipeline btc-observe      # paper/log prediction only, no orders
    python -m quant.pipeline btc-fees         # fee-drag table
    python -m quant.pipeline btc-all          # ingest -> train -> walkforward -> eval -> fees

Judge only on btc-eval / btc-walkforward (out-of-sample, net of IBKR fees).
Ignore btc-backtest's in-sample numbers. Signal/threshold/regime knobs and
the $5k fee-aware sizing all live in constants/parameters.py (BTC_* block).
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
from loguru import logger

# allow `python quant/pipeline.py` too
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant import config
from quant.data.market_data import MarketData
from quant.data.store import DataStore
from quant.features import assemble_features
from quant.models.boosted import GradientBoostingModel
from quant.retrain.scheduler import RetrainScheduler
from quant.universe.selection import UniverseSelector

DEFAULT_TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
                   "META", "TSLA", "AMD", "JPM", "XOM", "UNH", "LLY"]


def _tickers(args) -> List[str]:
    if args.watchlist:
        lines = Path(args.watchlist).read_text().splitlines()
        out = [ln.split("#")[0].strip().upper() for ln in lines]
        return [t for t in out if t]
    if args.tickers:
        return [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    return DEFAULT_TICKERS


def _label_and_features(store: DataStore, tickers: List[str]):
    """
    Stacked (X, y) across tickers: features at t, fwd return over horizon.
    Rows are keyed (date, ticker) and sorted by date so positional
    walk-forward splits stay temporally ordered across the whole panel.
    """
    frames = []
    for t in tickers:
        bars = store.load_bars(t, "1d")
        if bars is None or len(bars) < 120:
            continue
        f = assemble_features(bars)
        fwd = bars["Close"].shift(-config.TARGET_HORIZON_BARS) / bars["Close"] - 1
        df = f.assign(_y=fwd)
        df.index = pd.MultiIndex.from_product([[t], df.index], names=["ticker", "date"])
        frames.append(df)
    if not frames:
        raise SystemExit("no usable history - run `python -m quant.pipeline ingest` first")
    panel = (pd.concat(frames)
             .replace([float("inf"), float("-inf")], float("nan"))
             .dropna()
             .swaplevel()
             .sort_index())  # sorted by date first
    return panel.drop(columns="_y"), panel["_y"]


def cmd_ingest(args) -> None:
    tickers = _tickers(args)
    md = MarketData()
    md.fetch_history(tickers + ["^VIX"], interval="1d")
    logger.info("ingest complete: {} tickers cached under {}", len(tickers), config.BARS_DIR)


def cmd_universe(args) -> None:
    tickers = _tickers(args)
    report = UniverseSelector().select(tickers)
    print(json.dumps({"survivors": report.survivors,
                      "rejected": report.rejected}, indent=2))


def cmd_train(args) -> None:
    store = DataStore()
    tickers = _tickers(args)
    X, y = _label_and_features(store, tickers)
    logger.info("training matrix: {} rows x {} features", *X.shape)
    scheduler = RetrainScheduler()
    promoted = scheduler.retrain(lambda: GradientBoostingModel(), X, y)
    print(f"promoted version: {promoted}" if promoted
          else "challenger did not beat champion (or no valid folds)")


def cmd_backtest(args) -> None:
    from quant.backtest.engine import BacktestEngine
    from quant.models.registry import ModelRegistry

    store = DataStore()
    tickers = _tickers(args)
    bars = {t: store.load_bars(t, "1d") for t in tickers}
    bars = {t: b for t, b in bars.items() if b is not None}
    try:
        model, meta = ModelRegistry().load("prob_up")
        logger.info("backtesting model v{}", meta.get("version"))
    except FileNotFoundError:
        raise SystemExit("no trained model - run `python -m quant.pipeline train` first")

    def signal_fn(day, history: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        out = {}
        for t, h in history.items():
            f = assemble_features(h).iloc[[-1]].replace(
                [float("inf"), float("-inf")], float("nan")).dropna()
            if not f.empty:
                out[t] = float(model.predict_proba(f)[0])
        return out

    logger.warning("NOTE: this backtest overlaps the model's training window - "
                   "results are IN-SAMPLE and inflated. The honest out-of-sample "
                   "numbers are the walk-forward folds stored in the model registry "
                   "meta.json (quant_data/models/prob_up/).")
    result = BacktestEngine(bars).run(signal_fn)
    out = config.DATA_DIR / "backtest_result.json"
    out.write_text(json.dumps({
        "summary": result.summary,
        "trades": result.trades.to_dict("records") if result.trades is not None else [],
    }, indent=2, default=str))
    print(json.dumps(result.summary, indent=2))
    print(f"full result: {out}")


def cmd_report(args) -> None:
    path = config.DATA_DIR / "backtest_result.json"
    if not path.exists():
        raise SystemExit("no backtest_result.json - run backtest first")
    print(json.dumps(json.loads(path.read_text())["summary"], indent=2))


# ============================================================== BTCUSD ===
BTC_TICKER = "BTCUSD"
BTC_MODEL_NAME = "btc_linear_ar"


def _btc_model():
    """Feature-rich linear model, or the lagged-returns AR baseline, per config."""
    from quant.models.linear_ar import LinearARModel, LinearBTCModel
    if config.BTC_USE_FEATURES:
        return LinearBTCModel(n_lags=config.BTC_AR_LAGS,
                              forecast_horizon=config.BTC_FORECAST_HORIZON_BARS,
                              ma_window=config.BTC_MA_WINDOW)
    return LinearARModel(n_lags=config.BTC_AR_LAGS,
                         forecast_horizon=config.BTC_FORECAST_HORIZON_BARS)


def _btc_Xy(bars):
    """(X, y) matrix for the configured model type - for classification AUC."""
    model = _btc_model()
    from quant.models.linear_ar import LinearBTCModel
    if isinstance(model, LinearBTCModel):
        from quant.features.btc_features import forward_return_target
        X = model.build_features(bars)
        y = forward_return_target(bars["Close"].astype(float), config.BTC_FORECAST_HORIZON_BARS)
    else:
        X = model.build_features(bars["Close"])
        y = model.build_target(bars["Close"])
    df = X.assign(_y=y).dropna()
    return df.drop(columns="_y"), df["_y"]


def _btc_signal_fn(model, notional: float):
    """
    Fee- and regime-aware signal from a fitted BTC model. Returns
    {BTC: prob} only when every gate passes: enough confidence, above the
    entry threshold, live vol above the dead-tape floor, and implied edge
    clearing round-trip costs. Otherwise {} (stay flat - avoiding bad
    trades is the first real edge on a fee-heavy small account).
    """
    from quant.execution.fees import edge_clears_costs
    from quant.features.btc_features import latest_vol_pctile
    from quant.probability.engine import expected_return, prob_target_before_stop

    def signal_fn(day, history: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        h = history.get(BTC_TICKER)
        if h is None or len(h) < 60:
            return {}
        feat = model.latest_features(h)
        if feat.empty:
            return {}
        prob = float(model.predict_proba(feat)[0])
        if abs(prob - 0.5) < config.BTC_MIN_CONFIDENCE:   # near coin-flip
            return {}
        if prob < config.BTC_PROB_ENTRY:                  # not strong enough
            return {}
        if latest_vol_pctile(h) < config.BTC_MIN_VOL_PCTILE:  # dead/choppy tape
            return {}
        hit = prob_target_before_stop(prob, config.BTC_TARGET_PCT, config.BTC_STOP_PCT)
        edge_pct = expected_return(hit, config.BTC_TARGET_PCT, config.BTC_STOP_PCT)
        if not edge_clears_costs(edge_pct, notional):     # fees eat the edge
            return {}
        return {BTC_TICKER: prob}

    return signal_fn


def _btc_fit_fn(notional: float):
    """train_bars -> signal_fn closure (fits a fresh model). For walk-forward."""
    def fit(train_bars):
        model = _btc_model().fit_from_bars(train_bars)
        return _btc_signal_fn(model, notional)
    return fit


def _btc_engine_kwargs(notional: float) -> dict:
    from quant.execution.fees import ibkr_crypto_fee
    return dict(
        initial_equity=config.ACCOUNT_EQUITY_USD,
        max_positions=1,
        prob_entry=config.BTC_PROB_ENTRY,
        stop_pct=config.BTC_STOP_PCT, target_pct=config.BTC_TARGET_PCT,
        max_hold_bars=config.BTC_MAX_HOLD_BARS,
        commission_fn=ibkr_crypto_fee,
        slippage_bps=config.BTC_SLIPPAGE_BPS,
        allow_fractional=True,
        position_notional=notional,
        min_trade_notional=config.BTC_MIN_TRADE_NOTIONAL_USD,
    )


def cmd_btc_ingest(args) -> None:
    from quant.data.btc_data import get_btc_bars

    bars = get_btc_bars(interval=config.BTC_INTERVAL, refresh=not args.offline)
    logger.info("BTC ingest complete: {} bars ({} to {}) cached under {}",
               len(bars), bars.index.min().date(), bars.index.max().date(),
               config.BARS_DIR)


def cmd_btc_train(args) -> None:
    from quant.data.store import DataStore

    store = DataStore()
    bars = store.load_bars(BTC_TICKER, config.BTC_INTERVAL)
    if bars is None or len(bars) < 60:
        raise SystemExit("no BTC history - run `python -m quant.pipeline btc-ingest` first")

    X, y = _btc_Xy(bars)
    kind = "feature" if config.BTC_USE_FEATURES else "AR"
    logger.info("BTC {} training matrix: {} rows x {} features", kind, *X.shape)

    scheduler = RetrainScheduler(model_name=BTC_MODEL_NAME)
    promoted = scheduler.retrain(_btc_model, X, y, n_folds=5)
    if promoted:
        from quant.models.registry import ModelRegistry
        _, meta = ModelRegistry().load(BTC_MODEL_NAME)
        print(f"promoted version: {promoted}  walk_forward_auc={meta['walk_forward_auc']:.4f}")
    else:
        print("challenger did not beat champion (or no valid folds)")


def _btc_position_notional() -> float:
    """
    Risk-based size for the configured account: risking BTC_RISK_PER_TRADE_PCT
    of equity with a BTC_STOP_PCT stop implies notional = risk$/stop%. Clamped
    to [BTC_MIN_TRADE_NOTIONAL_USD, BTC_MAX_POSITION_PCT of equity] so the
    $1.75 fee floor can't dominate and one trade can't swamp a $5k book.
    """
    equity = config.ACCOUNT_EQUITY_USD
    risk_dollars = equity * config.BTC_RISK_PER_TRADE_PCT / 100
    raw = risk_dollars / (config.BTC_STOP_PCT / 100)
    cap = equity * config.BTC_MAX_POSITION_PCT / 100
    return min(max(raw, config.BTC_MIN_TRADE_NOTIONAL_USD), cap)


def _load_btc_bars():
    from quant.data.store import DataStore
    bars = DataStore().load_bars(BTC_TICKER, config.BTC_INTERVAL)
    if bars is None or len(bars) < 60:
        raise SystemExit("no BTC history - run `python -m quant.pipeline btc-ingest` first")
    return bars


def cmd_btc_backtest(args) -> None:
    """In-sample backtest (trains on all data). Kept for a quick sanity read;
    the honest number is btc-walkforward / btc-eval."""
    from quant.backtest.engine import BacktestEngine
    from quant.execution.fees import round_trip_cost_pct

    bars = _load_btc_bars()
    notional = _btc_position_notional()
    logger.info("account ${:,.0f} | position notional ${:,.0f} | round-trip cost "
                "{:.3f}% | edge gate {:.1f}x | prob_entry {:.2f}",
                config.ACCOUNT_EQUITY_USD, notional, round_trip_cost_pct(notional),
                config.BTC_MIN_EDGE_TO_COST, config.BTC_PROB_ENTRY)
    logger.warning("NOTE: IN-SAMPLE (model trained on the same bars). Use "
                   "btc-walkforward / btc-eval for the honest out-of-sample number.")

    model = _btc_model().fit_from_bars(bars)
    signal_fn = _btc_signal_fn(model, notional)
    result = BacktestEngine({BTC_TICKER: bars}, **_btc_engine_kwargs(notional)).run(signal_fn)
    out = config.DATA_DIR / "btc_backtest_result.json"
    out.write_text(json.dumps({"summary": result.summary,
                               "trades": result.trades.to_dict("records")
                               if result.trades is not None else []},
                              indent=2, default=str))
    print(json.dumps(result.summary, indent=2))
    print(f"full result: {out}")


def cmd_btc_walkforward(args) -> None:
    """The real scoreboard: out-of-sample walk-forward net backtest."""
    from quant.backtest.validation import walk_forward_oos_backtest, evaluate_walk_forward
    from quant.execution.fees import round_trip_cost_pct

    bars = _load_btc_bars()
    notional = _btc_position_notional()
    logger.info("walk-forward OOS | notional ${:,.0f} | round-trip {:.3f}% | "
                "features={} | prob_entry={:.2f}", notional,
                round_trip_cost_pct(notional), config.BTC_USE_FEATURES,
                config.BTC_PROB_ENTRY)

    result = walk_forward_oos_backtest(
        bars, _btc_fit_fn(notional), _btc_engine_kwargs(notional),
        ticker=BTC_TICKER, n_folds=5, min_train_frac=0.4, embargo=5, warmup=252,
    )
    # classification AUC per fold (separate lens on the same model)
    X, y = _btc_Xy(bars)
    auc = evaluate_walk_forward(_btc_model, X, y, n_folds=5)
    if not auc.empty:
        result["aggregate"]["mean_oos_auc"] = float(auc["auc"].mean())
    out = config.DATA_DIR / "btc_walkforward_result.json"
    out.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))
    print(f"full result: {out}")


def _check_btc_data_quality() -> None:
    """Block btc-eval if data fails quality gate."""
    from quant.data.btc_audit import build_audit
    audit = build_audit(include_yahoo=False, include_ibkr=False)
    gate = audit.get("quality_gate", {})
    if not gate.get("passes"):
        print(f"\n[FAIL] BTC DATA QUALITY GATE FAILED:\n{gate.get('verdict')}\n")
        print(f"Recommendation: {audit.get('recommendation')}\n")
        raise SystemExit(f"Cannot run btc-eval: {gate.get('verdict')}")
    print(f"[OK] BTC data quality: {gate.get('verdict')}\n")


def cmd_btc_eval(args) -> None:
    """Scoreboard: walk-forward OOS strategy vs dumb baselines on the same span."""
    from quant.backtest.baselines import run_all
    from quant.backtest.regimes import regime_breakdown
    from quant.backtest.validation import walk_forward_oos_backtest, walk_forward_splits
    from quant.execution.fees import round_trip_cost_pct

    _check_btc_data_quality()  # Fail fast if data is unsafe
    bars = _load_btc_bars()
    notional = _btc_position_notional()

    wf = walk_forward_oos_backtest(
        bars, _btc_fit_fn(notional), _btc_engine_kwargs(notional),
        ticker=BTC_TICKER, n_folds=5, min_train_frac=0.4, embargo=5, warmup=252,
        include_details=True,
    )
    # baselines over the same OOS span (from first test bar to the end)
    splits = walk_forward_splits(bars.index, n_folds=5, min_train_frac=0.4, embargo=5)
    oos_start = int(splits[0].test_idx[0]) if splits else 0
    oos_bars = bars.iloc[oos_start:]
    baselines = run_all(oos_bars, config.ACCOUNT_EQUITY_USD, ma_window=config.BTC_MA_WINDOW)
    regimes = regime_breakdown(oos_bars, wf.get("trades", []), wf.get("equity", []))

    strat_net = wf["aggregate"]["mean_fold_net_return_pct"]
    scoreboard = {
        "oos_span": {"from": str(oos_bars.index.min()), "to": str(oos_bars.index.max()),
                     "bars": len(oos_bars)},
        "position_notional_usd": notional,
        "round_trip_cost_pct": round(round_trip_cost_pct(notional), 4),
        "strategy_walk_forward": wf["aggregate"],
        "baselines": baselines,
        "regime_breakdown": regimes,
        "verdict": _btc_verdict(wf["aggregate"], baselines),
    }
    out = config.DATA_DIR / "btc_eval_result.json"
    out.write_text(json.dumps({**scoreboard, "folds": wf["folds"]}, indent=2, default=str))
    print(json.dumps(scoreboard, indent=2, default=str))
    print(f"full result: {out}")


def cmd_btc_data_audit(args) -> None:
    """Audit current BTC data sources without changing btc-eval."""
    from quant.data.btc_audit import build_audit

    report = build_audit(include_yahoo=not args.offline, include_ibkr=args.ibkr)
    out = config.DATA_DIR / "btc_data_audit.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    print(f"full result: {out}")


def cmd_btc_label_eval(args) -> None:
    """Research-only: compare BTC label definitions on the same OOS judge."""
    from quant.research.btc_label_eval import print_report, run_label_eval

    print_report(run_label_eval())


def cmd_btc_signal_search(args) -> None:
    """Research-only: signal hunt across labels/features/regimes/datasets."""
    from quant.research.btc_signal_search import print_report, run_signal_search

    print_report(run_signal_search(offline=args.offline))


def cmd_btc_observe_daily(args) -> None:
    """Paper-observe the promoted daily candidate. Never places orders."""
    from quant.data.btc_data import get_btc_daily_history
    from quant.monitor.btc_daily_observer import log_daily_observations
    from quant.research.btc_label_eval import _position_notional

    bars = get_btc_daily_history(refresh=not args.offline)
    report = log_daily_observations(bars, _position_notional(), count=args.count)
    out = config.DATA_DIR / "btc_observe_daily_result.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    print(f"full result: {out}")


def cmd_btc_paper_report(args) -> None:
    """Paper-observation scoreboard + gated verdict for the daily candidate."""
    from quant.monitor.btc_paper_report import build_paper_report, write_artifacts

    report = build_paper_report()
    artifacts = write_artifacts(report)
    report = {k: v for k, v in report.items() if not k.startswith("_")}
    report["artifacts"] = artifacts
    out = config.DATA_DIR / "btc_paper_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    print(f"full result: {out}")


def cmd_btc_daily_routine(args) -> None:
    """One-command daily paper routine: safety -> data gate -> observe -> report -> status."""
    from quant.monitor.btc_daily_routine import run_daily_routine

    report = run_daily_routine(offline=args.offline)
    out = config.DATA_DIR / "btc_daily_routine_result.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    print(f"full result: {out}")


def cmd_btc_context(args) -> None:
    """Fetch daily cross-asset context used by optional BTC external features."""
    from quant.data.market_data import MarketData
    from quant.features.btc_features import CONTEXT_TICKERS

    md = MarketData()
    fetched = md.fetch_history(list(CONTEXT_TICKERS), interval="1d", years=12)
    report = {
        "context_tickers": list(CONTEXT_TICKERS),
        "fetched": {t: {"bars": len(df), "from": str(df.index.min()),
                        "to": str(df.index.max())}
                    for t, df in fetched.items()},
        "note": "Features use these daily bars shifted by one day before aligning to BTC.",
    }
    out = config.DATA_DIR / "btc_context_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    print(f"full result: {out}")


def cmd_btc_observe(args) -> None:
    """Log BTC paper observations only. This command never places orders."""
    from quant.monitor.btc_observer import log_observations

    bars = _load_btc_bars()
    notional = _btc_position_notional()
    report = log_observations(bars, notional, count=args.count)
    out = config.DATA_DIR / "btc_observe_result.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    print(f"full result: {out}")


def _btc_verdict(agg: dict, baselines: list) -> str:
    bh = next((b for b in baselines if b["strategy"] == "buy_and_hold"), {})
    strat = agg.get("mean_fold_net_return_pct", 0.0)
    exp = agg.get("pooled_net_expectancy_usd", 0.0)
    n = agg.get("total_oos_trades", 0)
    if n == 0:
        return "NO OOS TRADES - filters rejected everything; loosen thresholds or the edge isn't there."
    if exp <= 0:
        return f"NOT VIABLE - negative net expectancy (${exp:.2f}/trade after fees) over {n} OOS trades."
    if strat <= bh.get("net_return_pct", 0.0):
        return (f"DOESN'T BEAT BUY-AND-HOLD - strat {strat:.1f}% vs hold "
                f"{bh.get('net_return_pct', 0.0):.1f}%; complexity not earning its keep.")
    return (f"MARGINAL EDGE - strat {strat:.1f}% beats hold {bh.get('net_return_pct', 0.0):.1f}% "
            f"at ${exp:.2f}/trade net over {n} trades; paper-trade before trusting.")


def cmd_btc_fees(args) -> None:
    """Fee-drag report: old flat assumption vs the real IBKR schedule."""
    from quant.execution.fees import fee_table, ibkr_crypto_fee, round_trip_cost_pct

    old_flat_pct = 0.045  # yesterday's Binance-taker placeholder
    notional = _btc_position_notional()
    rows = []
    for r in fee_table():
        n = r["notional_usd"]
        rows.append({**r,
                     "old_flat_round_trip_pct": round(old_flat_pct * 2, 4),
                     "delta_round_trip_pct": round(r["round_trip_pct"] - old_flat_pct * 2, 4)})
    report = {
        "fee_model": "IBKR spot crypto: max($1.75, 0.18% notional), capped at 1%",
        "reference_check": {
            "$100": ibkr_crypto_fee(100), "$500": ibkr_crypto_fee(500),
            "$1000": ibkr_crypto_fee(1000), "$10000": ibkr_crypto_fee(10000),
        },
        "table": rows,
        "configured_position": {
            "account_equity_usd": config.ACCOUNT_EQUITY_USD,
            "position_notional_usd": notional,
            "round_trip_cost_pct": round(round_trip_cost_pct(notional), 4),
            "breakeven_move_pct": round(round_trip_cost_pct(notional), 4),
            "edge_gate_multiple": config.BTC_MIN_EDGE_TO_COST,
        },
    }
    print(json.dumps(report, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description="quant pipeline")
    p.add_argument("step", choices=["ingest", "universe", "train", "backtest", "report", "all",
                                    "btc-ingest", "btc-train", "btc-backtest", "btc-walkforward",
                                    "btc-eval", "btc-data-audit", "btc-label-eval", "btc-context",
                                    "btc-observe", "btc-fees", "btc-signal-search",
                                    "btc-observe-daily", "btc-paper-report",
                                    "btc-daily-routine", "btc-all"])
    p.add_argument("--tickers", help="comma-separated symbols")
    p.add_argument("--watchlist", help="file with one symbol per line")
    p.add_argument("--offline", action="store_true",
                   help="btc-ingest: skip the Binance refresh, use only the bundled/cached seed")
    p.add_argument("--ibkr", action="store_true",
                   help="btc-data-audit: also try IBKR/TWS BTC crypto historical bars")
    p.add_argument("--count", type=int, default=1,
                   help="btc-observe: number of latest cached bars to log")
    args = p.parse_args()
    config.ensure_dirs()

    steps = {"ingest": cmd_ingest, "universe": cmd_universe, "train": cmd_train,
             "backtest": cmd_backtest, "report": cmd_report,
             "btc-ingest": cmd_btc_ingest, "btc-train": cmd_btc_train,
             "btc-backtest": cmd_btc_backtest, "btc-walkforward": cmd_btc_walkforward,
             "btc-eval": cmd_btc_eval, "btc-data-audit": cmd_btc_data_audit,
             "btc-label-eval": cmd_btc_label_eval,
             "btc-context": cmd_btc_context, "btc-observe": cmd_btc_observe,
             "btc-fees": cmd_btc_fees, "btc-signal-search": cmd_btc_signal_search,
             "btc-observe-daily": cmd_btc_observe_daily,
             "btc-paper-report": cmd_btc_paper_report,
             "btc-daily-routine": cmd_btc_daily_routine}
    if args.step == "all":
        for name in ("ingest", "universe", "train", "backtest", "report"):
            logger.info("=== pipeline step: {} ===", name)
            steps[name](args)
    elif args.step == "btc-all":
        for name in ("btc-ingest", "btc-train", "btc-walkforward", "btc-eval",
                     "btc-data-audit", "btc-label-eval", "btc-context",
                     "btc-observe", "btc-fees"):
            logger.info("=== pipeline step: {} ===", name)
            steps[name](args)
    else:
        steps[args.step](args)


if __name__ == "__main__":
    main()
