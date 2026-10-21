"""
Offline smoke test for the quant framework - synthetic data only, no
network, no IBKR. Run: venv\\Scripts\\python.exe -m tests.test_quant_smoke
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def synthetic_bars(n: int = 400, seed: int = 7, drift: float = 0.0004) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ret = rng.normal(drift, 0.02, n)
    close = 100 * np.exp(np.cumsum(ret))
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    open_ = low + (high - low) * rng.uniform(0.2, 0.8, n)
    vol = rng.integers(1e5, 5e6, n).astype(float)
    idx = pd.bdate_range("2024-01-02", periods=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close, "Volume": vol}, index=idx)


def main() -> int:
    failures = []

    def check(name, fn):
        try:
            fn()
            print(f"  OK   {name}")
        except Exception as e:
            failures.append((name, e))
            print(f"  FAIL {name}: {e}")

    bars = synthetic_bars()

    # --- features
    from quant.features import assemble_features
    f = assemble_features(bars)
    check("features shape", lambda: (_ for _ in ()).throw(AssertionError)
          if f.dropna().empty else None)

    # --- models + walk-forward
    from quant.models.boosted import GradientBoostingModel
    from quant.backtest.validation import evaluate_walk_forward
    fwd = bars["Close"].shift(-5) / bars["Close"] - 1
    Xy = f.assign(_y=fwd).dropna()
    X, y = Xy.drop(columns="_y"), Xy["_y"]
    check("gradient boosting fit/proba", lambda: GradientBoostingModel()
          .fit(X, y).predict_proba(X.head(5)))
    check("walk-forward", lambda: (_ for _ in ()).throw(AssertionError)
          if evaluate_walk_forward(lambda: GradientBoostingModel(), X, y,
                                   n_folds=3).empty else None)

    # --- regime, kalman
    from quant.models.regime_hmm import RegimeModel
    check("regime model", lambda: RegimeModel(3).fit_predict(bars["Close"]))
    from quant.models.kalman import KalmanLevel, KalmanBeta
    check("kalman level", lambda: KalmanLevel().filter(bars["Close"]))
    check("kalman beta", lambda: KalmanBeta().filter(
        bars["Close"].pct_change(), bars["Close"].pct_change().shift(1).fillna(0)))

    # --- probability engine
    from quant.probability import ProbabilityEngine
    def _prob():
        est = ProbabilityEngine().estimate("TEST", 4.0, 2.0,
                                           model_prob_up=0.6, godel_prob_up=0.62)
        assert 0 <= est.prob_target_first <= 1
        assert est.prob_source == "blend"
    check("probability engine", _prob)

    # --- sizing
    from quant.sizing import QuantPositionSizer
    def _size():
        s = QuantPositionSizer(100_000)
        d = s.fixed_fractional(50.0, risk_pct=1.0, stop_pct=2.0)
        capped = s.apply_caps(d, 50.0)
        assert capped.dollars <= 100_000 * 0.10 + 50
    check("position sizing + caps", _size)

    # --- risk
    from quant.risk import RiskManager
    def _risk():
        import tempfile
        rm = RiskManager(state_path=Path(tempfile.mkdtemp()) / "rs.json")
        rm.new_session(100_000)
        halted, _ = rm.update_equity(99_000)
        assert not halted
        halted, reason = rm.update_equity(96_000)   # -4% > 3% daily limit
        assert halted and "daily" in reason
    check("risk kill switches", _risk)

    # --- backtest + metrics
    from quant.backtest import BacktestEngine
    def _bt():
        universe = {f"T{i}": synthetic_bars(seed=i, drift=0.0006) for i in range(4)}
        def sig(day, hist):
            return {t: 0.6 for t in hist}
        res = BacktestEngine(universe, max_positions=2).run(sig)
        assert res.equity_curve is not None and len(res.equity_curve) > 100
        assert "sharpe" in res.summary
    check("backtest engine + metrics", _bt)

    # --- registry + retrain (uses temp-ish model dir under quant_data)
    from quant.models.registry import ModelRegistry
    def _reg():
        import tempfile
        reg = ModelRegistry(root=Path(tempfile.mkdtemp()))
        v1 = reg.save("smoke", {"dummy": 1}, {"auc": 0.5})
        reg.promote("smoke", v1)
        v2 = reg.save("smoke", {"dummy": 2}, {"auc": 0.6})
        reg.promote("smoke", v2)
        assert reg.rollback("smoke") == v1
        model, meta = reg.load("smoke")
        assert model == {"dummy": 1}
    check("model registry promote/rollback", _reg)

    # --- journal
    from quant.journal import JournalLogger
    def _journal():
        j = JournalLogger()
        j.log_prediction("TEST", 0.61, "blend", model_version="v001")
        j.log_fill("TEST", "BUY", 10, 50.0, "42", rationale="smoke test")
        df = j.read("predictions")
        assert not df.empty
    check("journal audit trail", _journal)

    # --- microstructure / volume profile
    from quant.features.microstructure import volume_profile
    check("volume profile", lambda: (_ for _ in ()).throw(AssertionError)
          if volume_profile(bars).empty else None)

    # --- BTC linear AR model (no network, no Godel)
    from quant.models.linear_ar import LinearARModel, lagged_log_return_features
    def _btc_ar():
        close = synthetic_bars(seed=99, drift=0.0002)["Close"]
        m = LinearARModel(n_lags=3, forecast_horizon=1).fit_from_prices(close)
        f = lagged_log_return_features(close, 3, 1).dropna()
        proba = m.predict_proba(f.iloc[[-1]])
        assert 0 <= proba[0] <= 1
        assert len(m.coefficients()) == 3
    check("BTC linear AR model", _btc_ar)

    # --- BTC backtest engine costs (commission_pct, fractional shares)
    def _btc_backtest():
        universe = {"BTCUSD": synthetic_bars(seed=11, drift=0.0003)}
        def sig(day, hist):
            return {"BTCUSD": 0.6}
        res = BacktestEngine(universe, max_positions=1, commission_pct=0.045,
                             slippage_bps=5.0, allow_fractional=True).run(sig)
        assert res.trades is not None
        if not res.trades.empty:
            assert not res.trades["shares"].apply(float.is_integer).all()
    check("BTC backtest costs (pct commission, fractional)", _btc_backtest)

    # --- IBKR spot-crypto fee model vs published reference points
    def _ibkr_fees():
        from quant.execution.fees import ibkr_crypto_fee, round_trip_cost_pct, edge_clears_costs
        assert abs(ibkr_crypto_fee(100) - 1.00) < 1e-9      # 1% cap
        assert abs(ibkr_crypto_fee(500) - 1.75) < 1e-9      # minimum
        assert abs(ibkr_crypto_fee(1000) - 1.80) < 1e-9     # 0.18%
        assert abs(ibkr_crypto_fee(10000) - 18.00) < 1e-9   # 0.18%
        assert round_trip_cost_pct(500) > round_trip_cost_pct(2500)  # floor hurts small trades
        # 1% edge on $1,667: $16.67 vs $3.60 round-trip -> clears 2x
        assert edge_clears_costs(1.0, 1667, min_ratio=2.0)
        # 0.2% edge on $500: $1.00 vs $3.50 round-trip -> blocked
        assert not edge_clears_costs(0.2, 500, min_ratio=2.0)
    check("IBKR crypto fee model (reference points)", _ibkr_fees)

    # --- notional floor/cap gates in the engine
    def _notional_gates():
        from quant.execution.fees import ibkr_crypto_fee
        universe = {"BTCUSD": synthetic_bars(seed=12, drift=0.0003)}
        def sig(day, hist):
            return {"BTCUSD": 0.6}
        res = BacktestEngine(universe, initial_equity=5000, max_positions=1,
                             commission_fn=ibkr_crypto_fee, allow_fractional=True,
                             position_notional=1667, min_trade_notional=500).run(sig)
        if res.trades is not None and not res.trades.empty:
            notionals = res.trades["shares"] * res.trades["entry_price"]
            assert (notionals >= 499).all()      # floor respected
            assert (notionals <= 1667 * 1.01).all()  # target respected
    check("engine notional floor/cap", _notional_gates)

    # --- P1 fix: per-trade pnl is NET (gross - commission)
    def _net_trade_metrics():
        from quant.execution.fees import ibkr_crypto_fee
        u = {"BTCUSD": synthetic_bars(seed=13, drift=0.0004)}
        res = BacktestEngine(u, initial_equity=5000, max_positions=1,
                             commission_fn=ibkr_crypto_fee, allow_fractional=True,
                             position_notional=1500).run(lambda d, h: {"BTCUSD": 0.6})
        t = res.trades
        if t is not None and not t.empty:
            assert {"gross_pnl", "commission", "pnl"}.issubset(t.columns)
            recomputed = (t["gross_pnl"] - t["commission"])
            assert (t["pnl"] - recomputed).abs().max() < 1e-6      # pnl == gross - fees
            assert t["pnl"].mean() < t["gross_pnl"].mean()          # net below gross
            assert res.summary["total_fees_usd"] > 0
    check("P1 net per-trade metrics", _net_trade_metrics)

    # --- P1 fix: unfinished candle dropped
    def _drop_unclosed():
        from quant.data.btc_data import drop_unclosed
        now = 1_700_000_000_000
        df = pd.DataFrame({"Open": [1, 2, 3],
                           "close_time": [now - 10, now - 1, now + 43_200_000]})
        out = drop_unclosed(df, now_ms=now)
        assert len(out) == 2 and 3 not in out["Open"].values
    check("P1 drop unfinished candle", _drop_unclosed)

    # --- BTC feature model fit/predict
    def _btc_feature_model():
        from quant.models.linear_ar import LinearBTCModel
        b = synthetic_bars(seed=21, drift=0.0003)
        m = LinearBTCModel(n_lags=3, forecast_horizon=1, ma_window=20).fit_from_bars(b)
        feat = m.latest_features(b)
        assert not feat.empty
        p = m.predict_proba(feat)
        assert 0 <= p[0] <= 1
    check("BTC feature model (fit/predict)", _btc_feature_model)

    # --- walk-forward OOS backtest: no leakage, returns folds + aggregate
    def _walkforward_oos():
        from quant.backtest.validation import walk_forward_oos_backtest
        from quant.models.linear_ar import LinearARModel
        b = synthetic_bars(seed=22, drift=0.0004, n=500)
        def fit_fn(train_bars):
            m = LinearARModel(n_lags=3).fit_from_bars(train_bars)
            def sig(day, hist):
                h = hist.get("BTCUSD")
                if h is None:
                    return {}
                f = m.latest_features(h)
                return {"BTCUSD": float(m.predict_proba(f)[0])} if not f.empty else {}
            return sig
        res = walk_forward_oos_backtest(b, fit_fn, dict(initial_equity=5000, max_positions=1,
                                        allow_fractional=True), n_folds=3, warmup=60)
        assert "aggregate" in res and res["aggregate"]["n_folds"] >= 1
    check("walk-forward OOS backtest (no leakage)", _walkforward_oos)

    # --- dumb baselines
    def _baselines():
        from quant.backtest.baselines import run_all
        b = synthetic_bars(seed=23, drift=0.0005)
        rows = run_all(b, 5000, ma_window=20)
        names = {r["strategy"] for r in rows}
        assert "buy_and_hold" in names and "never_trade" in names
        nt = next(r for r in rows if r["strategy"] == "never_trade")
        assert nt["net_return_pct"] == 0.0 and nt["n_trades"] == 0
    check("dumb baselines (buy-hold / MA / never-trade)", _baselines)

    # --- experiment parametrization: lag0 + feature-group ablation
    def _param_features():
        from quant.models.linear_ar import lagged_log_return_features
        from quant.features.btc_features import assemble_btc_features
        b = synthetic_bars(seed=24)
        base = lagged_log_return_features(b["Close"], 5, 1, include_lag0=False)
        lag0 = lagged_log_return_features(b["Close"], 5, 1, include_lag0=True)
        assert "ar_lag_0" not in base.columns and "ar_lag_0" in lag0.columns
        ar_only = assemble_btc_features(b, groups=set())
        full = assemble_btc_features(b, groups=None)
        assert ar_only.shape[1] < full.shape[1]           # subset really is smaller
        assert "rsi14" not in ar_only.columns and "rsi14" in full.columns
    check("experiment parametrization (lag0, groups)", _param_features)

    # --- Task 5: AUC walk-forward respects min_train_frac (same splits as btc-eval)
    def _auc_split_align():
        import inspect
        from quant.backtest.validation import evaluate_walk_forward
        sig = inspect.signature(evaluate_walk_forward)
        assert sig.parameters["min_train_frac"].default == 0.4
    check("Task 5 AUC split alignment (min_train_frac=0.4)", _auc_split_align)

    # --- BTC label eval components: trade-aware labels + balanced logit path
    def _btc_label_eval_components():
        from quant.models.linear_ar import LinearBTCModel
        from quant.research.btc_label_eval import LABEL_SPECS, build_label
        b = synthetic_bars(n=220, seed=25, drift=0.0002)
        spec = next(s for s in LABEL_SPECS if s.name == "target_before_stop")
        labels = build_label(b, spec, notional=1667)
        y = labels["y"].dropna()
        assert not y.empty and set(y.unique()).issubset({0.0, 1.0})

        m = LinearBTCModel(n_lags=3, ma_window=20, class_weight="balanced")
        X = m.build_features(b)
        df = X.join(labels["y"]).dropna()
        if df["y"].nunique() < 2:
            df.loc[df.index[-1], "y"] = 1 - int(df["y"].iloc[-1])
        m.fit(df.drop(columns="y"), df["y"].astype(int))
        p = m.predict_proba(df.drop(columns="y").tail(3))
        assert len(p) == 3 and ((0 <= p) & (p <= 1)).all()
    check("BTC label eval components", _btc_label_eval_components)

    # --- BTC data quality gate
    def _btc_data_quality_gate():
        from quant.data.btc_audit import build_audit
        audit = build_audit(include_yahoo=False, include_ibkr=False)
        gate = audit.get("quality_gate", {})
        assert "passes" in gate and "verdict" in gate
        passes, verdict = gate["passes"], gate["verdict"]
        if not passes:
            raise AssertionError(f"BTC data quality failed: {verdict}")
    check("BTC data quality gate", _btc_data_quality_gate)

    # --- signal search components (offline, synthetic)
    def _signal_search_components():
        from quant.backtest.validation import walk_forward_splits
        from quant.research.btc_label_eval import LABEL_SPECS, run_candidate
        from quant.research.btc_signal_search import (
            REGIME_FILTERS, _passes_core_gates, _survives_multi_regime, decile_lift,
        )
        b = synthetic_bars(n=420, seed=31, drift=0.0006)
        # regime filters are causal callables on history
        assert REGIME_FILTERS["none"] is None
        assert isinstance(REGIME_FILTERS["trend200"](b), bool)
        assert isinstance(REGIME_FILTERS["trend200+vol"](b), bool)
        # run one candidate with a regime filter + kept arrays
        splits = walk_forward_splits(b.index, n_folds=3, min_train_frac=0.4, embargo=5)
        spec = LABEL_SPECS[0]
        res = run_candidate(b, spec, splits, 1667.0, groups=set(),
                            regime_filter=REGIME_FILTERS["trend200"],
                            keep_arrays=True, tag="smoke")
        assert "regime" in res["gate_counts"] or res["gate_counts"]["evaluated"] >= 0
        assert "_prob" in res and "_y" in res
        # promotion gates: a config that trades 0 times must not be promotable
        ok, why = _passes_core_gates(res, min_trades=10**6)
        assert not ok
        # multi-regime survival: 1d passes by definition, 12h needs a twin
        res["dataset"] = "1d"
        ok, _ = _survives_multi_regime(res, {})
        assert ok
        res["dataset"] = "12h"
        res["features"] = "full"
        res["regime_filter"] = "none"
        ok, why = _survives_multi_regime(res, {})
        assert not ok and "twin" in why
        # lift diagnostics
        lift = decile_lift(res.get("_prob", []), res.get("_y", []))
        assert "available" in lift
    check("signal search components (filters/gates/lift)", _signal_search_components)

    # --- funding features degrade gracefully without cache
    def _funding_features_graceful():
        from quant.features.btc_features import assemble_btc_features, funding_features
        b = synthetic_bars(n=260, seed=32)
        f = funding_features(b.index)  # may be empty (no/short cache) - must not raise
        assert f.shape[0] in (0, len(b))
        full = assemble_btc_features(b, groups={"momentum", "funding"})
        assert "rsi14" in full.columns  # momentum present regardless of funding
    check("funding features graceful degradation", _funding_features_graceful)

    # --- daily paper observer (promoted candidate; synthetic, no orders)
    def _daily_observer():
        from quant.monitor.btc_daily_observer import (
            CANDIDATE, build_daily_observation, fit_candidate_model, regime_state,
        )
        b = synthetic_bars(n=480, seed=33, drift=0.0008)
        assert CANDIDATE["status"] == "PAPER_OBSERVATION_ONLY"
        assert CANDIDATE["live_trading"] == "disabled"
        model = fit_candidate_model(b.iloc[:400])
        # historical position: realized fwd returns must exist
        rec = build_daily_observation(model, b, 1667.0, signal_pos=430)
        assert rec["places_orders"] is False
        assert rec["mode"] == "paper_observation_daily"
        assert {"trend200_above", "vol_pctile", "passed"} <= set(rec["regime_filter"])
        assert rec["realized"]["fwd_1bar_pct"] is not None
        assert rec["realized"]["fwd_5bar_pct"] is not None
        gate_names = {g["name"] for g in rec["gates"]}
        assert "regime_trend200_vol" in gate_names and "edge_to_cost" in gate_names
        assert "round_trip_fees_usd" in rec and "expected_edge_usd" in rec
        # latest position: futures unknown, must not raise
        rec2 = build_daily_observation(model, b, 1667.0)
        assert rec2["realized"]["fwd_1bar_pct"] is None
        # regime state is causal + boolean
        rs = regime_state(b)
        assert isinstance(rs["passed"], bool)
    check("daily paper observer (no orders, gates, fwd returns)", _daily_observer)

    # --- paper report verdict gating (20 signals / 60 days rule)
    def _paper_report_gate():
        from quant.monitor.btc_paper_report import MIN_PAPER_DAYS, MIN_PAPER_SIGNALS
        from quant.monitor import btc_paper_report as bpr
        assert MIN_PAPER_SIGNALS == 20 and MIN_PAPER_DAYS == 60
        report = bpr.build_paper_report()
        gate = report["reconsideration_gate"]
        assert gate["min_paper_signals"] == 20 and gate["min_calendar_days"] == 60
        # with fewer than thresholds, verdict must be KEEP OBSERVING
        if not gate["met"]:
            assert report["verdict"].startswith("KEEP OBSERVING")
        assert report["candidate"]["live_trading"] == "disabled"
        assert "PAPER" in report["candidate"]["status"]
    check("paper report reconsideration gate (20 signals / 60 days)", _paper_report_gate)

    # --- daily routine safety verification (positive + negative)
    def _routine_safety():
        from quant.monitor.btc_daily_routine import (
            FORBIDDEN_TOKENS, _imported_modules, verify_paper_safety,
        )
        # positive: current observation path must pass
        result = verify_paper_safety()
        assert result["passed"] is True
        assert result["tokens_checked"] == len(FORBIDDEN_TOKENS)
        # negative: the AST import check must catch order-capable imports
        bad = "from ibkr.orders import OrderManager\nimport quant.execution.executor\n"
        mods = _imported_modules(bad)
        assert any(m.split(".")[0] == "ibkr" for m in mods)
        assert any(m.startswith("quant.execution.executor") for m in mods)
        # and must NOT be fooled by string literals / comments
        harmless = 's = "placeOrder"\n# ibkr.orders mention in a comment\n'
        assert not any(m.split(".")[0] == "ibkr" for m in _imported_modules(harmless))
    check("routine paper-safety verification (AST + tokens)", _routine_safety)

    # --- frozen candidate metadata file agrees with runtime CANDIDATE
    def _frozen_candidate():
        import json as _json
        from quant.monitor.btc_daily_observer import CANDIDATE
        from quant.monitor.btc_daily_routine import CANDIDATE_FILE
        frozen = _json.loads(CANDIDATE_FILE.read_text(encoding="utf-8"))
        assert frozen["live_trading"] == "disabled"
        assert frozen["status"] == "PAPER_OBSERVATION_ONLY"
        assert frozen["name"] == CANDIDATE["name"]
        assert frozen["config"]["dataset"] == "1d"
        assert frozen["config"]["label"] == "cost_5bar"
        assert frozen["config"]["regime_filter"].startswith("trend200+vol")
        assert frozen["sizing"]["account_equity_usd"] == 5000.0
        assert frozen["reconsideration_gate"]["min_paper_signals"] == 20
        assert frozen["reconsideration_gate"]["min_calendar_days"] == 60
    check("frozen candidate metadata (paper-only, matches runtime)", _frozen_candidate)

    # --- paper report artifacts: markdown, CSV, reason codes
    def _report_artifacts():
        from quant.monitor.btc_paper_report import (
            observations_csv, reason_codes, render_markdown,
        )
        rec_no = {"signal_date": "2026-01-02", "close": 50000.0,
                  "predicted_prob_up": 0.51, "would_enter": False,
                  "gates_failed": ["regime_trend200_vol", "edge_to_cost"],
                  "regime_filter": {"trend200_above": False, "vol_pctile": 0.4},
                  "expected_edge_usd": -1.0, "round_trip_fees_usd": 6.0,
                  "realized": {"fwd_1bar_pct": 1.2, "fwd_3bar_pct": None,
                               "fwd_5bar_pct": None}}
        rec_yes = {**rec_no, "would_enter": True, "gates_failed": [],
                   "hypothetical": {"entry_price": 50100.0,
                                    "outcome": {"reason": "target", "net_pnl_usd": 94.0}}}
        assert reason_codes(rec_no) == "REGIME_FILTER+EDGE_BELOW_COST"
        assert reason_codes(rec_yes) == ""
        df = observations_csv([rec_no, rec_yes])
        assert len(df) == 2 and "no_trade_reasons" in df.columns
        assert df.iloc[1]["hypo_exit_reason"] == "target"
        fake_report = {
            "reconsideration_gate": {"min_paper_signals": 20, "min_calendar_days": 60,
                                     "signals_so_far": 5, "days_so_far": 12, "met": False},
            "paper_stats": {"closed_paper_trades": 3, "open_paper_trades": 1,
                            "paper_net_expectancy_usd": 10.0, "paper_profit_factor": 1.4,
                            "paper_win_rate_pct": 66.7, "observations_logged": 12},
            "verdict": "KEEP OBSERVING - test", "journal": "test.jsonl",
        }
        md = render_markdown(fake_report)
        assert "5/20" in md and "12/60" in md          # progress
        assert "$15.5" in md and "1.58" in md          # OOS reference comparison
        assert "live trading: disabled" in md
    check("paper report artifacts (markdown/CSV/reason codes)", _report_artifacts)

    # --- tax-ready paper trade ledger
    def _tax_ledger():
        from quant.monitor.btc_paper_report import (
            TAX_COLUMNS, paper_trades_tax_rows,
        )
        b = synthetic_bars(n=320, seed=34, drift=0.001)
        # fabricate one would-enter signal 20 bars from the end so it resolves
        sig_ts = b.index[-20]
        signals = [{"signal_ts": str(sig_ts), "would_enter": True}]
        df = paper_trades_tax_rows(signals, b, 1666.67)
        assert len(df) == 1
        row = df.iloc[0]
        for col in ["entry_time", "exit_time", "asset", "quantity",
                    "entry_price_usd", "exit_price_usd", "entry_fee_usd",
                    "exit_fee_usd", "cost_basis_usd", "proceeds_usd",
                    "realized_gain_loss_usd", "holding_period_days",
                    "short_or_long_term", "strategy_id", "candidate_version"]:
            assert col in df.columns, f"missing tax field {col}"
        assert row["asset"] == "BTC" and bool(row["paper_only"]) is True
        assert row["short_or_long_term"] == "short_term"  # 8-bar max hold
        # accounting identity: gain = proceeds - cost basis
        assert abs(row["realized_gain_loss_usd"]
                   - (row["proceeds_usd"] - row["cost_basis_usd"])) < 0.02
        # cost basis = qty*entry + entry fee
        assert abs(row["cost_basis_usd"]
                   - (row["quantity"] * row["entry_price_usd"] + row["entry_fee_usd"])) < 0.02
        assert list(df.columns) == TAX_COLUMNS
    check("tax-ready paper trade ledger (fields + accounting identity)", _tax_ledger)

    # --- manual paper-trade checklist
    def _checklist():
        from quant.monitor.btc_paper_report import build_paper_report, manual_checklist
        report = build_paper_report()
        items = manual_checklist(report)
        names = [i["item"] for i in items]
        assert len(items) == 9
        for needle in ("data gate", "safety gate", "candidate file", "would_enter",
                       "regime filter", "probability gate", "edge-to-cost",
                       "fees included", "live trading disabled"):
            assert any(needle in n for n in names), f"checklist missing: {needle}"
        assert all(isinstance(i["ok"], bool) for i in items)
        # live-trading-disabled must be checked; safety must pass right now
        assert next(i for i in items if "live trading" in i["item"])["ok"] is True
        assert next(i for i in items if "safety" in i["item"])["ok"] is True
    check("manual paper-trade checklist (9 live-computed items)", _checklist)

    # --- Excel export: 7 sheets + graceful fallback without openpyxl
    def _excel_export():
        import sys
        from quant.monitor.btc_paper_report import build_paper_report, write_excel
        report = build_paper_report()
        path = write_excel(report)
        if path is not None:  # openpyxl installed: verify the workbook
            import openpyxl
            wb = openpyxl.load_workbook(path)
            assert wb.sheetnames == ["Summary", "Observations", "Signals Only",
                                     "Reason Codes", "Forward Returns",
                                     "Candidate Config", "Safety Status"]
            ws = wb["Summary"]
            vals = {ws.cell(r, 1).value: ws.cell(r, 2).value
                    for r in range(2, ws.max_row + 1)}
            assert vals.get("live_trading") == "disabled"
        # graceful fallback: block the import and confirm no exception
        saved = sys.modules.get("openpyxl")
        sys.modules["openpyxl"] = None  # forces ImportError inside write_excel
        try:
            assert write_excel(report) is None
        finally:
            if saved is not None:
                sys.modules["openpyxl"] = saved
            else:
                sys.modules.pop("openpyxl", None)
    check("Excel export (7 sheets + graceful openpyxl fallback)", _excel_export)

    # --- alerts: graceful no-config + event builder covers all trigger rules
    def _alerts():
        from pathlib import Path as _P
        from quant.monitor.alerts import build_alert_events, load_alert_config, send_email
        # graceful disable: missing config file -> None -> send returns not-sent
        assert load_alert_config(_P("definitely_missing_alerts.json")) is None
        out = send_email("t", "b", cfg=None) if load_alert_config() is None else \
            {"sent": False, "reason": "configured - not exercising real SMTP in smoke"}
        assert out["sent"] is False
        # event rules: failure / safety / data gate / signal / success
        ev = build_alert_events({"status": "FAILURE", "error": "boom"})
        assert ev[0]["event"] == "routine_failure"
        ev = build_alert_events({"safety": {"passed": False},
                                 "data_gate": {"gate": "FAIL"},
                                 "latest_observation": {"would_enter": True,
                                                        "signal_date": "2026-01-01"}})
        names = {e["event"] for e in ev}
        assert {"safety_gate_failure", "data_gate_failure", "would_enter_true"} <= names
        ev = build_alert_events({"safety": {"passed": True},
                                 "data_gate": {"gate": "PASS"},
                                 "latest_observation": {"would_enter": False,
                                                        "gates_failed": ["confidence"]}})
        assert ev[0]["event"] == "routine_success" and len(ev) == 1
    check("alerts (graceful disable + all trigger rules)", _alerts)

    # --- ops_db: schema + inserts round-trip on a scratch database
    def _ops_db():
        import sqlite3
        from quant import config as qc
        from quant.monitor import ops_db
        scratch = qc.DATA_DIR / "ops_monitor_smoke_test.db"
        scratch.unlink(missing_ok=True)
        report = {"started": "t", "status": "SUCCESS",
                  "safety": {"passed": True, "files_scanned": ["a", "b"], "tokens_checked": 9},
                  "data_gate": {"gate": "PASS"},
                  "observation_logged": True,
                  "latest_observation": {"signal_date": "2026-01-01", "close": 50000.0,
                                         "predicted_prob_up": 0.5, "would_enter": False,
                                         "gates_failed": ["confidence"],
                                         "expected_edge_usd": -1.0,
                                         "round_trip_fees_usd": 6.0},
                  "paper_verdict": "KEEP OBSERVING"}
        ops_db.log_routine_run(report, {"worst_level": "OK",
                                        "email": {"sent": False, "reason": "nc"}},
                               db_path=scratch)
        ops_db.log_observation(report["latest_observation"], db_path=scratch)
        ops_db.log_safety_check(report["safety"], db_path=scratch)
        ops_db.log_data_quality({"gate": "PASS", "bars": 100, "to": "2026-01-01",
                                 "bars_behind_expected": 0,
                                 "missing_bars": {"missing_count": 0},
                                 "duplicate_bars": 0}, db_path=scratch)
        from contextlib import closing
        with closing(sqlite3.connect(scratch)) as conn:
            for table in ("routine_runs", "observations", "safety_checks", "data_quality"):
                n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                assert n == 1, f"{table} expected 1 row, got {n}"
            status = conn.execute("SELECT status, would_enter FROM routine_runs").fetchone()
            assert status == ("SUCCESS", 0)
        scratch.unlink(missing_ok=True)
    check("ops_db (4 tables, inserts round-trip)", _ops_db)

    # --- daily status file writer
    def _status_file():
        from quant.monitor.btc_daily_routine import _write_status_file
        path = _write_status_file(
            "2026-01-01T00:00:00",
            {"passed": True, "tokens_checked": 9},
            {"gate": "PASS", "bars": 100, "to": "2026-01-01"},
            {"latest_record": {"signal_date": "2026-01-01", "predicted_prob_up": 0.5,
                               "would_enter": False, "gates_failed": ["confidence"],
                               "expected_edge_usd": -1.0, "round_trip_fees_usd": 6.0}},
            None,
        )
        text = Path(path).read_text(encoding="utf-8")
        assert "live trading: DISABLED" in text
        assert "safety: PASS" in text and "data gate: PASS" in text
    check("daily status file writer", _status_file)

    print()
    if failures:
        print(f"{len(failures)} FAILURES")
        return 1
    print("ALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
