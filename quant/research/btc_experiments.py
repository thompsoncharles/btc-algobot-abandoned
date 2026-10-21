"""
BTC AR/modeling-path OOS experiment harness.

Answers the review's questions WITHOUT changing trading architecture:
  1. AR lag-0 alignment (include just-completed return) A/B
  2. feature ablation: AR only / +momentum / +trend / +volatility / +flow / full
  3. logistic C sweep [0.01, 0.03, 0.1, 0.3, 1.0]
  4. probability diagnostics: calibration buckets, hit rate, avg fwd return,
     net expectancy by prob bucket
  5. AUC on the EXACT same walk-forward splits as btc-eval
  6. exposure (time in market) so partial-deployment is visible
  7. fold stability, so a lucky single fold can't win

Every config is scored with the identical splits, filters, fee model, and
sizing that `python -m quant.pipeline btc-eval` uses - only the model
(feature set / C / lag0) varies. Run:

    python -m quant.research.btc_experiments            # 12h seed data
    python -m quant.research.btc_experiments --daily    # + daily BTC-USD robustness
"""
import argparse
import itertools
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from quant import config
from quant.backtest.validation import walk_forward_splits

FEATURE_SETS = {
    "ar_only": set(),
    "ar+momentum": {"momentum"},
    "ar+trend": {"trend"},
    "ar+volatility": {"volatility"},
    "ar+flow": {"flow"},
    "ar+external": {"external"},
    "full": None,   # all groups
    "full+external": {"momentum", "trend", "volatility", "flow", "external"},
}
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0]


def _make_model_factory(groups, C, lag0):
    from quant.models.linear_ar import LinearBTCModel

    def factory():
        return LinearBTCModel(
            n_lags=config.BTC_AR_LAGS,
            forecast_horizon=config.BTC_FORECAST_HORIZON_BARS,
            ma_window=config.BTC_MA_WINDOW,
            C=C, include_lag0=lag0, groups=groups,
        )
    return factory


def run_config(bars: pd.DataFrame, groups, C: float, lag0: bool,
               splits, notional: float, engine_kwargs: dict,
               warmup: int = 252, horizon: int = 1) -> dict:
    """One config over all folds: net + AUC + calibration + exposure, OOS only."""
    from quant.backtest.engine import BacktestEngine
    from quant.features.btc_features import forward_return_target
    from quant.metrics.performance import max_drawdown, profit_factor
    from quant.pipeline import BTC_TICKER, _btc_signal_fn

    factory = _make_model_factory(groups, C, lag0)
    yfwd = forward_return_target(bars["Close"].astype(float), horizon)

    fold_returns, fold_dds, all_trades = [], [], []
    pooled_prob, pooled_fwd = [], []
    trade_bucket = []  # (entry_prob, net_pnl)
    total_oos_bars, bars_in_trade = 0, 0

    for split in splits:
        train_end = int(split.train_idx[-1]) + 1
        model = factory().fit_from_bars(bars.iloc[:train_end])

        ts0, ts1 = int(split.test_idx[0]), int(split.test_idx[-1])
        sub = bars.iloc[max(0, ts0 - warmup):ts1 + 1]
        signal_fn = _btc_signal_fn(model, notional)
        res = BacktestEngine({BTC_TICKER: sub}, trade_start=bars.index[ts0],
                             **engine_kwargs).run(signal_fn)
        eq = res.equity_curve
        tr = res.trades if res.trades is not None else pd.DataFrame()
        fold_returns.append(float((eq.iloc[-1] / eq.iloc[0] - 1) * 100) if len(eq) > 1 else 0.0)
        fold_dds.append(float(max_drawdown(eq) * 100) if len(eq) else 0.0)
        if not tr.empty:
            all_trades.append(tr)

        # OOS predictions at test timestamps (causal features on full bars)
        Xfull = model.build_features(bars)
        test_ts = bars.index[split.test_idx]
        rows = Xfull.reindex(test_ts).dropna()
        valid = rows.index.intersection(yfwd.dropna().index)
        if len(valid):
            p = model.predict_proba(rows.loc[valid])
            pooled_prob.extend(map(float, p))
            pooled_fwd.extend(map(float, yfwd.loc[valid].values))

        # exposure + per-trade entry prob (signal is the bar BEFORE entry)
        total_oos_bars += len(split.test_idx)
        for _, row in tr.iterrows():
            ei = bars.index.get_indexer([row["entry_date"]])[0]
            xi = bars.index.get_indexer([row["exit_date"]])[0]
            if ei >= 0 and xi >= 0:
                bars_in_trade += max(1, xi - ei)
            if ei - 1 >= 0:
                ts = bars.index[ei - 1]
                if ts in Xfull.index:
                    fr = Xfull.loc[[ts]].dropna()
                    if not fr.empty:
                        trade_bucket.append((float(model.predict_proba(fr)[0]),
                                             float(row["pnl"])))

    pooled = pd.concat(all_trades) if all_trades else pd.DataFrame()
    fr = np.array(fold_returns, dtype=float)
    auc = _safe_auc(pooled_prob, pooled_fwd)
    return {
        "features": _feat_label(groups), "C": C, "lag0": lag0,
        "n_trades": int(len(pooled)),
        "pooled_net_exp_usd": float(pooled["pnl"].mean()) if not pooled.empty else 0.0,
        "pooled_win_pct": float((pooled["pnl"] > 0).mean() * 100) if not pooled.empty else 0.0,
        "pooled_pf": float(profit_factor(pooled)) if not pooled.empty else 0.0,
        "mean_fold_ret_pct": float(fr.mean()) if len(fr) else 0.0,
        "std_fold_ret_pct": float(fr.std()) if len(fr) else 0.0,
        "min_fold_ret_pct": float(fr.min()) if len(fr) else 0.0,
        "frac_folds_positive": float((fr > 0).mean()) if len(fr) else 0.0,
        "worst_dd_pct": float(min(fold_dds)) if fold_dds else 0.0,
        "auc": auc,
        "time_in_market_pct": float(bars_in_trade / total_oos_bars * 100) if total_oos_bars else 0.0,
        "_pooled_prob": pooled_prob, "_pooled_fwd": pooled_fwd,
        "_trade_bucket": trade_bucket,
    }


def _safe_auc(probs, fwd) -> float:
    from sklearn.metrics import roc_auc_score
    y = (np.array(fwd) > 0).astype(int)
    if len(y) < 10 or y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, np.array(probs)))


def _feat_label(groups) -> str:
    if groups is None:
        return "full"
    if not groups:
        return "ar_only"
    return "ar+" + "+".join(sorted(groups))


def calibration_table(probs, fwd, edges=(0.5, 0.55, 0.6, 0.65, 0.7, 1.01)) -> pd.DataFrame:
    p = np.array(probs)
    r = np.array(fwd)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        rows.append({
            "prob_bucket": f"[{lo:.2f},{hi:.2f})",
            "n": int(m.sum()),
            "hit_rate_pct": float((r[m] > 0).mean() * 100),
            "avg_fwd_ret_pct": float(r[m].mean() * 100),
        })
    return pd.DataFrame(rows)


def trade_bucket_table(trade_bucket, edges=(0.5, 0.58, 0.62, 0.66, 0.70, 1.01)) -> pd.DataFrame:
    if not trade_bucket:
        return pd.DataFrame()
    p = np.array([t[0] for t in trade_bucket])
    pnl = np.array([t[1] for t in trade_bucket])
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        rows.append({
            "entry_prob_bucket": f"[{lo:.2f},{hi:.2f})",
            "n_trades": int(m.sum()),
            "net_exp_usd": float(pnl[m].mean()),
            "win_pct": float((pnl[m] > 0).mean() * 100),
        })
    return pd.DataFrame(rows)


def _engine_kwargs_and_notional():
    from quant.pipeline import _btc_engine_kwargs, _btc_position_notional
    notional = _btc_position_notional()
    return _btc_engine_kwargs(notional), notional


def run_grid(bars: pd.DataFrame, splits, label: str) -> pd.DataFrame:
    engine_kwargs, notional = _engine_kwargs_and_notional()
    results = []
    combos = list(itertools.product(FEATURE_SETS.items(), C_GRID, [False, True]))
    logger.info("[{}] running {} configs x {} folds ...", label, len(combos), len(splits))
    for (fname, groups), C, lag0 in combos:
        res = run_config(bars, groups, C, lag0, splits, notional, engine_kwargs)
        results.append(res)
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in results])
    df.attrs["raw"] = results
    return df


def _load_daily_btc() -> Optional[pd.DataFrame]:
    import yfinance as yf
    try:
        df = yf.download("BTC-USD", period="max", interval="1d",
                         auto_adjust=True, progress=False)
    except Exception as e:
        logger.warning("yfinance BTC-USD fetch failed: {}", e)
        return None
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.capitalize)[["Open", "High", "Low", "Close", "Volume"]]
    df.index.name = "datetime"
    return df.dropna()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily", action="store_true",
                    help="also run the winning config on multi-year daily BTC-USD (yfinance)")
    args = ap.parse_args()

    from quant.data.store import DataStore
    bars = DataStore().load_bars("BTCUSD", config.BTC_INTERVAL)
    if bars is None:
        raise SystemExit("no BTC bars - run `python -m quant.pipeline btc-ingest` first")
    splits = walk_forward_splits(bars.index, n_folds=5, min_train_frac=0.4, embargo=5)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    df = run_grid(bars, splits, "12h seed")

    baseline_mask = (df["features"] == "full") & (df["C"] == 1.0) & (~df["lag0"])
    baseline = df[baseline_mask].iloc[0]

    print("\n" + "=" * 90)
    print("CURRENT btc-eval CONFIG (full, C=1.0, lag0=False):")
    print(baseline[["n_trades", "pooled_net_exp_usd", "mean_fold_ret_pct",
                    "std_fold_ret_pct", "min_fold_ret_pct", "frac_folds_positive",
                    "auc", "time_in_market_pct"]].to_string())

    print("\n" + "=" * 90)
    print("TASK 1 - AR lag-0 A/B (same feature set + C, lag0 False vs True):")
    ab = df[(df["C"] == 1.0)].pivot_table(index="features", columns="lag0",
                                          values="pooled_net_exp_usd")
    print(ab.to_string())

    print("\n" + "=" * 90)
    print("TASK 2 - feature ablation (C=1.0, best lag0 per row):")
    abl = (df[df["C"] == 1.0]
           .sort_values("pooled_net_exp_usd", ascending=False)
           .groupby("features").first()
           [["C", "lag0", "n_trades", "pooled_net_exp_usd", "mean_fold_ret_pct",
             "frac_folds_positive", "auc", "time_in_market_pct"]])
    print(abl.sort_values("pooled_net_exp_usd", ascending=False).to_string())

    print("\n" + "=" * 90)
    print("TASK 3 - logistic C sweep (feature=full, lag0=False):")
    csweep = (df[(df["features"] == "full") & (~df["lag0"])]
              .sort_values("C")[["C", "n_trades", "pooled_net_exp_usd",
                                  "mean_fold_ret_pct", "frac_folds_positive", "auc"]])
    print(csweep.to_string(index=False))

    print("\n" + "=" * 90)
    print("STABILITY-GATED RANKING (net exp>0, >=60% folds positive, >=8 trades):")
    gated = df[(df["pooled_net_exp_usd"] > 0) & (df["frac_folds_positive"] >= 0.6)
               & (df["n_trades"] >= 8)].sort_values("pooled_net_exp_usd", ascending=False)
    cols = ["features", "C", "lag0", "n_trades", "pooled_net_exp_usd",
            "mean_fold_ret_pct", "min_fold_ret_pct", "frac_folds_positive", "auc"]
    print((gated[cols].head(12).to_string(index=False)) if not gated.empty
          else "  *** NO config clears the stability gate ***")

    # Task 4 - diagnostics for baseline
    raw = df.attrs["raw"]
    base_raw = next(r for r in raw if r["features"] == "full" and r["C"] == 1.0 and not r["lag0"])
    print("\n" + "=" * 90)
    print("TASK 4 - probability calibration, baseline (all OOS predictions):")
    print(calibration_table(base_raw["_pooled_prob"], base_raw["_pooled_fwd"]).to_string(index=False))
    print("\nnet expectancy by entry-prob bucket (executed trades):")
    tb = trade_bucket_table(base_raw["_trade_bucket"])
    print(tb.to_string(index=False) if not tb.empty else "  (no trades)")

    # Task 6 - exposure note
    print("\n" + "=" * 90)
    print("TASK 6 - exposure: baseline time_in_market = "
          f"{baseline['time_in_market_pct']:.1f}% of OOS bars; position deploys "
          f"{config.BTC_MAX_POSITION_PCT:.0f}%-cap capital. Buy-hold = 100% time, 100% capital.")

    # Robustness on daily data for the gated winner (Task 7)
    if args.daily and not gated.empty:
        win = gated.iloc[0]
        print("\n" + "=" * 90)
        print(f"ROBUSTNESS - top gated config on DAILY BTC-USD (multi-regime): "
              f"{win['features']} C={win['C']} lag0={win['lag0']}")
        daily = _load_daily_btc()
        if daily is None or len(daily) < 400:
            print("  daily data unavailable - cannot verify regime robustness")
        else:
            dsplits = walk_forward_splits(daily.index, n_folds=5, min_train_frac=0.4, embargo=5)
            groups = FEATURE_SETS[win["features"]] if win["features"] in FEATURE_SETS else None
            ek, notl = _engine_kwargs_and_notional()
            dres = run_config(daily, groups, float(win["C"]), bool(win["lag0"]),
                              dsplits, notl, ek)
            print(f"  daily span {daily.index.min().date()}..{daily.index.max().date()} "
                  f"({len(daily)} bars, includes bull+bear)")
            print(f"  net_exp=${dres['pooled_net_exp_usd']:.2f}  n_trades={dres['n_trades']}  "
                  f"mean_fold_ret={dres['mean_fold_ret_pct']:.2f}%  "
                  f"frac_folds_pos={dres['frac_folds_positive']:.2f}  auc={dres['auc']:.3f}")


if __name__ == "__main__":
    main()
