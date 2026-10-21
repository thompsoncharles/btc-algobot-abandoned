"""
Validation: walk-forward splits, out-of-sample evaluation, purged splits
with embargo (prevents leakage from overlapping forward-return labels).
"""
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class Split:
    train_idx: np.ndarray
    test_idx: np.ndarray
    label: str


def walk_forward_splits(index: pd.Index, n_folds: int = 5,
                        min_train_frac: float = 0.3,
                        embargo: int = 5) -> List[Split]:
    """
    Anchored walk-forward: train window always starts at 0 and grows;
    each test fold immediately follows its train window, separated by an
    `embargo` gap so overlapping labels can't leak.
    """
    n = len(index)
    first_test_start = int(n * min_train_frac)
    fold_size = (n - first_test_start) // n_folds
    splits: List[Split] = []
    for k in range(n_folds):
        test_start = first_test_start + k * fold_size
        test_end = test_start + fold_size if k < n_folds - 1 else n
        train_end = max(0, test_start - embargo)
        if train_end < 30 or test_end <= test_start:
            continue
        splits.append(Split(
            train_idx=np.arange(0, train_end),
            test_idx=np.arange(test_start, test_end),
            label=f"fold{k + 1}",
        ))
    return splits


def evaluate_walk_forward(model_factory, X: pd.DataFrame, y: pd.Series,
                          n_folds: int = 5, embargo: int = 5,
                          min_train_frac: float = 0.4) -> pd.DataFrame:
    """
    Fit a fresh model per fold, score out-of-sample.

    model_factory: zero-arg callable returning an unfit model with
                   fit/predict_proba. Returns per-fold accuracy, AUC, and
                   hit rate of the >0.5 bucket.

    min_train_frac defaults to 0.4 to match walk_forward_oos_backtest /
    btc-eval, so the classification AUC and the net backtest are scored on
    the SAME splits (they weren't before - AUC silently used 0.3).
    """
    from sklearn.metrics import roc_auc_score

    rows = []
    y_bin = (y > 0).astype(int)
    for split in walk_forward_splits(X.index, n_folds=n_folds,
                                     min_train_frac=min_train_frac, embargo=embargo):
        Xtr, ytr = X.iloc[split.train_idx], y_bin.iloc[split.train_idx]
        Xte, yte = X.iloc[split.test_idx], y_bin.iloc[split.test_idx]
        if ytr.nunique() < 2 or yte.nunique() < 2:
            continue
        model = model_factory()
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)
        pred = (proba > 0.5).astype(int)
        long_bucket = yte[proba > 0.5]
        rows.append({
            "fold": split.label,
            "train_size": len(Xtr), "test_size": len(Xte),
            "accuracy": float((pred == yte).mean()),
            "auc": float(roc_auc_score(yte, proba)),
            "long_hit_rate": float(long_bucket.mean()) if len(long_bucket) else np.nan,
            "n_longs": int((proba > 0.5).sum()),
        })
    return pd.DataFrame(rows)


def walk_forward_oos_backtest(
    bars: pd.DataFrame,
    fit_fn: Callable[[pd.DataFrame], Callable],
    engine_kwargs: dict,
    ticker: str = "BTCUSD",
    n_folds: int = 5,
    min_train_frac: float = 0.4,
    embargo: int = 5,
    warmup: int = 60,
    include_details: bool = False,
) -> dict:
    """
    The real scoreboard: for each walk-forward fold, fit a fresh model on the
    training slice only, then backtest on the out-of-sample slice with real
    fees. Warmup bars before each test window feed the signal's lookback but
    can't be traded (engine trade_start), so there's no leakage.

    fit_fn(train_bars) -> signal_fn(day, history) closure holding the fold's
    trained model. engine_kwargs are passed to BacktestEngine (fees, sizing,
    stops); initial_equity/max_positions/etc. come from there.

    Returns per-fold net results plus a pooled aggregate. In-sample numbers
    are deliberately not produced here - only OOS.
    """
    from quant.backtest.engine import BacktestEngine
    from quant.metrics.performance import max_drawdown, profit_factor

    bars = bars.sort_index()
    splits = walk_forward_splits(bars.index, n_folds=n_folds,
                                 min_train_frac=min_train_frac, embargo=embargo)
    fold_rows: List[dict] = []
    all_trades: List[pd.DataFrame] = []
    equity_rows: List[pd.DataFrame] = []

    for split in splits:
        train_end = int(split.train_idx[-1]) + 1
        train_bars = bars.iloc[:train_end]
        test_start_pos = int(split.test_idx[0])
        test_end_pos = int(split.test_idx[-1])
        warmup_start = max(0, test_start_pos - warmup)
        sub_bars = bars.iloc[warmup_start:test_end_pos + 1]
        trade_start = bars.index[test_start_pos]

        try:
            signal_fn = fit_fn(train_bars)
        except Exception as e:
            logger.warning("fold {} model fit failed: {}", split.label, e)
            continue

        engine = BacktestEngine({ticker: sub_bars}, trade_start=trade_start,
                                **engine_kwargs)
        result = engine.run(signal_fn)
        eq = result.equity_curve
        trades = result.trades if result.trades is not None else pd.DataFrame()
        if not trades.empty:
            trades = trades.copy()
            trades["fold"] = split.label
            all_trades.append(trades)
        if include_details and eq is not None and len(eq):
            equity_rows.append(pd.DataFrame({
                "fold": split.label,
                "date": eq.index,
                "equity": eq.values,
            }))
        fold_rows.append({
            "fold": split.label,
            "train_bars": len(train_bars),
            "oos_bars": len(split.test_idx),
            "n_trades": int(len(trades)),
            "net_return_pct": float((eq.iloc[-1] / eq.iloc[0] - 1) * 100) if len(eq) > 1 else 0.0,
            "max_drawdown_pct": float(max_drawdown(eq) * 100) if len(eq) else 0.0,
            "net_expectancy_usd": float(trades["pnl"].mean()) if not trades.empty else 0.0,
        })

    folds_df = pd.DataFrame(fold_rows)
    pooled = pd.concat(all_trades) if all_trades else pd.DataFrame()
    agg = {
        "n_folds": len(fold_rows),
        "total_oos_trades": int(len(pooled)),
        "mean_fold_net_return_pct": float(folds_df["net_return_pct"].mean()) if not folds_df.empty else 0.0,
        "median_fold_net_return_pct": float(folds_df["net_return_pct"].median()) if not folds_df.empty else 0.0,
        "worst_fold_drawdown_pct": float(folds_df["max_drawdown_pct"].min()) if not folds_df.empty else 0.0,
        "pooled_net_expectancy_usd": float(pooled["pnl"].mean()) if not pooled.empty else 0.0,
        "pooled_net_win_rate_pct": float((pooled["pnl"] > 0).mean() * 100) if not pooled.empty else 0.0,
        "pooled_profit_factor": profit_factor(pooled) if not pooled.empty else 0.0,
        "pooled_total_fees_usd": float(pooled["commission"].sum()) if "commission" in pooled.columns else 0.0,
    }
    out = {"folds": fold_rows, "aggregate": agg}
    if include_details:
        out["trades"] = (pd.concat(all_trades).to_dict("records")
                         if all_trades else [])
        out["equity"] = (pd.concat(equity_rows).to_dict("records")
                         if equity_rows else [])
    return out
