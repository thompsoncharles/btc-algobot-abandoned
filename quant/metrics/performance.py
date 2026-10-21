"""
Performance metrics beyond win rate: CAGR, Sharpe, Sortino, max drawdown,
Calmar, profit factor, expectancy, turnover, alpha/beta, information ratio.
"""
from typing import Optional

import numpy as np
import pandas as pd

from quant.config import TRADING_DAYS_PER_YEAR


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    years = max(len(equity) / TRADING_DAYS_PER_YEAR, 1e-9)
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1


def sharpe(returns: pd.Series, rf_annual: float = 0.0) -> float:
    r = returns.dropna()
    if r.std() == 0 or len(r) < 2:
        return 0.0
    excess = r - rf_annual / TRADING_DAYS_PER_YEAR
    return float(excess.mean() / r.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino(returns: pd.Series, rf_annual: float = 0.0) -> float:
    r = returns.dropna()
    downside = r[r < 0].std()
    if not downside or downside == 0 or np.isnan(downside):
        return 0.0
    excess = r - rf_annual / TRADING_DAYS_PER_YEAR
    return float(excess.mean() / downside * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough decline, as a negative fraction."""
    peak = equity.cummax()
    dd = equity / peak - 1
    return float(dd.min()) if len(dd) else 0.0


def calmar(equity: pd.Series) -> float:
    mdd = abs(max_drawdown(equity))
    return cagr(equity) / mdd if mdd > 0 else 0.0


def profit_factor(trades: pd.DataFrame) -> float:
    if trades is None or trades.empty or "pnl" not in trades:
        return 0.0
    wins = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    losses = abs(trades.loc[trades["pnl"] < 0, "pnl"].sum())
    return float(wins / losses) if losses > 0 else float("inf") if wins > 0 else 0.0


def expectancy(trades: pd.DataFrame) -> float:
    """Average P&L per trade."""
    if trades is None or trades.empty or "pnl" not in trades:
        return 0.0
    return float(trades["pnl"].mean())


def turnover(trades: pd.DataFrame, avg_equity: float) -> float:
    """Total traded notional / average equity (round trips count twice)."""
    if trades is None or trades.empty or avg_equity <= 0:
        return 0.0
    notional = (trades["entry_price"] * trades["shares"]).sum() + \
               (trades["exit_price"] * trades["shares"]).sum()
    return float(notional / avg_equity)


def alpha_beta(returns: pd.Series, benchmark_returns: pd.Series) -> tuple:
    """OLS alpha (annualized) and beta vs a benchmark return series."""
    df = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(df) < 20:
        return 0.0, 0.0
    y, x = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
    beta_val, alpha_val = np.polyfit(x, y, 1)
    return float(alpha_val * TRADING_DAYS_PER_YEAR), float(beta_val)


def information_ratio(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    df = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(df) < 20:
        return 0.0
    active = df.iloc[:, 0] - df.iloc[:, 1]
    te = active.std()
    if te == 0:
        return 0.0
    return float(active.mean() / te * np.sqrt(TRADING_DAYS_PER_YEAR))


def summarize(equity: pd.Series, trades: Optional[pd.DataFrame] = None,
              benchmark_returns: Optional[pd.Series] = None) -> dict:
    """Full scorecard for a backtest or live period."""
    returns = equity.pct_change().dropna()
    out = {
        "start_equity": float(equity.iloc[0]) if len(equity) else 0.0,
        "end_equity": float(equity.iloc[-1]) if len(equity) else 0.0,
        "total_return_pct": float((equity.iloc[-1] / equity.iloc[0] - 1) * 100) if len(equity) > 1 else 0.0,
        "cagr_pct": cagr(equity) * 100,
        "sharpe": sharpe(returns),
        "sortino": sortino(returns),
        "max_drawdown_pct": max_drawdown(equity) * 100,
        "calmar": calmar(equity),
    }
    if trades is not None and not trades.empty:
        # trades["pnl"] is net of commissions (see BacktestEngine._trade_row),
        # so win_rate / profit_factor / expectancy below are all net numbers.
        out.update({
            "n_trades": int(len(trades)),
            "win_rate_pct": float((trades["pnl"] > 0).mean() * 100),
            "profit_factor": profit_factor(trades),
            "expectancy_per_trade": expectancy(trades),         # net $/trade
            "turnover_x": turnover(trades, float(equity.mean())),
        })
        if "commission" in trades.columns:
            out["total_fees_usd"] = float(trades["commission"].sum())
        if "gross_pnl" in trades.columns:
            out["gross_expectancy_per_trade"] = float(trades["gross_pnl"].mean())
            out["gross_win_rate_pct"] = float((trades["gross_pnl"] > 0).mean() * 100)
    if benchmark_returns is not None:
        a, b = alpha_beta(returns, benchmark_returns)
        out.update({
            "alpha_annual_pct": a * 100, "beta": b,
            "information_ratio": information_ratio(returns, benchmark_returns),
        })
    return out
