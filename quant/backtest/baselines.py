"""
Dumb baselines every model must beat before it earns its complexity:

  * buy_and_hold  - own BTC the whole window (one round-trip fee)
  * ma_trend      - long while close > SMA(n), flat otherwise
  * never_trade   - stay in cash (the "can you beat doing nothing" bar)

Standalone net-return simulators (not the event engine) - an independent
second implementation of the cost math, which also cross-checks the engine.
All use the same IBKR spot-crypto fee model and mark-to-market on close.
"""
from typing import Callable, Optional

import numpy as np
import pandas as pd

from quant.execution.fees import ibkr_crypto_fee
from quant.metrics.performance import max_drawdown, sharpe


def _summarize(equity: pd.Series, n_trades: int, fees: float, label: str) -> dict:
    rets = equity.pct_change().dropna()
    return {
        "strategy": label,
        "net_return_pct": float((equity.iloc[-1] / equity.iloc[0] - 1) * 100) if len(equity) > 1 else 0.0,
        "max_drawdown_pct": float(max_drawdown(equity) * 100) if len(equity) else 0.0,
        "sharpe": float(sharpe(rets)),
        "n_trades": int(n_trades),
        "total_fees_usd": float(fees),
    }


def buy_and_hold(bars: pd.DataFrame, equity: float,
                 fee_fn: Callable[[float], float] = ibkr_crypto_fee) -> dict:
    close = bars["Close"].astype(float)
    entry_fee = fee_fn(equity)
    shares = (equity - entry_fee) / close.iloc[0]
    curve = shares * close
    # exit fee applied to the final mark
    exit_fee = fee_fn(float(curve.iloc[-1]))
    curve = curve.copy()
    curve.iloc[-1] -= exit_fee
    curve = pd.concat([pd.Series([equity], index=[bars.index[0]]), curve.iloc[1:]])
    return _summarize(curve, n_trades=1, fees=entry_fee + exit_fee, label="buy_and_hold")


def ma_trend(bars: pd.DataFrame, equity: float, window: int = 20,
             fee_fn: Callable[[float], float] = ibkr_crypto_fee) -> dict:
    """Long the whole position while close > SMA(window); flat otherwise.
    Trades next bar's open after a signal flip (no same-bar look-ahead)."""
    close = bars["Close"].astype(float)
    sma = close.rolling(window).mean()
    want_long = (close > sma).shift(1).fillna(False)  # act on prior close's signal

    cash, shares, fees, n_trades = equity, 0.0, 0.0, 0
    curve = []
    for t in bars.index:
        price = float(close.loc[t])
        target_long = bool(want_long.loc[t])
        if target_long and shares == 0:            # enter
            fee = fee_fn(cash)
            shares = (cash - fee) / price
            cash, fees, n_trades = 0.0, fees + fee, n_trades + 1
        elif not target_long and shares > 0:       # exit
            proceeds = shares * price
            fee = fee_fn(proceeds)
            cash, shares = proceeds - fee, 0.0
            fees, n_trades = fees + fee, n_trades + 1
        curve.append(cash + shares * price)
    return _summarize(pd.Series(curve, index=bars.index), n_trades, fees, f"ma_trend_{window}")


def never_trade(bars: pd.DataFrame, equity: float) -> dict:
    curve = pd.Series(equity, index=bars.index)
    return _summarize(curve, n_trades=0, fees=0.0, label="never_trade")


def run_all(bars: pd.DataFrame, equity: float, ma_window: int = 20) -> list:
    return [
        buy_and_hold(bars, equity),
        ma_trend(bars, equity, window=ma_window),
        never_trade(bars, equity),
    ]
