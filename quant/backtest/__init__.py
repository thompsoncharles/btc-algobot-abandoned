"""Section 8 — Backtesting."""
from quant.backtest.engine import BacktestEngine, BacktestResult
from quant.backtest.validation import walk_forward_splits

__all__ = ["BacktestEngine", "BacktestResult", "walk_forward_splits"]
