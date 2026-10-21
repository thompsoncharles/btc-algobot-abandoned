"""
Event-driven backtest over daily (or intraday) bars with realistic costs.

Realism handled here:
  * commissions (per-share with minimum, IBKR-style)
  * slippage (bps of fill price, applied against you)
  * next-bar-open fills (signals computed on bar close can't fill at that close)
  * delisting/missing data: a ticker whose bars end mid-test is force-closed
    at its last print (approximates delisting; survivorship caveat logged)

Signals come from a caller-provided function so the same engine can test
model output, Godel prob_up thresholds, or blended signals.

Cost model defaults to IBKR-style per-share commission (equities). Pass
commission_pct for percentage-of-notional venues (e.g. Binance spot taker
fee ~0.045%, used for the BTCUSD path) - see quant.pipeline's btc commands.
Hold-time is tracked by bar count, not calendar days, so it means the same
thing whether bars are daily (equities) or 12h (crypto).
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import pandas as pd
from loguru import logger

from quant import config


@dataclass
class Position:
    ticker: str
    shares: float
    entry_price: float
    entry_date: pd.Timestamp
    entry_index: int
    stop_price: float
    target_price: float
    entry_commission: float = 0.0


@dataclass
class BacktestResult:
    equity_curve: pd.Series = None
    trades: pd.DataFrame = None
    summary: dict = field(default_factory=dict)


class BacktestEngine:
    """
    signal_fn(date, history_by_ticker) -> {ticker: prob_up} for names to
    consider that day; history frames end at `date` (no look-ahead - the
    engine slices them, the signal function can't peek).
    """

    def __init__(self, bars_by_ticker: Dict[str, pd.DataFrame],
                 initial_equity: float = 100_000.0,
                 stop_pct: float = 2.0, target_pct: float = 4.0,
                 max_positions: int = 5, prob_entry: float = 0.55,
                 max_hold_bars: int = 10,
                 commission_pct: Optional[float] = None,
                 commission_fn: Optional[Callable[[float], float]] = None,
                 slippage_bps: Optional[float] = None,
                 allow_fractional: bool = False,
                 position_notional: Optional[float] = None,
                 min_trade_notional: float = 0.0,
                 trade_start: Optional[pd.Timestamp] = None):
        """
        commission_fn: notional -> fee in dollars; takes precedence over
            commission_pct. Use for schedules with minimums/caps (e.g. IBKR
            spot crypto's max($1.75, 0.18%) capped at 1%).
        position_notional: fixed target dollar size per position (risk-based
            sizing computed by the caller); default is equal-split of cash.
        min_trade_notional: skip any entry whose size would fall below this -
            prevents fee-floor-dominated trades on small accounts.
        trade_start: if set, no NEW positions open before this timestamp -
            earlier bars still feed the signal function's lookback window.
            This is what makes out-of-sample walk-forward backtests honest:
            warmup bars build features, but trades only fire in the OOS window.
        """
        self.bars = {t: df.sort_index() for t, df in bars_by_ticker.items() if not df.empty}
        self.initial_equity = initial_equity
        self.stop_pct = stop_pct
        self.target_pct = target_pct
        self.max_positions = max_positions
        self.prob_entry = prob_entry
        self.max_hold_bars = max_hold_bars
        self.commission_pct = commission_pct
        self.commission_fn = commission_fn
        self.slippage_bps = config.SLIPPAGE_BPS if slippage_bps is None else slippage_bps
        self.allow_fractional = allow_fractional
        self.position_notional = position_notional
        self.min_trade_notional = min_trade_notional
        self.trade_start = trade_start

    def _commission(self, shares: float, price: float) -> float:
        if self.commission_fn is not None:
            return self.commission_fn(shares * price)
        if self.commission_pct is not None:
            return shares * price * self.commission_pct / 100
        return max(config.COMMISSION_MIN, shares * config.COMMISSION_PER_SHARE)

    def _slip(self, price: float, side: str) -> float:
        slip = price * self.slippage_bps / 10_000
        return price + slip if side == "BUY" else price - slip

    def run(self, signal_fn: Callable[[pd.Timestamp, Dict[str, pd.DataFrame]], Dict[str, float]],
            ) -> BacktestResult:
        all_dates = sorted({d for df in self.bars.values() for d in df.index})
        cash = self.initial_equity
        positions: Dict[str, Position] = {}
        trades: List[dict] = []
        equity_hist: Dict[pd.Timestamp, float] = {}
        pending_entries: Dict[str, float] = {}  # ticker -> prob (fill next open)

        for i, day in enumerate(all_dates):
            # ---- 1. fill pending entries at today's open
            for ticker, prob in list(pending_entries.items()):
                df = self.bars.get(ticker)
                if df is None or day not in df.index or ticker in positions:
                    continue
                if len(positions) >= self.max_positions:
                    break
                open_px = float(df.loc[day, "Open"])
                fill = self._slip(open_px, "BUY")
                budget = (cash / (self.max_positions - len(positions))) * 0.95
                if self.position_notional is not None:
                    budget = min(budget, self.position_notional)
                if budget < self.min_trade_notional:
                    continue  # fee floor would dominate a trade this small
                shares = budget / fill if self.allow_fractional else int(budget / fill)
                if shares <= 0:
                    continue
                entry_comm = self._commission(shares, fill)
                cost = shares * fill + entry_comm
                if cost > cash:
                    continue
                cash -= cost
                positions[ticker] = Position(
                    ticker=ticker, shares=shares, entry_price=fill, entry_date=day,
                    entry_index=i,
                    stop_price=fill * (1 - self.stop_pct / 100),
                    target_price=fill * (1 + self.target_pct / 100),
                    entry_commission=entry_comm,
                )
            pending_entries.clear()

            # ---- 2. manage open positions on today's bar
            for ticker, pos in list(positions.items()):
                df = self.bars.get(ticker)
                if df is None or day not in df.index:
                    # missing bar; if the series has ended, treat as delisted
                    if df is not None and df.index[-1] < day:
                        last_px = float(df["Close"].iloc[-1])
                        fill = self._slip(last_px, "SELL")
                        exit_comm = self._commission(pos.shares, fill)
                        cash += pos.shares * fill - exit_comm
                        trades.append(self._trade_row(pos, day, fill, "delisted", exit_comm))
                        del positions[ticker]
                        logger.warning("{} disappeared mid-test; closed at last print "
                                       "(delisting approximation)", ticker)
                    continue
                row = df.loc[day]
                exit_px, reason = None, None
                if float(row["Low"]) <= pos.stop_price:
                    exit_px, reason = pos.stop_price, "stop"
                elif float(row["High"]) >= pos.target_price:
                    exit_px, reason = pos.target_price, "target"
                elif (i - pos.entry_index) >= self.max_hold_bars or \
                        i == len(all_dates) - 1:
                    exit_px, reason = float(row["Close"]), "time"
                if exit_px is not None:
                    fill = self._slip(exit_px, "SELL")
                    exit_comm = self._commission(pos.shares, fill)
                    cash += pos.shares * fill - exit_comm
                    trades.append(self._trade_row(pos, day, fill, reason, exit_comm))
                    del positions[ticker]

            # ---- 3. generate signals on today's close for tomorrow's open
            #        (no new entries before trade_start - warmup bars only feed
            #         the signal function's lookback, keeping OOS folds honest)
            next_day = all_dates[i + 1] if i < len(all_dates) - 1 else None
            entries_open = self.trade_start is None or (
                next_day is not None and next_day >= self.trade_start)
            if len(positions) < self.max_positions and i < len(all_dates) - 1 and entries_open:
                history = {t: df.loc[:day] for t, df in self.bars.items()
                           if day in df.index and len(df.loc[:day]) >= 30}
                try:
                    signals = signal_fn(day, history) or {}
                except Exception as e:
                    logger.error("signal_fn failed on {}: {}", day, e)
                    signals = {}
                ranked = sorted(signals.items(), key=lambda kv: kv[1], reverse=True)
                for ticker, prob in ranked:
                    if prob >= self.prob_entry and ticker not in positions:
                        pending_entries[ticker] = prob
                    if len(pending_entries) + len(positions) >= self.max_positions:
                        break

            # ---- 4. mark equity
            mtm = sum(
                pos.shares * float(self.bars[t].loc[day, "Close"])
                for t, pos in positions.items()
                if day in self.bars[t].index
            )
            equity_hist[day] = cash + mtm

        equity_curve = pd.Series(equity_hist).sort_index()
        trades_df = pd.DataFrame(trades)
        from quant.metrics.performance import summarize
        summary = summarize(equity_curve, trades_df)
        return BacktestResult(equity_curve=equity_curve, trades=trades_df, summary=summary)

    @staticmethod
    def _trade_row(pos: Position, exit_date, exit_price: float, reason: str,
                   exit_commission: float = 0.0) -> dict:
        gross_pnl = (exit_price - pos.entry_price) * pos.shares
        commission = pos.entry_commission + exit_commission
        net_pnl = gross_pnl - commission
        notional = pos.entry_price * pos.shares
        # `pnl` is NET (fees deducted) so all downstream stats - win_rate,
        # profit_factor, expectancy - are net, matching the equity curve.
        return {
            "ticker": pos.ticker, "shares": pos.shares,
            "entry_date": pos.entry_date, "entry_price": pos.entry_price,
            "exit_date": exit_date, "exit_price": exit_price,
            "gross_pnl": gross_pnl, "commission": commission, "pnl": net_pnl,
            "return_pct": (net_pnl / notional * 100) if notional else 0.0,
            "gross_return_pct": (exit_price / pos.entry_price - 1) * 100,
            "exit_reason": reason,
        }
