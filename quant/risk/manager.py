"""
Risk management: daily loss limits, max drawdown, correlation limits,
sector/factor exposure, circuit breakers.

House rule #8: every position must always carry a stop (the live bot's
executor places them); RiskManager.check_* functions are the pre-trade
gate and the kill switches. State persists to quant_data/risk_state.json
so limits survive process restarts.
"""
import json
from dataclasses import dataclass, asdict, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from quant import config


@dataclass
class RiskState:
    trading_date: str = ""
    day_start_equity: float = 0.0
    peak_equity: float = 0.0
    halted: bool = False
    halt_reason: str = ""
    realized_pnl_today: float = 0.0
    sector_exposure: Dict[str, float] = field(default_factory=dict)


class RiskManager:
    def __init__(self, state_path: Optional[Path] = None):
        self.state_path = state_path or (config.DATA_DIR / "risk_state.json")
        self.state = self._load()

    # -------------------------------------------------------- state io
    def _load(self) -> RiskState:
        if self.state_path.exists():
            try:
                return RiskState(**json.loads(self.state_path.read_text()))
            except Exception as e:
                logger.warning("risk state unreadable ({}); starting fresh", e)
        return RiskState()

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self.state), indent=2))
        tmp.replace(self.state_path)

    def new_session(self, equity: float) -> None:
        """Call once at the start of each trading day."""
        today = date.today().isoformat()
        if self.state.trading_date != today:
            self.state = RiskState(
                trading_date=today,
                day_start_equity=equity,
                peak_equity=max(equity, self.state.peak_equity),
            )
            self._save()

    # ----------------------------------------------------- kill switches
    def update_equity(self, equity: float) -> Tuple[bool, str]:
        """Feed current equity; returns (halted, reason)."""
        s = self.state
        s.peak_equity = max(s.peak_equity, equity)

        if s.day_start_equity > 0:
            day_loss_pct = (s.day_start_equity - equity) / s.day_start_equity * 100
            if day_loss_pct >= config.DAILY_LOSS_LIMIT_PCT:
                self._halt(f"daily loss limit hit ({day_loss_pct:.2f}%)")
        if s.peak_equity > 0:
            dd_pct = (s.peak_equity - equity) / s.peak_equity * 100
            if dd_pct >= config.MAX_DRAWDOWN_PCT:
                self._halt(f"max drawdown hit ({dd_pct:.2f}%)")
        self._save()
        return s.halted, s.halt_reason

    def circuit_breaker(self, vix_level: Optional[float] = None,
                        index_day_change_pct: Optional[float] = None,
                        data_stale: bool = False) -> Tuple[bool, str]:
        """Abnormal-conditions halt: VIX spike, index crash, stale data."""
        if vix_level is not None and vix_level >= config.CIRCUIT_VIX_LEVEL:
            self._halt(f"circuit breaker: VIX {vix_level:.1f}")
        if index_day_change_pct is not None and \
                index_day_change_pct <= -config.CIRCUIT_INDEX_DROP_PCT:
            self._halt(f"circuit breaker: index {index_day_change_pct:.2f}% on the day")
        if data_stale:
            self._halt("circuit breaker: market data stale")
        return self.state.halted, self.state.halt_reason

    def _halt(self, reason: str) -> None:
        if not self.state.halted:
            logger.error("TRADING HALTED - {}", reason)
        self.state.halted = True
        self.state.halt_reason = reason
        self._save()

    def resume(self) -> None:
        """Manual reset only - a human decides trading restarts."""
        self.state.halted = False
        self.state.halt_reason = ""
        self._save()
        logger.warning("risk halt manually cleared")

    # ------------------------------------------------------ pre-trade
    def check_new_position(self, ticker: str, dollars: float, equity: float,
                           sector: Optional[str] = None,
                           candidate_returns: Optional[pd.Series] = None,
                           book_returns: Optional[pd.DataFrame] = None) -> Tuple[bool, str]:
        """Gate for opening a position. Returns (allowed, reason_if_not)."""
        if self.state.halted:
            return False, f"halted: {self.state.halt_reason}"

        if dollars > equity * config.MAX_POSITION_PCT / 100:
            return False, "exceeds single-position cap"

        if sector:
            exposure = self.state.sector_exposure.get(sector, 0.0) + dollars
            if exposure > equity * config.MAX_SECTOR_PCT / 100:
                return False, f"sector cap ({sector})"

        if candidate_returns is not None and book_returns is not None and not book_returns.empty:
            aligned = book_returns.align(candidate_returns, axis=0, join="inner")[0]
            if len(aligned) >= 20:
                corr = aligned.corrwith(candidate_returns.loc[aligned.index]).abs()
                worst = corr.max()
                if worst > config.MAX_PAIRWISE_CORR:
                    return False, f"correlation {worst:.2f} with {corr.idxmax()}"
        return True, ""

    def register_fill(self, sector: Optional[str], dollars: float) -> None:
        if sector:
            self.state.sector_exposure[sector] = \
                self.state.sector_exposure.get(sector, 0.0) + dollars
            self._save()

    def register_close(self, sector: Optional[str], dollars: float, pnl: float) -> None:
        if sector and sector in self.state.sector_exposure:
            self.state.sector_exposure[sector] = \
                max(0.0, self.state.sector_exposure[sector] - dollars)
        self.state.realized_pnl_today += pnl
        self._save()
