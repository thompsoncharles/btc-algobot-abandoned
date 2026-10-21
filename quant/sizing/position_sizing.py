"""
Position sizing: fixed fractional, volatility targeting, fractional Kelly,
with portfolio-level exposure caps (single-name, sector, gross).

Complements strategy/position_sizing.py (the live bot's simple sizer);
this one is used by the quant pipeline and backtests.
"""
import math
from dataclasses import dataclass
from typing import Dict, Optional

from loguru import logger

from quant import config


@dataclass
class SizeDecision:
    shares: int
    dollars: float
    method: str
    capped_by: Optional[str] = None


class QuantPositionSizer:
    def __init__(self, equity: float):
        self.equity = float(equity)

    # ------------------------------------------------------- methods
    def fixed_fractional(self, price: float, risk_pct: float,
                         stop_pct: float) -> SizeDecision:
        """Risk risk_pct of equity; stop at stop_pct below entry."""
        risk_dollars = self.equity * risk_pct / 100
        per_share_risk = price * abs(stop_pct) / 100
        shares = int(risk_dollars / per_share_risk) if per_share_risk > 0 else 0
        return SizeDecision(shares, shares * price, "fixed_fractional")

    def volatility_target(self, price: float, annual_vol: float,
                          target_vol: Optional[float] = None) -> SizeDecision:
        """Scale position so its vol contribution matches the target."""
        tv = target_vol if target_vol is not None else config.VOL_TARGET_ANNUAL
        if annual_vol <= 0:
            return SizeDecision(0, 0.0, "volatility_target")
        dollars = self.equity * min(tv / annual_vol, 1.0)
        return SizeDecision(int(dollars / price), dollars, "volatility_target")

    def kelly(self, price: float, prob_win: float, win_pct: float,
              loss_pct: float, fraction: Optional[float] = None) -> SizeDecision:
        """
        Fractional Kelly. f* = p/a - q/b for win fraction b, loss fraction a
        (payoff-odds form); we then take only `fraction` of f*.
        """
        frac = config.KELLY_FRACTION if fraction is None else fraction
        b = abs(win_pct) / 100
        a = abs(loss_pct) / 100
        if a <= 0 or b <= 0:
            return SizeDecision(0, 0.0, "kelly")
        q = 1 - prob_win
        f_star = prob_win / a - q / b
        f = max(0.0, f_star * a) * frac  # convert odds-space f* to equity fraction
        dollars = self.equity * min(f, 1.0)
        return SizeDecision(int(dollars / price), dollars, "kelly")

    # --------------------------------------------------------- caps
    def apply_caps(self, decision: SizeDecision, price: float,
                   sector: Optional[str] = None,
                   current_sector_exposure: Optional[Dict[str, float]] = None,
                   current_gross_exposure: float = 0.0) -> SizeDecision:
        """Clamp a raw size to single-name / sector / gross exposure caps."""
        dollars = decision.shares * price
        capped_by = None

        max_name = self.equity * config.MAX_POSITION_PCT / 100
        if dollars > max_name:
            dollars, capped_by = max_name, "max_position_pct"

        if sector and current_sector_exposure is not None:
            max_sector = self.equity * config.MAX_SECTOR_PCT / 100
            room = max_sector - current_sector_exposure.get(sector, 0.0)
            if dollars > room:
                dollars, capped_by = max(room, 0.0), "max_sector_pct"

        max_gross = self.equity * config.MAX_GROSS_EXPOSURE_PCT / 100
        room = max_gross - current_gross_exposure
        if dollars > room:
            dollars, capped_by = max(room, 0.0), "max_gross_exposure"

        shares = int(dollars / price) if price > 0 else 0
        if capped_by:
            logger.debug("size capped by {}: {} -> {} shares", capped_by,
                         decision.shares, shares)
        return SizeDecision(shares, shares * price, decision.method, capped_by)
