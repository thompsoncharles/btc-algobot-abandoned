"""
Probability engine: instead of predicting price, estimate the quantities
that actually matter for a trade -

  * P(target hit before stop)   first-passage of drifted Brownian motion
  * expected return             probability-weighted target/stop outcome
  * expected drawdown           within-holding-period expected adverse move
  * confidence interval         Wilson interval on empirical win rates

Model P(up) and Godel PRT prob_up are blended with a configurable weight.
Godel is temporarily offline (see quant.models.linear_ar for the standalone
AR replacement used on BTCUSD) - the blend degrades cleanly to model-only
whenever godel_prob_up is omitted, so nothing here needs to change to run
without it; the Godel path just goes unused until it's reconnected.
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple

from quant import config


@dataclass
class TradeEstimate:
    ticker: str
    prob_target_first: float      # P(target before stop)
    expected_return_pct: float    # probability-weighted
    expected_drawdown_pct: float
    prob_source: str              # "model" | "godel" | "blend"
    blended_prob_up: float

    @property
    def edge(self) -> float:
        """Expected return - a plain 'is this worth doing' number."""
        return self.expected_return_pct


def blend_probabilities(model_prob: Optional[float], godel_prob: Optional[float],
                        godel_weight: Optional[float] = None) -> Tuple[float, str]:
    """Weighted blend of model P(up) and Godel PRT prob_up."""
    w = config.GODEL_BLEND_WEIGHT if godel_weight is None else godel_weight
    if model_prob is None and godel_prob is None:
        return 0.5, "none"
    if model_prob is None:
        return float(godel_prob), "godel"
    if godel_prob is None:
        return float(model_prob), "model"
    return w * float(godel_prob) + (1 - w) * float(model_prob), "blend"


def prob_target_before_stop(prob_up: float, target_pct: float, stop_pct: float) -> float:
    """
    P(hit target before stop) for a drifted random walk, using the classic
    gambler's-ruin / first-passage result. Drift is backed out of prob_up
    (prob_up 0.5 = driftless).

    For the driftless case the answer is stop/(target+stop); drift tilts it
    via the exponential martingale form.
    """
    target_pct, stop_pct = abs(target_pct), abs(stop_pct)
    if target_pct <= 0 or stop_pct <= 0:
        return 0.0
    p = min(max(prob_up, 1e-6), 1 - 1e-6)
    if abs(p - 0.5) < 1e-9:
        return stop_pct / (target_pct + stop_pct)
    # map prob_up to per-step odds; theta = log(q/p) is the ruin exponent
    theta = math.log((1 - p) / p)
    a, b = stop_pct, target_pct  # distance to stop, distance to target
    try:
        num = 1 - math.exp(theta * a)
        den = 1 - math.exp(theta * (a + b))
        return min(max(num / den, 0.0), 1.0)
    except OverflowError:
        return 1.0 if p > 0.5 else 0.0


def expected_return(prob_hit: float, target_pct: float, stop_pct: float) -> float:
    return prob_hit * abs(target_pct) - (1 - prob_hit) * abs(stop_pct)


def expected_drawdown(vol_pct_per_bar: float, holding_bars: int) -> float:
    """
    Expected maximum adverse excursion of a driftless walk over the holding
    period: E[max drawdown] ~ 1.2533 * sigma * sqrt(T) (expected max of
    |Brownian motion|, sqrt(pi/2) factor).
    """
    return 1.2533 * abs(vol_pct_per_bar) * math.sqrt(max(holding_bars, 1))


def wilson_interval(wins: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score CI for an empirical win rate."""
    if n == 0:
        return 0.0, 1.0
    phat = wins / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


class ProbabilityEngine:
    def __init__(self, godel_weight: Optional[float] = None):
        self.godel_weight = config.GODEL_BLEND_WEIGHT if godel_weight is None else godel_weight

    def estimate(self, ticker: str, target_pct: float, stop_pct: float,
                 model_prob_up: Optional[float] = None,
                 godel_prob_up: Optional[float] = None,
                 vol_pct_per_bar: float = 0.3,
                 holding_bars: Optional[int] = None,
                 model_name: Optional[str] = None) -> TradeEstimate:
        """
        model_name: label for what produced model_prob_up (e.g. "linear_ar",
        "gradient_boosting"). Only affects the reported prob_source when
        godel_prob_up is absent - otherwise it's still "blend"/"godel".
        """
        prob_up, source = blend_probabilities(model_prob_up, godel_prob_up, self.godel_weight)
        if source == "model" and model_name:
            source = model_name
        hit = prob_target_before_stop(prob_up, target_pct, stop_pct)
        bars = holding_bars if holding_bars is not None else config.TARGET_HORIZON_BARS
        return TradeEstimate(
            ticker=ticker.upper(),
            prob_target_first=hit,
            expected_return_pct=expected_return(hit, target_pct, stop_pct),
            expected_drawdown_pct=expected_drawdown(vol_pct_per_bar, bars),
            prob_source=source,
            blended_prob_up=prob_up,
        )
