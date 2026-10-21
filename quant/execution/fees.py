"""
Venue fee models. Currently: IBKR spot crypto (Paxos custody).

IBKR spot-crypto schedule (retail):
    0.18% of trade value  (monthly crypto volume <= $100k - where a $5k
                           account lives; higher tiers exist at $100k+ and
                           $1M+ monthly volume but are unreachable here)
    minimum $1.75/order, capped at 1% of trade value
    no added spread/markup/custody fees

Reference points (used as test fixtures in tests/test_quant_smoke.py):
    $100 -> $1.00 (1% cap)   $500 -> $1.75 (minimum)
    $1,000 -> $1.80          $10,000 -> $18.00

The shape matters for a small account: below ~$972 notional the $1.75
minimum dominates and the effective rate rises as the trade shrinks, until
the 1% cap takes over below $175. Position sizing must respect this floor
(see BTC_MIN_TRADE_NOTIONAL_USD) or fees eat any model edge.
"""
from quant import config


def ibkr_crypto_fee(notional: float,
                    tier_pct: float = None,
                    min_usd: float = None,
                    cap_pct: float = None) -> float:
    """One-side commission in dollars: max(min_usd, notional*tier%) capped at cap% of notional."""
    if notional <= 0:
        return 0.0
    tier_pct = config.BTC_FEE_TIER1_PCT if tier_pct is None else tier_pct
    min_usd = config.BTC_FEE_MIN_USD if min_usd is None else min_usd
    cap_pct = config.BTC_FEE_CAP_PCT if cap_pct is None else cap_pct
    fee = max(min_usd, notional * tier_pct / 100)
    return min(fee, notional * cap_pct / 100)


def round_trip_cost(notional: float) -> float:
    """Dollar cost of entering and exiting a position of this size."""
    return ibkr_crypto_fee(notional) * 2


def round_trip_cost_pct(notional: float) -> float:
    """Round-trip cost as percent of notional - the breakeven price move."""
    if notional <= 0:
        return float("inf")
    return round_trip_cost(notional) / notional * 100


def edge_clears_costs(expected_edge_pct: float, notional: float,
                      min_ratio: float = None) -> bool:
    """
    Gate: expected per-trade edge must exceed min_ratio x round-trip cost.
    An edge of 0.5% on a $500 position ($2.50) does NOT clear 2x the $3.50
    round-trip - this is what stops fee-dominated trades on a small account.
    """
    min_ratio = config.BTC_MIN_EDGE_TO_COST if min_ratio is None else min_ratio
    edge_dollars = expected_edge_pct / 100 * notional
    return edge_dollars >= min_ratio * round_trip_cost(notional)


def fee_table(notionals=(100, 250, 500, 1000, 2500, 5000, 10000)) -> list:
    """Fee schedule rows for reporting: per-side fee, round-trip %, breakeven move."""
    rows = []
    for n in notionals:
        fee = ibkr_crypto_fee(n)
        rows.append({
            "notional_usd": n,
            "fee_per_side_usd": round(fee, 2),
            "fee_per_side_pct": round(fee / n * 100, 4),
            "round_trip_pct": round(round_trip_cost_pct(n), 4),
        })
    return rows
