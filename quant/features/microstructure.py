"""
Microstructure features from NBBO quotes and bars: order imbalance,
bid-ask spread, volume profile. Queue position needs L2 (stubbed).
"""
from typing import Dict, Optional

import numpy as np
import pandas as pd

from quant.data.quotes import Quote


def spread_bps(quote: Quote) -> float:
    return quote.spread_bps


def order_imbalance(quote: Quote) -> float:
    return quote.order_imbalance


def quote_features(quotes: Dict[str, Quote]) -> pd.DataFrame:
    """Feature row per ticker from a batch of NBBO snapshots."""
    rows = []
    for t, q in quotes.items():
        rows.append({
            "ticker": t,
            "mid": q.mid,
            "spread_bps": q.spread_bps,
            "order_imbalance": q.order_imbalance,
        })
    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()


def volume_profile(bars: pd.DataFrame, bins: int = 20) -> pd.DataFrame:
    """
    Volume-at-price histogram from intraday bars.

    Returns DataFrame [price_low, price_high, volume, poc] where poc marks
    the point-of-control bin (highest volume node).
    """
    if bars.empty:
        return pd.DataFrame()
    typical = (bars["High"] + bars["Low"] + bars["Close"]) / 3
    lo, hi = float(bars["Low"].min()), float(bars["High"].max())
    if hi <= lo:
        return pd.DataFrame()
    edges = np.linspace(lo, hi, bins + 1)
    idx = np.clip(np.digitize(typical, edges) - 1, 0, bins - 1)
    vol = np.zeros(bins)
    np.add.at(vol, idx, bars["Volume"].to_numpy(dtype=float))
    out = pd.DataFrame({
        "price_low": edges[:-1], "price_high": edges[1:], "volume": vol,
    })
    out["poc"] = out["volume"] == out["volume"].max()
    return out


def queue_position(ticker: str):
    """Queue position estimation requires Level II depth; stubbed."""
    raise NotImplementedError("Queue position needs L2 market-depth data; stubbed.")
