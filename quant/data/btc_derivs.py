"""
BTC derivatives context: perpetual funding rate.

Availability audit (2026-07-04, this machine):
  - Binance futures API: geo-blocked (HTTP 451) - unusable.
  - OKX funding-rate history: public and reachable, BUT the API only serves
    ~3 months of history. Cached incrementally here so depth accumulates
    over time; the feature group auto-enables at >= 365 days of cache.
  - Bybit open interest: response unparseable from here - skipped.
  - OKX long/short ratio (rubik): only ~30 days of history - too short for
    walk-forward research - skipped.
  - Liquidation feeds: no clean free source with history - skipped.

Funding is settled every 8h on OKX. A settled funding rate is public at its
funding time, so aligning bar T to the last funding settled at-or-before T
is causal; features additionally shift by one settlement for safety margin.
"""
import time
from typing import Optional

import pandas as pd
import requests
from loguru import logger

from quant.data.store import DataStore

OKX_FUNDING_URL = "https://www.okx.com/api/v5/public/funding-rate-history"
FUNDING_TABLE = "BTC_FUNDING"          # quant_data/macro/BTC_FUNDING.parquet
INST_ID = "BTC-USDT-SWAP"


def fetch_okx_funding_page(before_ms: Optional[int] = None, limit: int = 100) -> pd.DataFrame:
    """One page of OKX funding history, newest first; `before_ms` pages backwards."""
    params = {"instId": INST_ID, "limit": limit}
    if before_ms is not None:
        params["after"] = before_ms  # OKX: 'after' = records EARLIER than this ts
    r = requests.get(OKX_FUNDING_URL, params=params, timeout=20)
    r.raise_for_status()
    rows = r.json().get("data", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms",
                                    utc=True).dt.tz_localize(None)
    df["funding_rate"] = df["fundingRate"].astype(float)
    return df.set_index("datetime")[["funding_rate"]].sort_index()


def fetch_okx_funding_history(max_pages: int = 120,
                              stop_before: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Page backwards through OKX funding history (3 settlements/day, 100/page
    = ~33 days per page; 120 pages ~ 10+ years, i.e. everything they have).
    stop_before: stop once records older than this exist (incremental refresh).
    """
    frames = []
    before = None
    for _ in range(max_pages):
        page = fetch_okx_funding_page(before_ms=before)
        if page.empty:
            break
        frames.append(page)
        oldest = page.index.min()
        before = int(oldest.timestamp() * 1000)
        if stop_before is not None and oldest <= stop_before:
            break
        time.sleep(0.12)  # public rate limit courtesy
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated(keep="last")]


def get_funding_series(store: Optional[DataStore] = None,
                       refresh: bool = True) -> pd.DataFrame:
    """Cached OKX funding history, incrementally refreshed."""
    store = store or DataStore()
    cached = store.load_table(FUNDING_TABLE, "macro")
    if cached is not None and not cached.empty:
        cached.index = pd.to_datetime(cached.index)

    if not refresh:
        return cached if cached is not None else pd.DataFrame()

    try:
        stop = cached.index.max() if cached is not None and not cached.empty else None
        fresh = fetch_okx_funding_history(stop_before=stop)
        if not fresh.empty:
            merged = fresh
            if cached is not None and not cached.empty:
                merged = pd.concat([cached, fresh]).sort_index()
                merged = merged[~merged.index.duplicated(keep="last")]
            store.save_table(FUNDING_TABLE, merged, "macro")
            logger.info("funding history: {} records ({} to {})", len(merged),
                        merged.index.min(), merged.index.max())
            return merged
    except Exception as e:
        logger.warning("OKX funding refresh failed ({}); using cache", e)
    return cached if cached is not None else pd.DataFrame()
