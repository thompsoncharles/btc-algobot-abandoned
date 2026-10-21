"""
Alternative data (optional per spec). Free tier: news headlines via
yfinance with a naive keyword sentiment score. Everything institutional
(options flow, ETF flows, satellite, card spending) is stubbed with the
provider it would need, so interfaces exist without pretending we have data.
"""
from typing import List, Optional

import pandas as pd
from loguru import logger

from quant.data.market_data import to_yahoo_symbol

_POS = {"beat", "beats", "surge", "soar", "record", "upgrade", "upgraded", "strong",
        "growth", "profit", "rally", "gain", "jump", "bullish", "raise", "raises"}
_NEG = {"miss", "misses", "fall", "falls", "plunge", "downgrade", "downgraded", "weak",
        "loss", "losses", "lawsuit", "probe", "recall", "cut", "cuts", "bearish", "warn"}


def news_sentiment(ticker: str) -> Optional[pd.DataFrame]:
    """Headlines + naive keyword sentiment in [-1, 1]. Use cautiously (per spec)."""
    import yfinance as yf

    try:
        items = yf.Ticker(to_yahoo_symbol(ticker)).news or []
    except Exception as e:
        logger.debug("news fetch failed for {}: {}", ticker, e)
        return None
    rows = []
    for it in items:
        content = it.get("content", it) or {}
        title = (content.get("title") or "").strip()
        if not title:
            continue
        words = {w.strip(".,!?:;()'\"").lower() for w in title.split()}
        pos, neg = len(words & _POS), len(words & _NEG)
        score = 0.0 if pos + neg == 0 else (pos - neg) / (pos + neg)
        rows.append({"ticker": ticker.upper(), "title": title,
                     "pubDate": content.get("pubDate"), "sentiment": score})
    return pd.DataFrame(rows) if rows else None


def options_flow(ticker: str):
    """Unusual options activity. Needs a paid feed (CBOE LiveVol, FlowAlgo...)."""
    raise NotImplementedError("Options flow requires a paid provider; stubbed.")


def etf_flows(tickers: List[str]):
    """Creation/redemption flows. Needs ETF.com / Bloomberg-grade data."""
    raise NotImplementedError("ETF flow data requires a paid provider; stubbed.")


def social_sentiment(ticker: str):
    """Reddit/X sentiment. Needs platform API keys; use cautiously per spec."""
    raise NotImplementedError("Social sentiment requires platform API access; stubbed.")
