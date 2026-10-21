"""
Macro data via FRED (free API key) plus economic-calendar stub.

Series fetched by default:
    FEDFUNDS   effective federal funds rate
    CPIAUCSL   CPI (inflation)
    UNRATE     unemployment rate
    PAYEMS     nonfarm payrolls
    DGS2/DGS10 2y / 10y treasury yields
    T10Y2Y     10y-2y spread (curve)
"""
from typing import Dict, Optional

import pandas as pd
import requests
from loguru import logger

from quant import config
from quant.data.store import DataStore

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"

DEFAULT_SERIES = {
    "FEDFUNDS": "fed_funds_rate",
    "CPIAUCSL": "cpi",
    "UNRATE": "unemployment_rate",
    "PAYEMS": "nonfarm_payrolls",
    "DGS2": "treasury_2y",
    "DGS10": "treasury_10y",
    "T10Y2Y": "curve_10y_2y",
}


class MacroData:
    def __init__(self, store: Optional[DataStore] = None, api_key: str = ""):
        self.store = store or DataStore()
        self.api_key = api_key or config.FRED_API_KEY

    def fetch_series(self, series_id: str) -> Optional[pd.Series]:
        if not self.api_key:
            logger.warning("FRED_API_KEY not set in constants/parameters.py - skipping {}", series_id)
            return None
        r = requests.get(FRED_URL, params={
            "series_id": series_id, "api_key": self.api_key, "file_type": "json",
        }, timeout=30)
        r.raise_for_status()
        obs = r.json().get("observations", [])
        if not obs:
            return None
        s = pd.Series(
            [float(o["value"]) if o["value"] not in (".", "") else float("nan") for o in obs],
            index=pd.to_datetime([o["date"] for o in obs]),
            name=series_id,
        ).dropna()
        self.store.save_table(series_id, s.to_frame(), subdir="macro")
        return s

    def fetch_all(self) -> Dict[str, pd.Series]:
        out = {}
        for sid in DEFAULT_SERIES:
            try:
                s = self.fetch_series(sid)
                if s is not None:
                    out[sid] = s
            except Exception as e:
                logger.warning("FRED fetch failed for {}: {}", sid, e)
        return out

    def load(self, series_id: str) -> Optional[pd.Series]:
        df = self.store.load_table(series_id, subdir="macro")
        if df is None:
            return None
        return df.iloc[:, 0]

    # ------------------------------------------------------------ stub
    def economic_calendar(self, start=None, end=None):
        """Upcoming economic events (FOMC, CPI prints, NFP).

        No good free structured feed; candidates: tradingeconomics (paid),
        investpy (fragile scraping). Deliberately stubbed.
        """
        raise NotImplementedError(
            "Economic calendar requires a paid feed (e.g. Trading Economics); "
            "stubbed until a provider is chosen."
        )
