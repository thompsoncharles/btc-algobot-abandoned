"""
Quant framework configuration.

All tunable values live in constants/parameters.py (house rule #4).
This module resolves them with safe defaults so an older parameters.py
without the QUANT_* section still works, and derives filesystem paths.
"""
from pathlib import Path

import constants.parameters as P

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _get(name: str, default):
    return getattr(P, name, default)


# --- Data layer ---
DATA_DIR = PROJECT_ROOT / _get("QUANT_DATA_DIR", "quant_data")
BARS_DIR = DATA_DIR / "bars"
FUNDAMENTALS_DIR = DATA_DIR / "fundamentals"
MACRO_DIR = DATA_DIR / "macro"
DUCKDB_PATH = DATA_DIR / "quant.duckdb"
JOURNAL_DIR = DATA_DIR / "journal"
FRED_API_KEY = _get("FRED_API_KEY", "")
HISTORY_YEARS = int(_get("QUANT_HISTORY_YEARS", 2))

# --- Universe ---
UNIVERSE_MIN_PRICE = float(_get("UNIVERSE_MIN_PRICE", 3.0))
UNIVERSE_MIN_ADDV_USD = float(_get("UNIVERSE_MIN_ADDV_USD", 5_000_000))
UNIVERSE_MIN_MARKET_CAP = float(_get("UNIVERSE_MIN_MARKET_CAP", 300_000_000))
UNIVERSE_MAX_SPREAD_BPS = float(_get("UNIVERSE_MAX_SPREAD_BPS", 20.0))
UNIVERSE_ALLOWED_EXCHANGES = tuple(
    _get("UNIVERSE_ALLOWED_EXCHANGES",
         ("NYQ", "NMS", "ASE", "PCX", "NGM", "NCM", "NYSE", "NASDAQ", "AMEX"))
)

# --- Modeling ---
TARGET_HORIZON_BARS = int(_get("QUANT_TARGET_HORIZON_BARS", 5))
LABEL_UP_THRESHOLD = float(_get("QUANT_LABEL_UP_THRESHOLD", 0.0))
MODEL_DIR = PROJECT_ROOT / _get("QUANT_MODEL_DIR", "quant_data/models")

# --- Probability engine ---
GODEL_BLEND_WEIGHT = float(_get("QUANT_GODEL_BLEND_WEIGHT", 0.5))

# --- Sizing ---
KELLY_FRACTION = float(_get("QUANT_KELLY_FRACTION", 0.25))
VOL_TARGET_ANNUAL = float(_get("QUANT_VOL_TARGET_ANNUAL", 0.15))
MAX_POSITION_PCT = float(_get("QUANT_MAX_POSITION_PCT", 10.0))
MAX_SECTOR_PCT = float(_get("QUANT_MAX_SECTOR_PCT", 30.0))
MAX_GROSS_EXPOSURE_PCT = float(_get("QUANT_MAX_GROSS_EXPOSURE_PCT", 100.0))

# --- Risk ---
DAILY_LOSS_LIMIT_PCT = float(_get("QUANT_DAILY_LOSS_LIMIT_PCT", 3.0))
MAX_DRAWDOWN_PCT = float(_get("QUANT_MAX_DRAWDOWN_PCT", 10.0))
MAX_PAIRWISE_CORR = float(_get("QUANT_MAX_PAIRWISE_CORR", 0.85))
CIRCUIT_VIX_LEVEL = float(_get("QUANT_CIRCUIT_VIX_LEVEL", 40.0))
CIRCUIT_INDEX_DROP_PCT = float(_get("QUANT_CIRCUIT_INDEX_DROP_PCT", 3.0))

# --- Backtest costs ---
COMMISSION_PER_SHARE = float(_get("QUANT_COMMISSION_PER_SHARE", 0.005))
COMMISSION_MIN = float(_get("QUANT_COMMISSION_MIN", 1.0))
SLIPPAGE_BPS = float(_get("QUANT_SLIPPAGE_BPS", 2.0))

# --- Retraining ---
RETRAIN_SCHEDULE = str(_get("QUANT_RETRAIN_SCHEDULE", "weekly"))
DRIFT_SHARPE_FLOOR = float(_get("QUANT_DRIFT_SHARPE_FLOOR", 0.0))
ROLLBACK_ON_UNDERPERFORM = bool(_get("QUANT_ROLLBACK_ON_UNDERPERFORM", True))

# --- Execution ---
TWAP_SLICES = int(_get("QUANT_TWAP_SLICES", 4))
TWAP_INTERVAL_SECONDS = int(_get("QUANT_TWAP_INTERVAL_SECONDS", 30))

# --- BTCUSD linear-AR path ---
BTC_INTERVAL = str(_get("BTC_INTERVAL", "12h"))
BTC_AR_LAGS = int(_get("BTC_AR_LAGS", 5))
BTC_FORECAST_HORIZON_BARS = int(_get("BTC_FORECAST_HORIZON_BARS", 1))
BTC_SLIPPAGE_BPS = float(_get("BTC_SLIPPAGE_BPS", 5.0))
BTC_STOP_PCT = float(_get("BTC_STOP_PCT", 3.0))
BTC_TARGET_PCT = float(_get("BTC_TARGET_PCT", 6.0))
BTC_MAX_HOLD_BARS = int(_get("BTC_MAX_HOLD_BARS", 8))

# --- Account + IBKR spot-crypto fees ---
ACCOUNT_EQUITY_USD = float(_get("ACCOUNT_EQUITY_USD", 5000.0))
BTC_FEE_TIER1_PCT = float(_get("BTC_FEE_TIER1_PCT", 0.18))
BTC_FEE_MIN_USD = float(_get("BTC_FEE_MIN_USD", 1.75))
BTC_FEE_CAP_PCT = float(_get("BTC_FEE_CAP_PCT", 1.0))

# --- $5k-aware position sizing ---
BTC_RISK_PER_TRADE_PCT = float(_get("BTC_RISK_PER_TRADE_PCT", 1.0))
BTC_MIN_TRADE_NOTIONAL_USD = float(_get("BTC_MIN_TRADE_NOTIONAL_USD", 500.0))
BTC_MAX_POSITION_PCT = float(_get("BTC_MAX_POSITION_PCT", 50.0))
BTC_MIN_EDGE_TO_COST = float(_get("BTC_MIN_EDGE_TO_COST", 2.0))

# --- Signal: features, entry threshold, no-trade regime filter ---
BTC_USE_FEATURES = bool(_get("BTC_USE_FEATURES", True))
BTC_PROB_ENTRY = float(_get("BTC_PROB_ENTRY", 0.58))
BTC_MIN_CONFIDENCE = float(_get("BTC_MIN_CONFIDENCE", 0.05))
BTC_MIN_VOL_PCTILE = float(_get("BTC_MIN_VOL_PCTILE", 0.10))
BTC_MA_WINDOW = int(_get("BTC_MA_WINDOW", 20))

TRADING_DAYS_PER_YEAR = 252


def ensure_dirs() -> None:
    """Create all storage directories. Safe to call repeatedly."""
    for d in (DATA_DIR, BARS_DIR, FUNDAMENTALS_DIR, MACRO_DIR, MODEL_DIR, JOURNAL_DIR):
        d.mkdir(parents=True, exist_ok=True)
