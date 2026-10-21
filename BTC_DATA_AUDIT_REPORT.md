# BTC Data Audit & Quality Gate Report
**Date:** 2026-07-04  
**Status:** SAFE FOR RESEARCH ✓

## Data Quality Gate Verdict

✓ **PASSES** - Cached BTC data meets all quality requirements for research and paper trading.

## Cached Data Summary

| Metric | Value | Status |
|--------|-------|--------|
| **Symbol** | BTCUSD (Binance BTCUSDT alias) | ✓ |
| **Interval** | 12h | ✓ |
| **Total bars** | 1,455 | ✓ |
| **Date range** | 2024-07-05 to 2026-07-03 12:00 | ✓ |
| **Source chain** | Binance seed + repair fallback (Kraken/Yahoo) | ✓ |
| **Staleness** | 0 hours (current) | ✓ |
| **Missing bars** | 3 of 1,458 expected (0.2%) | ✓ |
| **Duplicate bars** | 0 | ✓ |
| **Unfinished candle** | No | ✓ |
| **Volume coverage** | 73.9% non-null | ✓ |

## Quality Gate Checks

All 5 quality gates passed:

1. **No duplicates** ✓ — 0 duplicate timestamps found
2. **Missing bars <1%** ✓ — 0.2% missing (3/1458), well below 1% threshold
3. **Not stale** ✓ — Latest bar is at expected time (0 hours behind)
4. **No unfinished candles** ✓ — Latest bar is fully closed
5. **Volume coverage** ✓ — 73.9% positive volume (>50% threshold)

## Data Source Chain

```
Bundled seed CSV (qt4 project, 2024-07-05 start)
    ↓
Binance BTCUSDT 12h klines (primary, no geo-block)
    ↓
Fallback: Kraken XBTUSD (if Binance unavailable)
    ↓
Local parquet cache: quant_data/bars/12h/BTCUSD.parquet
```

## Robustness Checks

- **Yahoo BTC-USD daily:** Unavailable (network/yfinance config)
- **IBKR BTC/USD crypto:** Not checked in offline mode
- **Recommendation:** Use Binance/Yahoo for long-history research; validate against IBKR when live trading

## Verdict for btc-eval

**Recommendation:** SAFE FOR RESEARCH & PAPER TRADING

The cached data passes all quality gates and is suitable for:
- ✓ Strategy research and backtesting
- ✓ Model training and validation (out-of-sample)
- ✓ Paper trading simulation
- ✗ Live trading (validate against IBKR first)

## Missing Bars Detail

3 missing 12h bars were detected:
- First missing: 2025-10-28 00:00:00
- This represents only 0.2% of expected bars and is acceptable for research

These gaps may be due to:
1. Exchange maintenance windows
2. Data source gaps in Binance history
3. Timezone handling edge cases

**Action:** No remediation required; missing bars are immaterial (<1%) and do NOT block research.

## Implementation Notes

The quality gate is enforced in the pipeline:
- `python -m quant.pipeline btc-eval` checks quality gate BEFORE running evaluation
- `python -m quant.pipeline btc-data-audit --offline` shows full audit details
- `python -m quant.pipeline btc-data-audit --ibkr` adds IBKR BTC/USD comparison (requires TWS/Gateway)

If quality gate fails, btc-eval will exit with clear diagnostic message and recommendation.

## Code Changes

1. **Enhanced btc_audit.py:**
   - Added `_quality_gate()` function with 5 strict checks
   - Returns (passes: bool, verdict: str)
   - Updated `build_audit()` to include quality_gate section

2. **Updated pipeline.py:**
   - Added `_check_btc_data_quality()` function
   - btc-eval now calls quality gate check before proceeding
   - Fails fast with diagnostic message if data is unsafe

3. **Added smoke test:**
   - New test: "BTC data quality gate" verifies gate logic
   - All 27 smoke tests pass ✓

