"""
Quant research framework for algobot V4.

Maps to the 15-component institutional stack:
 1. Data layer .............. quant.data
 2. Universe selection ...... quant.universe
 3. Feature engineering ..... quant.features
 4. Models .................. quant.models
 5. Probability engine ...... quant.probability
 6. Position sizing ......... quant.sizing
 7. Risk management ......... quant.risk
 8. Backtesting ............. quant.backtest
 9. Performance metrics ..... quant.metrics
10. Execution ............... quant.execution
11. Monitoring .............. quant.monitor
12. Logging (audit) ......... quant.journal
13. Retraining .............. quant.retrain
14. Tech stack .............. requirements-quant.txt
15. Dev workflow ............ quant.pipeline (CLI orchestrator)

Godel Terminal PRT signals are ingested via quant.data.godel_signals and
blended with model output in quant.probability.engine.
"""

__version__ = "0.1.0"
