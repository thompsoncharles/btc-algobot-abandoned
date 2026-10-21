"""
Linear autoregressive model: P(up) from lagged log-returns via logistic
regression on standardized features - the sklearn-native equivalent of the
qt4 reference repo's single nn.Linear layer trained with LBFGS on log
returns (models.py:LinearModel + research.py:add_log_return_features).

Kept genuinely linear (no hidden layers, no gradient boosting) so it's a
readable baseline: coefficients are inspectable per-lag AR weights, same
spirit as get_linear_params() in the reference repo.

Used as the primary signal for BTCUSD while Godel PRT is unavailable -
see quant.probability.engine, which accepts this model's predict_proba()
output as model_prob_up with no godel_prob_up required.
"""
from typing import List, Optional

import numpy as np
import pandas as pd

from quant.models.baseline import LogisticModel


def log_return(close: pd.Series, step: int = 1) -> pd.Series:
    return np.log(close / close.shift(step))


def lagged_log_return_features(close: pd.Series, n_lags: int = 5,
                               forecast_horizon: int = 1,
                               include_lag0: bool = False) -> pd.DataFrame:
    """
    AR feature matrix: n_lags columns of past log-returns, each shifted by
    forecast_horizon steps so column i is known forecast_horizon*i bars
    before the target - mirrors research.py's add_lags(step=forecast_horizon).

    include_lag0=True adds ar_lag_0 = the just-completed bar's log return
    r_t = log(close_t/close_{t-1}). At decision time (bar t has closed, trade
    fills at open_{t+1}) r_t is known, so this is NOT look-ahead - it's the
    standard AR(p) formulation that the default (lags 1..n) omits.
    """
    r = log_return(close)
    start = 0 if include_lag0 else 1
    cols = {f"ar_lag_{i}": r.shift(forecast_horizon * i) for i in range(start, n_lags + 1)}
    return pd.DataFrame(cols, index=close.index)


class LinearARModel:
    """AR(n_lags) classifier: P(up) over the next `forecast_horizon` bars."""

    def __init__(self, n_lags: int = 5, forecast_horizon: int = 1, C: float = 1.0,
                 include_lag0: bool = False, class_weight=None):
        self.n_lags = n_lags
        self.forecast_horizon = forecast_horizon
        self.include_lag0 = include_lag0
        self.logit = LogisticModel(C=C, class_weight=class_weight)
        self.feature_names: Optional[List[str]] = None

    def build_features(self, close: pd.Series) -> pd.DataFrame:
        return lagged_log_return_features(close, self.n_lags, self.forecast_horizon,
                                          include_lag0=self.include_lag0)

    def build_target(self, close: pd.Series) -> pd.Series:
        """Forward log-return over forecast_horizon bars - positive = up."""
        return log_return(close, self.forecast_horizon).shift(-self.forecast_horizon)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearARModel":
        self.feature_names = list(X.columns)
        self.logit.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.logit.predict_proba(X)

    def coefficients(self) -> pd.Series:
        """Per-lag AR weights (standardized-feature scale) - the readable part."""
        clf = self.logit.pipe.named_steps["logit"]
        return pd.Series(clf.coef_.flatten(), index=self.feature_names)

    def fit_from_prices(self, close: pd.Series) -> "LinearARModel":
        """Convenience: build lagged features + forward-return target from a
        raw close series and fit in one call."""
        X = self.build_features(close)
        y = self.build_target(close)
        df = X.assign(_y=y).dropna()
        return self.fit(df.drop(columns="_y"), df["_y"])

    # Uniform bars-in interface so callers don't care which model they hold.
    def fit_from_bars(self, bars) -> "LinearARModel":
        return self.fit_from_prices(bars["Close"].astype(float))

    def latest_features(self, bars) -> pd.DataFrame:
        """Last usable feature row from a bars DataFrame (for live/backtest scoring)."""
        return self.build_features(bars["Close"].astype(float)).iloc[[-1]].dropna()


class LinearBTCModel(LinearARModel):
    """
    Same linear/logistic core as LinearARModel, but on the richer BTC feature
    set (momentum + trend + volatility + flow) instead of lagged returns
    alone. Features come from quant.features.btc_features; still fully causal.
    """

    def __init__(self, n_lags: int = 5, forecast_horizon: int = 1,
                 ma_window: int = 20, C: float = 1.0, include_lag0: bool = False,
                 groups=None, class_weight=None):
        super().__init__(n_lags=n_lags, forecast_horizon=forecast_horizon, C=C,
                         include_lag0=include_lag0, class_weight=class_weight)
        self.ma_window = ma_window
        self.groups = groups  # None = all feature groups; a set selects a subset

    def build_features(self, bars) -> pd.DataFrame:
        # accepts a full OHLCV DataFrame (not just close) - richer features need it
        from quant.features.btc_features import assemble_btc_features
        return assemble_btc_features(bars, self.n_lags, self.forecast_horizon,
                                     self.ma_window, groups=self.groups,
                                     include_lag0=self.include_lag0)

    def fit_from_bars(self, bars) -> "LinearBTCModel":
        from quant.features.btc_features import forward_return_target
        X = self.build_features(bars)
        y = forward_return_target(bars["Close"].astype(float), self.forecast_horizon)
        df = X.assign(_y=y).dropna()
        self.fit(df.drop(columns="_y"), df["_y"])
        return self

    def latest_features(self, bars) -> pd.DataFrame:
        return self.build_features(bars).iloc[[-1]].dropna()
