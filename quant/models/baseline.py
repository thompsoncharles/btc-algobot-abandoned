"""
Baseline models: linear regression, logistic regression, autoregression.

Every model exposes fit(X, y) / predict(X), and classifiers add
predict_proba(X) returning P(up) as a 1-D array. Baselines are the
benchmark any fancier model must beat out-of-sample.
"""
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class LinearModel:
    """Ridge-free OLS on standardized features - forward-return regression."""

    def __init__(self):
        self.pipe = Pipeline([("scale", StandardScaler()), ("ols", LinearRegression())])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearModel":
        self.pipe.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipe.predict(X)


class LogisticModel:
    """P(up) classifier on standardized features."""

    def __init__(self, C: float = 1.0, class_weight=None):
        self.pipe = Pipeline([
            ("scale", StandardScaler()),
            ("logit", LogisticRegression(C=C, max_iter=1000,
                                          class_weight=class_weight)),
        ])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticModel":
        self.pipe.fit(X, (y > 0).astype(int) if y.dtype != int else y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipe.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipe.predict_proba(X)[:, 1]


class ARModel:
    """AR(p) on returns. Uses statsmodels when available, else OLS lags."""

    def __init__(self, lags: int = 5):
        self.lags = lags
        self._sm_result = None
        self._coef: Optional[np.ndarray] = None
        self._last_values: Optional[np.ndarray] = None

    def fit(self, returns: pd.Series) -> "ARModel":
        r = returns.dropna().to_numpy(dtype=float)
        self._last_values = r[-self.lags:]
        try:
            from statsmodels.tsa.ar_model import AutoReg
            self._sm_result = AutoReg(r, lags=self.lags, old_names=False).fit()
        except ImportError:
            X = np.column_stack([r[self.lags - i - 1: len(r) - i - 1] for i in range(self.lags)])
            y = r[self.lags:]
            X = np.column_stack([np.ones(len(X)), X])
            self._coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return self

    def forecast(self, steps: int = 1) -> np.ndarray:
        if self._sm_result is not None:
            return np.asarray(self._sm_result.forecast(steps))
        history = list(self._last_values)
        out = []
        for _ in range(steps):
            x = np.concatenate([[1.0], history[-self.lags:][::-1]])
            nxt = float(x @ self._coef)
            out.append(nxt)
            history.append(nxt)
        return np.asarray(out)
