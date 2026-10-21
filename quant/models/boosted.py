"""
Gradient boosting. Prefers XGBoost/LightGBM when installed; otherwise
uses scikit-learn's HistGradientBoosting (same family, zero extra deps).
The backend in use is recorded so the model registry can track it.
"""
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


def _pick_backend(preferred: Optional[str] = None) -> str:
    order = [preferred] if preferred else []
    order += ["xgboost", "lightgbm", "sklearn"]
    for name in order:
        if name == "sklearn":
            return name
        try:
            __import__(name)
            return name
        except ImportError:
            continue
    return "sklearn"


class GradientBoostingModel:
    """Classifier producing P(up). backend: 'xgboost' | 'lightgbm' | 'sklearn'."""

    def __init__(self, backend: Optional[str] = None, **params):
        self.backend = _pick_backend(backend)
        self.params = params
        self.model = self._build()
        logger.info("gradient boosting backend: {}", self.backend)

    def _build(self):
        if self.backend == "xgboost":
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=self.params.get("n_estimators", 300),
                max_depth=self.params.get("max_depth", 4),
                learning_rate=self.params.get("learning_rate", 0.05),
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss",
            )
        if self.backend == "lightgbm":
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                n_estimators=self.params.get("n_estimators", 300),
                max_depth=self.params.get("max_depth", -1),
                learning_rate=self.params.get("learning_rate", 0.05),
                subsample=0.8, colsample_bytree=0.8, verbosity=-1,
            )
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=self.params.get("n_estimators", 300),
            max_depth=self.params.get("max_depth", None),
            learning_rate=self.params.get("learning_rate", 0.05),
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GradientBoostingModel":
        self.feature_names = list(X.columns)
        self.model.fit(X, (y > 0).astype(int) if y.dtype != int else y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def feature_importances(self) -> pd.Series:
        """Importance per feature (permutation-free, backend-native)."""
        imp = getattr(self.model, "feature_importances_", None)
        if imp is None:
            return pd.Series(dtype=float)
        return pd.Series(imp, index=self.feature_names).sort_values(ascending=False)


class RandomForestModel:
    """Random forest classifier (spec item; sklearn-native)."""

    def __init__(self, **params):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 400),
            max_depth=params.get("max_depth", 8),
            min_samples_leaf=params.get("min_samples_leaf", 20),
            n_jobs=-1,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestModel":
        self.feature_names = list(X.columns)
        self.model.fit(X, (y > 0).astype(int) if y.dtype != int else y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def feature_importances(self) -> pd.Series:
        return pd.Series(self.model.feature_importances_,
                         index=self.feature_names).sort_values(ascending=False)
