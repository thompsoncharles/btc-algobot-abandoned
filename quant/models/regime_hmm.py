"""
Market regime classification. Prefers a Gaussian HMM (hmmlearn) on daily
returns; falls back to a volatility-quantile classifier when hmmlearn is
not installed - both return an integer regime series where higher regime
numbers mean rougher markets.
"""
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


class RegimeModel:
    def __init__(self, n_states: int = 3):
        self.n_states = n_states
        self.backend = "hmm"
        try:
            from hmmlearn.hmm import GaussianHMM
            self._hmm = GaussianHMM(n_components=n_states, covariance_type="full",
                                    n_iter=200, random_state=7)
        except ImportError:
            self._hmm = None
            self.backend = "vol_quantile"
            logger.info("hmmlearn not installed - using volatility-quantile regimes")
        self._state_order: Optional[np.ndarray] = None

    def fit_predict(self, close: pd.Series) -> pd.Series:
        """Label each bar with a regime (0 = calmest)."""
        ret = np.log(close / close.shift(1)).dropna()
        if self._hmm is not None:
            X = ret.to_numpy().reshape(-1, 1)
            self._hmm.fit(X)
            states = self._hmm.predict(X)
            # Re-index states so 0 = lowest variance regime
            variances = np.array([self._hmm.covars_[i].item() for i in range(self.n_states)])
            self._state_order = np.argsort(variances)
            remap = {orig: rank for rank, orig in enumerate(self._state_order)}
            labeled = pd.Series([remap[s] for s in states], index=ret.index, name="regime")
            return labeled.reindex(close.index)
        # Fallback: bucket 20d realized vol into quantiles
        vol = ret.rolling(20).std()
        q = vol.rank(pct=True)
        edges = np.linspace(0, 1, self.n_states + 1)[1:-1]
        regime = pd.Series(np.digitize(q, edges), index=ret.index, name="regime")
        return regime.reindex(close.index)
