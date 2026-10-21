"""
Kalman filters (pure NumPy): a 1-D level filter for denoising a price
series, and a dynamic-beta filter for hedging / pairs work.
"""
import numpy as np
import pandas as pd


class KalmanLevel:
    """Random-walk-plus-noise filter; smooths price without lag of an SMA."""

    def __init__(self, process_var: float = 1e-5, measure_var: float = 1e-2):
        self.q = process_var
        self.r = measure_var

    def filter(self, series: pd.Series) -> pd.Series:
        z = series.to_numpy(dtype=float)
        n = len(z)
        x = np.zeros(n)   # state estimate
        p = np.zeros(n)   # estimate variance
        x[0], p[0] = z[0], 1.0
        for k in range(1, n):
            # predict
            x_prior, p_prior = x[k - 1], p[k - 1] + self.q
            # update
            gain = p_prior / (p_prior + self.r)
            x[k] = x_prior + gain * (z[k] - x_prior)
            p[k] = (1 - gain) * p_prior
        return pd.Series(x, index=series.index, name="kalman_level")


class KalmanBeta:
    """Time-varying beta of asset returns vs market returns."""

    def __init__(self, process_var: float = 1e-6, measure_var: float = 1e-4):
        self.q = process_var
        self.r = measure_var

    def filter(self, asset_ret: pd.Series, market_ret: pd.Series) -> pd.Series:
        df = pd.concat([asset_ret, market_ret], axis=1).dropna()
        y = df.iloc[:, 0].to_numpy(dtype=float)
        x = df.iloc[:, 1].to_numpy(dtype=float)
        n = len(y)
        beta = np.zeros(n)
        p = 1.0
        b = 1.0
        for k in range(n):
            p_prior = p + self.q
            denom = x[k] * p_prior * x[k] + self.r
            gain = p_prior * x[k] / denom if denom != 0 else 0.0
            b = b + gain * (y[k] - x[k] * b)
            p = (1 - gain * x[k]) * p_prior
            beta[k] = b
        return pd.Series(beta, index=df.index, name="kalman_beta")
