from __future__ import annotations

import numpy as np
from .exceptions import InputValidationError


def _as_2d_float(name: str, x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise InputValidationError(f"{name} must be a 1D or 2D numeric array.")
    if not np.all(np.isfinite(arr)):
        raise InputValidationError(f"{name} contains NaN or infinite values.")
    return arr


def validate_survival_inputs(T, delta, X, Z=None):
    T = np.asarray(T, dtype=np.float64).reshape(-1)
    
    delta_raw = np.asarray(delta).reshape(-1)
    if not np.all(np.isin(delta_raw, [0, 1, 0.0, 1.0, False, True])):
        raise InputValidationError("delta must contain only 0/1 censoring indicators.")
    delta = delta_raw.astype(np.int64)
    
    X = _as_2d_float("X", X)
    if T.ndim != 1 or delta.ndim != 1:
        raise InputValidationError("T and delta must be one-dimensional arrays.")
    if T.shape[0] != delta.shape[0] or T.shape[0] != X.shape[0]:
        raise InputValidationError("T, delta, and X must have the same number of rows.")
    if T.shape[0] < 2:
        raise InputValidationError("At least two observations are required.")
    if not np.all(np.isfinite(T)) or np.any(T < 0):
        raise InputValidationError("T must contain finite nonnegative times.")
    if not set(np.unique(delta)).issubset({0, 1}):
        raise InputValidationError("delta must contain only 0/1 censoring indicators.")
    if np.sum(delta) == 0:
        raise InputValidationError("At least one observed event is required.")
    Z_arr = None
    if Z is not None:
        Z_arr = _as_2d_float("Z", Z)
        if Z_arr.shape[0] != T.shape[0]:
            raise InputValidationError("Z must have the same number of rows as T.")
    return T, delta, X, Z_arr


def truncate_at_tau(T: np.ndarray, delta: np.ndarray, tau: float | None):
    if tau is None:
        tau = float(np.max(T))
    tau = float(tau)
    if not np.isfinite(tau) or tau <= 0:
        raise InputValidationError("tau must be a positive finite number.")
    original_T = T.copy()
    T_trunc = np.minimum(T, tau)
    delta_trunc = delta.copy().astype(np.int64)
    delta_trunc[original_T > tau] = 0
    return T_trunc, delta_trunc, tau


class Standardizer:
    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray):
        self.mean_ = np.mean(X, axis=0, keepdims=True)
        self.scale_ = np.std(X, axis=0, keepdims=True)
        self.scale_[self.scale_ < 1e-12] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer is not fitted.")
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
