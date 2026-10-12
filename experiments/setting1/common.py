"""Shared constants and helpers for the setting-1 (six-setting) simulation.

Importing this module first also makes the `flexihaz` package importable from a
source checkout: set FLEXIHAZ_SOURCE_DIR=<repo>/src if the package is not installed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

REFINE_LBFGS_OPTIONS = {"gtol": 1e-10, "ftol": 1e-14, "maxiter": 1000}

_src = os.environ.get("FLEXIHAZ_SOURCE_DIR")
if _src and _src not in sys.path:
    sys.path.insert(0, _src)

HERE = Path(__file__).resolve().parent

BASE_SEED = 221100          # replication r uses data seed BASE_SEED + r
TUNING_SEED = 500_000       # independent tuning dataset
THETA_TRUE = float(np.log(1.5))
LR_PHASES = [1e-3, 1e-4, 1e-5]   # warm-started Adam phases, both sample sizes
PROJECTION_LR = 1e-3


def theta_original_scale(model):
    theta_std = model.net_.theta().astype(np.float64)
    return theta_std / model.z_scaler_.scale_.reshape(-1)


def refine_theta(model, expanded):
    X = expanded["X"]
    Z = expanded["Z"].astype(np.float64)
    t = expanded["t"]
    event = expanded["event"].astype(np.float64)
    length = expanded["length"].astype(np.float64)
    g = model._predict_g_on_pairs(X, t, batch_size=model.batch_size).reshape(-1, 1)
    theta0 = model.net_.theta().astype(np.float64)

    # normalised per EVENT, not per expanded row: with millions of expanded rows a
    # per-row mean has gradients of order 1e-5 and L-BFGS-B's default tolerances
    # stop before moving theta at all
    n_events = max(float(np.sum(event)), 1.0)

    def obj(theta):
        eta = g + Z @ theta.reshape(-1, 1)
        eta = np.clip(eta, -80.0, 80.0)
        return float(np.sum(length * np.exp(eta) - event * eta) / n_events)

    def grad(theta):
        eta = g + Z @ theta.reshape(-1, 1)
        eta = np.clip(eta, -80.0, 80.0)
        gr = Z * (length * np.exp(eta) - event)
        return np.sum(gr, axis=0) / n_events

    res = minimize(obj, theta0, jac=grad, method="L-BFGS-B", options=REFINE_LBFGS_OPTIONS)
    if res.success:
        with torch.no_grad():
            model.net_.linear.weight.copy_(
                torch.as_tensor(res.x.reshape(1, -1), dtype=torch.float32, device=model.device)
            )
