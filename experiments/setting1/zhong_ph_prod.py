"""Zhong, Mueller & Wang (2022) proportional-hazards DNN: model (CoxPHDNN), additive
residual network for the SE, Breslow partial-likelihood loss and the exact profile-likelihood
refinement of theta with g fixed. Used by tune_zhong_grid.py and run_zhong_reps.py.

Original notes:
PyTorch port of the Zhong, Mueller & Wang (2022) partially linear Cox
model -- proportional hazards h(t|X,Z) = h0(t)*exp(theta^T Z + g(X)), with
g(X) having NO time dependence (that's what makes it proportional, unlike
FlexiHaz's g(t,X)). Estimated via Breslow (no-ties) Cox partial likelihood,
full-batch, sorted-cumsum -- no piecewise-exponential time-grid expansion
needed, so each rep processes n rows instead of n*grid_size rows.

Ported from the reference implementation at ~/Downloads/PH_model.ipynb,
adapted to:
- run on OUR DGM (dgm_additional.py) instead of the notebook's own toy
  simulator, for direct comparability with the FlexiHaz Method-1 results on
  the same 6-setting grid (gamma x rho, n=8000, 30% censoring, 200 reps).
- standardize X/Z like the rest of our pipeline (theta_hat rescaled back to
  original Z units at the end).
- reuse flexihaz's existing _fit_projection_net/_predict_projection training
  utilities for the SE step, but with a NEW AdditiveProjectionNet (below)
  instead of flexihaz's own ProjectionNet: Zhong et al.'s residual network
  for E[Z|X,T] is g_X(X) + g_T(T), two independent branches summed with NO
  X-T interaction (their build_mse_separate), whereas FlexiHaz's own
  ProjectionNet concatenates (t,X) into one joint network and so DOES allow
  interaction -- this is a real, deliberate modeling difference between the
  two papers' constructions, not just an implementation detail, so it's kept
  even though this model's own hazard g(X) has no time dependence: the
  residual/orthogonalization step still conditions on T because the Cox
  partial-likelihood risk set at each event time depends on T.
- single init per rep (K=1 -- no multi-init/bootstrap-of-3).

Variance estimator: Var(theta_hat) = inv(r_oof^T r_oof), r_oof = K-fold
cross-fitted Z-minus-predicted-Z residuals on TRAIN-subject events only,
matching the notebook's crossfit_event_residuals + inv(r.T @ r) formula
(the model's own asymptotic theory), NOT FlexiHaz's estimate_theta_variance
formula (which is derived for the expanded piecewise-exponential likelihood
and isn't the right plug-in estimator for a Cox partial-likelihood model).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from lifelines import CoxPHFitter
from scipy.optimize import minimize
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402,F401  (adds FLEXIHAZ_SOURCE_DIR to sys.path)

from flexihaz import FlexiHazardModel  # noqa: E402
from flexihaz.networks import MLP  # noqa: E402
from flexihaz.preprocessing import Standardizer, validate_survival_inputs, truncate_at_tau  # noqa: E402
from flexihaz.inference import _fit_projection_net, _predict_projection  # noqa: E402
from flexihaz.utils import set_seed  # noqa: E402

from dgm_additional import DGMConfig, calibrate_censoring_c, simulate_dataset  # noqa: E402

BASE_SEED = 221100
INIT_SEED = 910_000
MAX_EPOCHS = 5000
PATIENCE = 35
VAL_FRACTION = 0.33

PROJ_ARCH = dict(hidden_layers=(64, 64), activation="relu", dropout=0.0)
PROJ_LR = 1e-3
PROJ_BATCH = 2048
PROJ_MAX_EPOCHS = 300
PROJ_PATIENCE = 30
PROJ_K_FOLDS = 5



class CoxPHDNN(nn.Module):
    """chi = g(X) + theta^T Z, g has no time input (proportional)."""

    def __init__(self, x_dim, z_dim, hidden_layers, activation, dropout):
        super().__init__()
        self.g_net = MLP(input_dim=x_dim, hidden_layers=hidden_layers, activation=activation, dropout=dropout)
        self.beta = nn.Linear(z_dim, 1, bias=False)

    def forward(self, X, Z):
        g = self.g_net(X).squeeze(-1)
        beta_z = self.beta(Z).squeeze(-1)
        return g + beta_z, g

    def theta(self):
        return self.beta.weight.detach().cpu().numpy().reshape(-1)


class AdditiveProjectionNet(nn.Module):
    """g_X(X) + g_T(t) -- two independent branches summed, NO interaction
    between X and T. Matches Zhong et al.'s build_mse_separate construction
    for the E[Z|X,T] residual/orthogonalization step, as opposed to
    FlexiHaz's own ProjectionNet which concatenates (t,X) into one joint
    network and so DOES allow X-T interaction."""

    def __init__(self, x_dim, hidden_layers=(64, 64), activation="relu", dropout=0.0):
        super().__init__()
        self.gx_net = MLP(input_dim=x_dim, hidden_layers=hidden_layers, activation=activation, dropout=dropout)
        self.gt_net = MLP(input_dim=1, hidden_layers=hidden_layers, activation=activation, dropout=dropout)

    def forward(self, t, X):
        return self.gx_net(X) + self.gt_net(t)


def breslow_loss(chi, T, delta, clip=80.0, eps=1e-12):
    """Negative Cox partial log-likelihood, no ties, full-batch."""
    order = torch.argsort(T, descending=True)
    chi_sorted = torch.clamp(chi[order], -clip, clip)
    delta_sorted = delta[order]
    cumexp = torch.cumsum(torch.exp(chi_sorted), dim=0)
    log_denom = torch.log(torch.clamp(cumexp, min=eps))
    contrib = delta_sorted * (chi_sorted - log_denom)
    n_events = torch.clamp(delta_sorted.sum(), min=1.0)
    return -contrib.sum() / n_events


def refine_theta_zhong(g_fixed, Z, T, delta, theta0, clip=80.0, eps=1e-300):
    """Exact profile-likelihood correction of theta (Breslow, no ties),
    holding g FIXED as an offset -- the Cox-partial-likelihood analog of
    FlexiHaz's own refine_theta (run_reps.py), which does the same thing for
    the piecewise-exponential likelihood. Both exist for the same reason:
    joint (theta,g) optimization by SGD/Adam does not exactly satisfy the
    theta score equation at convergence, especially once g is flexible
    enough to be correlated with Z -- an exact re-solve for theta given the
    fitted g removes that residual finite-sample bias, which multi-init
    restarts cannot (it's a property of the optimum, not of which basin the
    optimizer lands in).

    g_fixed, Z, T, delta: 1D numpy arrays, same length (n subjects).
    theta0: scalar warm start (e.g. the joint-trained theta).
    Returns the refined scalar theta (same scale as theta0/Z, i.e. call this
    on STANDARDIZED Z to match the rest of this module's "_scaled" convention).
    """
    order = np.argsort(-T)
    g_s = g_fixed[order].astype(np.float64)
    Z_s = Z[order].astype(np.float64)
    d_s = delta[order].astype(np.float64)
    n_events = max(float(d_s.sum()), 1.0)

    def obj_grad(theta_arr):
        theta = float(theta_arr[0])
        eta = np.clip(g_s + theta * Z_s, -clip, clip)
        w = np.exp(eta)
        cumw = np.maximum(np.cumsum(w), eps)
        cumZw = np.cumsum(Z_s * w)
        ll = np.sum(d_s * (eta - np.log(cumw)))
        grad = np.sum(d_s * (Z_s - cumZw / cumw))
        return -ll / n_events, np.array([-grad / n_events])

    res = minimize(lambda th: obj_grad(th)[0], x0=[theta0], jac=lambda th: obj_grad(th)[1], method="L-BFGS-B")
    return float(res.x[0])
