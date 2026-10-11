"""PyTorch port of the Zhong, Mueller & Wang (2022) partially linear Cox
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
ARCH = dict(hidden_layers=(128,) * 6, activation="relu", dropout=0.0)
LEARNING_RATE = 1e-3
MAX_EPOCHS = 5000
PATIENCE = 35
VAL_FRACTION = 0.33

PROJ_ARCH = dict(hidden_layers=(64, 64), activation="relu", dropout=0.0)
PROJ_LR = 1e-3
PROJ_BATCH = 2048
PROJ_MAX_EPOCHS = 300
PROJ_PATIENCE = 30
PROJ_K_FOLDS = 5

FIELDS = ["rep_id", "theta_hat", "se", "n_epochs_run", "event_rate", "obs_cens", "fit_time_sec"]


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


def run_one_rep(rep_id, cfg, calibrated_c, device, base_seed):
    data_seed = base_seed + rep_id
    t0 = time.perf_counter()

    data = simulate_dataset(cfg, seed=data_seed, calibrated_c=calibrated_c)
    T_raw, delta_raw, X_raw, Z_raw = data["T"], data["delta"], data["X"], data["Z"]
    T, delta, X, Z = validate_survival_inputs(T_raw, delta_raw, X_raw, Z_raw)
    T, delta, tau = truncate_at_tau(T, delta, None)

    x_scaler = Standardizer()
    X_scaled = x_scaler.fit_transform(X).astype(np.float32)
    z_scaler = Standardizer()
    Z_scaled = z_scaler.fit_transform(Z).astype(np.float32)

    n = X_scaled.shape[0]

    # ---- theta warm-start: standard Cox PH fit on Z alone (matches
    # FlexiHaz's own theta_init="cox" convention -- Z-only, not Z+X) ----
    cox_df = pd.DataFrame(np.concatenate([Z, delta.reshape(-1, 1)], axis=1), columns=["Z0", "event"])
    cox_df["duration"] = T
    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col="duration", event_col="event", show_progress=False)
    theta_init_raw = float(cph.params_["Z0"])
    theta_init_scaled = theta_init_raw * float(z_scaler.scale_.reshape(-1)[0])

    dummy = FlexiHazardModel(validation_fraction=VAL_FRACTION, device=device, seed=data_seed, verbose=0)
    train_idx, val_idx = dummy._train_val_indices(np.arange(n))

    X_t = torch.as_tensor(X_scaled, dtype=torch.float32, device=device)
    Z_t = torch.as_tensor(Z_scaled, dtype=torch.float32, device=device)
    T_t = torch.as_tensor(T, dtype=torch.float32, device=device)
    delta_t = torch.as_tensor(delta, dtype=torch.float32, device=device)

    set_seed(INIT_SEED)
    model = CoxPHDNN(X_scaled.shape[1], Z_scaled.shape[1], **ARCH).to(device)
    with torch.no_grad():
        model.beta.weight.fill_(theta_init_scaled)

    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    tr_idx_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    va_idx_t = torch.as_tensor(val_idx, dtype=torch.long, device=device)

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    wait = 0
    epochs_run = 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        opt.zero_grad(set_to_none=True)
        chi_tr, _ = model(X_t[tr_idx_t], Z_t[tr_idx_t])
        loss = breslow_loss(chi_tr, T_t[tr_idx_t], delta_t[tr_idx_t])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            chi_va, _ = model(X_t[va_idx_t], Z_t[va_idx_t])
            val_loss = float(breslow_loss(chi_va, T_t[va_idx_t], delta_t[va_idx_t]))

        epochs_run = epoch + 1
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    model.load_state_dict(best_state)
    theta_hat_scaled = model.theta()[0]
    scale = float(z_scaler.scale_.reshape(-1)[0])
    theta_hat = theta_hat_scaled / scale

    # ---- SE: K-fold cross-fitted Z~(X,T) residuals on TRAIN-subject events,
    # using the ADDITIVE g_X(X)+g_T(t) net (no X-T interaction, matching
    # Zhong et al.'s build_mse_separate), then Var(theta_hat_scaled) =
    # inv(r_oof^T r_oof) (z_dim=1 here, so this is just 1/sum(r_oof**2)) ----
    train_event_idx = train_idx[delta[train_idx] == 1]
    X_ev = X_scaled[train_event_idx]
    Z_ev = Z_scaled[train_event_idx].reshape(-1)
    T_ev = (T[train_event_idx] / tau).astype(np.float32).reshape(-1, 1)
    n_ev = len(train_event_idx)

    rng = np.random.default_rng(data_seed + 777)
    fold_ids = rng.integers(0, PROJ_K_FOLDS, size=n_ev)
    r_oof = np.zeros(n_ev, dtype=np.float64)
    for k in range(PROJ_K_FOLDS):
        te_mask = fold_ids == k
        tr_mask = ~te_mask
        if te_mask.sum() == 0 or tr_mask.sum() < 10:
            continue
        n_tr_fold = tr_mask.sum()
        perm = rng.permutation(n_tr_fold)
        n_inner_val = max(1, int(0.2 * n_tr_fold))
        tr_idx_fold = np.where(tr_mask)[0][perm[n_inner_val:]]
        va_idx_fold = np.where(tr_mask)[0][perm[:n_inner_val]]
        te_idx_fold = np.where(te_mask)[0]

        set_seed(INIT_SEED + k)
        proj = AdditiveProjectionNet(X_ev.shape[1], **PROJ_ARCH)
        proj = _fit_projection_net(
            proj, X_ev[tr_idx_fold], T_ev[tr_idx_fold], Z_ev[tr_idx_fold],
            X_ev[va_idx_fold], T_ev[va_idx_fold], Z_ev[va_idx_fold],
            device=device, lr=PROJ_LR, weight_decay=0.0, batch_size=PROJ_BATCH,
            max_epochs=PROJ_MAX_EPOCHS, patience=PROJ_PATIENCE,
        )
        pred_te = _predict_projection(proj, X_ev[te_idx_fold], T_ev[te_idx_fold],
                                       device=device, batch_size=PROJ_BATCH)
        r_oof[te_idx_fold] = Z_ev[te_idx_fold] - pred_te.reshape(-1)

    var_scaled = 1.0 / max(float(np.sum(r_oof ** 2)), 1e-12)
    se = float(np.sqrt(var_scaled)) / scale

    fit_time = time.perf_counter() - t0
    event_rate = float(np.mean(delta))
    return dict(
        theta_hat=float(theta_hat), se=se, n_epochs_run=epochs_run,
        event_rate=event_rate, obs_cens=1.0 - event_rate, fit_time_sec=fit_time,
    )


def already_done(out_csv, rep_id):
    if not out_csv.exists():
        return False
    with open(out_csv) as f:
        return any(int(row["rep_id"]) == rep_id for row in csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--censoring-rate", type=float, required=True)
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--n-reps", type=int, default=200)
    ap.add_argument("--base-seed", type=int, default=BASE_SEED)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = DGMConfig(gamma=args.gamma, rho=args.rho, censoring_rate=args.censoring_rate, n=args.n)
    tag = f"gamma{cfg.gamma}_rho{cfg.rho}_cens{int(cfg.censoring_rate*100)}"
    out_csv = Path(args.out) if args.out else HERE / f"zhong_ph_{tag}.csv"

    calibrated_c = calibrate_censoring_c(gamma=cfg.gamma, rho=cfg.rho, target_rate=cfg.censoring_rate)
    theta_true = float(np.log(1.5))
    print(f"config: gamma={cfg.gamma} rho={cfg.rho} censoring_rate={cfg.censoring_rate} n={cfg.n} "
          f"n_reps={args.n_reps} calibrated_c={calibrated_c:.4f}", flush=True)
    print(f"theta_true = {theta_true:.4f}\n", flush=True)

    write_header = not out_csv.exists()
    with open(out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
            f.flush()

        for rep_id in range(args.n_reps):
            if already_done(out_csv, rep_id):
                print(f"rep={rep_id:03d} already done, skipping", flush=True)
                continue
            try:
                result = run_one_rep(rep_id, cfg, calibrated_c, device, args.base_seed)
            except Exception as exc:  # noqa: BLE001
                print(f"rep={rep_id:03d} FAILED: {type(exc).__name__}: {exc}", flush=True)
                torch.cuda.empty_cache()
                continue
            row = dict(rep_id=rep_id, **result)
            writer.writerow(row)
            f.flush()
            bias_pct = 100 * (result["theta_hat"] - theta_true) / theta_true
            print(
                f"rep={rep_id:03d} theta_hat={result['theta_hat']:.4f} bias%={bias_pct:+.2f} "
                f"se={result['se']:.4f} epochs={result['n_epochs_run']} fit_time={result['fit_time_sec']:.1f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
