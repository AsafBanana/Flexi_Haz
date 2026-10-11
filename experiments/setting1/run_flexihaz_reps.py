"""FLEXI-Haz replications for setting 1.

Per replication: simulate the data (seed BASE_SEED + rep), standardize on the training
subjects, build the piecewise-exponential rows, compute the 5-fold cross-fitted residuals
Z - E[Z | X, t] ONCE (training subjects; they give both the selection score and the SE),
then train --n-init random restarts of the main net with warm-started learning-rate phases
(--lr-phases; theta branch at --theta-lr-mult times the phase rate), exact L-BFGS refinement
of theta on the training rows after every phase, and report the restart whose efficient
(orthogonalised) score is closest to zero, with SE = (R'R)^-1 from the shared residuals.
The one-step and root efficient-score corrections and the val-loss-selected restart are
recorded as diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402,F401  (adds FLEXIHAZ_SOURCE_DIR to sys.path)

from flexihaz import FlexiHazardModel  # noqa: E402
from flexihaz.data import build_time_grid, make_expanded_data  # noqa: E402
from flexihaz.networks import FlexiHazardNet  # noqa: E402
from flexihaz.preprocessing import Standardizer, validate_survival_inputs, truncate_at_tau  # noqa: E402
from flexihaz.utils import set_seed  # noqa: E402

from dgm_additional import DGMConfig, calibrate_censoring_c, paper_scales, simulate_dataset  # noqa: E402
from common import refine_theta, LR_PHASES, BASE_SEED, PROJECTION_LR  # noqa: E402
import crossfit_residuals as _cfr  # noqa: E402
from crossfit_residuals import compute_crossfitted_residuals, residuals_on_heldout_rows  # noqa: E402

BATCH_SIZE = 100_000
# if set, overrides BATCH_SIZE per rep: batch = ceil(#expanded TRAIN rows / this), so the main net
# sees the same number of minibatches per epoch at every n ((n/8000)^2 scaling gives ~213 at 100000)
MAIN_BATCHES_PER_EPOCH = None
LR_SCHEDULE = list(LR_PHASES)  # overridable via --lr-phases
# --theta-lr-mult: learning-rate multiplier for the theta (linear) branch
THETA_LR_MULT = 1.0


def make_init_seeds(n_init):
    return [910_000 + i for i in range(n_init)]


def make_fields(n_init):
    fields = ["rep_id", "trial_selected", "theta_hat", "se", "score_selected",
               "trial_selected_valloss", "theta_hat_valloss"]
    fields += [f"theta_trial{k}" for k in range(n_init)]
    fields += [f"score_trial{k}" for k in range(n_init)]
    fields += [f"val_loss_trial{k}" for k in range(n_init)]
    fields += ["trial_selected_valscore", "theta_hat_valscore"]
    # efficient-score corrections (g fixed, theta solved): one Newton step from the restart's theta, and the
    # converged root of S(theta)=0; "_sel" = selected restart, "_mean" = average over the K restarts
    fields += ["theta_hat_onestep_sel", "theta_hat_onestep_mean", "theta_hat_root_sel", "theta_hat_root_mean"]
    fields += [f"theta_onestep_trial{k}" for k in range(n_init)]
    fields += [f"theta_root_trial{k}" for k in range(n_init)]
    fields += [f"root_iters_trial{k}" for k in range(n_init)]
    fields += [f"score_val_trial{k}" for k in range(n_init)]
    fields += ["main_batch_size", "n_train_subj", "lr_start", "auto_lr_at_min", "fit_time_sec"]
    return fields


def build_trial_model(hidden_layers, init_seed, X_scaled, Z_scaled, Z_raw, T, delta, tau, time_grid,
                      z_scaler, device, X_raw, train_subj, lr0):
    model = FlexiHazardModel(
        hidden_layers=hidden_layers, activation="relu", dropout=0.0, weight_decay=0.0, grid_size=2,
        learning_rate=lr0, batch_size=BATCH_SIZE, tau=tau, max_epochs=500, patience=30,
        theta_init="cox", refine_theta=True, compute_theta_variance=False,
        device=device, seed=init_seed, verbose=0,
    )
    set_seed(init_seed)
    net = FlexiHazardNet(
        x_dim=X_scaled.shape[1], z_dim=Z_scaled.shape[1],
        hidden_layers=hidden_layers, activation="relu", dropout=0.0,
    ).to(device)

    model.time_grid = time_grid
    model.time_scale_ = float(tau) if model.scale_time else 1.0
    model.has_linear_ = True
    model.x_dim_ = X_scaled.shape[1]
    model.z_dim_ = Z_scaled.shape[1]
    model.net_ = net
    model.z_scaler_ = z_scaler
    model.theta_lr_multiplier = THETA_LR_MULT
    # Cox warm start on TRAIN subjects only (validation subjects must not inform theta at all)
    model._initialize_theta_from_cox(T[train_subj], delta[train_subj], X_raw[train_subj], Z_raw[train_subj],
                                     Z_scaled[train_subj])
    return model


def fit_one_trial(hidden_layers, init_seed, X_scaled, Z_scaled, Z_raw, T, delta, tau, time_grid,
                   expanded, train_idx, val_idx, z_scaler, device, X_raw, train_subj, lr_schedule,
                   resid_all=None, info_scaled=None):
    model = build_trial_model(hidden_layers, init_seed, X_scaled, Z_scaled, Z_raw, T, delta, tau, time_grid,
                              z_scaler, device, X_raw, train_subj, lr_schedule[0])

    # refine_theta must see TRAIN-split rows only -- val rows are used solely
    # for early stopping inside _fit_expanded and must never enter the point
    # estimate (its SE, computed separately, is already train-subject-only).
    train_expanded = dict(
        X=expanded["X"][train_idx], Z=expanded["Z"][train_idx],
        t=expanded["t"][train_idx], event=expanded["event"][train_idx],
        length=expanded["length"][train_idx],
    )

    # best_val is carried across phases (not reset to +inf each phase), so a
    # later (lower-LR) phase's early stopping only ever accepts a model
    # better than the WHOLE run's best so far -- matches the fix validated
    # in the Zhong best_val-carryover experiment.
    carried_best_val = None
    for lr in lr_schedule:
        model.learning_rate = lr
        carried_best_val = model._fit_expanded(expanded, train_idx, val_idx, init_best_val=carried_best_val)
        model.theta_unrefined_ = model.net_.theta().copy()
        refine_theta(model, train_expanded)

    return model, carried_best_val


def score_rows(model, expanded, row_idx):
    """Pull the row-level pieces of the efficient score once per (restart, row set): g is the
    restart's own fitted g(t,X); theta enters only through eta = g + theta*Z."""
    X_all = expanded["X"][row_idx].astype(np.float32)
    t_raw_all = expanded["t"][row_idx].reshape(-1).astype(np.float64)
    g_all = model._predict_g_on_pairs(X_all, t_raw_all, batch_size=model.batch_size).reshape(-1).astype(np.float64)
    return dict(
        g=g_all,
        Z=expanded["Z"][row_idx].reshape(-1).astype(np.float64),
        event=expanded["event"][row_idx].reshape(-1).astype(np.float64),
        length=expanded["length"][row_idx].reshape(-1).astype(np.float64),
    )


def efficient_score(rows, theta_scaled, resid):
    """S_code(theta) = sum_rows resid * (length*exp(eta) - event)  [= -dloglik/dtheta along the
    efficient direction], and its derivative dS_code/dtheta = sum_rows resid * Z * length*exp(eta)."""
    eta = np.clip(rows["g"] + theta_scaled * rows["Z"], -80.0, 80.0)
    haz = rows["length"] * np.exp(eta)
    return float(np.sum(resid * (haz - rows["event"]))), float(np.sum(resid * rows["Z"] * haz))


def score_from_shared_residuals(model, theta_hat_scaled, resid_all, expanded, train_idx):
    """Score = sum over ALL expanded train rows of resid_i * (length_i*exp(eta_i) - event_i),
    using the trial's OWN fitted g/theta but the SHARED cross-fitted residuals.
    (train_idx may also be the VALIDATION rows, with resid_all from residuals_on_heldout_rows,
    giving a fully out-of-sample score: neither g/theta nor the projection saw those subjects.)"""
    return efficient_score(score_rows(model, expanded, train_idx), theta_hat_scaled, resid_all)[0]


def solve_efficient_score(rows, theta0_scaled, resid, info_scaled, max_iter=20, tol=1e-7):
    """With g fixed, (a) one Newton step theta0 - S/I using the cross-fitted information I (the
    one-step efficient estimator), and (b) Newton iteration with the exact dS/dtheta until
    |step| < tol -- the root of the orthogonalized estimating equation S(theta) = 0.
    Returns (theta_onestep, theta_root, n_iter), all on the scaled-Z scale."""
    s0, _ = efficient_score(rows, theta0_scaled, resid)
    theta_onestep = theta0_scaled - s0 / info_scaled
    theta = theta0_scaled
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        s, ds = efficient_score(rows, theta, resid)
        if not np.isfinite(ds) or ds <= 0:          # fall back to the information if the curvature is off
            ds = info_scaled
        step = s / ds
        theta = theta - step
        if abs(step) < tol:
            break
    return float(theta_onestep), float(theta), n_iter


def run_one_rep(rep_id, cfg, calibrated_c, device, base_seed, hidden_layers, proj_hidden_layers,
                 n_init, init_seeds, dataset_dir=None):
    data_seed = base_seed + rep_id
    t0 = time.perf_counter()

    data = simulate_dataset(cfg, seed=data_seed, calibrated_c=calibrated_c)
    if dataset_dir is not None:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dataset_dir / f"rep_{rep_id:03d}_dataset.npz",
            X=data["X"], Z=data["Z"], T=data["T"], T_true=data["T_true"], C=data["C"],
            delta=data["delta"], seed=data_seed, calibrated_c=calibrated_c,
        )
    T_raw, delta_raw, X_raw, Z_raw = data["T"], data["delta"], data["X"], data["Z"]
    T, delta, X, Z = validate_survival_inputs(T_raw, delta_raw, X_raw, Z_raw)
    T, delta, tau = truncate_at_tau(T, delta, None)

    global BATCH_SIZE
    dummy = FlexiHazardModel(
        hidden_layers=(64, 64), activation="relu", dropout=0.0, weight_decay=0.0, grid_size=2,
        learning_rate=LR_SCHEDULE[0], batch_size=BATCH_SIZE, tau=tau, device=device, seed=data_seed, verbose=0,
    )
    dummy.time_scale_ = float(tau) if dummy.scale_time else 1.0
    # subject-level train/val split FIRST (same GroupShuffleSplit/seed as splitting the expanded rows
    # by subject_id, so the same subjects), so the scalers are fit on TRAIN subjects only
    train_subj, _ = dummy._train_val_indices(np.arange(len(T)))
    train_subj = np.sort(train_subj)

    x_scaler = Standardizer()
    x_scaler.fit_transform(X[train_subj])
    X_scaled = x_scaler.transform(X)
    z_scaler = Standardizer()
    z_scaler.fit_transform(Z[train_subj])
    Z_scaled = z_scaler.transform(Z)

    time_grid = build_time_grid(T, tau, 2, True)
    expanded = make_expanded_data(T, delta, X_scaled, Z_scaled, time_grid)
    train_idx, val_idx = dummy._train_val_indices(expanded["subject_id"])
    assert np.array_equal(np.unique(expanded["subject_id"][train_idx]), train_subj), "subject split mismatch"

    if MAIN_BATCHES_PER_EPOCH is not None:
        BATCH_SIZE = max(1, int(np.ceil(len(train_idx) / MAIN_BATCHES_PER_EPOCH)))

    # ---- STEP 1: cross-fitted residuals, computed ONCE, before any main-net
    # training -- doesn't depend on theta/g at all ----
    resid_all, var_theta_hat_scaled, projs = compute_crossfitted_residuals(
        dummy, expanded, train_idx, T, delta, X_scaled, Z_scaled,
        proj_hidden_layers, PROJECTION_LR, seed=data_seed, device=device, n_splits=5, return_projs=True,
    )
    # residuals on VALIDATION subjects' rows -- no projection net ever trained on them
    resid_val = residuals_on_heldout_rows(projs, dummy, expanded, val_idx)

    scale = float(z_scaler.scale_.reshape(-1)[0])
    se = float(np.sqrt(max(var_theta_hat_scaled, 0.0))) / scale
    info_scaled = 1.0 / var_theta_hat_scaled                     # = sum of squared cross-fitted event residuals

    lr_schedule, auto_lr_at_min = list(LR_SCHEDULE), float("nan")

    # ---- STEP 2: train n_init restarts, score each with the SHARED residuals ----
    thetas, scores, val_losses, scores_val = [], [], [], []
    thetas_onestep, thetas_root, root_iters = [], [], []
    for k in range(n_init):
        model, best_val = fit_one_trial(hidden_layers, init_seeds[k], X_scaled, Z_scaled, Z, T, delta, tau,
                               time_grid, expanded, train_idx, val_idx, z_scaler, device, X, train_subj,
                               lr_schedule, resid_all, info_scaled)
        theta_hat_scaled = float(model.net_.theta()[0])
        theta_hat = theta_hat_scaled / scale

        rows_tr = score_rows(model, expanded, train_idx)
        score, _ = efficient_score(rows_tr, theta_hat_scaled, resid_all)
        scores_val.append(score_from_shared_residuals(model, theta_hat_scaled, resid_val, expanded, val_idx))
        t1, t_root, n_it = solve_efficient_score(rows_tr, theta_hat_scaled, resid_all, info_scaled)
        thetas_onestep.append(t1 / scale)
        thetas_root.append(t_root / scale)
        root_iters.append(n_it)
        del rows_tr
        thetas.append(theta_hat)
        scores.append(score)
        val_losses.append(best_val)
        torch.cuda.empty_cache()

    i_best = int(np.argmin(np.abs(scores)))
    theta_hat = thetas[i_best]

    i_best_val = int(np.argmin(val_losses))
    theta_hat_valloss = thetas[i_best_val]

    i_best_vs = int(np.argmin(np.abs(scores_val)))

    fit_time = time.perf_counter() - t0
    result = dict(
        trial_selected=i_best, theta_hat=float(theta_hat), se=se, score_selected=scores[i_best],
        trial_selected_valloss=i_best_val, theta_hat_valloss=float(theta_hat_valloss),
        trial_selected_valscore=i_best_vs, theta_hat_valscore=float(thetas[i_best_vs]),
        theta_hat_onestep_sel=float(thetas_onestep[i_best]), theta_hat_onestep_mean=float(np.mean(thetas_onestep)),
        theta_hat_root_sel=float(thetas_root[i_best]), theta_hat_root_mean=float(np.mean(thetas_root)),
        main_batch_size=BATCH_SIZE, n_train_subj=len(train_subj),
        lr_start=lr_schedule[0], auto_lr_at_min=auto_lr_at_min,
        fit_time_sec=fit_time,
    )
    for k in range(n_init):
        result[f"theta_trial{k}"] = thetas[k]
        result[f"score_trial{k}"] = scores[k]
        result[f"val_loss_trial{k}"] = val_losses[k]
        result[f"score_val_trial{k}"] = scores_val[k]
        result[f"theta_onestep_trial{k}"] = thetas_onestep[k]
        result[f"theta_root_trial{k}"] = thetas_root[k]
        result[f"root_iters_trial{k}"] = root_iters[k]
    return result


def already_done(out_csv, rep_id):
    if not out_csv.exists():
        return False
    with open(out_csv) as f:
        return any(int(row["rep_id"]) == rep_id for row in csv.DictReader(f))


def main():
    global BATCH_SIZE, MAIN_BATCHES_PER_EPOCH, LR_SCHEDULE
    global THETA_LR_MULT
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--censoring-rate", type=float, default=0.3)
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--k", type=int, default=3, choices=[0, 1, 2, 3],
                     help="0 = original additive setting, 1 = with interactions (I1/I2), "
                          "2 = extreme (squared terms -> cubed, interactions Xi*Xj -> Xi^2*Xj), "
                          "3 = pure interaction (a/b are ONLY Xi^2*Xj, no base terms, "
                          "scaled so PH/non-PH/linear all have comparable variance)")
    ap.add_argument("--nonlinear-scale", type=float, default=1.0)
    ap.add_argument("--ph-scale", type=float, default=None, help="c_b, multiplies ONLY b(X) (PH part); default: the paper constant for gamma")
    ap.add_argument("--nonph-scale", type=float, default=None, help="c_a, multiplies ONLY a(X) (non-PH part); default: the paper constant for gamma")
    ap.add_argument("--n-reps", type=int, default=200)
    ap.add_argument("--rep-start", type=int, default=0,
                     help="first rep_id to run (for splitting one setting's reps across multiple machines "
                          "while keeping data_seed=base_seed+rep_id globally unique per machine)")
    ap.add_argument("--n-init", type=int, default=3, help="number of multi-init restarts (best of n-init)")
    ap.add_argument("--base-seed", type=int, default=BASE_SEED)
    ap.add_argument("--config", type=str, required=True, help="path to tune_hyperparams_zhonggrid.py JSON output")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--proj-batch-size", type=int, default=_cfr.PROJECTION_TRAIN_BATCH,
                     help="minibatch size for the cross-fitting projection nets (training only)")
    ap.add_argument("--main-batches-per-epoch", type=int, default=None,
                     help="main-net batch = ceil(#expanded train rows / this); overrides --batch-size")
    ap.add_argument("--proj-batches-per-epoch", type=int, default=None,
                     help="projection-net batch = ceil(fold inner-train size / this); overrides --proj-batch-size")
    ap.add_argument("--lr-phases", type=str, default=None, help="comma-separated LR schedule, e.g. 1e-3,1e-4,1e-5,1e-6")
    ap.add_argument("--theta-lr-mult", type=float, default=1.0, help="learning-rate multiplier for the theta branch")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--save-datasets", action="store_true",
                     help="save each rep's simulated dataset (X,Z,T,delta,...) as an .npz for later inspection")
    args = ap.parse_args()

    BATCH_SIZE = args.batch_size
    _cfr.PROJECTION_TRAIN_BATCH = args.proj_batch_size
    _cfr.PROJECTION_BATCHES_PER_EPOCH = args.proj_batches_per_epoch
    MAIN_BATCHES_PER_EPOCH = args.main_batches_per_epoch
    if args.lr_phases:
        LR_SCHEDULE = [float(x) for x in args.lr_phases.split(",")]
    THETA_LR_MULT = args.theta_lr_mult

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = DGMConfig(gamma=args.gamma, rho=args.rho, censoring_rate=args.censoring_rate, n=args.n, k=args.k,
                     nonlinear_scale=args.nonlinear_scale, ph_scale=args.ph_scale, nonph_scale=args.nonph_scale)
    tag = f"gamma{cfg.gamma}_rho{cfg.rho}_cens{int(cfg.censoring_rate*100)}"
    if cfg.k:
        tag += f"_k{cfg.k}"
    tag += f"_n{cfg.n}"
    if cfg.nonlinear_scale != 1.0:
        tag += f"_scale{cfg.nonlinear_scale:g}"
    if dict(ph_scale=cfg.ph_scale, nonph_scale=cfg.nonph_scale) != paper_scales(cfg.gamma, cfg.k):
        tag += f"_ph{cfg.ph_scale:g}_nonph{cfg.nonph_scale:g}"
    if args.n_init != 3:
        tag += f"_ninit{args.n_init}"
    if args.rep_start:
        tag += f"_repstart{args.rep_start}"
    out_csv = Path(args.out) if args.out else HERE / "outputs" / f"flexihaz_{tag}.csv"
    dataset_dir = (out_csv.parent / f"{out_csv.stem}_datasets") if args.save_datasets else None

    init_seeds = make_init_seeds(args.n_init)
    fields = make_fields(args.n_init)

    with open(args.config) as f:
        tuning = json.load(f)
    hidden_layers = tuple(tuning["main_net"]["selected"]["hidden_layers"])
    proj_hidden_layers = tuple(tuning["projection_net"]["selected"]["hidden_layers"])

    calibrated_c = tuning.get("calibrated_c") or calibrate_censoring_c(
        gamma=cfg.gamma, rho=cfg.rho, target_rate=cfg.censoring_rate, k=cfg.k,
        nonlinear_scale=cfg.nonlinear_scale, ph_scale=cfg.ph_scale, nonph_scale=cfg.nonph_scale,
    )
    theta_true = float(np.log(1.5))
    print(f"config: gamma={cfg.gamma} rho={cfg.rho} censoring_rate={cfg.censoring_rate} n={cfg.n} k={cfg.k} "
          f"nonlinear_scale={cfg.nonlinear_scale} ph_scale={cfg.ph_scale} nonph_scale={cfg.nonph_scale} "
          f"n_reps={args.n_reps} n_init={args.n_init} "
          f"calibrated_c={calibrated_c:.4f} batch_size={BATCH_SIZE} proj_batch_size={_cfr.PROJECTION_TRAIN_BATCH} "
          f"main_batches_per_epoch={MAIN_BATCHES_PER_EPOCH} proj_batches_per_epoch={_cfr.PROJECTION_BATCHES_PER_EPOCH} "
          f"lr_phases={LR_SCHEDULE}"
          + f" theta_lr_mult={THETA_LR_MULT}", flush=True)
    print(f"main net arch: {hidden_layers}  projection net arch: {proj_hidden_layers}", flush=True)
    if dataset_dir is not None:
        print(f"saving per-rep datasets to: {dataset_dir}", flush=True)
    print(f"theta_true = {theta_true:.4f}\n", flush=True)

    write_header = not out_csv.exists()
    with open(out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
            f.flush()

        for rep_id in range(args.rep_start, args.rep_start + args.n_reps):
            if already_done(out_csv, rep_id):
                print(f"rep={rep_id:03d} already done, skipping", flush=True)
                continue
            try:
                result = run_one_rep(rep_id, cfg, calibrated_c, device, args.base_seed,
                                      hidden_layers, proj_hidden_layers,
                                      args.n_init, init_seeds, dataset_dir)
            except Exception as exc:  # noqa: BLE001
                print(f"rep={rep_id:03d} FAILED: {type(exc).__name__}: {exc}", flush=True)
                torch.cuda.empty_cache()
                continue
            row = dict(rep_id=rep_id, **result)
            writer.writerow(row)
            f.flush()
            bias_pct = 100 * (result["theta_hat"] - theta_true) / theta_true
            print(
                f"rep={rep_id:03d} trial_sel={result['trial_selected']} theta_hat={result['theta_hat']:.4f} "
                f"bias%={bias_pct:+.2f} onestep_mean={result['theta_hat_onestep_mean']:.4f} "
                f"root_mean={result['theta_hat_root_mean']:.4f} se={result['se']:.4f} lr_start={result['lr_start']:.2e} "
                f"fit_time={result['fit_time_sec']:.1f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
