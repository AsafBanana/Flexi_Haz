"""Zhong, Mueller & Wang (2022) proportional DNN model, K=5 multi-init restarts
selected by lowest VALIDATION LOSS (Breslow partial-likelihood loss on the
held-out split), instead of zhong_ph_multi_init_tuned.py's |efficient score|
selection.

Why: zhong_ph_multi_init_tuned.py's compute_score() takes NO model/theta/g
argument at all -- it is a function of (X_ev, T_ev, Z_ev, seed) only, so the
"selection score" for a given restart is actually independent of that
restart's fitted model. Trial selection there is effectively seed-driven
noise, not a real selection signal. This script replaces that broken
mechanism with the simplest well-defined alternative: pick the restart with
the lowest validation loss from its own training run (a real, per-trial
signal, unlike the broken score).

Architecture and learning rate come from zhong_tune_hyperparams.py's single
grid-search JSON (run ONCE per setting on n=8000, unchanged). For EACH of the
200 replications, 5 independent restarts (different init seed) are trained
with the same 3-phase warm-started LR schedule and refine_theta_zhong
correction as zhong_ph_multi_init_tuned.py; the restart with lowest final-
phase validation loss is selected. The SE uses the SAME K-fold cross-fitted
additive-projection-net construction as zhong_ph_prod.py / the score-based
script, applied to the selected trial.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402,F401  (adds FLEXIHAZ_SOURCE_DIR to sys.path)

from flexihaz import FlexiHazardModel  # noqa: E402
from flexihaz.preprocessing import Standardizer, validate_survival_inputs, truncate_at_tau  # noqa: E402
from flexihaz.inference import _fit_projection_net, _predict_projection  # noqa: E402
from flexihaz.utils import set_seed  # noqa: E402

from dgm_additional import DGMConfig, calibrate_censoring_c, paper_scales, simulate_dataset  # noqa: E402
from zhong_ph_prod import (  # noqa: E402
    CoxPHDNN, AdditiveProjectionNet, breslow_loss, refine_theta_zhong,
    BASE_SEED, MAX_EPOCHS, PATIENCE, VAL_FRACTION,
    PROJ_BATCH, PROJ_MAX_EPOCHS, PROJ_PATIENCE, PROJ_K_FOLDS,
)

K = 5
# projection-net TRAINING batch (prediction still uses PROJ_BATCH); overridable
# via --proj-batch-size so it can scale with n like the FlexiHaz n-sweeps.
# The main CoxPHDNN is full-batch (Breslow loss over the whole train split),
# so there is no main-net batch to scale.
PROJ_TRAIN_BATCH = PROJ_BATCH
INIT_SEEDS = [910_000, 910_001, 910_002, 910_003, 910_004]

FIELDS = [
    "rep_id", "trial_selected", "theta_hat_unrefined", "theta_hat_refined", "se", "val_loss_selected",
    "theta_trial0_unrefined", "theta_trial1_unrefined", "theta_trial2_unrefined",
    "theta_trial3_unrefined", "theta_trial4_unrefined",
    "theta_trial0_refined", "theta_trial1_refined", "theta_trial2_refined",
    "theta_trial3_refined", "theta_trial4_refined",
    "val_loss_trial0", "val_loss_trial1", "val_loss_trial2", "val_loss_trial3", "val_loss_trial4",
    "fit_time_sec",
]


def fit_one_trial(hidden_layers, lr_phases, init_seed, X_scaled, Z_scaled, T, delta,
                   theta_init_scaled, train_idx, val_idx, device):
    X_t = torch.as_tensor(X_scaled, dtype=torch.float32, device=device)
    Z_t = torch.as_tensor(Z_scaled, dtype=torch.float32, device=device)
    T_t = torch.as_tensor(T, dtype=torch.float32, device=device)
    delta_t = torch.as_tensor(delta, dtype=torch.float32, device=device)
    tr_idx_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    va_idx_t = torch.as_tensor(val_idx, dtype=torch.long, device=device)

    set_seed(init_seed)
    model = CoxPHDNN(X_scaled.shape[1], Z_scaled.shape[1], hidden_layers, "relu", 0.0).to(device)
    with torch.no_grad():
        model.beta.weight.fill_(theta_init_scaled)

    # best_val/best_state are NOT reset per phase -- they carry across all
    # phases, so a later (lower-LR) phase's early stopping only ever accepts
    # a model that is better than anything seen so far in the WHOLE run, not
    # just within that phase (matching the fix validated in the FlexiHaz
    # full-batch/SGD diagnostic).
    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    final_val_loss = None
    for lr in lr_phases:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        wait = 0
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

            if val_loss < best_val - 1e-8:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= PATIENCE:
                    break
        model.load_state_dict(best_state)
        final_val_loss = best_val

        # ---- exact profile-likelihood refinement of theta at the phase boundary, holding g
        # fixed (TRAIN-SPLIT SUBJECTS ONLY -- the val split is used solely for early stopping
        # and must never enter the point estimate or its SE); the next phase starts from the
        # refined coefficient ----
        model.eval()
        with torch.no_grad():
            g_fixed_tr = model.g_net(X_t[tr_idx_t]).squeeze(-1).cpu().numpy().astype(np.float64)
        theta_joint_scaled = float(model.theta()[0])  # jointly trained coefficient of this phase
        theta_refined_scaled = refine_theta_zhong(g_fixed_tr, Z_scaled.reshape(-1)[train_idx], T[train_idx],
                                                  delta[train_idx], theta_joint_scaled)
        with torch.no_grad():
            model.beta.weight.fill_(float(theta_refined_scaled))
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    return theta_joint_scaled, theta_refined_scaled, float(final_val_loss)


def run_one_rep(rep_id, cfg, calibrated_c, device, base_seed, hidden_layers, lr):
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

    # Cox fit is JOINT on (X,Z) -- assumed linear -- so theta_init is the
    # Z-coefficient from that joint fit, not confounded by X's linear effect
    # leaking through X-Z correlation (fitting Cox on Z alone would let
    # theta absorb that leakage).
    x_cols = [f"X{j}" for j in range(X.shape[1])]
    cox_df = pd.DataFrame(X, columns=x_cols)
    cox_df["Z0"] = Z.reshape(-1)
    cox_df["event"] = delta
    cox_df["duration"] = T
    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col="duration", event_col="event", show_progress=False)
    theta_init_raw = float(cph.params_["Z0"])
    theta_init_scaled = theta_init_raw * float(z_scaler.scale_.reshape(-1)[0])

    dummy = FlexiHazardModel(validation_fraction=VAL_FRACTION, device=device, seed=data_seed, verbose=0)
    train_idx, val_idx = dummy._train_val_indices(np.arange(n))

    lr_phases = [lr, lr / 10.0, lr / 100.0]

    train_event_idx = train_idx[delta[train_idx] == 1]
    X_ev = X_scaled[train_event_idx]
    Z_ev = Z_scaled[train_event_idx].reshape(-1)
    T_ev = (T[train_event_idx] / tau).astype(np.float32).reshape(-1, 1)

    scale = float(z_scaler.scale_.reshape(-1)[0])
    thetas_unrefined, thetas_refined, val_losses = [], [], []
    for k in range(K):
        theta_joint_scaled, theta_refined_scaled, val_loss = fit_one_trial(
            hidden_layers, lr_phases, INIT_SEEDS[k], X_scaled, Z_scaled, T, delta,
            theta_init_scaled, train_idx, val_idx, device
        )
        thetas_unrefined.append(theta_joint_scaled / scale)
        thetas_refined.append(theta_refined_scaled / scale)
        val_losses.append(val_loss)
        torch.cuda.empty_cache()

    i_best = int(np.argmin(val_losses))
    theta_hat_unrefined = thetas_unrefined[i_best]
    theta_hat_refined = thetas_refined[i_best]

    # ---- SE: K-fold cross-fitted additive-projection residuals on
    # TRAIN-subject events, same construction as zhong_ph_prod.py ----
    n_ev = len(X_ev)
    rng = np.random.default_rng(data_seed + 777)
    fold_ids = rng.integers(0, PROJ_K_FOLDS, size=n_ev)
    r_oof = np.zeros(n_ev, dtype=np.float64)
    for kf in range(PROJ_K_FOLDS):
        te_mask = fold_ids == kf
        tr_mask = ~te_mask
        if te_mask.sum() == 0 or tr_mask.sum() < 10:
            continue
        n_tr_fold = tr_mask.sum()
        perm = rng.permutation(n_tr_fold)
        n_inner_val = max(1, int(0.2 * n_tr_fold))
        tr_idx_fold = np.where(tr_mask)[0][perm[n_inner_val:]]
        va_idx_fold = np.where(tr_mask)[0][perm[:n_inner_val]]
        te_idx_fold = np.where(te_mask)[0]

        set_seed(INIT_SEEDS[i_best] + kf)
        proj = AdditiveProjectionNet(X_ev.shape[1], hidden_layers=hidden_layers, activation="relu", dropout=0.0)
        proj = _fit_projection_net(
            proj, X_ev[tr_idx_fold], T_ev[tr_idx_fold], Z_ev[tr_idx_fold],
            X_ev[va_idx_fold], T_ev[va_idx_fold], Z_ev[va_idx_fold],
            device=device, lr=lr, weight_decay=0.0, batch_size=PROJ_TRAIN_BATCH,
            max_epochs=PROJ_MAX_EPOCHS, patience=PROJ_PATIENCE,
        )
        pred_te = _predict_projection(proj, X_ev[te_idx_fold], T_ev[te_idx_fold],
                                       device=device, batch_size=PROJ_BATCH)
        r_oof[te_idx_fold] = Z_ev[te_idx_fold] - pred_te.reshape(-1)

    var_scaled = 1.0 / max(float(np.sum(r_oof ** 2)), 1e-12)
    se = float(np.sqrt(var_scaled)) / scale

    fit_time = time.perf_counter() - t0
    result = dict(
        trial_selected=i_best, theta_hat_unrefined=float(theta_hat_unrefined),
        theta_hat_refined=float(theta_hat_refined), se=se, val_loss_selected=val_losses[i_best],
        fit_time_sec=fit_time,
    )
    for k in range(K):
        result[f"theta_trial{k}_unrefined"] = thetas_unrefined[k]
        result[f"theta_trial{k}_refined"] = thetas_refined[k]
        result[f"val_loss_trial{k}"] = val_losses[k]
    return result


def already_done(out_csv, rep_id):
    if not out_csv.exists():
        return False
    with open(out_csv) as f:
        return any(int(row["rep_id"]) == rep_id for row in csv.DictReader(f))


def main():
    global PROJ_TRAIN_BATCH
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
    ap.add_argument("--rep-start", type=int, default=0, help="first replication id (for splitting over machines)")
    ap.add_argument("--base-seed", type=int, default=BASE_SEED)
    ap.add_argument("--config", type=str, required=True, help="path to zhong_tune_hyperparams.py JSON output")
    ap.add_argument("--proj-batch-size", type=int, default=PROJ_BATCH,
                     help="projection-net training batch (default PROJ_BATCH=2048 at n=8000)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    PROJ_TRAIN_BATCH = args.proj_batch_size

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
    out_csv = Path(args.out) if args.out else HERE / "outputs" / f"zhong_{tag}.csv"

    with open(args.config) as f:
        tuning = json.load(f)
    selected = tuning["selected"]
    hidden_layers = tuple(selected["hidden_layers"])
    lr = selected["learning_rate"]

    calibrated_c = tuning.get("calibrated_c") or calibrate_censoring_c(
        gamma=cfg.gamma, rho=cfg.rho, target_rate=cfg.censoring_rate, k=cfg.k,
        nonlinear_scale=cfg.nonlinear_scale, ph_scale=cfg.ph_scale, nonph_scale=cfg.nonph_scale,
    )
    theta_true = float(np.log(1.5))
    print(f"config: gamma={cfg.gamma} rho={cfg.rho} censoring_rate={cfg.censoring_rate} n={cfg.n} k={cfg.k} "
          f"nonlinear_scale={cfg.nonlinear_scale} ph_scale={cfg.ph_scale} nonph_scale={cfg.nonph_scale} "
          f"n_reps={args.n_reps} K={K} calibrated_c={calibrated_c:.4f} "
          f"proj_batch_size={PROJ_TRAIN_BATCH} (select by val_loss)", flush=True)
    print(f"architecture: hidden_layers={hidden_layers} lr_phases={[lr, lr/10, lr/100]}", flush=True)
    print(f"theta_true = {theta_true:.4f}\n", flush=True)

    write_header = not out_csv.exists()
    with open(out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
            f.flush()

        for rep_id in range(args.rep_start, args.rep_start + args.n_reps):
            if already_done(out_csv, rep_id):
                print(f"rep={rep_id:03d} already done, skipping", flush=True)
                continue
            try:
                result = run_one_rep(rep_id, cfg, calibrated_c, device, args.base_seed, hidden_layers, lr)
            except Exception as exc:  # noqa: BLE001
                print(f"rep={rep_id:03d} FAILED: {type(exc).__name__}: {exc}", flush=True)
                torch.cuda.empty_cache()
                continue
            row = dict(rep_id=rep_id, **result)
            writer.writerow(row)
            f.flush()
            bias_pct_unref = 100 * (result["theta_hat_unrefined"] - theta_true) / theta_true
            bias_pct_ref = 100 * (result["theta_hat_refined"] - theta_true) / theta_true
            print(
                f"rep={rep_id:03d} trial_sel={result['trial_selected']} "
                f"theta_unref={result['theta_hat_unrefined']:.4f} (bias%={bias_pct_unref:+.2f}) "
                f"theta_ref={result['theta_hat_refined']:.4f} (bias%={bias_pct_ref:+.2f}) "
                f"se={result['se']:.4f} fit_time={result['fit_time_sec']:.1f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
