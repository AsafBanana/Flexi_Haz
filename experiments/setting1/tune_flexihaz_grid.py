"""Hyperparameter grid search for FlexiHaz's main net, using the SAME
(layers x neurons) grid as zhong_tune_hyperparams.py -- {1,2,3,4,5,6,8}
layers x {25,50,100,150,200,300,500} neurons, 49 configs -- MINUS the
learning-rate dimension: FlexiHaz keeps its own fixed 3-phase warm-start
schedule (1e-3->1e-4->1e-5), not searched. Dropout fixed at 0 (matches
tune_hyperparams.py's own earlier finding that dropout hurts here).

Selection metric: lowest val_loss over the 3-phase schedule, same as
tune_hyperparams.py.

The projection (residual) network does NOT get an independent grid search
here -- it REUSES the selected main-net architecture, matching Zhong's own
convention (zhong_tune_hyperparams.py's residual net "is similar to the
network we use to maximize log partial likelihoods," i.e., same architecture,
no separate search).
"""

from __future__ import annotations

import argparse
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

from dgm_additional import DGMConfig, calibrate_censoring_c, simulate_dataset  # noqa: E402

TUNING_SEED = 500_000
BATCH_SIZE = 100_000
LR_PHASES = [1e-3, 1e-4, 1e-5]
MAIN_MAX_EPOCHS = 500
MAIN_PATIENCE = 30

LAYERS = [2, 3, 4, 5, 6]
NEURONS = [25, 50, 100, 150, 200, 300, 500]


def _fit_one_seed(hidden_layers, seed, X_scaled, Z_scaled, Z_raw, T, delta, tau, time_grid,
                   expanded, device, X_raw, z_scaler=None):
    model = FlexiHazardModel(
        hidden_layers=hidden_layers, activation="relu", dropout=0.0, weight_decay=0.0, grid_size=2,
        learning_rate=LR_PHASES[0], batch_size=BATCH_SIZE, tau=tau, max_epochs=MAIN_MAX_EPOCHS,
        patience=MAIN_PATIENCE, theta_init="cox", refine_theta=False, compute_theta_variance=False,
        device=device, seed=TUNING_SEED, verbose=0,
    )
    set_seed(seed)
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
    model.z_scaler_ = z_scaler  # Cox warm start of theta on the standardized-Z scale the net trains on
    model._initialize_theta_from_cox(T, delta, X_raw, Z_raw, Z_scaled)

    # train/val split uses model.seed (=TUNING_SEED, fixed) -- SAME split every rep, so only
    # init/minibatch-order stochasticity varies across reps, not the held-out subjects.
    train_idx, val_idx = model._train_val_indices(expanded["subject_id"])

    # best_val is carried across phases (not reset to +inf each phase), matching
    # the fix validated in the Zhong best_val-carryover experiment and already
    # applied to multi_init_crossfit.py -- otherwise each phase's early stopping
    # only competes against its own +inf, understating how good the earlier
    # (higher-LR) phase's fit already was.
    carried_best_val = None
    for lr in LR_PHASES:
        model.learning_rate = lr
        carried_best_val = model._fit_expanded(expanded, train_idx, val_idx, init_best_val=carried_best_val)
    return carried_best_val


def evaluate_main_config(hidden_layers, X_scaled, Z_scaled, Z_raw, T, delta, tau, time_grid,
                          expanded, device, X_raw, z_scaler=None):
    t0 = time.perf_counter()
    best_val = _fit_one_seed(hidden_layers, TUNING_SEED, X_scaled, Z_scaled, Z_raw, T, delta, tau,
                              time_grid, expanded, device, X_raw, z_scaler)
    fit_time = time.perf_counter() - t0
    return dict(hidden_layers=list(hidden_layers), dropout=0.0, val_loss=best_val, fit_time_sec=fit_time)


def select_preferred(grid_results, tol_frac=0.01):
    """Among configs within tol_frac of the grid's minimum val_loss (the
    landscape is typically flat, so the literal argmin is often noise), pick
    the one with the FEWEST layers -- not the narrowest width -- tie-broken
    by lowest val_loss. Cheaper to train and just as accurate as the
    (frequently much larger, much slower) literal argmin."""
    min_val = min(r["val_loss"] for r in grid_results)
    candidates = [r for r in grid_results if r["val_loss"] <= min_val * (1.0 + tol_frac)]
    return min(candidates, key=lambda r: (len(r["hidden_layers"]), r["val_loss"]))


def main():
    global TUNING_SEED, BATCH_SIZE
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--censoring-rate", type=float, required=True)
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--k", type=int, default=0, choices=[0, 1, 2, 3],
                     help="0 = original additive setting, 1 = with interactions (I1/I2), "
                          "2 = extreme (squared terms -> cubed, interactions Xi*Xj -> Xi^2*Xj), "
                          "3 = pure interaction (a/b are ONLY Xi^2*Xj, no base terms, "
                          "scaled so PH/non-PH/linear all have comparable variance)")
    ap.add_argument("--nonlinear-scale", type=float, default=1.0,
                     help="multiplies a(X)/b(X); >1 raises Var(nonlinear part) relative to "
                          "Var(theta0*Z), which is fixed at theta0^2 regardless of rho/k")
    ap.add_argument("--ph-scale", type=float, default=1.0, help="multiplies ONLY b(X) (PH part); 0.0 removes it")
    ap.add_argument("--nonph-scale", type=float, default=1.0, help="multiplies ONLY a(X) (non-PH part)")
    ap.add_argument("--seed", type=int, default=TUNING_SEED,
                     help="overrides TUNING_SEED -- controls the tuning dataset draw, net init, "
                          "and train/val split all at once (all runs so far used the default)")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--prefer-fewer-layers", action="store_true",
                     help="instead of the literal grid-argmin, select the fewest-layer config within "
                          "1%% of the minimum val_loss (tie-break: lowest val_loss)")
    ap.add_argument("--layers", type=str, default=None, help="comma-separated depths (default: the full grid)")
    ap.add_argument("--neurons", type=str, default=None, help="comma-separated widths (default: the full grid)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    global LAYERS, NEURONS
    if args.layers:
        LAYERS = [int(v) for v in args.layers.split(",")]
    if args.neurons:
        NEURONS = [int(v) for v in args.neurons.split(",")]

    TUNING_SEED = args.seed
    BATCH_SIZE = args.batch_size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = DGMConfig(gamma=args.gamma, rho=args.rho, censoring_rate=args.censoring_rate, n=args.n, k=args.k,
                     nonlinear_scale=args.nonlinear_scale, ph_scale=args.ph_scale, nonph_scale=args.nonph_scale)
    tag = f"gamma{cfg.gamma}_rho{cfg.rho}_cens{int(cfg.censoring_rate*100)}"
    if cfg.k:
        tag += f"_k{cfg.k}"
    tag += f"_n{cfg.n}"
    if cfg.nonlinear_scale != 1.0:
        tag += f"_scale{cfg.nonlinear_scale:g}"
    if cfg.ph_scale != 1.0 or cfg.nonph_scale != 1.0:
        tag += f"_ph{cfg.ph_scale:g}_nonph{cfg.nonph_scale:g}"
    if TUNING_SEED != 500_000:
        tag += f"_seed{TUNING_SEED}"
    out_json = Path(args.out) if args.out else HERE / "outputs" / f"tuning_flexihaz_{tag}.json"

    calibrated_c = calibrate_censoring_c(gamma=cfg.gamma, rho=cfg.rho, target_rate=cfg.censoring_rate, k=cfg.k,
                                          nonlinear_scale=cfg.nonlinear_scale, ph_scale=cfg.ph_scale,
                                          nonph_scale=cfg.nonph_scale)
    print(f"config: gamma={cfg.gamma} rho={cfg.rho} censoring_rate={cfg.censoring_rate} n={cfg.n} k={cfg.k} "
          f"nonlinear_scale={cfg.nonlinear_scale} ph_scale={cfg.ph_scale} nonph_scale={cfg.nonph_scale} "
          f"tuning_seed={TUNING_SEED} calibrated_c={calibrated_c:.4f}",
          flush=True)

    data = simulate_dataset(cfg, seed=TUNING_SEED, calibrated_c=calibrated_c)
    T_raw, delta_raw, X_raw, Z_raw = data["T"], data["delta"], data["X"], data["Z"]
    T, delta, X, Z = validate_survival_inputs(T_raw, delta_raw, X_raw, Z_raw)
    T, delta, tau = truncate_at_tau(T, delta, None)

    x_scaler = Standardizer()
    X_scaled = x_scaler.fit_transform(X)
    z_scaler = Standardizer()
    Z_scaled = z_scaler.fit_transform(Z)

    time_grid = build_time_grid(T, tau, 2, True)
    expanded = make_expanded_data(T, delta, X_scaled, Z_scaled, time_grid)

    main_results = []
    n_configs = len(LAYERS) * len(NEURONS)
    i = 0
    for n_layers in LAYERS:
        for n_neurons in NEURONS:
            i += 1
            hidden_layers = (n_neurons,) * n_layers
            r = evaluate_main_config(hidden_layers, X_scaled, Z_scaled, Z, T, delta, tau, time_grid,
                                      expanded, device, X, z_scaler)
            main_results.append(r)
            print(f"[{i}/{n_configs}] layers={n_layers} neurons={n_neurons}: "
                  f"val_loss={r['val_loss']:.6f} time={r['fit_time_sec']:.1f}s", flush=True)
            torch.cuda.empty_cache()

    if args.prefer_fewer_layers:
        main_selected = select_preferred(main_results)
    else:
        main_selected = min(main_results, key=lambda r: r["val_loss"])
    print(f"main net selected: {main_selected}", flush=True)
    # projection net reuses the main net's architecture -- no separate search
    proj_selected = dict(hidden_layers=main_selected["hidden_layers"])
    print(f"projection net (reused from main): {proj_selected}", flush=True)

    out = dict(
        gamma=cfg.gamma, rho=cfg.rho, censoring_rate=cfg.censoring_rate, n=cfg.n, k=cfg.k,
        nonlinear_scale=cfg.nonlinear_scale, calibrated_c=calibrated_c, tuning_seed=TUNING_SEED,
        grid=dict(layers=LAYERS, neurons=NEURONS, dropout_fixed_at=0.0),
        main_net=dict(grid=main_results, selected=main_selected),
        projection_net=dict(selected=proj_selected, note="reuses main net architecture, no separate search"),
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
