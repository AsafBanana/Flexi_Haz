"""Hyperparameter grid search for the Zhong, Mueller & Wang (2022) proportional
DNN model, following the paper's OWN protocol (Section 4.1 + Supplement S4,
Table 1): grid search over number of hidden layers, neurons per layer, and
learning rate, selected by minimum partial-likelihood loss on a held-out
validation set. Dropout is EXCLUDED from the grid and fixed at 0 -- the
paper's theory (Assumption A1) requires a complexity-controlled/sparse
network, but early stopping already provides that complexity control (as it
does for FlexiHaz), so dropout is not needed as a second mechanism for it.

Grid (paper's exact values, dropout removed):
  layers  in {1,2,3,4,5,6,8}
  neurons in {25,50,100,150,200,300,500}
  lr      in {1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,1e-1}
  => 343 configs.

Run on ONE held-out simulated dataset per (gamma,rho,censoring_rate) setting
(same TUNING_SEED convention as tune_hyperparams.py -- independent of the 200
replication seeds used for the reported results).

The residual/projection network used for the standard error (AdditiveProjectionNet
in zhong_ph_prod.py) reuses the SAME selected (hidden_layers, learning_rate) as
the main network -- the paper states its implementation "is similar to the
network we use to maximize log partial likelihoods," and does not give it a
separate grid.
"""

from __future__ import annotations

import argparse
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
from flexihaz.utils import set_seed  # noqa: E402

from dgm_additional import DGMConfig, calibrate_censoring_c, paper_scales, simulate_dataset  # noqa: E402
from zhong_ph_prod import CoxPHDNN, breslow_loss, INIT_SEED  # noqa: E402

TUNING_SEED = 500_000  # same tuning dataset seed convention as tune_hyperparams.py
VAL_FRACTION = 0.33
MAX_EPOCHS = 5000
PATIENCE = 35

LAYERS = [2, 3, 4, 5, 6]  # matches tune_hyperparams_zhonggrid.py's (FlexiHaz's own) restricted grid
NEURONS = [25, 50, 100, 150, 200, 300, 500]
LEARNING_RATES = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]


def evaluate_config(hidden_layers, lr, X_scaled, Z_scaled, Z_raw, T, delta, theta_init_scaled,
                     train_idx, val_idx, device):
    t0 = time.perf_counter()
    set_seed(INIT_SEED)
    model = CoxPHDNN(X_scaled.shape[1], Z_scaled.shape[1], hidden_layers, "relu", 0.0).to(device)
    with torch.no_grad():
        model.beta.weight.fill_(theta_init_scaled)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    X_t = torch.as_tensor(X_scaled, dtype=torch.float32, device=device)
    Z_t = torch.as_tensor(Z_scaled, dtype=torch.float32, device=device)
    T_t = torch.as_tensor(T, dtype=torch.float32, device=device)
    delta_t = torch.as_tensor(delta, dtype=torch.float32, device=device)
    tr_idx_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    va_idx_t = torch.as_tensor(val_idx, dtype=torch.long, device=device)

    best_val = float("inf")
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
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    fit_time = time.perf_counter() - t0
    return dict(hidden_layers=list(hidden_layers), learning_rate=lr, val_loss=best_val,
                epochs_run=epochs_run, fit_time_sec=fit_time)


def main():
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
    ap.add_argument("--layers", type=str, default=None, help="comma-separated depths (default: the full grid)")
    ap.add_argument("--neurons", type=str, default=None, help="comma-separated widths (default: the full grid)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    global LAYERS, NEURONS
    if args.layers:
        LAYERS = [int(v) for v in args.layers.split(",")]
    if args.neurons:
        NEURONS = [int(v) for v in args.neurons.split(",")]

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
    out_json = Path(args.out) if args.out else HERE / "outputs" / f"tuning_zhong_{tag}.json"

    calibrated_c = calibrate_censoring_c(gamma=cfg.gamma, rho=cfg.rho, target_rate=cfg.censoring_rate, k=cfg.k,
                                          nonlinear_scale=cfg.nonlinear_scale, ph_scale=cfg.ph_scale,
                                          nonph_scale=cfg.nonph_scale)
    print(f"config: gamma={cfg.gamma} rho={cfg.rho} censoring_rate={cfg.censoring_rate} n={cfg.n} k={cfg.k} "
          f"nonlinear_scale={cfg.nonlinear_scale} ph_scale={cfg.ph_scale} nonph_scale={cfg.nonph_scale} "
          f"calibrated_c={calibrated_c:.4f}", flush=True)

    data = simulate_dataset(cfg, seed=TUNING_SEED, calibrated_c=calibrated_c)
    T_raw, delta_raw, X_raw, Z_raw = data["T"], data["delta"], data["X"], data["Z"]
    T, delta, X, Z = validate_survival_inputs(T_raw, delta_raw, X_raw, Z_raw)
    T, delta, tau = truncate_at_tau(T, delta, None)

    x_scaler = Standardizer()
    X_scaled = x_scaler.fit_transform(X).astype(np.float32)
    z_scaler = Standardizer()
    Z_scaled = z_scaler.fit_transform(Z).astype(np.float32)
    n = X_scaled.shape[0]

    x_cols = [f"X{j}" for j in range(X.shape[1])]
    cox_df = pd.DataFrame(X, columns=x_cols)
    cox_df["Z0"] = Z.reshape(-1)
    cox_df["event"] = delta
    cox_df["duration"] = T
    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col="duration", event_col="event", show_progress=False)
    theta_init_scaled = float(cph.params_["Z0"]) * float(z_scaler.scale_.reshape(-1)[0])

    dummy = FlexiHazardModel(validation_fraction=VAL_FRACTION, device=device, seed=TUNING_SEED, verbose=0)
    train_idx, val_idx = dummy._train_val_indices(np.arange(n))

    results = []
    n_configs = len(LAYERS) * len(NEURONS) * len(LEARNING_RATES)
    i = 0
    for n_layers in LAYERS:
        for n_neurons in NEURONS:
            for lr in LEARNING_RATES:
                i += 1
                hidden_layers = (n_neurons,) * n_layers
                r = evaluate_config(hidden_layers, lr, X_scaled, Z_scaled, Z, T, delta,
                                     theta_init_scaled, train_idx, val_idx, device)
                results.append(r)
                print(f"[{i}/{n_configs}] layers={n_layers} neurons={n_neurons} lr={lr}: "
                      f"val_loss={r['val_loss']:.6f} epochs={r['epochs_run']} time={r['fit_time_sec']:.1f}s",
                      flush=True)
                torch.cuda.empty_cache()

    selected = min(results, key=lambda r: r["val_loss"])
    print(f"selected: {selected}", flush=True)

    out = dict(
        gamma=cfg.gamma, rho=cfg.rho, censoring_rate=cfg.censoring_rate, n=cfg.n, k=cfg.k,
        nonlinear_scale=cfg.nonlinear_scale, calibrated_c=calibrated_c, tuning_seed=TUNING_SEED,
        grid=dict(layers=LAYERS, neurons=NEURONS, learning_rates=LEARNING_RATES, dropout_fixed_at=0.0),
        results=results, selected=selected,
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
