"""Rotterdam 5-fold CV treatment-effect analysis with exhaustive architecture grid tuning.

Same protocol for FLEXI-Haz (model_type="flexihaz") and the Zhong, Mueller & Wang
(2022) proportional-hazards DNN (model_type="zhong"), both fitted with
flexihaz.FlexiHazardModel:

* design "standard" (Z = chemo, hormon) or "er_interaction" (Z = hormon/chemo x
  low/high ER, threshold 61); X = age, nodes, pgr, er, meno, grade + one-hot tumour
  size (rotterdam_io.make_design_dataframe); X and Z standardized inside the model;
* five outer folds (KFold, shuffled, random_state=--seed, stored as
  results/folds_seed<seed>.csv), fit on each fold's training rows only; --seed also
  seeds the network initialisation (seed + fold) and the restarts;
* per fold, an exhaustive grid over depth L in {2,3,4,5,6} x width w in
  {25,50,100,150,200,300,500} (35 architectures, hidden_layers=(w,)*L), weight
  decay 0, dropout 0.  FLEXI-Haz: learning rate FIXED at 1e-3 (architecture only).
  Zhong: the same architectures x the learning-rate grid
  {1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2} (245 fits per fold), as Zhong et al.
  tune both.  --learning-rates overrides either.  Each candidate is trained once on
  the fold's training rows with the estimator's own 67%/33% subject split for early
  stopping and scored by its validation loss; the argmin is selected;
* the selected architecture is refitted with --n-restarts (default 5) random
  restarts, keeping the restart with the lowest validation loss (or the smallest
  efficient score with --restart-selection efficient_score); theta, its SE (residual
  projection nets: joint (t,X) net for FLEXI-Haz, additive g_X(X)+g_T(t) net for
  Zhong), 90% and 95% CIs are recorded per fold;
* across folds: mean estimate, SD of the fold estimates, mean SE, mean CI bounds.

Outputs under results/<model_type>_gridcv[_thetalr<M>][_plain]/<design>_seed<seed>/:
    fold_XX_grid.csv               validation loss of every architecture
    fold_XX_treatment_effects.csv  theta, se, CIs for the selected architecture
    fold_XX_log.json               selected architecture, restarts, timings
    summary_treatment_effects.csv  across-fold aggregation

Examples
--------
    python run_rotterdam_grid_cv.py --model-type flexihaz --design er_interaction --theta-lr-mult 20 --seed 20260303
    python run_rotterdam_grid_cv.py --model-type zhong --design er_interaction --zhong-plain --seed 20260707
    python run_rotterdam_grid_cv.py --model-type flexihaz --design standard --start-fold 2 --end-fold 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from rotterdam_io import DEFAULT_DATA_PATH, DEFAULT_RESULTS_DIR, ensure_dir, load_arrays_and_folds, save_json

DEFAULT_LAYERS = [2, 3, 4, 5, 6]
DEFAULT_WIDTHS = [25, 50, 100, 150, 200, 300, 500]
DEFAULT_LEARNING_RATES = {"flexihaz": [1e-3], "zhong": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]}


def import_flexihaz_model():
    try:
        from flexihaz import FlexiHazardModel
        return FlexiHazardModel
    except ImportError:
        source_dir = os.environ.get("FLEXIHAZ_SOURCE_DIR")
        if source_dir:
            sys.path.insert(0, source_dir)
            from flexihaz import FlexiHazardModel
            return FlexiHazardModel
        raise


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rotterdam 5-fold CV with exhaustive architecture grid (FLEXI-Haz or Zhong).")
    p.add_argument("--model-type", choices=["flexihaz", "zhong"], default="flexihaz")
    p.add_argument("--design", choices=["standard", "er_interaction"], default="er_interaction")
    p.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    p.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    p.add_argument("--start-fold", type=int, default=0)
    p.add_argument("--end-fold", type=int, default=5)
    p.add_argument("--layers", type=str, default=",".join(map(str, DEFAULT_LAYERS)), help="comma-separated depths")
    p.add_argument("--widths", type=str, default=",".join(map(str, DEFAULT_WIDTHS)), help="comma-separated widths")
    p.add_argument("--learning-rates", type=str, default=None,
                   help="comma-separated learning-rate grid; default 1e-3 (fixed) for flexihaz, "
                        "1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,1e-2 for zhong")
    p.add_argument("--batch-size", type=int, default=100000)
    p.add_argument("--max-epochs", type=int, default=None, help="None -> model default (500 flexihaz / 5000 zhong)")
    p.add_argument("--patience", type=int, default=None, help="None -> model default (30 flexihaz / 35 zhong)")
    p.add_argument("--tuning-max-epochs", type=int, default=None, help="None -> same as --max-epochs")
    p.add_argument("--n-restarts", type=int, default=5)
    p.add_argument("--restart-selection", choices=["val_loss", "efficient_score"], default="val_loss")
    p.add_argument("--theta-lr-mult", type=float, default=1.0, help="learning-rate multiplier for the theta branch")
    p.add_argument("--zhong-plain", action="store_true",
                   help="single constant-lr phase and no profile-likelihood refinement of theta (Zhong's own recipe)")
    p.add_argument("--no-three-phase-lr", action="store_true", help="disable the lr, lr/10, lr/100 schedule")
    p.add_argument("--no-refine-theta", action="store_true", help="disable the exact theta refinement after training")
    p.add_argument("--projection-max-epochs", type=int, default=300)
    p.add_argument("--projection-patience", type=int, default=30)
    p.add_argument("--projection-n-splits", type=int, default=5)
    p.add_argument("--tau", type=float, default=None)
    p.add_argument("--seed", type=int, default=221100,
                   help="seeds the outer folds (results/folds_seed<seed>.csv), the network init and the restarts; "
                        "our runs used 20260303 for flexihaz and 20260707 for zhong")
    p.add_argument("--device", default="auto")
    p.add_argument("--output-tag", type=str, default=None, help="override the output folder name")
    p.add_argument("--skip-completed", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def output_name(args: argparse.Namespace) -> str:
    if args.output_tag:
        return args.output_tag
    name = f"{args.model_type}_gridcv"
    if args.theta_lr_mult != 1.0:
        name += f"_thetalr{args.theta_lr_mult:g}"
    if args.zhong_plain:
        name += "_plain"
    elif args.no_three_phase_lr or args.no_refine_theta:
        name += ("_singlelr" if args.no_three_phase_lr else "") + ("_norefine" if args.no_refine_theta else "")
    return name


def model_kwargs(args: argparse.Namespace, hidden_layers, learning_rate, *, final: bool, seed: int, tau: float | None) -> dict:
    kw = dict(
        model_type=args.model_type,
        hidden_layers=tuple(hidden_layers),
        activation="relu",
        dropout=0.0,
        weight_decay=0.0,
        learning_rate=float(learning_rate),
        batch_size=args.batch_size,
        grid_size="n",
        include_observed_times=True,
        validation_fraction=0.33,
        tau=tau,
        theta_init="cox",
        device=args.device,
        seed=seed,
        verbose=0,
        theta_lr_multiplier=args.theta_lr_mult,
    )
    if args.max_epochs is not None:
        kw["max_epochs"] = args.max_epochs
    if args.patience is not None:
        kw["patience"] = args.patience
    plain = args.zhong_plain
    kw["three_phase_lr"] = not (plain or args.no_three_phase_lr)
    kw["refine_theta"] = final and not (plain or args.no_refine_theta)
    if final:
        kw.update(
            n_restarts=args.n_restarts,
            restart_selection=args.restart_selection,
            compute_theta_variance=True,
            projection_max_epochs=args.projection_max_epochs,
            projection_patience=args.projection_patience,
            projection_n_splits=args.projection_n_splits,
        )
    else:
        kw.update(n_restarts=1, compute_theta_variance=False)
        if args.tuning_max_epochs is not None:
            kw["max_epochs"] = args.tuning_max_epochs
    return kw


def best_val_loss(model) -> float:
    hist = model.train_history
    v = hist["val_loss"] if isinstance(hist, dict) else hist
    return float(np.min(np.asarray(v, dtype=np.float64)))


def main() -> None:
    args = parse_args()
    layers = [int(v) for v in args.layers.split(",") if v.strip()]
    widths = [int(v) for v in args.widths.split(",") if v.strip()]
    lrs = ([float(v) for v in args.learning_rates.split(",") if v.strip()] if args.learning_rates
           else list(DEFAULT_LEARNING_RATES[args.model_type]))
    if args.smoke_test:
        layers, widths, lrs = layers[:1], widths[:2], lrs[:2]
        args.max_epochs = args.max_epochs or 5
        args.tuning_max_epochs = 3
        args.projection_max_epochs = 3
        args.n_restarts = min(args.n_restarts, 2)
        args.end_fold = min(args.end_fold, args.start_fold + 1)
        args.results_dir = str(Path(args.results_dir) / "smoke_test")
    grid = [((w,) * L, lr) for L in layers for w in widths for lr in lrs]

    FlexiHazardModel = import_flexihaz_model()
    data = load_arrays_and_folds(data_path=args.data_path, results_dir=args.results_dir, design=args.design, seed=args.seed)
    X, Z, T, delta, folds = data["X"], data["Z"], data["T"], data["delta"], data["folds"]
    z_cols = data["metadata"]["z_columns"]
    tau = float(np.max(T)) if args.tau is None else float(args.tau)

    out_dir = ensure_dir(Path(args.results_dir) / output_name(args) / f"{args.design}_seed{args.seed}")
    print(f"model_type={args.model_type} design={args.design} grid={len(grid)} configs "
          f"(L={layers}, w={widths}, lr={lrs}) theta_lr_mult={args.theta_lr_mult} "
          f"restarts={args.n_restarts} three_phase_lr={not (args.zhong_plain or args.no_three_phase_lr)} "
          f"refine_theta={not (args.zhong_plain or args.no_refine_theta)} -> {out_dir}", flush=True)

    for fold_id in range(args.start_fold, args.end_fold):
        log_path = out_dir / f"fold_{fold_id:02d}_log.json"
        if args.skip_completed and log_path.exists() and json.load(open(log_path)).get("status") == "success":
            print(f"fold {fold_id}: already completed, skipping", flush=True)
            continue
        tr = folds.loc[folds.fold_id != fold_id, "row_id"].to_numpy()
        Xf, Zf, Tf, df = X[tr], Z[tr], T[tr], delta[tr]
        fold_seed = args.seed + fold_id
        t0 = time.perf_counter()

        # ---- exhaustive architecture grid at fixed lr (inner 67/33 split for early stopping)
        grid_rows = []
        for i, (hl, lr) in enumerate(grid):
            t1 = time.perf_counter()
            m = FlexiHazardModel(**model_kwargs(args, hl, lr, final=False, seed=fold_seed, tau=tau))
            m.fit(Tf, df, Xf, Z=Zf)
            vl = best_val_loss(m)
            grid_rows.append(dict(fold=fold_id, config_idx=i, depth=len(hl), width=hl[0], hidden_layers=str(list(hl)),
                                  learning_rate=lr, val_loss=vl,
                                  epochs_run=int(np.size(m.train_history["val_loss"])) if isinstance(m.train_history, dict) else None,
                                  fit_time_sec=round(time.perf_counter() - t1, 1)))
            print(f"fold {fold_id} [{i + 1}/{len(grid)}] {hl} lr={lr:g}: val_loss={vl:.6f} ({grid_rows[-1]['fit_time_sec']}s)", flush=True)
        grid_df = pd.DataFrame(grid_rows)
        grid_df.to_csv(out_dir / f"fold_{fold_id:02d}_grid.csv", index=False)
        best = grid_df.loc[grid_df.val_loss.idxmin()]
        selected = tuple(int(v) for v in json.loads(best.hidden_layers))
        selected_lr = float(best.learning_rate)
        print(f"fold {fold_id}: selected {selected} lr={selected_lr:g} (val_loss={best.val_loss:.6f})", flush=True)

        # ---- final fit: restarts + theta inference
        t2 = time.perf_counter()
        model = FlexiHazardModel(**model_kwargs(args, selected, selected_lr, final=True, seed=fold_seed, tau=tau))
        model.fit(Tf, df, Xf, Z=Zf)
        theta = np.asarray(model.theta_hat_original_scale, dtype=np.float64).reshape(-1)
        se = np.sqrt(np.maximum(np.diag(np.asarray(model.var_theta_hat, dtype=np.float64)), 0.0))
        z90, z95 = norm.ppf(0.95), norm.ppf(0.975)
        eff = pd.DataFrame(dict(fold=fold_id, term=z_cols, estimate=theta, se=se,
                                ci90_lower=theta - z90 * se, ci90_upper=theta + z90 * se,
                                ci95_lower=theta - z95 * se, ci95_upper=theta + z95 * se,
                                hazard_ratio=np.exp(theta)))
        eff.to_csv(out_dir / f"fold_{fold_id:02d}_treatment_effects.csv", index=False)
        restarts = None
        if getattr(model, "restart_results_", None):
            restarts = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()} for r in model.restart_results_]
        save_json(log_path, dict(
            status="success", model_type=args.model_type, design=args.design, fold=fold_id,
            n_train=int(tr.size), n_events_train=int(df.sum()), selected_hidden_layers=list(selected),
            selected_val_loss=float(best.val_loss), selected_learning_rate=selected_lr, learning_rate_grid=lrs,
            theta_lr_multiplier=args.theta_lr_mult,
            three_phase_lr=bool(model.three_phase_lr), refine_theta=bool(model.refine_theta),
            n_restarts=args.n_restarts, restart_selection=args.restart_selection,
            selected_restart=getattr(model, "selected_restart_", 0), restarts=restarts,
            grid_layers=layers, grid_widths=widths, grid_learning_rates=lrs, n_grid=len(grid),
            tuning_time_sec=round(t2 - t0, 1), final_fit_time_sec=round(time.perf_counter() - t2, 1),
            z_columns=z_cols, theta=theta.tolist(), se=se.tolist(),
        ))
        for _, r in eff.iterrows():
            print(f"  fold {fold_id} {r.term}: estimate={r.estimate:.4f} se={r.se:.4f} "
                  f"ci95=({r.ci95_lower:.4f}, {r.ci95_upper:.4f})", flush=True)

    # ---- across-fold summary (all folds present on disk)
    effs = sorted(out_dir.glob("fold_*_treatment_effects.csv"))
    if effs:
        all_eff = pd.concat([pd.read_csv(f) for f in effs], ignore_index=True)
        summ = (all_eff.groupby("term", sort=False)
                .agg(n_folds=("fold", "nunique"), mean_estimate=("estimate", "mean"), sd_estimate=("estimate", "std"),
                     mean_se=("se", "mean"), mean_ci90_lower=("ci90_lower", "mean"), mean_ci90_upper=("ci90_upper", "mean"),
                     mean_ci95_lower=("ci95_lower", "mean"), mean_ci95_upper=("ci95_upper", "mean"))
                .reset_index())
        summ["mean_hazard_ratio"] = np.exp(summ.mean_estimate)
        summ["significant_95"] = (summ.mean_ci95_upper < 0) | (summ.mean_ci95_lower > 0)
        summ.to_csv(out_dir / "summary_treatment_effects.csv", index=False)
        sel = [(json.load(open(p)).get("selected_hidden_layers"), json.load(open(p)).get("selected_learning_rate"))
               for p in sorted(out_dir.glob("fold_*_log.json"))]
        print("\nselected (architecture, learning rate) by fold:", sel)
        print(summ.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
