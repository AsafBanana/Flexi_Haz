"""FLEXI-Haz architecture selection for setting 2: exhaustive grid on an independent tuning
dataset with five common subject-level folds (Supplementary A.2.2). The Zhong comparator is
tuned by tune_zhong_optuna.py.

Protocol (no Optuna, no weight-decay or dropout search):

* one independent tuning dataset per sample size (seed --tuning-seed; same DGM, n, tau,
  censoring and theta_true as the replications, read from config.json);
* five COMMON outer folds (KFold, shuffle, random_state=--fold-seed), identical for every
  configuration;
* inside each outer training fold the estimator's own 67%/33% subject split is used for
  early stopping (validation_fraction=0.33);
* the outer held-out fold is scored with the fitted model's negative log-likelihood
  (piecewise-exponential NLL for FLEXI-Haz, Breslow partial-likelihood NLL for Zhong);
* all 27 architectures of Table 7 (135 fits), learning rate FIXED at 1e-3, the usual
  three-phase schedule inside each fit;
* the architecture minimising the mean outer-fold NLL is selected.

The grid can be split across machines with --arch-ids / --arch-start / --arch-end; every
fold result is appended to a CSV as it finishes (resume-safe with --skip-completed).
`--combine` merges the fold CSVs of a model type, selects the argmin and writes
<model>_best_params.json in the format read by run_replications.py.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from simulation_dgm import simulate_dataset, save_npz, load_npz
from simulation_io import DEFAULT_OUTPUT_DIR, ensure_dir, load_json, save_json

ARCHITECTURES = [
    [16], [32], [64], [128], [256],
    [16, 16], [16, 32], [32, 32], [32, 64], [64, 64], [64, 128], [128, 128],
    [16, 16, 16], [16, 16, 32], [32, 32, 32], [32, 32, 64], [64, 64, 64], [64, 64, 128], [128, 128, 128],
    [16, 16, 16, 16], [16, 16, 32, 32], [32, 32, 32, 32], [32, 32, 64, 64], [64, 64, 64, 64],
    [16, 16, 16, 16, 16], [16, 16, 32, 32, 32], [32, 32, 32, 32, 32],
]
MODEL_DEFAULTS = {  # max_epochs, patience, learning rates, file prefix
    "flexihaz": dict(max_epochs=2000, patience=15, learning_rates=[1e-3], prefix="flexihaz_partial_linear"),
}
FIELDS = ["arch_idx", "hidden_layers", "learning_rate", "fold", "holdout_nll", "best_inner_val_loss",
          "epochs_run", "fit_time_sec"]


def import_flexihaz():
    try:
        import flexihaz  # noqa: F401
    except ImportError:
        source_dir = os.environ.get("FLEXIHAZ_SOURCE_DIR")
        if source_dir:
            sys.path.insert(0, source_dir)
        else:
            raise
    from flexihaz import FlexiHazardModel
    from flexihaz.preprocessing import truncate_at_tau
    from flexihaz.data import make_expanded_data
    return FlexiHazardModel, truncate_at_tau, make_expanded_data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid search for FLEXI-Haz / Zhong on an independent tuning dataset.")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--model-type", choices=["flexihaz"], default="flexihaz")
    p.add_argument("--tuning-seed", type=int, default=500000, help="seed of the independent tuning dataset")
    p.add_argument("--fold-seed", type=int, default=221100, help="KFold random_state (common folds for all configs)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--learning-rates", type=str, default=None, help="comma-separated grid; default fixed 1e-3")
    p.add_argument("--max-epochs", type=int, default=None, help="default 2000")
    p.add_argument("--patience", type=int, default=None, help="default 15")
    p.add_argument("--batch-size", type=int, default=100000)
    p.add_argument("--arch-start", type=int, default=0)
    p.add_argument("--arch-end", type=int, default=len(ARCHITECTURES))
    p.add_argument("--arch-ids", type=str, default=None,
                   help="comma-separated architecture indices (overrides --arch-start/--arch-end)")
    p.add_argument("--tag", type=str, default=None, help="suffix for the fold CSV name (default: start_end or ids)")
    p.add_argument("--device", default="auto")
    p.add_argument("--results-csv", default=None, help="fold-level results CSV (default tuning/<prefix>_grid_folds_<tag>.csv)")
    p.add_argument("--skip-completed", action="store_true")
    p.add_argument("--combine", nargs="*", default=None, metavar="CSV",
                   help="combine fold CSVs (default: all tuning/<prefix>_grid_folds*.csv), select argmin, write best_params json")
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def tuning_dataset(output_dir: Path, seed: int):
    cfg = load_json(output_dir / "config.json")
    path = output_dir / "datasets" / f"tuning_seed{seed}.npz"
    if path.exists():
        data = load_npz(path)
    else:
        data = simulate_dataset(
            n=int(cfg["n"]), seed=seed, tau=float(cfg["tau"]), censoring_mean=float(cfg["censoring_mean"]),
            theta_true=np.asarray(cfg["theta_true"], dtype=np.float64),
        )
        save_npz(path, data)
        print(f"generated tuning dataset n={cfg['n']} seed={seed} theta_true={cfg['theta_true']} -> {path}", flush=True)
    return data, float(cfg["tau"])


def evaluate_on_holdout(model, T_val, delta_val, X_val, Z_val, truncate_at_tau, make_expanded_data):
    """Outer-fold NLL of a fitted model (same construction as flexihaz.tuning)."""
    T_val, delta_val, _ = truncate_at_tau(T_val, delta_val, model.tau)
    X_val = model.x_scaler_.transform(X_val) if model.x_scaler_ is not None else np.asarray(X_val, dtype=np.float64)
    if Z_val is not None:
        Z_val = model.z_scaler_.transform(Z_val) if model.z_scaler_ is not None else np.asarray(Z_val, dtype=np.float64)
    if model.model_type == "zhong":
        rows = model._subject_rows(T_val, delta_val, X_val, Z_val)
    else:
        rows = make_expanded_data(T_val, delta_val, X_val, Z_val, model.time_grid)
    idx = np.arange(rows["t"].shape[0])
    return float(model._validation_loss(model._tensors_from_expanded(rows, idx)))


def model_kwargs(args, hl, lr, seed, tau):
    kw = dict(
        model_type=args.model_type, hidden_layers=tuple(hl), activation="relu", dropout=0.0, learning_rate=lr,
        weight_decay=0.0, batch_size=args.batch_size, grid_size="n", tau=tau, max_epochs=args.max_epochs,
        patience=args.patience, validation_fraction=0.33, compute_theta_variance=False, refine_theta=False,
        device=args.device, seed=seed, verbose=0, num_workers=2, pin_memory=True, persistent_workers=True,
        non_blocking=True,
    )
    return kw


def combine(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    tuning_dir = ensure_dir(output_dir / "tuning")
    prefix = MODEL_DEFAULTS[args.model_type]["prefix"]
    files = args.combine if args.combine else sorted(str(p) for p in tuning_dir.glob(f"{prefix}_grid_folds*.csv"))
    if not files:
        raise SystemExit("no fold CSVs found to combine")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "learning_rate" not in df.columns:  # CSVs written before the learning-rate column existed
        df["learning_rate"] = MODEL_DEFAULTS[args.model_type]["learning_rates"][0]
    df = df.drop_duplicates(subset=["arch_idx", "learning_rate", "fold"], keep="last")
    agg = (df.groupby(["arch_idx", "hidden_layers", "learning_rate"])
             .agg(n_folds=("fold", "nunique"), mean_holdout_nll=("holdout_nll", "mean"),
                  sd_holdout_nll=("holdout_nll", "std"), mean_epochs=("epochs_run", "mean"),
                  mean_fit_time_sec=("fit_time_sec", "mean"))
             .reset_index().sort_values("mean_holdout_nll"))
    agg.to_csv(tuning_dir / f"{prefix}_grid_summary.csv", index=False)
    complete = agg[agg.n_folds == args.folds]
    n_expected = len(ARCHITECTURES) * len(args.learning_rates)
    if len(complete) < n_expected:
        print(f"WARNING: {len(complete)} of {n_expected} configurations have all {args.folds} folds")
    if complete.empty:
        raise SystemExit("no configuration has all folds")
    best = complete.iloc[0]
    hidden = [int(v) for v in str(best.hidden_layers).strip("[]()").split(",") if v.strip()]
    payload = {
        "model_type": args.model_type,
        "tuning_seed": args.tuning_seed,
        "fold_seed": args.fold_seed,
        "folds": args.folds,
        "selection": "argmin mean outer-fold NLL over the 27-architecture grid"
                     + (" x learning-rate grid" if len(args.learning_rates) > 1 else " at fixed lr")
                     + "; weight_decay 0, dropout 0",
        "best_value": float(best.mean_holdout_nll),
        "best_params": {
            "hidden_layers": hidden,
            "learning_rate": float(best.learning_rate),
            "batch_size": float(args.batch_size),
            "activation": "relu",
            "grid_size": "n",
            "dropout": 0.0,
            "weight_decay": 0.0,
        },
        "max_epochs_used_for_tuning": args.max_epochs,
        "patience_used_for_tuning": args.patience,
        "n_configurations_complete": int(len(complete)),
        "search_space": {"hidden_layers": ARCHITECTURES, "learning_rate": list(args.learning_rates),
                         "weight_decay": [0.0], "dropout": [0.0], "batch_size": [float(args.batch_size)],
                         "activation": ["relu"], "grid_size": ["n"]},
    }
    save_json(tuning_dir / f"{prefix}_best_params.json", payload)
    print(agg.head(10).to_string(index=False))
    print(f"\nselected: hidden_layers={hidden} learning_rate={best.learning_rate:g} mean_holdout_nll={best.mean_holdout_nll:.6f}")
    print(f"wrote {tuning_dir / f'{prefix}_best_params.json'} and {tuning_dir / f'{prefix}_grid_summary.csv'}")


def main() -> None:
    args = parse_args()
    defaults = MODEL_DEFAULTS[args.model_type]
    args.max_epochs = defaults["max_epochs"] if args.max_epochs is None else args.max_epochs
    args.patience = defaults["patience"] if args.patience is None else args.patience
    args.learning_rates = ([float(v) for v in args.learning_rates.split(",") if v.strip()] if args.learning_rates
                           else list(defaults["learning_rates"]))
    prefix = defaults["prefix"]
    if args.combine is not None:
        combine(args)
        return

    if args.smoke_test:
        args.max_epochs = 3
        args.folds = 2
        args.arch_end = min(args.arch_end, args.arch_start + 2)
        args.arch_ids = None
        args.learning_rates = args.learning_rates[:2]

    FlexiHazardModel, truncate_at_tau, make_expanded_data = import_flexihaz()
    output_dir = Path(args.output_dir)
    tuning_dir = ensure_dir(output_dir / "tuning")
    arch_ids = ([int(v) for v in args.arch_ids.split(",") if v.strip()] if args.arch_ids
                else list(range(args.arch_start, args.arch_end)))
    tag = args.tag or (args.arch_ids.replace(",", "-") if args.arch_ids else f"{args.arch_start}_{args.arch_end}")
    results_csv = Path(args.results_csv) if args.results_csv else tuning_dir / f"{prefix}_grid_folds_{tag}.csv"

    data, tau = tuning_dataset(output_dir, args.tuning_seed)
    T, delta, X, Z = data["T"], data["delta"], data["X"], data["Z"]

    done = set()
    if results_csv.exists() and args.skip_completed:
        prev = pd.read_csv(results_csv)
        if "learning_rate" not in prev.columns:
            prev["learning_rate"] = args.learning_rates[0]
        done = set(zip(prev.arch_idx.astype(int), prev.learning_rate.astype(float), prev.fold.astype(int)))
    write_header = not results_csv.exists()

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.fold_seed)
    splits = list(kf.split(X))  # common folds for every configuration

    print(f"model_type={args.model_type} archs {arch_ids} of {len(ARCHITECTURES)}, lrs={args.learning_rates}, "
          f"folds={args.folds}, max_epochs={args.max_epochs}, patience={args.patience}, n={T.shape[0]}", flush=True)

    with results_csv.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if write_header:
            writer.writeheader(); fh.flush()
        for arch_idx in arch_ids:
            hidden = tuple(ARCHITECTURES[arch_idx])
            for lr in args.learning_rates:
                for fold_id, (tr, va) in enumerate(splits):
                    if (arch_idx, float(lr), fold_id) in done:
                        print(f"arch {arch_idx} {hidden} lr {lr:g} fold {fold_id}: already done, skipping", flush=True)
                        continue
                    t0 = time.perf_counter()
                    model = FlexiHazardModel(**model_kwargs(args, hidden, lr, args.fold_seed + arch_idx * 1000 + fold_id, tau))
                    model.fit(T[tr], delta[tr], X[tr], Z=Z[tr])
                    nll = evaluate_on_holdout(model, T[va], delta[va], X[va], Z[va], truncate_at_tau, make_expanded_data)
                    row = dict(arch_idx=arch_idx, hidden_layers=str(list(hidden)), learning_rate=lr, fold=fold_id,
                               holdout_nll=nll, best_inner_val_loss=float(np.min(model.train_history["val_loss"])),
                               epochs_run=int(len(model.train_history["val_loss"])),
                               fit_time_sec=round(time.perf_counter() - t0, 1))
                    writer.writerow(row); fh.flush()
                    print(f"arch {arch_idx} {hidden} lr {lr:g} fold {fold_id}: holdout_nll={nll:.6f} "
                          f"epochs={row['epochs_run']} time={row['fit_time_sec']}s", flush=True)
    print(f"GRID DONE archs {arch_ids} -> {results_csv}", flush=True)


if __name__ == "__main__":
    main()
