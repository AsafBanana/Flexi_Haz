"""Per-fold FLEXI-Haz tuning for the Rotterdam experiment.

This script tunes hyperparameters inside one outer training fold using a 20%
validation split, matching the Section 7.2 protocol.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rotterdam_io import (
    DEFAULT_DATA_PATH,
    DEFAULT_RESULTS_DIR,
    best_params_path,
    ensure_dir,
    inner_train_val_indices,
    load_arrays_and_folds,
    save_json,
    tuning_dir,
)

MODEL_NAME_BY_TYPE = {
    "partial_linear": "flexihaz_partial_linear",
    "nonlinear": "flexihaz_nonlinear",
}


def import_tune_flexihaz():
    try:
        from flexihaz import tune_flexihaz
        return tune_flexihaz
    except ImportError:
        source_dir = os.environ.get("FLEXIHAZ_SOURCE_DIR")
        if source_dir:
            sys.path.insert(0, source_dir)
            from flexihaz import tune_flexihaz
            return tune_flexihaz
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune FLEXI-Haz on a Rotterdam outer fold.")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument(
        "--design",
        choices=["standard", "er_interaction"],
        default="standard",
        help="Rotterdam design: standard uses Z=[chemo,hormon]; er_interaction uses ER-stratified treatment interactions.",
    )
    parser.add_argument("--model-type", choices=["partial_linear", "nonlinear"], default="partial_linear")
    parser.add_argument("--fold-id", type=int, default=0)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--seed", type=int, default=221100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.n_trials = min(args.n_trials, 2)
        args.max_epochs = min(args.max_epochs, 5)
        args.verbose = 0

    model_name = MODEL_NAME_BY_TYPE[args.model_type]
    out_path = best_params_path(args.results_dir, args.fold_id, model_name)
    if out_path.exists() and not args.overwrite:
        print(f"Tuning already exists, skipping: {out_path}")
        return

    tune_flexihaz = import_tune_flexihaz()
    data = load_arrays_and_folds(
        args.data_path,
        args.results_dir,
        seed=args.seed,
        design=args.design,
    )
    X, Z, T, delta = data["X"], data["Z"], data["T"], data["delta"]
    folds = data["folds"]

    train_idx = folds.loc[folds["fold_id"] != args.fold_id, "row_id"].to_numpy(dtype=int)
    inner_train_idx, inner_val_idx = inner_train_val_indices(train_idx, val_fraction=0.20, seed=args.seed + args.fold_id)

    # tune_flexihaz already uses its own validation split when folds=1.  To force
    # the paper-style 20% validation subset, we tune on the outer-training data
    # with folds=1 and validation_fraction=0.20.  The saved inner indices allow
    # exact reproducibility of the intended split if a future tune_flexihaz API
    # accepts explicit validation indices.
    X_fit = X[train_idx]
    Z_fit = Z[train_idx]
    T_fit = T[train_idx]
    delta_fit = delta[train_idx]
    if args.model_type == "nonlinear":
        X_fit = np.column_stack([X_fit, Z_fit])
        Z_fit = None

    # search_space = {
    #     "hidden_layers": [
    #         [16], [32], [64], [128], [256],
    #         [16, 16], [32, 32], [64, 64], [128, 128],
    #         [32, 64], [64, 128], [128, 64],
    #         [32, 32, 32], [64, 64, 64], [128, 128, 128],
    #     ],
    #     "learning_rate": (1e-4, 5e-2),
    #     "weight_decay": (1e-6, 1e-2),
    #     "dropout": (0.0, 0.3),
    #     "batch_size": [100000],
    #     "activation": ["relu"],
    #     "grid_size": ["n"],
    # }

    search_space = {
        "hidden_layers": [
            [16],
            [32],
            [64],
            [128],
            [256],
            [16, 16],
            [16, 32],
            [32, 32],
            [32, 64],
            [64, 64],
            [64, 128],
            [128, 128],
            [16, 16, 16],
            [16, 16, 32],
            [32, 32, 32],
            [32, 32, 64],
            [64, 64, 64],
            [64, 64, 128],
            [128, 128, 128],
            [16, 16, 16, 16],
            [16, 16, 32, 32],
            [32, 32, 32, 32],
            [32, 32, 64, 64],
            [64, 64, 64, 64],
            [16, 16, 16, 16, 16],
            [16, 16, 32, 32, 32],
            [32, 32, 32, 32, 32],
        ],
        "learning_rate": (1e-3, 5e-2),
        #"weight_decay": (1e-6, 1e-2),
        #"dropout": (0.0, 0.3),
        "batch_size": [1e5],
        "activation": ["relu"],
        "grid_size": ["n"],
    }

    result = tune_flexihaz(
        T=T_fit,
        delta=delta_fit,
        X=X_fit,
        Z=Z_fit,
        tau=args.tau,
        n_trials=args.n_trials,
        seed=args.seed + 100 * args.fold_id,
        device=args.device,
        verbose=args.verbose,
        search_space=search_space,
        folds=1,
        max_epochs=args.max_epochs,
        n_jobs=args.n_jobs,
        enable_pruning=True,
        model_kwargs={
            "validation_fraction": 0.20,
            "num_workers": 2,
            "pin_memory": True,
            "persistent_workers": True,
            "non_blocking": True,
        },
    )

    tdir = tuning_dir(args.results_dir, args.fold_id)
    payload = {
        "model": model_name,
        "model_type": args.model_type,
        "fold_id": int(args.fold_id),
        "best_value": float(result["best_value"]),
        "best_params": result["best_params"],
        "tau": args.tau,
        "n_trials": int(args.n_trials),
        "max_epochs_used_for_tuning": int(args.max_epochs),
        "validation_fraction": 0.20,
        "outer_train_size": int(train_idx.size),
        "inner_train_size_reference": int(inner_train_idx.size),
        "inner_val_size_reference": int(inner_val_idx.size),
        "seed": int(args.seed),
        "search_space": search_space,
        "note": "tune_flexihaz uses validation_fraction=0.20 on the outer-training fold.",
    }
    save_json(out_path, payload)

    pd.DataFrame({"inner_train_row_id": pd.Series(inner_train_idx), "inner_val_row_id": pd.Series(inner_val_idx)}).to_csv(
        tdir / f"{model_name}_inner_split_reference.csv", index=False
    )

    try:
        result["study"].trials_dataframe().to_csv(tdir / f"{model_name}_trials.csv", index=False)
    except Exception as exc:  # noqa: BLE001
        pd.DataFrame([{"warning": f"Could not save trials dataframe: {exc}"}]).to_csv(tdir / f"{model_name}_trials.csv", index=False)

    try:
        with (tdir / f"{model_name}_study.pkl").open("wb") as f:
            pickle.dump(result["study"], f)
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not pickle Optuna study: {exc}")

    print(f"Saved FLEXI-Haz tuning: {out_path}")
    print(result["best_params"])


if __name__ == "__main__":
    main()
