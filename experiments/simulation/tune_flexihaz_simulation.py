"""Hyperparameter selection for FLEXI-Haz on one simulated dataset.

Uses the package function tune_flexihaz with 5-fold validation by default.

Examples
--------
Partial-linear smoke test:
    python tune_flexihaz_simulation.py --smoke-test --model-type partial_linear

No-linear smoke test:
    python tune_flexihaz_simulation.py --smoke-test --model-type nonlinear

Full tuning:
    python tune_flexihaz_simulation.py --model-type partial_linear --n-trials 30 --folds 5
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import pandas as pd

from simulation_io import DEFAULT_OUTPUT_DIR, ensure_dir, load_json, load_train_dataset, save_json


def import_flexihaz():
    """Import flexihaz, with an optional source-root fallback.

    If your package is not installed, set FLEXIHAZ_SOURCE_DIR to the directory
    containing the flexihaz package folder, or run `pip install -e .` first.
    """
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
    parser = argparse.ArgumentParser(description="Tune FLEXI-Haz on one simulated dataset.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-type", choices=["partial_linear", "nonlinear"], default="partial_linear")
    parser.add_argument("--tuning-rep", type=int, default=0)
    parser.add_argument("--tau", type=float, default=10.0)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=221100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.output_dir = str(Path(args.output_dir) / "smoke_test")
        args.n_trials = min(args.n_trials, 2)
        args.folds = min(args.folds, 2)
        args.max_epochs = min(args.max_epochs, 5)

    tune_flexihaz = import_flexihaz()
    output_dir = Path(args.output_dir)
    tuning_dir = ensure_dir(output_dir / "tuning")

    data = load_train_dataset(output_dir, args.tuning_rep)
    T = data["T"]
    delta = data["delta"]
    X = data["X"]
    Z = data["Z"]

    if args.model_type == "nonlinear":
        # No-linear version receives all covariates through X and no Z.
        X_fit = __import__("numpy").column_stack([X, Z])
        Z_fit = None
    else:
        X_fit = X
        Z_fit = Z

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
        T=T,
        delta=delta,
        X=X_fit,
        Z=Z_fit,
        tau=args.tau,
        n_trials=args.n_trials,
        seed=args.seed,
        device=args.device,
        verbose=args.verbose,
        search_space=search_space,
        folds=args.folds,
        max_epochs=args.max_epochs,
        n_jobs=args.n_jobs,
        enable_pruning=True,
        model_kwargs={
            "num_workers": 2,
            "pin_memory": True,
            "persistent_workers": True,
            "non_blocking": True,
            },
    )

    best_payload = {
        "model_type": args.model_type,
        "tuning_rep": args.tuning_rep,
        "best_value": result["best_value"],
        "best_params": result["best_params"],
        "tau": args.tau,
        "folds": args.folds,
        "n_trials": args.n_trials,
        "max_epochs_used_for_tuning": args.max_epochs,
        "n_jobs": args.n_jobs,
        "seed": args.seed,
        "search_space": search_space,
    }

    prefix = f"flexihaz_{args.model_type}"
    save_json(tuning_dir / f"{prefix}_best_params.json", best_payload)

    try:
        trials_df = result["study"].trials_dataframe()
        trials_df.to_csv(tuning_dir / f"{prefix}_trials.csv", index=False)
    except Exception as exc:  # noqa: BLE001
        pd.DataFrame([{"warning": f"Could not save trials dataframe: {exc}"}]).to_csv(
            tuning_dir / f"{prefix}_trials.csv", index=False
        )

    try:
        with (tuning_dir / f"{prefix}_study.pkl").open("wb") as f:
            pickle.dump(result["study"], f)
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not pickle Optuna study: {exc}")

    print(f"Saved best parameters to {tuning_dir / f'{prefix}_best_params.json'}")
    print(result["best_params"])


if __name__ == "__main__":
    main()
