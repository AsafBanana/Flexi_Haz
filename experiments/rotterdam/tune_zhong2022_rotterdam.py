"""Per-fold tuning for Zhong et al. (2022) neural PH on Rotterdam.

This script reuses the uploaded Zhong implementation without changing the model
itself.  It only adapts the data interface to the Rotterdam experiment:
- same X/Z split used by the other Rotterdam runners;
- 5 outer folds supplied by rotterdam_io;
- inside each outer training fold, 20% is used as validation for tuning.
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rotterdam_io import (
    DEFAULT_DATA_PATH,
    DEFAULT_RESULTS_DIR,
    best_params_path,
    inner_train_val_indices,
    load_arrays_and_folds,
    save_json,
    tuning_dir,
)

import zhong2022_ph_common as zhong_common

MODEL_NAME = "zhong2022_ph"

ZHONG_SEARCH_SPACE = {
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
}


def patch_zhong_column_names(input_dim_X: int, input_dim_Z: int) -> None:
    """Patch only column labels used by Cox initialization.

    The uploaded simulation helper has X_COLUMNS fixed to X0..X2 because the
    simulation had three X variables. Rotterdam has more X columns, so the Cox
    initialization DataFrame needs matching labels. This does not alter the
    Zhong model architecture, loss, prediction, or training procedure.
    """
    zhong_common.X_COLUMNS = [f"X{j}" for j in range(int(input_dim_X))]
    zhong_common.Z_COLUMNS = [f"Z{j}" for j in range(int(input_dim_Z))]
    zhong_common.ALL_COLUMNS = zhong_common.X_COLUMNS + zhong_common.Z_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune Zhong et al. (2022) neural PH on one Rotterdam outer fold.")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument(
        "--design",
        choices=["standard", "er_interaction"],
        default="standard",
        help="Rotterdam design: standard uses Z=[chemo,hormon]; er_interaction uses ER-stratified treatment interactions.",
    )
    parser.add_argument("--fold-id", type=int, default=0)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=100000)
    parser.add_argument("--val-frac", type=float, default=0.20)
    parser.add_argument("--cox-penalizer", type=float, default=0.0)
    parser.add_argument("--lmbd-l1", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=221100)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def sample_params(trial) -> dict[str, Any]:
    hidden_layer_options = ZHONG_SEARCH_SPACE["hidden_layers"]
    hidden_layers_id = trial.suggest_categorical(
        "hidden_layers_id",
        list(range(len(hidden_layer_options))),
    )
    hidden_layers = hidden_layer_options[int(hidden_layers_id)]
    return {
        "hidden_layers": [int(h) for h in hidden_layers],
        "hidden_layers_id": int(hidden_layers_id),
        "learning_rate": float(
            trial.suggest_float(
                "learning_rate",
                ZHONG_SEARCH_SPACE["learning_rate"][0],
                ZHONG_SEARCH_SPACE["learning_rate"][1],
                log=True,
            )
        ),
    }


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.n_trials = min(args.n_trials, 2)
        args.max_epochs = min(args.max_epochs, 5)
        args.patience = min(args.patience, 2)

    out_path = best_params_path(args.results_dir, args.fold_id, MODEL_NAME)
    if out_path.exists() and not args.overwrite:
        print(f"Zhong tuning already exists, skipping: {out_path}")
        return

    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Zhong tuning requires optuna and tensorflow.") from exc

    data = load_arrays_and_folds(
        args.data_path,
        args.results_dir,
        seed=args.seed,
        design=args.design,
    )
    X, Z, T, delta = data["X"], data["Z"], data["T"], data["delta"]
    folds = data["folds"]
    patch_zhong_column_names(X.shape[1], Z.shape[1])

    outer_train_idx = folds.loc[folds["fold_id"] != args.fold_id, "row_id"].to_numpy(dtype=int)
    train_idx, val_idx = inner_train_val_indices(outer_train_idx, val_fraction=args.val_frac, seed=args.seed + args.fold_id)

    trial_rows: list[dict[str, Any]] = []

    def objective(trial) -> float:
        params = sample_params(trial)
        t0 = time.perf_counter()
        notes = ""
        try:
            result = zhong_common.fit_zhong_model(
                X=X,
                Z=Z,
                T=T,
                delta=delta,
                train_idx=train_idx,
                val_idx=val_idx,
                hidden_layers=params["hidden_layers"],
                learning_rate=params["learning_rate"],
                max_epochs=args.max_epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                lmbd_l1=args.lmbd_l1,
                seed=args.seed + 1000 * trial.number + args.fold_id,
                cox_penalizer=args.cox_penalizer,
                verbose=0,
            )
            value = float(result.val_loss)
            n_epochs_run = int(result.n_epochs_run)
            best_epoch = result.best_epoch if result.best_epoch is not None else np.nan
            train_loss = float(result.train_loss)
        except Exception as exc:  # noqa: BLE001
            value = float("inf")
            n_epochs_run = 0
            best_epoch = np.nan
            train_loss = np.nan
            notes = f"failed: {type(exc).__name__}: {exc}"

        row = {
            "trial_number": int(trial.number),
            "value": value,
            "val_loss": value,
            "train_loss": train_loss,
            "trial_time_sec": float(time.perf_counter() - t0),
            "n_epochs_run": n_epochs_run,
            "best_epoch": best_epoch,
            "notes": notes,
            "hidden_layers": str(params["hidden_layers"]),
            "hidden_layers_id": int(params["hidden_layers_id"]),
            "learning_rate": float(params["learning_rate"]),
            "lmbd_l1": float(args.lmbd_l1),
            "batch_size": int(args.batch_size),
            "max_epochs": int(args.max_epochs),
            "patience": int(args.patience),
            "val_frac": float(args.val_frac),
            "cox_penalizer": float(args.cox_penalizer),
            "inner_train_size": int(train_idx.size),
            "inner_val_size": int(val_idx.size),
        }
        trial_rows.append(row)
        tdir = tuning_dir(args.results_dir, args.fold_id)
        pd.DataFrame(trial_rows).to_csv(tdir / f"{MODEL_NAME}_trials.csv", index=False)
        if not np.isfinite(value):
            raise RuntimeError(notes)
        return value

    sampler = optuna.samplers.TPESampler(seed=args.seed + args.fold_id)
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name=f"zhong2022_ph_rotterdam_fold_{args.fold_id}")
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, n_jobs=1)

    best_params_raw = dict(study.best_params)
    hidden_layers_id = int(best_params_raw["hidden_layers_id"])
    best_params = {
        "hidden_layers": [int(h) for h in ZHONG_SEARCH_SPACE["hidden_layers"][hidden_layers_id]],
        "hidden_layers_id": hidden_layers_id,
        "learning_rate": float(best_params_raw["learning_rate"]),
        "batch_size": int(args.batch_size),
        "max_epochs": int(args.max_epochs),
        "patience": int(args.patience),
        "val_frac": float(args.val_frac),
        "cox_penalizer": float(args.cox_penalizer),
        "lmbd_l1": float(args.lmbd_l1),
    }
    payload = {
        "model": MODEL_NAME,
        "fold_id": int(args.fold_id),
        "objective": "validation_cox_partial_negative_log_likelihood",
        "best_value": float(study.best_value),
        "best_trial_number": int(study.best_trial.number),
        "best_params": best_params,
        "validation_fraction": float(args.val_frac),
        "inner_train_size": int(train_idx.size),
        "inner_val_size": int(val_idx.size),
        "seed": int(args.seed),
        "x_columns": list(data["metadata"]["x_columns"]),
        "z_columns": list(data["metadata"]["z_columns"]),
    }
    save_json(out_path, payload)
    tdir = tuning_dir(args.results_dir, args.fold_id)
    pd.DataFrame(trial_rows).to_csv(tdir / f"{MODEL_NAME}_trials.csv", index=False)
    pd.DataFrame({"inner_train_row_id": pd.Series(train_idx), "inner_val_row_id": pd.Series(val_idx)}).to_csv(
        tdir / f"{MODEL_NAME}_inner_split.csv", index=False
    )
    try:
        with (tdir / f"{MODEL_NAME}_study.pkl").open("wb") as f:
            pickle.dump(study, f)
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not pickle study: {exc}")

    print(f"Saved Zhong tuning: {out_path}")
    print(best_params)


if __name__ == "__main__":
    main()
