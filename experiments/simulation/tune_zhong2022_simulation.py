"""Tune the Zhong et al. (2022) neural proportional hazards comparator.

This script follows the saved-data simulation workflow used by the FLEXI-Haz
simulation framework. It tunes the model once on a saved training replication
and stores the best hyperparameters under simulation_outputs/tuning/.
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from simulation_io import DEFAULT_OUTPUT_DIR, ensure_dir, load_train_dataset, save_json
from zhong2022_ph_common import fit_zhong_model, make_random_train_val_split, set_all_seeds

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune Zhong et al. (2022) neural PH simulation comparator.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tuning-rep", type=int, default=0)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=5000)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=100000)
    parser.add_argument("--val-frac", type=float, default=0.33)
    parser.add_argument("--seed", type=int, default=20240222)
    parser.add_argument("--cox-penalizer", type=float, default=0.0)
    parser.add_argument("--lmbd-l1", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.output_dir = str(Path(args.output_dir) / "smoke_test")
        args.n_trials = min(args.n_trials, 3)
        args.folds = min(args.folds, 2)
        args.max_epochs = min(args.max_epochs, 20)
        args.patience = min(args.patience, 5)

    try:
        import optuna
        from sklearn.model_selection import KFold
    except ImportError as exc:
        raise ImportError("tune_zhong2022_simulation.py requires optuna and scikit-learn.") from exc

    output_dir = Path(args.output_dir)
    tuning_dir = ensure_dir(output_dir / "tuning")
    data = load_train_dataset(output_dir, args.tuning_rep)
    X, Z, T, delta = data["X"], data["Z"], data["T"], data["delta"]

    n = int(T.shape[0])
    set_all_seeds(args.seed)
    trial_rows: list[dict] = []

    def objective(trial):
        hidden_layer_options = ZHONG_SEARCH_SPACE["hidden_layers"]
        hidden_layers_id = trial.suggest_categorical(
            "hidden_layers_id",
            list(range(len(hidden_layer_options))),
        )
        hidden_layers = hidden_layer_options[int(hidden_layers_id)]

        params = {
            "hidden_layers": [int(h) for h in hidden_layers],
            "hidden_layers_id": int(hidden_layers_id),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                ZHONG_SEARCH_SPACE["learning_rate"][0],
                ZHONG_SEARCH_SPACE["learning_rate"][1],
                log=True,
            ),
            "lmbd_l1": float(args.lmbd_l1),
        }

        fold_losses = []
        fold_epochs = []
        fold_times = []
        notes = []
        t_trial = time.perf_counter()

        if args.folds <= 1:
            splits = [make_random_train_val_split(n, args.val_frac, args.seed + trial.number)]
        else:
            kf = KFold(n_splits=int(args.folds), shuffle=True, random_state=args.seed + trial.number)
            splits = list(kf.split(np.arange(n)))

        for fold_id, (train_idx, val_idx) in enumerate(splits):
            t_fold = time.perf_counter()
            try:
                result = fit_zhong_model(
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
                    lmbd_l1=params["lmbd_l1"],
                    seed=args.seed + 1000 * trial.number + fold_id,
                    cox_penalizer=args.cox_penalizer,
                    verbose=0,
                )
                fold_losses.append(float(result.val_loss))
                fold_epochs.append(int(result.n_epochs_run))
            except Exception as exc:  # noqa: BLE001
                notes.append(f"fold {fold_id} failed: {type(exc).__name__}: {exc}")
            fold_times.append(time.perf_counter() - t_fold)

        if len(fold_losses) == 0:
            raise RuntimeError("All folds failed. " + " | ".join(notes))

        value = float(np.mean(fold_losses))
        trial_rows.append(
            {
                "trial_number": trial.number,
                "value": value,
                "mean_val_loss": value,
                "sd_val_loss": float(np.std(fold_losses, ddof=0)) if len(fold_losses) else np.nan,
                "n_successful_folds": int(len(fold_losses)),
                "n_folds": int(len(splits)),
                "mean_epochs_run": float(np.mean(fold_epochs)) if fold_epochs else np.nan,
                "trial_time_sec": float(time.perf_counter() - t_trial),
                "mean_fold_time_sec": float(np.mean(fold_times)) if fold_times else np.nan,
                "notes": " | ".join(notes),
                "hidden_layers": str(params["hidden_layers"]),
                "hidden_layers_id": int(params["hidden_layers_id"]),
                "learning_rate": float(params["learning_rate"]),
                "lmbd_l1": float(params["lmbd_l1"]),
                "n_hidden_layers": int(len(params["hidden_layers"])),
                "hidden_layers_nodes": str(params["hidden_layers"]),
                "batch_size": int(args.batch_size),
                "max_epochs": int(args.max_epochs),
                "patience": int(args.patience),
                "val_frac": float(args.val_frac),
                "cox_penalizer": float(args.cox_penalizer),
            }
        )
        pd.DataFrame(trial_rows).to_csv(tuning_dir / f"{MODEL_NAME}_trials.csv", index=False)
        return value

    study = optuna.create_study(direction="minimize", study_name=f"{MODEL_NAME}_tuning")
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    best_params_raw = dict(study.best_params)
    best_hidden_layers_id = int(best_params_raw["hidden_layers_id"])
    best_hidden_layers = ZHONG_SEARCH_SPACE["hidden_layers"][best_hidden_layers_id]

    best_params = {
        "hidden_layers": [int(h) for h in best_hidden_layers],
        "hidden_layers_id": best_hidden_layers_id,
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
        "objective": "mean_validation_cox_partial_negative_log_likelihood",
        "best_value": float(study.best_value),
        "best_trial_number": int(study.best_trial.number),
        "best_params": best_params,
        "folds": int(args.folds),
        "tuning_rep": int(args.tuning_rep),
        "seed": int(args.seed),
        "n_train_subjects": int(n),
    }
    save_json(tuning_dir / f"{MODEL_NAME}_best_params.json", payload)
    pd.DataFrame(trial_rows).to_csv(tuning_dir / f"{MODEL_NAME}_trials.csv", index=False)
    with open(tuning_dir / f"{MODEL_NAME}_study.pkl", "wb") as f:
        pickle.dump(study, f)

    print("Best value:", study.best_value)
    print("Best params:", best_params)
    print("Saved to:", tuning_dir)


if __name__ == "__main__":
    main()
