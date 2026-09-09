"""Per-fold Random Survival Forest tuning for the Rotterdam experiment."""

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
    ensure_dir,
    inner_train_val_indices,
    load_arrays_and_folds,
    save_json,
    tuning_dir,
)
from rotterdam_metrics import predict_rsf_survival_matrix, structured_y

MODEL_NAME = "random_survival_forest"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune RSF structure parameters on a Rotterdam outer fold.")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument(
        "--design",
        choices=["standard", "er_interaction"],
        default="standard",
        help="Rotterdam design: standard uses Z=[chemo,hormon]; er_interaction uses ER-stratified treatment interactions.",
    )
    parser.add_argument("--fold-id", type=int, default=0)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-estimators-tuning", type=int, default=150)
    parser.add_argument("--final-n-estimators", type=int, default=500)
    parser.add_argument("--seed", type=int, default=332211)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def valid_times_for_ibs(train_T: np.ndarray, val_T: np.ndarray, val_delta: np.ndarray, candidate_times: np.ndarray) -> np.ndarray:
    train_T = np.asarray(train_T, dtype=float)
    val_T = np.asarray(val_T, dtype=float)
    val_delta = np.asarray(val_delta).astype(bool)
    eps = 1e-8
    max_allowed = min(float(np.max(train_T)), float(np.max(val_T))) - eps
    min_allowed = max(0.0, float(np.min(val_T)))
    times = np.asarray(candidate_times, dtype=float)
    mask = (times >= min_allowed) & (times <= max_allowed)
    mask &= np.array([np.any((val_T <= t) & val_delta) for t in times], dtype=bool)
    mask &= np.array([np.any(val_T > t) for t in times], dtype=bool)
    return times[mask]


def sample_params(trial) -> dict[str, Any]:
    min_samples_leaf = trial.suggest_categorical("min_samples_leaf", [5, 10, 15, 25, 50, 100])
    min_samples_split = trial.suggest_categorical("min_samples_split", [10, 20, 50, 100, 200])
    if min_samples_split < 2 * min_samples_leaf:
        min_samples_split = 2 * min_samples_leaf
    return {
        "min_samples_split": int(min_samples_split),
        "min_samples_leaf": int(min_samples_leaf),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0]),
        "max_depth": trial.suggest_categorical("max_depth", [None, 5, 10, 20, 40]),
        "max_samples": trial.suggest_categorical("max_samples", [None, 0.5, 0.7, 0.9]),
    }


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.n_trials = min(args.n_trials, 3)
        args.n_estimators_tuning = min(args.n_estimators_tuning, 20)
        args.n_jobs = 1

    out_path = best_params_path(args.results_dir, args.fold_id, MODEL_NAME)
    if out_path.exists() and not args.overwrite:
        print(f"RSF tuning already exists, skipping: {out_path}")
        return

    try:
        import optuna
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.metrics import integrated_brier_score
    except ImportError as exc:
        raise ImportError("RSF tuning requires optuna and scikit-survival.") from exc

    data = load_arrays_and_folds(
        args.data_path,
        args.results_dir,
        seed=args.seed,
        design=args.design,
    )
    X = np.column_stack([data["X"], data["Z"]])
    T = data["T"]
    delta = data["delta"]
    folds = data["folds"]
    eval_times = data["eval_times"]

    outer_train_idx = folds.loc[folds["fold_id"] != args.fold_id, "row_id"].to_numpy(dtype=int)
    train_idx, val_idx = inner_train_val_indices(outer_train_idx, val_fraction=0.20, seed=args.seed + args.fold_id)

    X_train, X_val = X[train_idx], X[val_idx]
    T_train, T_val = T[train_idx], T[val_idx]
    d_train, d_val = delta[train_idx], delta[val_idx]
    y_train = structured_y(T_train, d_train)
    y_val = structured_y(T_val, d_val)
    times = valid_times_for_ibs(T_train, T_val, d_val, eval_times)
    if times.size < 2:
        raise ValueError("Too few valid validation times for RSF IBS tuning.")

    trial_rows: list[dict[str, Any]] = []

    def objective(trial) -> float:
        params = sample_params(trial)
        t0 = time.perf_counter()
        model = RandomSurvivalForest(
            n_estimators=args.n_estimators_tuning,
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            max_depth=params["max_depth"],
            max_samples=params["max_samples"],
            n_jobs=args.n_jobs,
            random_state=args.seed + 1000 * trial.number + args.fold_id,
        )
        model.fit(X_train, y_train)
        S_val = predict_rsf_survival_matrix(model, X_val, times)
        value = float(integrated_brier_score(y_train, y_val, S_val, times))
        row = {
            "trial_number": int(trial.number),
            "value": value,
            "ibs": value,
            "trial_time_sec": float(time.perf_counter() - t0),
            "n_valid_times": int(times.size),
            "n_estimators_tuning": int(args.n_estimators_tuning),
            **params,
        }
        trial_rows.append(row)
        for k, v in row.items():
            try:
                trial.set_user_attr(k, v)
            except Exception:
                pass
        return value

    sampler = optuna.samplers.TPESampler(seed=args.seed + args.fold_id)
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name=f"rsf_rotterdam_fold_{args.fold_id}")
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, n_jobs=1)

    best_params = sample_params(study.best_trial)
    payload = {
        "model": MODEL_NAME,
        "fold_id": int(args.fold_id),
        "objective": "validation_integrated_brier_score",
        "best_value": float(study.best_value),
        "best_trial_number": int(study.best_trial.number),
        "best_params": best_params,
        "n_estimators_tuning": int(args.n_estimators_tuning),
        "recommended_final_n_estimators": int(args.final_n_estimators),
        "validation_fraction": 0.20,
        "inner_train_size": int(train_idx.size),
        "inner_val_size": int(val_idx.size),
        "n_valid_times": int(times.size),
        "seed": int(args.seed),
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

    print(f"Saved RSF tuning: {out_path}")
    print(best_params)


if __name__ == "__main__":
    main()
