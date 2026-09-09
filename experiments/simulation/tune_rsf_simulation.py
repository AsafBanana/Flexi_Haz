"""Tune Random Survival Forest hyperparameters on one saved simulation dataset.

This script tunes the structural RSF hyperparameters once using K-fold CV on one
saved training replication, usually train_rep_000.npz.  It follows the same
simulation-output layout used by the other simulation scripts.

Recommended workflow
--------------------
Tune structure with a moderate number of trees:
    python tune_rsf_simulation.py --n-trials 50 --folds 3 --n-estimators-tuning 150 --n-jobs 8

Then run final replications with more trees using the tuned structure:
    python run_rsf_replications.py --use-tuned-params --n-estimators 500 --n-jobs 16
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from simulation_io import DEFAULT_OUTPUT_DIR, ensure_dir, load_train_dataset, save_json

MODEL_NAME = "random_survival_forest"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune Random Survival Forest for the simulation study.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tuning-rep", type=int, default=0)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--n-estimators-tuning", type=int, default=150)
    parser.add_argument("--final-n-estimators", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=332211)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--timeout", type=float, default=None, help="Optional Optuna timeout in seconds.")
    return parser.parse_args()


def structured_y(T: np.ndarray, delta: np.ndarray):
    from sksurv.util import Surv

    return Surv.from_arrays(event=np.asarray(delta).astype(bool), time=np.asarray(T, dtype=np.float64))


def make_X_full(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    return np.column_stack([X, Z]).astype(np.float64)


def predict_survival_matrix(model, X_eval: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Predict survival on requested times with constant-tail extrapolation."""
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    surv_fns = model.predict_survival_function(X_eval)
    rows = []
    for fn in surv_fns:
        safe_times = times.copy()

        lower = 0.0
        upper = None
        domain = getattr(fn, "domain", None)
        if domain is not None and len(domain) == 2:
            lower = float(domain[0])
            upper = float(domain[1])

        x_grid = getattr(fn, "x", None)
        if x_grid is not None and len(x_grid) > 0:
            x_grid = np.asarray(x_grid, dtype=np.float64)
            lower = min(lower, float(x_grid[0]), 0.0)
            upper = float(x_grid[-1])

        safe_times = np.maximum(safe_times, lower)
        if upper is not None and np.isfinite(upper):
            safe_times = np.minimum(safe_times, upper)

        rows.append(fn(safe_times))
    return np.vstack(rows).astype(np.float64)


def valid_metric_times(train_T: np.ndarray, val_T: np.ndarray, val_delta: np.ndarray, candidate_times: np.ndarray) -> np.ndarray:
    """Keep times valid for IPCW IBS/AUC-like metrics.

    For tuning we primarily use IBS.  We still require at least one event by t
    and at least one subject with follow-up beyond t to avoid degenerate folds.
    """
    train_T = np.asarray(train_T, dtype=np.float64).reshape(-1)
    val_T = np.asarray(val_T, dtype=np.float64).reshape(-1)
    val_delta = np.asarray(val_delta).astype(bool).reshape(-1)
    candidate_times = np.asarray(candidate_times, dtype=np.float64).reshape(-1)

    if train_T.size == 0 or val_T.size == 0 or candidate_times.size == 0:
        return np.asarray([], dtype=np.float64)

    eps = 1e-8
    max_allowed = min(float(np.nanmax(train_T)), float(np.nanmax(val_T))) - eps
    min_allowed = max(0.0, float(np.nanmin(val_T)))
    mask = (candidate_times >= min_allowed) & (candidate_times <= max_allowed)
    mask &= np.array([np.any((val_T <= t) & val_delta) for t in candidate_times], dtype=bool)
    mask &= np.array([np.any(val_T > t) for t in candidate_times], dtype=bool)
    return candidate_times[mask]


def sample_rsf_params(trial) -> dict[str, Any]:
    """Sample structural RSF hyperparameters.

    n_estimators is intentionally not tuned here.  We tune tree structure using
    n_estimators_tuning, then use a larger fixed n_estimators in the final runs.
    """
    min_samples_leaf = trial.suggest_categorical("min_samples_leaf", [5, 10, 15, 25, 50, 100])
    min_samples_split = trial.suggest_categorical("min_samples_split", [10, 20, 50, 100, 200])

    # Keep a valid and interpretable tree configuration.
    if min_samples_split < 2 * min_samples_leaf:
        min_samples_split = 2 * min_samples_leaf

    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0])
    max_depth = trial.suggest_categorical("max_depth", [None, 5, 10, 20, 40])
    max_samples = trial.suggest_categorical("max_samples", [None, 0.5, 0.7, 0.9])

    return {
        "min_samples_split": int(min_samples_split),
        "min_samples_leaf": int(min_samples_leaf),
        "max_features": max_features,
        "max_depth": None if max_depth is None else int(max_depth),
        "max_samples": max_samples,
    }


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.output_dir = str(Path(args.output_dir) / "smoke_test")
        args.n_trials = min(args.n_trials, 3)
        args.folds = min(args.folds, 2)
        args.n_estimators_tuning = min(args.n_estimators_tuning, 20)
        args.n_jobs = 1

    try:
        import optuna
        from sklearn.model_selection import KFold
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.metrics import integrated_brier_score
    except ImportError as exc:
        raise ImportError(
            "tune_rsf_simulation.py requires optuna, scikit-learn, and scikit-survival. "
            "Install missing packages before running RSF tuning."
        ) from exc

    output_dir = Path(args.output_dir)
    tuning_dir = ensure_dir(output_dir / "tuning")
    data = load_train_dataset(output_dir, args.tuning_rep)

    X_full = make_X_full(data["X"], data["Z"])
    T = np.asarray(data["T"], dtype=np.float64)
    delta = np.asarray(data["delta"]).astype(bool)

    # Default metric times follow the simulation assignment convention.
    tau = float(data.get("tau", 30.0))
    candidate_times = np.arange(1.0, tau, 1.0, dtype=np.float64)

    if args.folds < 2:
        raise ValueError("RSF tuning requires --folds >= 2.")

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    trial_rows: list[dict[str, Any]] = []

    def objective(trial) -> float:
        params = sample_rsf_params(trial)
        fold_values: list[float] = []
        fold_times: list[float] = []
        fold_notes: list[str] = []
        t_trial = time.perf_counter()

        for fold_id, (train_idx, val_idx) in enumerate(kf.split(X_full)):
            X_train = X_full[train_idx]
            X_val = X_full[val_idx]
            T_train = T[train_idx]
            T_val = T[val_idx]
            delta_train = delta[train_idx]
            delta_val = delta[val_idx]

            times = valid_metric_times(T_train, T_val, delta_val, candidate_times)
            if times.size < 2:
                fold_notes.append(f"fold {fold_id}: fewer than 2 valid metric times")
                continue

            y_train = structured_y(T_train, delta_train)
            y_val = structured_y(T_val, delta_val)

            model = RandomSurvivalForest(
                n_estimators=args.n_estimators_tuning,
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
                max_features=params["max_features"],
                max_depth=params["max_depth"],
                max_samples=params["max_samples"],
                n_jobs=args.n_jobs,
                random_state=args.seed + 1000 * trial.number + fold_id,
            )

            t_fold = time.perf_counter()
            model.fit(X_train, y_train)
            S_val = predict_survival_matrix(model, X_val, times)
            fold_times.append(time.perf_counter() - t_fold)

            try:
                ibs = float(integrated_brier_score(y_train, y_val, S_val, times))
            except Exception as exc:  # noqa: BLE001
                fold_notes.append(f"fold {fold_id}: IBS failed: {type(exc).__name__}: {exc}")
                continue

            if np.isfinite(ibs):
                fold_values.append(ibs)

        if not fold_values:
            value = float("inf")
        else:
            value = float(np.mean(fold_values))

        row = {
            "trial_number": int(trial.number),
            "value": value,
            "mean_ibs": value,
            "sd_ibs": float(np.std(fold_values, ddof=1)) if len(fold_values) > 1 else np.nan,
            "n_successful_folds": int(len(fold_values)),
            "n_folds": int(args.folds),
            "mean_fold_time_sec": float(np.mean(fold_times)) if fold_times else np.nan,
            "trial_time_sec": float(time.perf_counter() - t_trial),
            "notes": " | ".join(fold_notes),
            "n_estimators_tuning": int(args.n_estimators_tuning),
            **params,
        }
        trial_rows.append(row)

        for key, val in row.items():
            if key not in {"notes"}:
                try:
                    trial.set_user_attr(key, val)
                except Exception:
                    pass
        if row["notes"]:
            trial.set_user_attr("notes", row["notes"])

        return value

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name="tune_rsf_simulation")
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, n_jobs=1)

    trials_df = pd.DataFrame(trial_rows)
    trials_path = tuning_dir / "random_survival_forest_trials.csv"
    trials_df.to_csv(trials_path, index=False)

    best_params = sample_rsf_params(study.best_trial)
    # sample_rsf_params is deterministic for a fixed completed trial's sampled values;
    # however, min_samples_split may be adjusted by the validity constraint above.
    best_payload = {
        "model": MODEL_NAME,
        "objective": "mean_cross_validated_integrated_brier_score",
        "best_value": float(study.best_value),
        "best_trial_number": int(study.best_trial.number),
        "best_params": best_params,
        "n_estimators_tuning": int(args.n_estimators_tuning),
        "recommended_final_n_estimators": int(args.final_n_estimators),
        "folds": int(args.folds),
        "tuning_rep": int(args.tuning_rep),
        "seed": int(args.seed),
    }
    save_json(tuning_dir / "random_survival_forest_best_params.json", best_payload)

    with (tuning_dir / "random_survival_forest_study.pkl").open("wb") as f:
        pickle.dump(study, f)

    print("RSF tuning finished")
    print(f"best_value={study.best_value}")
    print(f"best_params={best_params}")
    print(f"saved: {tuning_dir / 'random_survival_forest_best_params.json'}")
    print(f"saved: {trials_path}")


if __name__ == "__main__":
    main()
