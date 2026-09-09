"""Run Random Survival Forest on the Rotterdam breast-cancer data."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

import numpy as np

from rotterdam_io import DEFAULT_DATA_PATH, DEFAULT_RESULTS_DIR, best_params_path, completed, fold_file, load_arrays_and_folds, load_json, save_json, write_summary
from rotterdam_metrics import evaluate_real_survival_predictions, predict_rsf_survival_matrix, structured_y

MODEL_NAME = "random_survival_forest"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RSF Rotterdam folds.")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument(
        "--design",
        choices=["standard", "er_interaction"],
        default="standard",
        help="Rotterdam design: standard uses Z=[chemo,hormon]; er_interaction uses ER-stratified treatment interactions.",
    )
    parser.add_argument("--start-fold", type=int, default=0)
    parser.add_argument("--end-fold", type=int, default=5)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-estimators-tuning", type=int, default=150)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=332211)
    parser.add_argument("--skip-completed", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-tuning", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def run_tuning_if_needed(args: argparse.Namespace, fold_id: int) -> None:
    path = best_params_path(args.results_dir, fold_id, MODEL_NAME)
    if path.exists() and not args.overwrite_tuning:
        return
    cmd = [
        sys.executable, "tune_rsf_rotterdam.py",
        "--data-path", args.data_path,
        "--results-dir", args.results_dir,
        "--design", args.design,
        "--fold-id", str(fold_id),
        "--n-trials", str(args.n_trials),
        "--n-estimators-tuning", str(args.n_estimators_tuning),
        "--final-n-estimators", str(args.n_estimators),
        "--seed", str(args.seed),
        "--n-jobs", str(args.n_jobs),
    ]
    from pathlib import Path
    cmd[1] = str(Path(__file__).with_name("tune_rsf_rotterdam.py"))
    if args.overwrite_tuning:
        cmd.append("--overwrite")
    if args.smoke_test:
        cmd.append("--smoke-test")
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.end_fold = min(args.end_fold, 1)
        args.n_trials = min(args.n_trials, 3)
        args.n_estimators_tuning = min(args.n_estimators_tuning, 20)
        args.n_estimators = min(args.n_estimators, 20)
        args.n_jobs = 1

    try:
        from sksurv.ensemble import RandomSurvivalForest
    except ImportError as exc:
        raise ImportError("RSF runner requires scikit-survival.") from exc

    data = load_arrays_and_folds(
        args.data_path,
        args.results_dir,
        seed=args.seed,
        design=args.design,
    )
    X_full = np.column_stack([data["X"], data["Z"]])
    T, delta = data["T"], data["delta"]
    folds, eval_times = data["folds"], data["eval_times"]

    for fold_id in range(args.start_fold, args.end_fold):
        if args.skip_completed and not args.overwrite and completed(args.results_dir, MODEL_NAME, fold_id):
            print(f"fold={fold_id:02d} already completed, skipping")
            continue
        print(f"Running {MODEL_NAME}, fold={fold_id:02d}")
        try:
            run_tuning_if_needed(args, fold_id)
            payload = load_json(best_params_path(args.results_dir, fold_id, MODEL_NAME))
            params = payload["best_params"]
            train_idx = folds.loc[folds["fold_id"] != fold_id, "row_id"].to_numpy(dtype=int)
            test_idx = folds.loc[folds["fold_id"] == fold_id, "row_id"].to_numpy(dtype=int)

            model = RandomSurvivalForest(
                n_estimators=args.n_estimators,
                min_samples_split=int(params["min_samples_split"]),
                min_samples_leaf=int(params["min_samples_leaf"]),
                max_features=params["max_features"],
                max_depth=params["max_depth"],
                max_samples=params["max_samples"],
                n_jobs=args.n_jobs,
                random_state=args.seed + fold_id,
            )
            t0 = time.perf_counter()
            model.fit(X_full[train_idx], structured_y(T[train_idx], delta[train_idx]))
            training_time = time.perf_counter() - t0

            t1 = time.perf_counter()
            S_hat = predict_rsf_survival_matrix(model, X_full[test_idx], eval_times)
            prediction_time = time.perf_counter() - t1

            extra = {
                "params_path": str(best_params_path(args.results_dir, fold_id, MODEL_NAME)),
                "n_estimators": int(args.n_estimators),
                "n_events_train": int(delta[train_idx].sum()),
                "n_events_test": int(delta[test_idx].sum()),
                **params,
            }
            summary, auc_df, brier_df = evaluate_real_survival_predictions(
                model_name=MODEL_NAME,
                fold_id=fold_id,
                train_T=T[train_idx],
                train_delta=delta[train_idx],
                test_T=T[test_idx],
                test_delta=delta[test_idx],
                S_hat=S_hat,
                eval_times=eval_times,
                training_time_sec=training_time,
                prediction_time_sec=prediction_time,
                extra_summary=extra,
            )
            write_summary(fold_file(args.results_dir, MODEL_NAME, fold_id, "summary"), summary)
            auc_df.to_csv(fold_file(args.results_dir, MODEL_NAME, fold_id, "auc_by_time"), index=False)
            brier_df.to_csv(fold_file(args.results_dir, MODEL_NAME, fold_id, "brier_by_time"), index=False)
            if args.save_predictions:
                np.savez_compressed(fold_file(args.results_dir, MODEL_NAME, fold_id, "predictions", suffix="npz"), S_hat=S_hat, eval_times=eval_times, test_idx=test_idx)
            save_json(fold_file(args.results_dir, MODEL_NAME, fold_id, "log", suffix="json"), {"status": "success", "fold_id": fold_id, "params": params})
            print(f"fold={fold_id:02d} done, ibs={summary['ibs']}, c_index={summary['c_index']}")
        except Exception as exc:  # noqa: BLE001
            row = {"model": MODEL_NAME, "fold_id": int(fold_id), "status": "failed", "error_type": type(exc).__name__, "error_message": str(exc)}
            write_summary(fold_file(args.results_dir, MODEL_NAME, fold_id, "summary"), row)
            save_json(fold_file(args.results_dir, MODEL_NAME, fold_id, "log", suffix="json"), row)
            print(f"fold={fold_id:02d} failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
