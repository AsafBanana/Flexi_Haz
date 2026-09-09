"""Run partially linear FLEXI-Haz on the Rotterdam breast-cancer data."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from rotterdam_io import (
    DEFAULT_DATA_PATH,
    DEFAULT_RESULTS_DIR,
    best_params_path,
    completed,
    fold_file,
    load_arrays_and_folds,
    load_json,
    save_json,
    write_summary,
)
from rotterdam_metrics import evaluate_real_survival_predictions, flexihaz_treatment_effects

MODEL_NAME = "flexihaz_partial_linear"


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
    parser = argparse.ArgumentParser(description="Run partially linear FLEXI-Haz Rotterdam folds.")
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
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--tuning-max-epochs", type=int, default=120)
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--projection-max-epochs", type=int, default=2000)
    parser.add_argument("--projection-patience", type=int, default=30)
    parser.add_argument("--projection-n-splits", type=int, default=5)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--seed", type=int, default=221100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-jobs-tuning", type=int, default=1)
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
        sys.executable, str(Path(__file__).with_name("tune_flexihaz_rotterdam.py")),
        "--data-path", args.data_path,
        "--results-dir", args.results_dir,
        "--design", args.design,
        "--model-type", "partial_linear",
        "--fold-id", str(fold_id),
        "--n-trials", str(args.n_trials),
        "--max-epochs", str(args.tuning_max_epochs),
        "--seed", str(args.seed),
        "--device", args.device,
        "--n-jobs", str(args.n_jobs_tuning),
    ]
    if args.tau is not None:
        cmd += ["--tau", str(args.tau)]
    if args.overwrite_tuning:
        cmd.append("--overwrite")
    if args.smoke_test:
        cmd.append("--smoke-test")
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.end_fold = min(args.end_fold, 1)
        args.n_trials = min(args.n_trials, 2)
        args.tuning_max_epochs = min(args.tuning_max_epochs, 5)
        args.max_epochs = min(args.max_epochs, 5)
        args.patience = min(args.patience, 2)
        args.projection_max_epochs = min(args.projection_max_epochs, 5)
        args.projection_patience = min(args.projection_patience, 2)
        args.projection_n_splits = 2

    FlexiHazardModel = import_flexihaz_model()
    data = load_arrays_and_folds(
        args.data_path,
        args.results_dir,
        seed=args.seed,
        design=args.design,
    )
    X, Z, T, delta = data["X"], data["Z"], data["T"], data["delta"]
    folds, eval_times = data["folds"], data["eval_times"]

    for fold_id in range(args.start_fold, args.end_fold):
        if args.skip_completed and not args.overwrite and completed(args.results_dir, MODEL_NAME, fold_id):
            print(f"fold={fold_id:02d} already completed, skipping")
            continue
        print(f"Running {MODEL_NAME}, fold={fold_id:02d}")
        try:
            run_tuning_if_needed(args, fold_id)
            best_payload = load_json(best_params_path(args.results_dir, fold_id, MODEL_NAME))
            best_params = dict(best_payload["best_params"])
            lr = best_params.get("learning_rate")

            train_idx = folds.loc[folds["fold_id"] != fold_id, "row_id"].to_numpy(dtype=int)
            test_idx = folds.loc[folds["fold_id"] == fold_id, "row_id"].to_numpy(dtype=int)

            model = FlexiHazardModel(
                **best_params,
                tau=args.tau,
                include_observed_times=True,
                max_epochs=args.max_epochs,
                patience=args.patience,
                compute_theta_variance=True,
                refine_theta=True,
                projection_max_epochs=args.projection_max_epochs,
                projection_patience=args.projection_patience,
                projection_n_splits=args.projection_n_splits,
                device=args.device,
                seed=args.seed + fold_id,
                verbose=0,
                num_workers=8,
                pin_memory=True,
                persistent_workers=True,
                non_blocking=True,
            )
            t0 = time.perf_counter()
            model.fit(T[train_idx], delta[train_idx], X[train_idx], Z=Z[train_idx])
            training_time = time.perf_counter() - t0

            t1 = time.perf_counter()
            S_hat = model.predict_survival(X[test_idx], eval_times, Z_new=Z[test_idx])
            prediction_time = time.perf_counter() - t1

            theta_df, theta_summary = flexihaz_treatment_effects(model, MODEL_NAME, fold_id, data["metadata"]["treatment_columns"])
            theta_df.to_csv(fold_file(args.results_dir, MODEL_NAME, fold_id, "treatment_effects"), index=False)

            extra = {
                "params_path": str(best_params_path(args.results_dir, fold_id, MODEL_NAME)),
                "n_events_train": int(delta[train_idx].sum()),
                "n_events_test": int(delta[test_idx].sum()),
                **theta_summary,
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
            save_json(fold_file(args.results_dir, MODEL_NAME, fold_id, "log", suffix="json"), {"status": "success", "fold_id": fold_id})
            print(f"fold={fold_id:02d} done, ibs={summary['ibs']}, c_index={summary['c_index']}")
        except Exception as exc:  # noqa: BLE001
            row = {"model": MODEL_NAME, "fold_id": int(fold_id), "status": "failed", "error_type": type(exc).__name__, "error_message": str(exc)}
            write_summary(fold_file(args.results_dir, MODEL_NAME, fold_id, "summary"), row)
            save_json(fold_file(args.results_dir, MODEL_NAME, fold_id, "log", suffix="json"), row)
            print(f"fold={fold_id:02d} failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
