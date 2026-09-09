"""Run Zhong et al. (2022) neural PH on Rotterdam breast-cancer folds.

The model implementation is reused from zhong2022_ph_common.py.  This runner
only provides the Rotterdam data split, per-fold tuning, result saving, and
metric aggregation interface used by the existing Rotterdam experiment.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from rotterdam_io import (
    DEFAULT_DATA_PATH,
    DEFAULT_RESULTS_DIR,
    best_params_path,
    completed,
    fold_file,
    inner_train_val_indices,
    load_arrays_and_folds,
    load_json,
    save_json,
    write_summary,
)
from rotterdam_metrics import evaluate_real_survival_predictions

import zhong2022_ph_common as zhong_common

MODEL_NAME = "zhong2022_ph"

DEFAULT_PARAMS = {
    "hidden_layers": [128, 128, 128, 128, 128, 128],
    "hidden_layers_id": -1,
    "learning_rate": 1e-3,
    "batch_size": 100000,
    "max_epochs": 5000,
    "patience": 35,
    "val_frac": 0.20,
    "cox_penalizer": 0.0,
    "lmbd_l1": 0.0,
}


def patch_zhong_column_names(input_dim_X: int, input_dim_Z: int) -> None:
    """Patch only column labels used by Cox initialization.

    The uploaded simulation helper has X_COLUMNS fixed to X0..X2. Rotterdam X
    has more columns, so the helper's Cox initialization DataFrame needs labels
    with the right length. This does not alter the Zhong model itself.
    """
    zhong_common.X_COLUMNS = [f"X{j}" for j in range(int(input_dim_X))]
    zhong_common.Z_COLUMNS = [f"Z{j}" for j in range(int(input_dim_Z))]
    zhong_common.ALL_COLUMNS = zhong_common.X_COLUMNS + zhong_common.Z_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Zhong et al. (2022) neural PH Rotterdam folds.")
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
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=100000)
    parser.add_argument("--val-frac", type=float, default=0.20)
    parser.add_argument("--cox-penalizer", type=float, default=0.0)
    parser.add_argument("--lmbd-l1", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=221100)
    parser.add_argument("--skip-completed", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-tuning", action="store_true")
    parser.add_argument("--use-default-params", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def run_tuning_if_needed(args: argparse.Namespace, fold_id: int) -> None:
    path = best_params_path(args.results_dir, fold_id, MODEL_NAME)
    if args.use_default_params:
        return
    if path.exists() and not args.overwrite_tuning:
        return
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("tune_zhong2022_rotterdam.py")),
        "--data-path", args.data_path,
        "--results-dir", args.results_dir,
        "--design", args.design,
        "--fold-id", str(fold_id),
        "--n-trials", str(args.n_trials),
        "--max-epochs", str(args.tuning_max_epochs),
        "--patience", str(args.patience),
        "--batch-size", str(args.batch_size),
        "--val-frac", str(args.val_frac),
        "--cox-penalizer", str(args.cox_penalizer),
        "--lmbd-l1", str(args.lmbd_l1),
        "--seed", str(args.seed),
    ]
    if args.overwrite_tuning:
        cmd.append("--overwrite")
    if args.smoke_test:
        cmd.append("--smoke-test")
    subprocess.run(cmd, check=True)


def load_params(args: argparse.Namespace, fold_id: int) -> dict:
    params = dict(DEFAULT_PARAMS)
    if args.use_default_params:
        params.update({
            "max_epochs": int(args.max_epochs),
            "patience": int(args.patience),
            "batch_size": int(args.batch_size),
            "val_frac": float(args.val_frac),
            "cox_penalizer": float(args.cox_penalizer),
            "lmbd_l1": float(args.lmbd_l1),
        })
        return params
    payload = load_json(best_params_path(args.results_dir, fold_id, MODEL_NAME))
    params.update(payload.get("best_params", payload))
    params["max_epochs"] = int(args.max_epochs)
    params["patience"] = int(args.patience)
    params["batch_size"] = int(params.get("batch_size", args.batch_size))
    params["val_frac"] = float(params.get("val_frac", args.val_frac))
    params["cox_penalizer"] = float(params.get("cox_penalizer", args.cox_penalizer))
    params["lmbd_l1"] = float(params.get("lmbd_l1", args.lmbd_l1))
    return params


def zhong_treatment_effects(fit, model_name: str, fold_id: int, treatment_columns: list[str]) -> tuple[pd.DataFrame, dict]:
    beta_hat = np.asarray(fit.beta_hat, dtype=float).reshape(-1)
    beta_init = np.asarray(fit.beta_init, dtype=float).reshape(-1)
    beta_cov = np.asarray(fit.beta_cov, dtype=float)
    se = np.sqrt(np.maximum(np.diag(beta_cov), 0.0)) if beta_cov.ndim == 2 else np.full_like(beta_hat, np.nan)
    z = float(norm.ppf(0.975))

    rows = []
    summary = {}
    for j, name in enumerate(treatment_columns):
        est = float(beta_hat[j])
        se_j = float(se[j]) if j < se.shape[0] and np.isfinite(se[j]) else np.nan
        init_j = float(beta_init[j]) if j < beta_init.shape[0] else np.nan
        lo = float(est - z * se_j) if np.isfinite(se_j) else np.nan
        hi = float(est + z * se_j) if np.isfinite(se_j) else np.nan
        row = {
            "model": model_name,
            "fold_id": int(fold_id),
            "parameter": name,
            "theta_index": int(j),
            "estimate": est,
            "se": se_j,
            "ci95_low": lo,
            "ci95_high": hi,
            "hazard_ratio": float(np.exp(est)),
            "hr_ci95_low": float(np.exp(lo)) if np.isfinite(lo) else np.nan,
            "hr_ci95_high": float(np.exp(hi)) if np.isfinite(hi) else np.nan,
            "cox_init": init_j,
            "ci_note": "SE computed from Cox beta Hessian with learned g(X) treated as fixed.",
        }
        rows.append(row)
        summary[f"{name}_coef"] = est
        summary[f"{name}_se"] = se_j
        summary[f"{name}_hr"] = row["hazard_ratio"]
    return pd.DataFrame(rows), summary


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.end_fold = min(args.end_fold, 1)
        args.n_trials = min(args.n_trials, 2)
        args.tuning_max_epochs = min(args.tuning_max_epochs, 5)
        args.max_epochs = min(args.max_epochs, 5)
        args.patience = min(args.patience, 2)

    data = load_arrays_and_folds(
        args.data_path,
        args.results_dir,
        seed=args.seed,
        design=args.design,
    )
    X, Z, T, delta = data["X"], data["Z"], data["T"], data["delta"]
    folds, eval_times = data["folds"], data["eval_times"]
    patch_zhong_column_names(X.shape[1], Z.shape[1])

    for fold_id in range(args.start_fold, args.end_fold):
        if args.skip_completed and not args.overwrite and completed(args.results_dir, MODEL_NAME, fold_id):
            print(f"fold={fold_id:02d} already completed, skipping")
            continue
        print(f"Running {MODEL_NAME}, fold={fold_id:02d}")
        try:
            run_tuning_if_needed(args, fold_id)
            params = load_params(args, fold_id)

            train_idx = folds.loc[folds["fold_id"] != fold_id, "row_id"].to_numpy(dtype=int)
            test_idx = folds.loc[folds["fold_id"] == fold_id, "row_id"].to_numpy(dtype=int)
            inner_train_idx, inner_val_idx = inner_train_val_indices(train_idx, val_fraction=float(params.get("val_frac", 0.20)), seed=args.seed + fold_id)

            t0 = time.perf_counter()
            fit = zhong_common.fit_zhong_model(
                X=X,
                Z=Z,
                T=T,
                delta=delta,
                train_idx=inner_train_idx,
                val_idx=inner_val_idx,
                hidden_layers=[int(h) for h in params["hidden_layers"]],
                learning_rate=float(params["learning_rate"]),
                max_epochs=int(params["max_epochs"]),
                patience=int(params["patience"]),
                batch_size=int(params["batch_size"]),
                lmbd_l1=float(params.get("lmbd_l1", 0.0)),
                seed=int(args.seed + fold_id),
                cox_penalizer=float(params.get("cox_penalizer", 0.0)),
                verbose=0,
            )
            training_time = time.perf_counter() - t0

            t1 = time.perf_counter()
            S_hat = zhong_common.predict_survival_from_ph(
                model=fit.model,
                train_X=X[inner_train_idx],
                train_Z=Z[inner_train_idx],
                train_T=T[inner_train_idx],
                train_delta=delta[inner_train_idx],
                test_X=X[test_idx],
                test_Z=Z[test_idx],
                eval_times=eval_times,
                batch_size=int(params["batch_size"]),
            )
            prediction_time = time.perf_counter() - t1

            theta_df, theta_summary = zhong_treatment_effects(fit, MODEL_NAME, fold_id, data["metadata"]["treatment_columns"])
            theta_df.to_csv(fold_file(args.results_dir, MODEL_NAME, fold_id, "treatment_effects"), index=False)

            extra = {
                "params_path": str(best_params_path(args.results_dir, fold_id, MODEL_NAME)) if not args.use_default_params else "default_params",
                "n_events_train": int(delta[train_idx].sum()),
                "n_events_test": int(delta[test_idx].sum()),
                "n_train_fit": int(inner_train_idx.size),
                "n_val_fit": int(inner_val_idx.size),
                "train_loss": float(fit.train_loss),
                "val_loss": float(fit.val_loss),
                "n_epochs_run": int(fit.n_epochs_run),
                "best_epoch": fit.best_epoch if fit.best_epoch is not None else np.nan,
                "hidden_layers": str(params["hidden_layers"]),
                "n_hidden_layers": int(len(params["hidden_layers"])),
                "learning_rate": float(params["learning_rate"]),
                "batch_size": int(params["batch_size"]),
                "max_epochs": int(params["max_epochs"]),
                "patience": int(params["patience"]),
                "val_frac": float(params.get("val_frac", 0.20)),
                "lmbd_l1": float(params.get("lmbd_l1", 0.0)),
                "cox_penalizer": float(params.get("cox_penalizer", 0.0)),
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
            save_json(
                fold_file(args.results_dir, MODEL_NAME, fold_id, "log", suffix="json"),
                {
                    "status": "success",
                    "fold_id": int(fold_id),
                    "params": params,
                    "beta_hat": fit.beta_hat.tolist(),
                    "beta_init": fit.beta_init.tolist(),
                    "beta_cov": fit.beta_cov.tolist(),
                    "history": fit.history,
                },
            )
            print(f"fold={fold_id:02d} done, ibs={summary['ibs']}, c_index={summary['c_index']}, beta={fit.beta_hat}")
        except Exception as exc:  # noqa: BLE001
            row = {"model": MODEL_NAME, "fold_id": int(fold_id), "status": "failed", "error_type": type(exc).__name__, "error_message": str(exc)}
            write_summary(fold_file(args.results_dir, MODEL_NAME, fold_id, "summary"), row)
            save_json(fold_file(args.results_dir, MODEL_NAME, fold_id, "log", suffix="json"), row)
            print(f"fold={fold_id:02d} failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
