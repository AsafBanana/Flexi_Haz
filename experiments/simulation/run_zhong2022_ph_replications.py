"""Run Zhong et al. (2022) neural PH comparator on saved simulation replications."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from metrics_survival import evaluate_survival_predictions
from simulation_io import (
    DEFAULT_OUTPUT_DIR,
    completed,
    ensure_dir,
    load_fixed_test,
    load_json,
    load_train_dataset,
    rep_auc_path,
    rep_bias_path,
    rep_log_path,
    rep_predictions_path,
    rep_summary_path,
    rep_theta_path,
    replication_dir,
    save_json,
    write_summary_row,
)
from zhong2022_ph_common import (
    TRUE_THETA,
    fit_zhong_model,
    make_random_train_val_split,
    predict_survival_from_ph,
    write_empirical_theta_summary,
)

try:
    from simulation_io import rep_brier_path
except ImportError:  # compatibility with earlier framework versions
    def rep_brier_path(output_dir, model_name, rep_id):
        return replication_dir(output_dir, model_name) / f"rep_{int(rep_id):03d}_brier_by_time.csv"

MODEL_NAME = "zhong2022_ph"

DEFAULT_PARAMS = {
    "n_hidden_layers": 6,
    "hidden_layers_nodes": 128,
    "learning_rate": 1e-3,
    "batch_size": 100000,
    "max_epochs": 5000,
    "patience": 35,
    "val_frac": 0.33,
    "cox_penalizer": 0.0,
    "lmbd_l1": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Zhong et al. (2022) neural PH replications.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-rep", type=int, default=0)
    parser.add_argument("--end-rep", type=int, default=200)
    parser.add_argument("--params-path", default=None)
    parser.add_argument("--use-default-params", action="store_true", help="Ignore tuning JSON and use notebook-style defaults.")
    parser.add_argument("--seed", type=int, default=20240222)
    parser.add_argument("--skip-completed", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def load_params(output_dir: Path, args: argparse.Namespace) -> dict:
    if args.use_default_params:
        return dict(DEFAULT_PARAMS)

    params_path = Path(args.params_path) if args.params_path is not None else output_dir / "tuning" / f"{MODEL_NAME}_best_params.json"
    if params_path.exists():
        payload = load_json(params_path)
        params = payload.get("best_params", payload)
        out = dict(DEFAULT_PARAMS)
        out.update(params)
        return out

    print(f"Tuned params not found at {params_path}; using default Zhong notebook-style params.")
    return dict(DEFAULT_PARAMS)


def save_theta_csv(
    path: Path,
    rep_id: int,
    beta_hat: np.ndarray,
    beta_init: np.ndarray,
    beta_cov: np.ndarray,
) -> None:
    beta_hat = np.asarray(beta_hat, dtype=np.float64).reshape(-1)
    beta_init = np.asarray(beta_init, dtype=np.float64).reshape(-1)
    beta_cov = np.asarray(beta_cov, dtype=np.float64)

    se = np.sqrt(np.maximum(np.diag(beta_cov), 0.0))

    z90 = float(norm.ppf(0.95))
    z95 = float(norm.ppf(0.975))

    rows = []
    for j, (hat, init, true, se_j) in enumerate(zip(beta_hat, beta_init, TRUE_THETA, se)):
        ci90_lower = float(hat - z90 * se_j)
        ci90_upper = float(hat + z90 * se_j)
        ci95_lower = float(hat - z95 * se_j)
        ci95_upper = float(hat + z95 * se_j)

        rows.append(
            {
                "model": MODEL_NAME,
                "rep_id": int(rep_id),

                # Zhong-specific names.
                "theta_index": int(j),
                "theta_name": f"Z{j}",
                "theta_hat": float(hat),
                "theta_true": float(true),

                # Standard names expected by aggregate_results.py.
                "parameter": f"theta_{j}",
                "estimate": float(hat),
                "true_value": float(true),

                "bias": float(hat - true),
                "cox_init": float(init),

                # Notebook-style estimated SE from covariance matrix.
                "se": float(se_j),

                "ci90_lower": ci90_lower,
                "ci90_upper": ci90_upper,
                "ci95_lower": ci95_lower,
                "ci95_upper": ci95_upper,

                # Standard coverage names expected by aggregate_results.py.
                "in_ci90": bool(ci90_lower <= true <= ci90_upper),
                "in_ci95": bool(ci95_lower <= true <= ci95_upper),

                # Extra aliases, in case you used these names elsewhere.
                "covered90": bool(ci90_lower <= true <= ci90_upper),
                "covered95": bool(ci95_lower <= true <= ci95_upper),

                "ci_note": "SE computed from Cox beta Hessian with learned g(X) treated as fixed offset.",
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.output_dir = str(Path(args.output_dir) / "smoke_test")
        args.start_rep = 0
        args.end_rep = min(args.end_rep, 3)

    output_dir = Path(args.output_dir)
    ensure_dir(replication_dir(output_dir, MODEL_NAME))
    params = load_params(output_dir, args)

    if args.smoke_test:
        params = dict(params)
        params["max_epochs"] = min(int(params.get("max_epochs", 5000)), 20)
        params["patience"] = min(int(params.get("patience", 35)), 5)

    test = load_fixed_test(output_dir)
    eval_times_survival = test["eval_times_survival"]
    eval_times_metrics = test["eval_times_metrics"]
    S_true = test["true_survival"]

    for rep_id in range(args.start_rep, args.end_rep):
        if args.skip_completed and not args.overwrite and completed(output_dir, MODEL_NAME, rep_id):
            print(f"rep={rep_id:03d} already completed, skipping")
            continue

        print(f"Running {MODEL_NAME}, rep={rep_id:03d}")
        try:
            train = load_train_dataset(output_dir, rep_id)
            train_idx, val_idx = make_random_train_val_split(
                n=int(train["T"].shape[0]),
                val_frac=float(params.get("val_frac", 0.33)),
                seed=int(args.seed + rep_id),
            )

            t0 = time.perf_counter()
            fit = fit_zhong_model(
                X=train["X"],
                Z=train["Z"],
                T=train["T"],
                delta=train["delta"],
                train_idx=train_idx,
                val_idx=val_idx,
                hidden_layers=[int(h) for h in params["hidden_layers"]],
                learning_rate=float(params["learning_rate"]),
                max_epochs=int(params["max_epochs"]),
                patience=int(params["patience"]),
                batch_size=int(params["batch_size"]),
                lmbd_l1=float(params.get("lmbd_l1", 0.0)),
                seed=int(args.seed + rep_id),
                cox_penalizer=float(params.get("cox_penalizer", 0.0)),
                verbose=0,
            )
            training_time = time.perf_counter() - t0

            t1 = time.perf_counter()
            S_hat = predict_survival_from_ph(
                model=fit.model,
                train_X=train["X"],
                train_Z=train["Z"],
                train_T=train["T"],
                train_delta=train["delta"],
                test_X=test["X"],
                test_Z=test["Z"],
                eval_times=eval_times_survival,
                batch_size=int(params["batch_size"]),
            )
            prediction_time = time.perf_counter() - t1

            extra = {
                "n_train": int(train["T"].shape[0]),
                "n_train_fit": int(len(train_idx)),
                "n_val_fit": int(len(val_idx)),
                "event_rate": float(train["event_rate"]),
                "observed_censoring_rate": float(train["observed_censoring_rate"]),
                "random_censoring_rate": float(train["random_censoring_rate"]),
                "administrative_censoring_rate": float(train["administrative_censoring_rate"]),
                "tau": float(train["tau"]) if "tau" in train else np.nan,
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
                "val_frac": float(params.get("val_frac", 0.33)),
                "lmbd_l1": float(params.get("lmbd_l1", 0.0)),
            }

            result = evaluate_survival_predictions(
                model_name=MODEL_NAME,
                rep_id=rep_id,
                train_T=train["T"],
                train_delta=train["delta"],
                test_T=test["T"],
                test_delta=test["delta"],
                S_hat=S_hat,
                S_true=S_true,
                eval_times_survival=eval_times_survival,
                eval_times_metrics=eval_times_metrics,
                training_time_sec=training_time,
                prediction_time_sec=prediction_time,
                extra_summary=extra,
            )

            if len(result) == 4:
                summary, bias_df, auc_df, brier_df = result
            else:
                summary, bias_df, auc_df = result
                brier_df = None

            write_summary_row(rep_summary_path(output_dir, MODEL_NAME, rep_id), summary)
            bias_df.to_csv(rep_bias_path(output_dir, MODEL_NAME, rep_id), index=False)
            auc_df.to_csv(rep_auc_path(output_dir, MODEL_NAME, rep_id), index=False)
            if brier_df is not None:
                brier_df.to_csv(rep_brier_path(output_dir, MODEL_NAME, rep_id), index=False)
            save_theta_csv(
                rep_theta_path(output_dir, MODEL_NAME, rep_id),
                rep_id,
                fit.beta_hat,
                fit.beta_init,
                fit.beta_cov,
            )

            save_json(
                rep_log_path(output_dir, MODEL_NAME, rep_id),
                {
                    "status": "success",
                    "rep_id": int(rep_id),
                    "params": params,
                    "beta_hat": fit.beta_hat.tolist(),
                    "beta_init": fit.beta_init.tolist(),
                    "beta_cov": fit.beta_cov.tolist(),
                    "beta_se": np.sqrt(np.maximum(np.diag(fit.beta_cov), 0.0)).tolist(),
                    "history": fit.history,
                },
            )
            if args.save_predictions:
                np.savez_compressed(rep_predictions_path(output_dir, MODEL_NAME, rep_id), S_hat=S_hat)

            print(f"rep={rep_id:03d} done, ibs={summary.get('ibs')}, c_index={summary.get('c_index')}, beta={fit.beta_hat}")

        except Exception as exc:  # noqa: BLE001
            error_row = {
                "model": MODEL_NAME,
                "rep_id": int(rep_id),
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            write_summary_row(rep_summary_path(output_dir, MODEL_NAME, rep_id), error_row)
            save_json(rep_log_path(output_dir, MODEL_NAME, rep_id), error_row)
            print(f"rep={rep_id:03d} failed: {type(exc).__name__}: {exc}")

    theta_paths = sorted(replication_dir(output_dir, MODEL_NAME).glob("rep_*_theta.csv"))
    write_empirical_theta_summary(
        theta_paths,
        replication_dir(output_dir, MODEL_NAME) / "theta_empirical_ci_summary.csv",
        model_name=MODEL_NAME,
    )


if __name__ == "__main__":
    main()
