"""Run Cox proportional hazards on saved simulation replications.

The Cox model uses all five covariates linearly: X1, X2, X3, Z1, Z2.
This is intentionally a standard baseline and is misspecified for the paper DGM.

Examples
--------
Smoke test:
    python run_cox_ph_replications.py --smoke-test

Full run:
    python run_cox_ph_replications.py --start-rep 0 --end-rep 200
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from metrics_survival import evaluate_survival_predictions
from simulation_io import (
    DEFAULT_OUTPUT_DIR,
    completed,
    ensure_dir,
    load_fixed_test,
    load_train_dataset,
    rep_auc_path,
    rep_bias_path,
    rep_brier_path,
    rep_log_path,
    rep_predictions_path,
    rep_summary_path,
    replication_dir,
    save_json,
    write_summary_row,
)

MODEL_NAME = "cox_ph"
FEATURE_COLUMNS = ["x1", "x2", "x3", "z1", "z2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Cox PH simulation replications.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-rep", type=int, default=0)
    parser.add_argument("--end-rep", type=int, default=200)
    parser.add_argument("--penalizer", type=float, default=0.0)
    parser.add_argument("--l1-ratio", type=float, default=0.0)
    parser.add_argument("--skip-completed", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def make_dataframe(X: np.ndarray, Z: np.ndarray, T: np.ndarray | None = None, delta: np.ndarray | None = None) -> pd.DataFrame:
    X_full = np.column_stack([X, Z])
    df = pd.DataFrame(X_full, columns=FEATURE_COLUMNS)
    if T is not None:
        df["T"] = np.asarray(T, dtype=np.float64)
    if delta is not None:
        df["event"] = np.asarray(delta, dtype=np.int64)
    return df


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.output_dir = str(Path(args.output_dir) / "smoke_test")
        args.start_rep = 0
        args.end_rep = min(args.end_rep, 3)

    try:
        from lifelines import CoxPHFitter
    except ImportError as exc:
        raise ImportError("run_cox_ph_replications.py requires lifelines. Install with `pip install lifelines`.") from exc

    output_dir = Path(args.output_dir)
    ensure_dir(replication_dir(output_dir, MODEL_NAME))

    test = load_fixed_test(output_dir)
    test_df = make_dataframe(test["X"], test["Z"])
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
            train_df = make_dataframe(train["X"], train["Z"], train["T"], train["delta"])

            model = CoxPHFitter(penalizer=args.penalizer, l1_ratio=args.l1_ratio)
            t0 = time.perf_counter()
            model.fit(train_df, duration_col="T", event_col="event", show_progress=False)
            training_time = time.perf_counter() - t0

            t1 = time.perf_counter()
            surv_df = model.predict_survival_function(test_df, times=eval_times_survival)
            S_hat = surv_df.T.to_numpy(dtype=np.float64)
            prediction_time = time.perf_counter() - t1

            extra = {
                "n_train": int(train["T"].shape[0]),
                "event_rate": float(train["event_rate"]),
                "observed_censoring_rate": float(train["observed_censoring_rate"]),
                "random_censoring_rate": float(train["random_censoring_rate"]),
                "administrative_censoring_rate": float(train["administrative_censoring_rate"]),
                "penalizer": args.penalizer,
                "l1_ratio": args.l1_ratio,
            }

            summary, bias_df, auc_df, brier_df = evaluate_survival_predictions(
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
            write_summary_row(rep_summary_path(output_dir, MODEL_NAME, rep_id), summary)
            bias_df.to_csv(rep_bias_path(output_dir, MODEL_NAME, rep_id), index=False)
            auc_df.to_csv(rep_auc_path(output_dir, MODEL_NAME, rep_id), index=False)
            brier_df.to_csv(rep_brier_path(output_dir, MODEL_NAME, rep_id), index=False)
            save_json(
                rep_log_path(output_dir, MODEL_NAME, rep_id),
                {
                    "status": "success",
                    "rep_id": rep_id,
                    "coefficients": model.params_.to_dict(),
                },
            )
            if args.save_predictions:
                np.savez_compressed(rep_predictions_path(output_dir, MODEL_NAME, rep_id), S_hat=S_hat)

            print(f"rep={rep_id:03d} done, ibs={summary['ibs']}, c_index={summary['c_index']}")

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


if __name__ == "__main__":
    main()
