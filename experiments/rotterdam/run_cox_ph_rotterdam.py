"""Run a notebook-style full-data Cox PH analysis on Rotterdam data.

This runner is designed to align the Cox results with the
`roterdam_study_strat_interaction.ipynb` approach:

- fit one Cox PH model on the full patient-level Rotterdam dataset;
- do not refit Cox separately over the 5 outer folds;
- keep the same Rotterdam design options used by the experiment code;
- save outputs using the same file naming convention as the other experiment
  runners, under fold_00, so the existing aggregator can read them;
- compute Cox standard errors as sqrt(diag(cph.variance_matrix_)).

Important: predictive metrics from this runner are in-sample, because the
model is fit on the full data and evaluated on the full data. This matches the
notebook-style Cox coefficient analysis, but it is not an out-of-sample CV
performance estimate.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rotterdam_io import (
    DEFAULT_DATA_PATH,
    DEFAULT_RESULTS_DIR,
    completed,
    fold_file,
    load_arrays_and_folds,
    save_json,
    write_summary,
)
from rotterdam_metrics import evaluate_real_survival_predictions

MODEL_NAME = "cox_ph"
DEFAULT_OUTPUT_FOLD_ID = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run notebook-style full-data Cox PH on Rotterdam. "
            "Fits once on all patients and writes outputs as fold_00."
        )
    )
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument(
        "--design",
        choices=["standard", "er_interaction"],
        default="standard",
        help=(
            "Rotterdam design: standard uses Z=[chemo,hormon]; "
            "er_interaction uses ER-stratified treatment interactions."
        ),
    )

    # Kept for compatibility with run_all_rotterdam.py and older commands.
    # This full-data Cox runner ignores the fold range and always fits once.
    parser.add_argument("--start-fold", type=int, default=0)
    parser.add_argument("--end-fold", type=int, default=5)

    parser.add_argument("--output-fold-id", type=int, default=DEFAULT_OUTPUT_FOLD_ID)
    parser.add_argument("--penalizer", type=float, default=0.0)
    parser.add_argument("--l1-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=221100)
    parser.add_argument("--skip-completed", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def make_cox_df(
    X_df: pd.DataFrame,
    Z_df: pd.DataFrame,
    T: np.ndarray | None = None,
    delta: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build a patient-level Cox dataframe from X, Z, T, and event."""
    df = pd.concat([X_df.reset_index(drop=True), Z_df.reset_index(drop=True)], axis=1).astype(float)
    if T is not None:
        df["T"] = np.asarray(T, dtype=float)
    if delta is not None:
        df["event"] = np.asarray(delta, dtype=int)
    return df


def _normal_two_sided_p_value(z: float) -> float:
    """Two-sided normal p-value without requiring scipy."""
    if not np.isfinite(z):
        return float("nan")
    return float(math.erfc(abs(float(z)) / math.sqrt(2.0)))


def cox_treatment_effects_sqrt_variance(
    cph: Any,
    model_name: str,
    fold_id: int,
    treatment_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Extract Cox treatment effects using sqrt(diag(variance_matrix_)) as SE.

    This is the statistically standard Cox CI calculation and differs from the
    notebook's displayed Cox CIs if that notebook used diag(variance_matrix_)
    directly as the SE.
    """
    rows: list[dict[str, float | int | str]] = []
    params = cph.params_
    variances = np.diag(cph.variance_matrix_.loc[params.index, params.index].to_numpy(dtype=float))
    se_by_name = pd.Series(np.sqrt(np.maximum(variances, 0.0)), index=params.index)

    for parameter in treatment_columns:
        if parameter not in params.index:
            rows.append(
                {
                    "model": model_name,
                    "fold_id": int(fold_id),
                    "parameter": parameter,
                    "estimate": np.nan,
                    "se": np.nan,
                    "ci95_low": np.nan,
                    "ci95_high": np.nan,
                    "hazard_ratio": np.nan,
                    "hr_ci95_low": np.nan,
                    "hr_ci95_high": np.nan,
                    "p_value": np.nan,
                    "status": "missing_parameter",
                }
            )
            continue

        estimate = float(params.loc[parameter])
        se = float(se_by_name.loc[parameter])
        ci_low = estimate - 1.96 * se
        ci_high = estimate + 1.96 * se
        z = estimate / se if se > 0 else float("nan")
        rows.append(
            {
                "model": model_name,
                "fold_id": int(fold_id),
                "parameter": parameter,
                "estimate": estimate,
                "se": se,
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
                "hazard_ratio": float(np.exp(estimate)),
                "hr_ci95_low": float(np.exp(ci_low)),
                "hr_ci95_high": float(np.exp(ci_high)),
                "p_value": _normal_two_sided_p_value(z),
                "status": "success",
            }
        )

    theta_df = pd.DataFrame(rows)
    theta_summary: dict[str, float] = {}
    for _, row in theta_df.iterrows():
        safe_name = str(row["parameter"]).replace(" ", "_")
        theta_summary[f"theta_{safe_name}"] = float(row["estimate"])
        theta_summary[f"theta_{safe_name}_se"] = float(row["se"])
        theta_summary[f"theta_{safe_name}_ci95_low"] = float(row["ci95_low"])
        theta_summary[f"theta_{safe_name}_ci95_high"] = float(row["ci95_high"])
        theta_summary[f"theta_{safe_name}_hr"] = float(row["hazard_ratio"])
        theta_summary[f"theta_{safe_name}_p_value"] = float(row["p_value"])
    return theta_df, theta_summary


def main() -> None:
    args = parse_args()
    fold_id = int(args.output_fold_id)

    try:
        from lifelines import CoxPHFitter
    except ImportError as exc:
        raise ImportError("Cox PH runner requires lifelines. Install with pip install lifelines.") from exc

    data = load_arrays_and_folds(
        args.data_path,
        args.results_dir,
        seed=args.seed,
        design=args.design,
    )
    X_df, Z_df, T, delta = data["X_df"], data["Z_df"], data["T"], data["delta"]
    eval_times = data["eval_times"]

    if args.skip_completed and not args.overwrite and completed(args.results_dir, MODEL_NAME, fold_id):
        print(f"full-data Cox output fold={fold_id:02d} already completed, skipping")
        return

    print(f"Running {MODEL_NAME} notebook-style full-data fit; output fold={fold_id:02d}")
    print("Note: --start-fold and --end-fold are ignored by this full-data Cox runner.")

    try:
        full_df = make_cox_df(X_df, Z_df, T, delta)
        pred_df = make_cox_df(X_df, Z_df)

        model = CoxPHFitter(penalizer=args.penalizer, l1_ratio=args.l1_ratio)
        t0 = time.perf_counter()
        model.fit(full_df, duration_col="T", event_col="event", show_progress=False)
        training_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        surv_df = model.predict_survival_function(pred_df, times=eval_times)
        S_hat = surv_df.T.to_numpy(dtype=float)
        prediction_time = time.perf_counter() - t1

        theta_df, theta_summary = cox_treatment_effects_sqrt_variance(
            model,
            MODEL_NAME,
            fold_id,
            data["metadata"]["treatment_columns"],
        )
        theta_df.to_csv(fold_file(args.results_dir, MODEL_NAME, fold_id, "treatment_effects"), index=False)

        extra = {
            "fit_style": "full_data_notebook_style",
            "prediction_evaluation_style": "in_sample_full_data",
            "uses_outer_folds": False,
            "cox_se_formula": "sqrt(diag(cph.variance_matrix_))",
            "penalizer": float(args.penalizer),
            "l1_ratio": float(args.l1_ratio),
            "n_train": int(len(T)),
            "n_test": int(len(T)),
            "n_events_train": int(delta.sum()),
            "n_events_test": int(delta.sum()),
            **theta_summary,
        }
        summary, auc_df, brier_df = evaluate_real_survival_predictions(
            model_name=MODEL_NAME,
            fold_id=fold_id,
            train_T=T,
            train_delta=delta,
            test_T=T,
            test_delta=delta,
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
            np.savez_compressed(
                fold_file(args.results_dir, MODEL_NAME, fold_id, "predictions", suffix="npz"),
                S_hat=S_hat,
                eval_times=eval_times,
                test_idx=np.arange(len(T), dtype=int),
            )
        save_json(
            fold_file(args.results_dir, MODEL_NAME, fold_id, "log", suffix="json"),
            {
                "status": "success",
                "fold_id": fold_id,
                "fit_style": "full_data_notebook_style",
                "uses_outer_folds": False,
                "cox_se_formula": "sqrt(diag(cph.variance_matrix_))",
                "coefficients": model.params_.to_dict(),
                "standard_errors": theta_df.set_index("parameter")["se"].to_dict(),
            },
        )
        print(f"full-data Cox done, ibs={summary['ibs']}, c_index={summary['c_index']}")
    except Exception as exc:  # noqa: BLE001
        row = {
            "model": MODEL_NAME,
            "fold_id": int(fold_id),
            "status": "failed",
            "fit_style": "full_data_notebook_style",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        write_summary(fold_file(args.results_dir, MODEL_NAME, fold_id, "summary"), row)
        save_json(fold_file(args.results_dir, MODEL_NAME, fold_id, "log", suffix="json"), row)
        print(f"full-data Cox failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
