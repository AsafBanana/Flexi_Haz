"""Run Random Survival Forest on saved simulation replications.

Uses scikit-survival's RandomSurvivalForest with all five covariates:
X1, X2, X3, Z1, Z2.

Examples
--------
Smoke test:
    python run_rsf_replications.py --smoke-test --n-estimators 20

Full run:
    python run_rsf_replications.py --start-rep 0 --end-rep 200 --n-estimators 500 --n-jobs -1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

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

MODEL_NAME = "random_survival_forest"

DEFAULT_TUNED_PARAMS_FILENAME = "random_survival_forest_best_params.json"


def _parse_optional_int(value):
    """Parse CLI values like None, none, null, or an integer."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    value_str = str(value).strip().lower()
    if value_str in {"none", "null", ""}:
        return None
    return int(value)


def _parse_optional_float(value):
    """Parse CLI values like None, none, null, or a float."""
    if value is None:
        return None
    if isinstance(value, float):
        return value
    value_str = str(value).strip().lower()
    if value_str in {"none", "null", ""}:
        return None
    return float(value)


def _normalize_max_features(value):
    """Convert max_features from JSON/CLI to a value accepted by scikit-survival."""
    if value is None:
        return None

    value_str = str(value).strip()
    value_lower = value_str.lower()

    if value_lower in {"none", "null"}:
        return None
    if value_lower in {"sqrt", "log2"}:
        return value_lower

    try:
        numeric = float(value_str)
    except ValueError:
        return value

    if numeric.is_integer() and numeric > 1:
        return int(numeric)
    return float(numeric)


def load_tuned_rsf_params(output_dir: Path, tuned_params_path: str | None = None) -> dict:
    """Load tuned RSF structure parameters.

    Expected JSON location by default:
        <output_dir>/tuning/random_survival_forest_best_params.json

    The tuning script may save metadata plus a nested "best_params" dictionary.
    This loader supports both:
        {"best_params": {...}, ...}
    and:
        {"min_samples_leaf": ..., ...}
    """
    if tuned_params_path is None:
        tuned_path = output_dir / "tuning" / DEFAULT_TUNED_PARAMS_FILENAME
    else:
        tuned_path = Path(tuned_params_path)

    if not tuned_path.exists():
        raise FileNotFoundError(
            f"Tuned RSF params file not found: {tuned_path}. "
            "Run tune_rsf_simulation.py first or pass --tuned-params-path."
        )

    with open(tuned_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    params = payload.get("best_params", payload)

    cleaned = {}
    if "min_samples_split" in params:
        cleaned["min_samples_split"] = int(params["min_samples_split"])
    if "min_samples_leaf" in params:
        cleaned["min_samples_leaf"] = int(params["min_samples_leaf"])
    if "max_features" in params:
        cleaned["max_features"] = _normalize_max_features(params["max_features"])
    if "max_depth" in params:
        cleaned["max_depth"] = _parse_optional_int(params["max_depth"])
    if "max_samples" in params:
        cleaned["max_samples"] = _parse_optional_float(params["max_samples"])

    return cleaned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RSF simulation replications.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-rep", type=int, default=0)
    parser.add_argument("--end-rep", type=int, default=200)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--min-samples-split", type=int, default=10)
    parser.add_argument("--min-samples-leaf", type=int, default=15)
    parser.add_argument("--max-features", default="sqrt")
    parser.add_argument("--max-depth", default=None)
    parser.add_argument("--max-samples", default=None)
    parser.add_argument("--use-tuned-params", action="store_true")
    parser.add_argument("--tuned-params-path", default=None)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=221100)
    parser.add_argument("--skip-completed", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def structured_y(T: np.ndarray, delta: np.ndarray):
    from sksurv.util import Surv

    return Surv.from_arrays(event=np.asarray(delta).astype(bool), time=np.asarray(T, dtype=np.float64))


def make_X_full(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    return np.column_stack([X, Z]).astype(np.float64)


def predict_survival_matrix(model, X_test: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Predict survival on requested times, with constant-tail extrapolation.

    scikit-survival's StepFunction objects cannot be evaluated beyond their
    internal domain, which in small smoke tests may end before tau=30.
    For t larger than the last model time, the survival curve is kept constant
    at its last available value. This avoids errors such as
    ``ValueError: x must be within [0; max_time]``.
    """
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    surv_fns = model.predict_survival_function(X_test)
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
            # The x-grid is the most reliable upper bound for RSF survival steps.
            upper = float(np.asarray(x_grid, dtype=np.float64)[-1])
            lower = min(lower, float(np.asarray(x_grid, dtype=np.float64)[0]), 0.0)

        safe_times = np.maximum(safe_times, lower)
        if upper is not None and np.isfinite(upper):
            safe_times = np.minimum(safe_times, upper)

        rows.append(fn(safe_times))
    return np.vstack(rows).astype(np.float64)


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.output_dir = str(Path(args.output_dir) / "smoke_test")
        args.start_rep = 0
        args.end_rep = min(args.end_rep, 3)
        args.n_estimators = min(args.n_estimators, 20)
        args.n_jobs = 1

    try:
        from sksurv.ensemble import RandomSurvivalForest
    except ImportError as exc:
        raise ImportError(
            "run_rsf_replications.py requires scikit-survival. "
            "Install with `pip install scikit-survival` or conda."
        ) from exc

    output_dir = Path(args.output_dir)
    ensure_dir(replication_dir(output_dir, MODEL_NAME))

    fixed_cli_params = {
        "min_samples_split": int(args.min_samples_split),
        "min_samples_leaf": int(args.min_samples_leaf),
        "max_features": _normalize_max_features(args.max_features),
        "max_depth": _parse_optional_int(args.max_depth),
        "max_samples": _parse_optional_float(args.max_samples),
    }

    if args.use_tuned_params:
        rsf_structure_params = load_tuned_rsf_params(
            output_dir=output_dir,
            tuned_params_path=args.tuned_params_path,
        )
        print("Using tuned RSF structure parameters:")
        print(rsf_structure_params)
    else:
        rsf_structure_params = fixed_cli_params
        print("Using fixed RSF structure parameters:")
        print(rsf_structure_params)

    test = load_fixed_test(output_dir)
    X_test_full = make_X_full(test["X"], test["Z"])
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
            X_train_full = make_X_full(train["X"], train["Z"])
            y_train = structured_y(train["T"], train["delta"])

            model = RandomSurvivalForest(
                n_estimators=args.n_estimators,
                min_samples_split=rsf_structure_params["min_samples_split"],
                min_samples_leaf=rsf_structure_params["min_samples_leaf"],
                max_features=rsf_structure_params["max_features"],
                max_depth=rsf_structure_params["max_depth"],
                max_samples=rsf_structure_params["max_samples"],
                n_jobs=args.n_jobs,
                random_state=args.seed + rep_id,
            )

            t0 = time.perf_counter()
            model.fit(X_train_full, y_train)
            training_time = time.perf_counter() - t0

            t1 = time.perf_counter()
            S_hat = predict_survival_matrix(model, X_test_full, eval_times_survival)
            prediction_time = time.perf_counter() - t1

            extra = {
                "n_train": int(train["T"].shape[0]),
                "event_rate": float(train["event_rate"]),
                "observed_censoring_rate": float(train["observed_censoring_rate"]),
                "random_censoring_rate": float(train["random_censoring_rate"]),
                "administrative_censoring_rate": float(train["administrative_censoring_rate"]),
                "n_estimators": int(args.n_estimators),
                "use_tuned_params": bool(args.use_tuned_params),
                "min_samples_split": rsf_structure_params["min_samples_split"],
                "min_samples_leaf": rsf_structure_params["min_samples_leaf"],
                "max_features": rsf_structure_params["max_features"],
                "max_depth": rsf_structure_params["max_depth"],
                "max_samples": rsf_structure_params["max_samples"],
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
                    "n_estimators": int(args.n_estimators),
                    "use_tuned_params": bool(args.use_tuned_params),
                    "rsf_structure_params": rsf_structure_params,
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
