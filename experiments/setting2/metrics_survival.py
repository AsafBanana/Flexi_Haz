"""Survival prediction and theta-inference metrics for simulation replications.

This module is shared by all model runners.  It is intentionally defensive:
small smoke-test datasets may have a maximum observed follow-up below tau,
so IPCW metrics from scikit-survival are evaluated only on time points that are
inside the empirical follow-up range and have a non-degenerate test risk set.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


THETA_TRUE_DEFAULT = np.array([2.0, -1.0], dtype=np.float64)


def _structured_y(T: np.ndarray, delta: np.ndarray):
    """Create a scikit-survival structured response array."""
    try:
        from sksurv.util import Surv
    except ImportError as exc:
        raise ImportError("scikit-survival is required for IBS, time-dependent AUC, and C-index.") from exc
    return Surv.from_arrays(event=np.asarray(delta).astype(bool), time=np.asarray(T, dtype=np.float64))


def _valid_metric_time_mask(
    *,
    eval_times_metrics: np.ndarray,
    train_T: np.ndarray,
    test_T: np.ndarray,
    test_delta: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Return a mask of time points that are safe for IPCW metrics.

    scikit-survival's IPCW metrics require evaluation times to be within the
    observed follow-up range. In smoke tests, it is common that no subject has
    observed time close to tau, so times such as 20, ..., 29 are invalid.
    For time-dependent AUC, each time point also needs at least one case by t
    and at least one control after t in the test set.
    """
    times = np.asarray(eval_times_metrics, dtype=np.float64).reshape(-1)
    train_T = np.asarray(train_T, dtype=np.float64).reshape(-1)
    test_T = np.asarray(test_T, dtype=np.float64).reshape(-1)
    test_delta = np.asarray(test_delta).astype(bool).reshape(-1)

    notes: list[str] = []
    if times.size == 0:
        return np.zeros(0, dtype=bool), ["No metric evaluation times were provided."]

    if train_T.size == 0 or test_T.size == 0:
        return np.zeros(times.shape[0], dtype=bool), ["Empty train or test follow-up array."]

    # Strictly below the smaller maximum avoids errors such as
    # "all times must be within follow-up time of test data".
    eps = 1e-8
    max_allowed = min(float(np.nanmax(train_T)), float(np.nanmax(test_T))) - eps
    min_allowed = max(0.0, float(np.nanmin(test_T)))

    mask = (times >= min_allowed) & (times <= max_allowed)

    # cumulative_dynamic_auc additionally needs cases and controls at each time.
    has_case = np.array([np.any((test_T <= t) & test_delta) for t in times], dtype=bool)
    has_control = np.array([np.any(test_T > t) for t in times], dtype=bool)
    mask &= has_case & has_control

    if not np.all(mask):
        dropped = times[~mask]
        if dropped.size:
            shown = ", ".join(str(int(x)) if float(x).is_integer() else f"{x:.3g}" for x in dropped[:10])
            suffix = "..." if dropped.size > 10 else ""
            notes.append(
                f"Dropped invalid metric times ({shown}{suffix}); valid range depends on observed train/test follow-up and non-degenerate test risk sets."
            )
            notes.append(f"max_allowed_metric_time={max_allowed:.6g}")

    return mask, notes


def survival_bias_by_time(
    S_hat: np.ndarray,
    S_true: np.ndarray,
    eval_times: np.ndarray,
    model_name: str,
    rep_id: int,
) -> pd.DataFrame:
    """Bias summaries at each evaluation time."""
    S_hat = np.asarray(S_hat, dtype=np.float64)
    S_true = np.asarray(S_true, dtype=np.float64)
    eval_times = np.asarray(eval_times, dtype=np.float64).reshape(-1)
    if S_hat.shape != S_true.shape:
        raise ValueError(f"S_hat shape {S_hat.shape} does not match S_true shape {S_true.shape}.")
    diff = S_hat - S_true
    return pd.DataFrame(
        {
            "model": model_name,
            "rep_id": int(rep_id),
            "time": eval_times,
            "mean_bias": np.nanmean(diff, axis=0),
            "mean_abs_bias": np.nanmean(np.abs(diff), axis=0),
            "rmse": np.sqrt(np.nanmean(diff**2, axis=0)),
            "mean_s_hat": np.nanmean(S_hat, axis=0),
            "mean_s_true": np.nanmean(S_true, axis=0),
        }
    )


def evaluate_survival_predictions(
    *,
    model_name: str,
    rep_id: int,
    train_T: np.ndarray,
    train_delta: np.ndarray,
    test_T: np.ndarray,
    test_delta: np.ndarray,
    S_hat: np.ndarray,
    S_true: np.ndarray,
    eval_times_survival: np.ndarray,
    eval_times_metrics: np.ndarray,
    training_time_sec: float,
    prediction_time_sec: float,
    extra_summary: dict[str, Any] | None = None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute common prediction metrics for one replication.

    Returns
    -------
    summary_row, bias_by_time_df, auc_by_time_df, brier_by_time_df
    """
    eval_times_survival = np.asarray(eval_times_survival, dtype=np.float64).reshape(-1)
    eval_times_metrics = np.asarray(eval_times_metrics, dtype=np.float64).reshape(-1)
    S_hat = np.asarray(S_hat, dtype=np.float64)
    S_true = np.asarray(S_true, dtype=np.float64)

    # Clip only for metric stability. We do not modify saved predictions elsewhere.
    S_hat_clipped = np.clip(S_hat, 0.0, 1.0)
    S_true_clipped = np.clip(S_true, 0.0, 1.0)

    bias_df = survival_bias_by_time(
        S_hat_clipped,
        S_true_clipped,
        eval_times_survival,
        model_name=model_name,
        rep_id=rep_id,
    )

    # Keep a row for every requested AUC time; invalid times stay NaN.
    auc_values_all = np.full(eval_times_metrics.shape[0], np.nan, dtype=np.float64)
    brier_values_all = np.full(eval_times_metrics.shape[0], np.nan, dtype=np.float64)

    mean_brier_score = float("nan")
    ibs = np.nan
    mean_auc = np.nan
    c_index = np.nan
    metric_notes: list[str] = []

    valid_mask, validity_notes = _valid_metric_time_mask(
        eval_times_metrics=eval_times_metrics,
        train_T=train_T,
        test_T=test_T,
        test_delta=test_delta,
    )
    metric_notes.extend(validity_notes)
    valid_times = eval_times_metrics[valid_mask]

    if valid_times.size == 0:
        metric_notes.append("No valid metric times available. This is common in very small smoke tests.")
    else:
        # Map valid metric times back to columns in S_hat/S_true.
        metric_indices = []
        for t in valid_times:
            idx = np.where(np.isclose(eval_times_survival, t))[0]
            if idx.size == 0:
                metric_notes.append(f"Metric time {t} was not found in eval_times_survival and was skipped.")
            else:
                metric_indices.append(int(idx[0]))

        if metric_indices:
            valid_times = valid_times[: len(metric_indices)]
            S_hat_metrics = S_hat_clipped[:, metric_indices]

            try:
                from sksurv.metrics import brier_score, integrated_brier_score, cumulative_dynamic_auc, concordance_index_censored

                y_train = _structured_y(train_T, train_delta)
                y_test = _structured_y(test_T, test_delta)

                try:
                    ibs = float(integrated_brier_score(y_train, y_test, S_hat_metrics, valid_times))
                except Exception as exc:  # noqa: BLE001
                    metric_notes.append(f"IBS failed: {type(exc).__name__}: {exc}")

                try:
                    brier_times, brier_values = brier_score(
                        y_train,
                        y_test,
                        S_hat_metrics,
                        valid_times,
                    )
                    brier_values = np.asarray(brier_values, dtype=np.float64)
                    valid_positions = np.where(valid_mask)[0][: len(brier_values)]
                    brier_values_all[valid_positions] = brier_values

                    if np.isfinite(brier_values).any():
                        mean_brier_score = float(np.nanmean(brier_values))
                except Exception as exc:  # noqa: BLE001
                    metric_notes.append(f"Brier by time failed: {type(exc).__name__}: {exc}")

                try:
                    risk_matrix = 1.0 - S_hat_metrics
                    auc_values, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_matrix, valid_times)
                    auc_values = np.asarray(auc_values, dtype=np.float64)
                    mean_auc = float(mean_auc)
                    auc_values_all[np.where(valid_mask)[0][: len(auc_values)]] = auc_values
                except Exception as exc:  # noqa: BLE001
                    metric_notes.append(f"AUC failed: {type(exc).__name__}: {exc}")

                try:
                    risk_last = 1.0 - S_hat_metrics[:, -1]
                    c_index = float(concordance_index_censored(test_delta.astype(bool), test_T.astype(float), risk_last)[0])
                except Exception as exc:  # noqa: BLE001
                    metric_notes.append(f"C-index failed: {type(exc).__name__}: {exc}")

            except ImportError as exc:
                metric_notes.append(str(exc))

    auc_df = pd.DataFrame(
        {
            "model": model_name,
            "rep_id": int(rep_id),
            "time": eval_times_metrics,
            "auc": auc_values_all,
            "valid_for_metric": valid_mask.astype(int),
        }
    )

    brier_df = pd.DataFrame(
        {
            "model": model_name,
            "rep_id": int(rep_id),
            "time": eval_times_metrics,
            "brier_score": brier_values_all,
            "valid_for_metric": valid_mask.astype(int),
        }
    )

    summary = {
        "model": model_name,
        "rep_id": int(rep_id),
        "status": "success",
        "training_time_sec": float(training_time_sec),
        "prediction_time_sec": float(prediction_time_sec),
        "ibs": ibs,
        "mean_brier_score": mean_brier_score,
        "mean_auc": mean_auc,
        "c_index": c_index,
        "mean_bias_all_times": float(bias_df["mean_bias"].mean()),
        "mean_abs_bias_all_times": float(bias_df["mean_abs_bias"].mean()),
        "mean_rmse_all_times": float(bias_df["rmse"].mean()),
        "n_requested_metric_times": int(eval_times_metrics.shape[0]),
        "n_valid_metric_times": int(valid_mask.sum()),
        "metric_notes": " | ".join(metric_notes),
    }
    if extra_summary:
        summary.update(extra_summary)
    return summary, bias_df, auc_df, brier_df


def theta_metrics_from_model(
    model,
    model_name: str,
    rep_id: int,
    theta_true: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Extract theta estimates, SE, and CI coverage from a fitted FLEXI-Haz model."""
    theta_true = THETA_TRUE_DEFAULT if theta_true is None else np.asarray(theta_true, dtype=np.float64).reshape(-1)
    theta_hat = np.asarray(model.theta_hat_original_scale, dtype=np.float64).reshape(-1)

    rows = []
    summary_extra = {}

    # theta_summary(alpha) returns CI at the requested confidence level.
    ts95 = model.theta_summary(alpha=0.05)
    ts90 = model.theta_summary(alpha=0.10)

    se = np.full(theta_hat.shape[0], np.nan, dtype=np.float64)
    ci95 = np.full((theta_hat.shape[0], 2), np.nan, dtype=np.float64)
    ci90 = np.full((theta_hat.shape[0], 2), np.nan, dtype=np.float64)

    if "se_theta_hat" in ts95:
        se = np.asarray(ts95["se_theta_hat"], dtype=np.float64).reshape(-1)
    if "ci_95" in ts95:
        ci95 = np.asarray(ts95["ci_95"], dtype=np.float64)
    if "ci_90" in ts90:
        ci90 = np.asarray(ts90["ci_90"], dtype=np.float64)

    for j in range(theta_hat.shape[0]):
        pname = f"theta{j + 1}"
        in90 = bool(ci90[j, 0] <= theta_true[j] <= ci90[j, 1]) if np.all(np.isfinite(ci90[j])) else False
        in95 = bool(ci95[j, 0] <= theta_true[j] <= ci95[j, 1]) if np.all(np.isfinite(ci95[j])) else False
        rows.append(
            {
                "model": model_name,
                "rep_id": int(rep_id),
                "parameter": pname,
                "true_value": float(theta_true[j]),
                "estimate": float(theta_hat[j]),
                "bias": float(theta_hat[j] - theta_true[j]),
                "se": float(se[j]) if np.isfinite(se[j]) else np.nan,
                "ci90_low": float(ci90[j, 0]) if np.isfinite(ci90[j, 0]) else np.nan,
                "ci90_high": float(ci90[j, 1]) if np.isfinite(ci90[j, 1]) else np.nan,
                "ci95_low": float(ci95[j, 0]) if np.isfinite(ci95[j, 0]) else np.nan,
                "ci95_high": float(ci95[j, 1]) if np.isfinite(ci95[j, 1]) else np.nan,
                "in_ci90": int(in90),
                "in_ci95": int(in95),
            }
        )
        summary_extra[f"{pname}_hat"] = float(theta_hat[j])
        summary_extra[f"{pname}_se"] = float(se[j]) if np.isfinite(se[j]) else np.nan
        summary_extra[f"{pname}_in_ci90"] = int(in90)
        summary_extra[f"{pname}_in_ci95"] = int(in95)

    return pd.DataFrame(rows), summary_extra
