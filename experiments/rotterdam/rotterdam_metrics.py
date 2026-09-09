"""Metrics for the Rotterdam real-data survival experiment."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def structured_y(T: np.ndarray, delta: np.ndarray):
    try:
        from sksurv.util import Surv
    except ImportError as exc:
        raise ImportError("scikit-survival is required for IBS, AUC, and C-index.") from exc
    return Surv.from_arrays(event=np.asarray(delta).astype(bool), time=np.asarray(T, dtype=np.float64))


def valid_metric_time_mask(eval_times: np.ndarray, train_T: np.ndarray, test_T: np.ndarray, test_delta: np.ndarray) -> tuple[np.ndarray, list[str]]:
    times = np.asarray(eval_times, dtype=np.float64).reshape(-1)
    train_T = np.asarray(train_T, dtype=np.float64).reshape(-1)
    test_T = np.asarray(test_T, dtype=np.float64).reshape(-1)
    test_delta = np.asarray(test_delta).astype(bool).reshape(-1)
    notes: list[str] = []
    if times.size == 0:
        return np.zeros(0, dtype=bool), ["No evaluation times were provided."]
    eps = 1e-8
    max_allowed = min(float(np.nanmax(train_T)), float(np.nanmax(test_T))) - eps
    min_allowed = max(0.0, float(np.nanmin(test_T)))
    mask = (times >= min_allowed) & (times <= max_allowed)
    mask &= np.array([np.any((test_T <= t) & test_delta) for t in times], dtype=bool)
    mask &= np.array([np.any(test_T > t) for t in times], dtype=bool)
    if not np.all(mask):
        dropped = times[~mask]
        if dropped.size:
            preview = ", ".join(str(int(t)) if float(t).is_integer() else f"{t:.3g}" for t in dropped[:12])
            suffix = "..." if dropped.size > 12 else ""
            notes.append(f"Dropped invalid metric times: {preview}{suffix}")
            notes.append(f"max_allowed_metric_time={max_allowed:.6g}")
    return mask, notes


def evaluate_real_survival_predictions(
    *,
    model_name: str,
    fold_id: int,
    train_T: np.ndarray,
    train_delta: np.ndarray,
    test_T: np.ndarray,
    test_delta: np.ndarray,
    S_hat: np.ndarray,
    eval_times: np.ndarray,
    training_time_sec: float,
    prediction_time_sec: float,
    extra_summary: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Evaluate one real-data outer fold.

    Returns a summary row, AUC-by-time dataframe, and Brier-by-time dataframe.
    No simulation bias metrics are computed because true survival is unknown.
    """
    eval_times = np.asarray(eval_times, dtype=np.float64).reshape(-1)
    S_hat = np.asarray(S_hat, dtype=np.float64)
    if S_hat.shape != (len(test_T), len(eval_times)):
        raise ValueError(f"S_hat shape {S_hat.shape} does not match expected {(len(test_T), len(eval_times))}.")
    S_hat = np.clip(S_hat, 0.0, 1.0)

    auc_values_all = np.full(eval_times.shape[0], np.nan, dtype=np.float64)
    brier_values_all = np.full(eval_times.shape[0], np.nan, dtype=np.float64)
    ibs = np.nan
    mean_auc = np.nan
    c_index = np.nan
    metric_notes: list[str] = []

    valid_mask, notes = valid_metric_time_mask(eval_times, train_T, test_T, test_delta)
    metric_notes.extend(notes)
    valid_times = eval_times[valid_mask]

    if valid_times.size >= 2:
        metric_indices = np.where(valid_mask)[0]
        S_metric = S_hat[:, metric_indices]
        try:
            from sksurv.metrics import brier_score, concordance_index_censored, cumulative_dynamic_auc, integrated_brier_score
            y_train = structured_y(train_T, train_delta)
            y_test = structured_y(test_T, test_delta)

            try:
                ibs = float(integrated_brier_score(y_train, y_test, S_metric, valid_times))
            except Exception as exc:  # noqa: BLE001
                metric_notes.append(f"IBS failed: {type(exc).__name__}: {exc}")

            try:
                _, brier_values = brier_score(y_train, y_test, S_metric, valid_times)
                brier_values = np.asarray(brier_values, dtype=np.float64)
                brier_values_all[metric_indices[: len(brier_values)]] = brier_values
            except Exception as exc:  # noqa: BLE001
                metric_notes.append(f"Brier failed: {type(exc).__name__}: {exc}")

            try:
                risk_matrix = 1.0 - S_metric
                auc_values, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_matrix, valid_times)
                auc_values = np.asarray(auc_values, dtype=np.float64)
                auc_values_all[metric_indices[: len(auc_values)]] = auc_values
                mean_auc = float(mean_auc)
            except Exception as exc:  # noqa: BLE001
                metric_notes.append(f"AUC failed: {type(exc).__name__}: {exc}")

            try:
                risk_last = 1.0 - S_metric[:, -1]
                c_index = float(concordance_index_censored(np.asarray(test_delta).astype(bool), np.asarray(test_T, dtype=float), risk_last)[0])
            except Exception as exc:  # noqa: BLE001
                metric_notes.append(f"C-index failed: {type(exc).__name__}: {exc}")
        except ImportError as exc:
            metric_notes.append(str(exc))
    else:
        metric_notes.append("Fewer than two valid metric times; IPCW metrics were skipped.")

    brier_df = pd.DataFrame({
        "model": model_name,
        "fold_id": int(fold_id),
        "time": eval_times,
        "brier_score": brier_values_all,
        "valid_for_metric": valid_mask.astype(int),
    })
    auc_df = pd.DataFrame({
        "model": model_name,
        "fold_id": int(fold_id),
        "time": eval_times,
        "auc": auc_values_all,
        "valid_for_metric": valid_mask.astype(int),
    })
    summary = {
        "model": model_name,
        "fold_id": int(fold_id),
        "status": "success",
        "n_train": int(len(train_T)),
        "n_test": int(len(test_T)),
        "train_event_rate": float(np.mean(train_delta)),
        "test_event_rate": float(np.mean(test_delta)),
        "training_time_sec": float(training_time_sec),
        "prediction_time_sec": float(prediction_time_sec),
        "ibs": float(ibs) if np.isfinite(ibs) else np.nan,
        "mean_auc": float(mean_auc) if np.isfinite(mean_auc) else np.nan,
        "c_index": float(c_index) if np.isfinite(c_index) else np.nan,
        "mean_brier_score": float(np.nanmean(brier_values_all)) if np.isfinite(brier_values_all).any() else np.nan,
        "n_requested_metric_times": int(eval_times.shape[0]),
        "n_valid_metric_times": int(valid_mask.sum()),
        "metric_notes": " | ".join(metric_notes),
    }
    if extra_summary:
        summary.update(extra_summary)
    return summary, auc_df, brier_df


def flexihaz_treatment_effects(model, model_name: str, fold_id: int, treatment_columns: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    theta_hat = np.asarray(model.theta_hat_original_scale, dtype=np.float64).reshape(-1)
    ts95 = model.theta_summary(alpha=0.05)
    se = np.asarray(ts95.get("se_theta_hat", np.full_like(theta_hat, np.nan)), dtype=np.float64).reshape(-1)
    ci = np.asarray(ts95.get("ci_95", np.full((theta_hat.shape[0], 2), np.nan)), dtype=np.float64)
    rows = []
    summary = {}
    for j, name in enumerate(treatment_columns):
        est = float(theta_hat[j])
        se_j = float(se[j]) if j < se.shape[0] and np.isfinite(se[j]) else np.nan
        lo = float(ci[j, 0]) if j < ci.shape[0] and np.isfinite(ci[j, 0]) else np.nan
        hi = float(ci[j, 1]) if j < ci.shape[0] and np.isfinite(ci[j, 1]) else np.nan
        row = {
            "model": model_name,
            "fold_id": int(fold_id),
            "parameter": name,
            "estimate": est,
            "se": se_j,
            "ci95_low": lo,
            "ci95_high": hi,
            "hazard_ratio": float(np.exp(est)),
            "hr_ci95_low": float(np.exp(lo)) if np.isfinite(lo) else np.nan,
            "hr_ci95_high": float(np.exp(hi)) if np.isfinite(hi) else np.nan,
        }
        rows.append(row)
        summary[f"{name}_coef"] = est
        summary[f"{name}_se"] = se_j
        summary[f"{name}_hr"] = row["hazard_ratio"]
    return pd.DataFrame(rows), summary


def cox_treatment_effects(cox_model, model_name: str, fold_id: int, treatment_columns: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    summary_df = cox_model.summary
    rows = []
    summary = {}
    for name in treatment_columns:
        if name not in summary_df.index:
            continue
        r = summary_df.loc[name]
        est = float(r["coef"])
        se = float(r["se(coef)"])
        lo = float(r.get("coef lower 95%", est - 1.96 * se))
        hi = float(r.get("coef upper 95%", est + 1.96 * se))
        row = {
            "model": model_name,
            "fold_id": int(fold_id),
            "parameter": name,
            "estimate": est,
            "se": se,
            "ci95_low": lo,
            "ci95_high": hi,
            "hazard_ratio": float(np.exp(est)),
            "hr_ci95_low": float(np.exp(lo)),
            "hr_ci95_high": float(np.exp(hi)),
            "p_value": float(r.get("p", np.nan)),
        }
        rows.append(row)
        summary[f"{name}_coef"] = est
        summary[f"{name}_se"] = se
        summary[f"{name}_hr"] = row["hazard_ratio"]
    return pd.DataFrame(rows), summary


def predict_rsf_survival_matrix(model, X_test: np.ndarray, times: np.ndarray) -> np.ndarray:
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
            x_grid = np.asarray(x_grid, dtype=np.float64)
            lower = min(lower, float(x_grid[0]), 0.0)
            upper = float(x_grid[-1])
        safe_times = np.maximum(safe_times, lower)
        if upper is not None and np.isfinite(upper):
            safe_times = np.minimum(safe_times, upper)
        rows.append(fn(safe_times))
    return np.vstack(rows).astype(np.float64)
