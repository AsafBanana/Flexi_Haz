"""Aggregate simulation-replication outputs into report-ready CSV files.

Examples
--------
Aggregate all available models:
    python aggregate_results.py --model all

Aggregate one model:
    python aggregate_results.py --model flexihaz_partial_linear
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from simulation_io import DEFAULT_OUTPUT_DIR, ensure_dir, model_results_dir

KNOWN_MODELS: list[str] = []  # discovered from the results folder (see main)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate simulation results.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="all")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def read_many(paths: list[Path]) -> pd.DataFrame:
    dfs = []
    for path in paths:
        try:
            dfs.append(pd.read_csv(path))
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: could not read {path}: {exc}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True, sort=False)


def summarize_replications(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    out_rows = []
    for model, grp in summary_df.groupby("model", dropna=False):
        success = grp[grp["status"] == "success"].copy()
        row = {
            "model": model,
            "n_replications_total": int(len(grp)),
            "n_successful": int(len(success)),
            "failure_rate": float(1.0 - len(success) / max(len(grp), 1)),
        }
        for col in [
            "ibs",
            "mean_brier_score",
            "mean_auc",
            "c_index",
            "mean_bias_all_times",
            "mean_abs_bias_all_times",
            "mean_rmse_all_times",
            "training_time_sec",
            "prediction_time_sec",
            "event_rate",
            "observed_censoring_rate",
            "random_censoring_rate",
            "administrative_censoring_rate",
        ]:
            if col in success.columns:
                values = pd.to_numeric(success[col], errors="coerce")
                row[f"{col}_mean"] = float(values.mean())
                row[f"{col}_sd"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else np.nan
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def summarize_theta(theta_df: pd.DataFrame) -> pd.DataFrame:
    if theta_df.empty:
        return pd.DataFrame()
    rows = []
    for (model, parameter), grp in theta_df.groupby(["model", "parameter"], dropna=False):
        estimates = pd.to_numeric(grp["estimate"], errors="coerce")
        ses = pd.to_numeric(grp["se"], errors="coerce")
        true_value = float(pd.to_numeric(grp["true_value"], errors="coerce").dropna().iloc[0])
        rows.append(
            {
                "model": model,
                "parameter": parameter,
                "true_value": true_value,
                "mean_estimate": float(estimates.mean()),
                "bias": float(estimates.mean() - true_value),
                "empirical_sd": float(estimates.std(ddof=1)) if estimates.notna().sum() > 1 else np.nan,
                "mean_estimated_se": float(ses.mean()),
                "coverage_90": float(pd.to_numeric(grp["in_ci90"], errors="coerce").mean()),
                "coverage_95": float(pd.to_numeric(grp["in_ci95"], errors="coerce").mean()),
                "n_successful": int(estimates.notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_by_time(df: pd.DataFrame, value_cols: list[str], group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for keys, grp in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for col in value_cols:
            if col in grp.columns:
                values = pd.to_numeric(grp[col], errors="coerce")
                row[f"{col}_mean"] = float(values.mean())
                row[f"{col}_sd"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_model(output_dir: Path, model_name: str) -> None:
    result_dir = model_results_dir(output_dir, model_name)
    rep_dir = result_dir / "replications"
    agg_dir = ensure_dir(result_dir / "aggregated")

    summary_df = read_many(sorted(rep_dir.glob("rep_*_summary.csv")))
    bias_df = read_many(sorted(rep_dir.glob("rep_*_bias_by_time.csv")))
    auc_df = read_many(sorted(rep_dir.glob("rep_*_auc_by_time.csv")))
    brier_df = read_many(sorted(rep_dir.glob("rep_*_brier_by_time.csv")))
    theta_df = read_many(sorted(rep_dir.glob("rep_*_theta.csv")))

    if summary_df.empty:
        print(f"No summary files found for {model_name} in {rep_dir}")
        return

    summary_df.to_csv(agg_dir / "all_replication_summaries.csv", index=False)
    summarize_replications(summary_df).to_csv(agg_dir / "summary_prediction_metrics.csv", index=False)

    success = summary_df[summary_df["status"] == "success"].copy()
    if not success.empty:
        censor_cols = [
            "rep_id",
            "event_rate",
            "observed_censoring_rate",
            "random_censoring_rate",
            "administrative_censoring_rate",
        ]
        existing = [c for c in censor_cols if c in success.columns]
        success[existing].to_csv(agg_dir / "censoring_rates.csv", index=False)

    failures = summary_df[summary_df["status"] != "success"].copy()
    failures.to_csv(agg_dir / "failures.csv", index=False)

    if not bias_df.empty:
        bias_df.to_csv(agg_dir / "all_bias_by_time.csv", index=False)
        summarize_by_time(
            bias_df,
            value_cols=["mean_bias", "mean_abs_bias", "rmse", "mean_s_hat", "mean_s_true"],
            group_cols=["model", "time"],
        ).to_csv(agg_dir / "summary_bias_by_time.csv", index=False)

    if not auc_df.empty:
        auc_df.to_csv(agg_dir / "all_auc_by_time.csv", index=False)
        summarize_by_time(auc_df, value_cols=["auc"], group_cols=["model", "time"]).to_csv(
            agg_dir / "summary_auc_by_time.csv", index=False
        )

    if not brier_df.empty:
        brier_df.to_csv(agg_dir / "all_brier_by_time.csv", index=False)
        summarize_by_time(
            brier_df,
            value_cols=["brier_score"],
            group_cols=["model", "time"],
        ).to_csv(agg_dir / "summary_brier_by_time.csv", index=False)

    if not theta_df.empty:
        theta_df.to_csv(agg_dir / "all_theta.csv", index=False)
        summarize_theta(theta_df).to_csv(agg_dir / "summary_theta.csv", index=False)

    print(f"Aggregated {model_name} results into {agg_dir}")


def aggregate_combined(output_dir: Path, models: list[str]) -> None:
    combined_dir = ensure_dir(output_dir / "results" / "combined")
    summary_tables = []
    theta_tables = []
    bias_tables = []
    auc_tables = []
    brier_tables = []
    for model_name in models:
        agg_dir = model_results_dir(output_dir, model_name) / "aggregated"
        for filename, target in [
            ("summary_prediction_metrics.csv", summary_tables),
            ("summary_theta.csv", theta_tables),
            ("summary_bias_by_time.csv", bias_tables),
            ("summary_auc_by_time.csv", auc_tables),
            ("summary_brier_by_time.csv", brier_tables),
        ]:
            path = agg_dir / filename
            if path.exists():
                target.append(pd.read_csv(path))

    if summary_tables:
        pd.concat(summary_tables, ignore_index=True, sort=False).to_csv(combined_dir / "summary_prediction_metrics_all_models.csv", index=False)
    if theta_tables:
        pd.concat(theta_tables, ignore_index=True, sort=False).to_csv(combined_dir / "summary_theta_all_models.csv", index=False)
    if bias_tables:
        pd.concat(bias_tables, ignore_index=True, sort=False).to_csv(combined_dir / "summary_bias_by_time_all_models.csv", index=False)
    if auc_tables:
        pd.concat(auc_tables, ignore_index=True, sort=False).to_csv(combined_dir / "summary_auc_by_time_all_models.csv", index=False)
    if brier_tables:
        pd.concat(brier_tables, ignore_index=True, sort=False).to_csv(
            combined_dir / "summary_brier_by_time_all_models.csv",
            index=False,
        )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.smoke_test:
        output_dir = output_dir / "smoke_test"

    if args.model == "all":
        results_root = output_dir / "results"
        models = sorted(p.name for p in results_root.iterdir() if p.is_dir() and p.name != "combined") if results_root.exists() else []
    else:
        models = [args.model]

    for model_name in models:
        aggregate_model(output_dir, model_name)
    aggregate_combined(output_dir, models)


if __name__ == "__main__":
    main()
