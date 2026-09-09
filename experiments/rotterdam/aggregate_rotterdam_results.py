"""Aggregate Rotterdam experiment fold results."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rotterdam_io import DEFAULT_RESULTS_DIR, ensure_dir

MODEL_NAMES = [
    "flexihaz_partial_linear",
    "flexihaz_nonlinear",
    "cox_ph",
    "random_survival_forest",
    "zhong_mueller_wang_placeholder",
    "zhong2022_ph",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Rotterdam experiment results.")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    return parser.parse_args()


def read_many(paths: list[Path]) -> pd.DataFrame:
    dfs = []
    for path in paths:
        if path.exists():
            try:
                dfs.append(pd.read_csv(path))
            except Exception as exc:  # noqa: BLE001
                dfs.append(pd.DataFrame([{"source_path": str(path), "status": "read_failed", "error_message": str(exc)}]))
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True, sort=False)


def summarize_numeric(df: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    existing = [c for c in value_cols if c in df.columns]
    if df.empty or not existing:
        return pd.DataFrame()
    agg = df.groupby(group_cols, dropna=False)[existing].agg(["mean", "std", "min", "max", "count"])
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    return agg.reset_index()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = ensure_dir(results_dir / "aggregated")

    summary_paths, auc_paths, brier_paths, treatment_paths = [], [], [], []
    for model in MODEL_NAMES:
        model_dir = results_dir / model
        summary_paths += sorted(model_dir.glob("fold_*_summary.csv"))
        auc_paths += sorted(model_dir.glob("fold_*_auc_by_time.csv"))
        brier_paths += sorted(model_dir.glob("fold_*_brier_by_time.csv"))
        treatment_paths += sorted(model_dir.glob("fold_*_treatment_effects.csv"))

    summary = read_many(summary_paths)
    auc = read_many(auc_paths)
    brier = read_many(brier_paths)
    treatment = read_many(treatment_paths)

    summary.to_csv(out_dir / "summary_by_fold.csv", index=False)
    auc.to_csv(out_dir / "auc_by_time.csv", index=False)
    brier.to_csv(out_dir / "brier_by_time.csv", index=False)
    treatment.to_csv(out_dir / "treatment_effects_by_fold.csv", index=False)

    perf_cols = [
        "ibs", "mean_auc", "c_index", "mean_brier_score",
        "training_time_sec", "prediction_time_sec", "train_event_rate", "test_event_rate",
    ]
    perf_summary = summarize_numeric(summary[summary.get("status", "success") == "success"] if not summary.empty else summary, ["model"], perf_cols)
    if not summary.empty and "status" in summary.columns:
        counts = summary.groupby(["model", "status"], dropna=False).size().reset_index(name="n_folds").pivot(index="model", columns="status", values="n_folds").fillna(0).reset_index()
        perf_summary = perf_summary.merge(counts, on="model", how="left") if not perf_summary.empty else counts
    perf_summary.to_csv(out_dir / "summary_by_model.csv", index=False)

    if not auc.empty:
        summarize_numeric(auc, ["model", "time"], ["auc"]).to_csv(out_dir / "auc_by_time_summary.csv", index=False)
    if not brier.empty:
        summarize_numeric(brier, ["model", "time"], ["brier_score"]).to_csv(out_dir / "brier_by_time_summary.csv", index=False)
    if not treatment.empty:
        tx_cols = ["estimate", "se", "ci95_low", "ci95_high", "hazard_ratio", "hr_ci95_low", "hr_ci95_high", "p_value"]
        summarize_numeric(treatment, ["model", "parameter"], tx_cols).to_csv(out_dir / "treatment_effects_summary.csv", index=False)

    print(f"Saved aggregated results to {out_dir}")


if __name__ == "__main__":
    main()
