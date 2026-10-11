"""Shared I/O helpers for the FLEXI-Haz simulation scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from simulation_dgm import load_npz


DEFAULT_OUTPUT_DIR = "outputs"


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_default(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=json_default)


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def train_dataset_path(output_dir: str | Path, rep_id: int) -> Path:
    return Path(output_dir) / "datasets" / f"train_rep_{int(rep_id):03d}.npz"


def fixed_test_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "test" / "fixed_test_set.npz"


def load_train_dataset(output_dir: str | Path, rep_id: int) -> dict:
    return load_npz(train_dataset_path(output_dir, rep_id))


def load_fixed_test(output_dir: str | Path) -> dict:
    return load_npz(fixed_test_path(output_dir))


def model_results_dir(output_dir: str | Path, model_name: str) -> Path:
    return Path(output_dir) / "results" / model_name


def replication_dir(output_dir: str | Path, model_name: str) -> Path:
    return model_results_dir(output_dir, model_name) / "replications"


def rep_summary_path(output_dir: str | Path, model_name: str, rep_id: int) -> Path:
    return replication_dir(output_dir, model_name) / f"rep_{int(rep_id):03d}_summary.csv"


def rep_bias_path(output_dir: str | Path, model_name: str, rep_id: int) -> Path:
    return replication_dir(output_dir, model_name) / f"rep_{int(rep_id):03d}_bias_by_time.csv"


def rep_auc_path(output_dir: str | Path, model_name: str, rep_id: int) -> Path:
    return replication_dir(output_dir, model_name) / f"rep_{int(rep_id):03d}_auc_by_time.csv"


def rep_brier_path(output_dir: str | Path, model_name: str, rep_id: int) -> Path:
    return replication_dir(output_dir, model_name) / f"rep_{int(rep_id):03d}_brier_by_time.csv"


def rep_theta_path(output_dir: str | Path, model_name: str, rep_id: int) -> Path:
    return replication_dir(output_dir, model_name) / f"rep_{int(rep_id):03d}_theta.csv"


def rep_predictions_path(output_dir: str | Path, model_name: str, rep_id: int) -> Path:
    return replication_dir(output_dir, model_name) / f"rep_{int(rep_id):03d}_survival_predictions.npz"


def rep_log_path(output_dir: str | Path, model_name: str, rep_id: int) -> Path:
    return replication_dir(output_dir, model_name) / f"rep_{int(rep_id):03d}_log.json"


def write_summary_row(path: str | Path, row: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, index=False)


def completed(output_dir: str | Path, model_name: str, rep_id: int) -> bool:
    path = rep_summary_path(output_dir, model_name, rep_id)
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
        return len(df) == 1 and str(df.loc[0, "status"]).lower() == "success"
    except Exception:
        return False


def load_best_params(path: str | Path) -> dict:
    params = load_json(path)
    if "best_params" in params:
        params = params["best_params"]
    if "hidden_layers" in params:
        params["hidden_layers"] = tuple(params["hidden_layers"])
    return params
