"""Input/output helpers for the Rotterdam breast-cancer real-data experiment.

The experiment follows Section 7.2 of the FLEXI-Haz paper:
- outcome: time to the earlier of recurrence or death;
- linear treatment variables: chemotherapy and hormonal therapy;
- flexible covariates: age, nodes, progesterone receptor, estrogen receptor,
  menopausal status, tumor grade, and one-hot encoded tumor size.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

DEFAULT_EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = DEFAULT_EXPERIMENT_DIR / "rotterdam_breast_cancer.csv"
DEFAULT_RESULTS_DIR = DEFAULT_EXPERIMENT_DIR / "results"

MODEL_DIRS = {
    "flexihaz_partial_linear": "flexihaz_partial_linear",
    "flexihaz_nonlinear": "flexihaz_nonlinear",
    "cox_ph": "cox_ph",
    "random_survival_forest": "random_survival_forest",
    "zhong_mueller_wang_placeholder": "zhong_mueller_wang_placeholder",
    "zhong2022_ph": "zhong2022_ph",
}

STANDARD_TREATMENT_COLUMNS = ["chemo", "hormon"]

ER_INTERACTION_THRESHOLD = 61.0
ER_INTERACTION_TREATMENT_COLUMNS = [
    "hormon_low_er",
    "hormon_high_er",
    "chemo_low_er",
    "chemo_high_er",
]

CONTINUOUS_X_COLUMNS = ["age", "nodes", "pgr", "er"]
BINARY_OR_ORDINAL_X_COLUMNS = ["meno", "grade"]
SIZE_COLUMN = "size"


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=_json_default)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def load_rotterdam_dataframe(data_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    expected = {
        "pid", "year", "age", "meno", "size", "grade", "nodes", "pgr", "er",
        "hormon", "chemo", "rtime", "recur", "dtime", "death",
    }
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Rotterdam CSV is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["T"] = np.minimum(df["rtime"].astype(float), df["dtime"].astype(float))
    df["event"] = ((df["recur"].astype(int) == 1) | (df["death"].astype(int) == 1)).astype(int)

    if (df["T"] <= 0).any():
        raise ValueError("All observed times must be positive after constructing T=min(rtime,dtime).")

    return df


def make_design_dataframe(
    df: pd.DataFrame,
    design: str = "standard",
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    """Create X, Z, T, delta from a Rotterdam dataframe.

    design="standard":
        Z = chemo, hormon

    design="er_interaction":
        Z = hormon_low_er, hormon_high_er, chemo_low_er, chemo_high_er
        where low/high ER are defined by ER threshold 61.
    """
    if design not in {"standard", "er_interaction"}:
        raise ValueError(f"Unknown design={design!r}. Use 'standard' or 'er_interaction'.")

    required = (
        CONTINUOUS_X_COLUMNS
        + BINARY_OR_ORDINAL_X_COLUMNS
        + [SIZE_COLUMN]
        + STANDARD_TREATMENT_COLUMNS
        + ["T", "event"]
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required design columns: {missing}")

    x_base = df[CONTINUOUS_X_COLUMNS + BINARY_OR_ORDINAL_X_COLUMNS].copy()
    size_dummies = pd.get_dummies(
        df[SIZE_COLUMN].astype(str),
        prefix="size",
        drop_first=False,
        dtype=float,
    )
    X_df = pd.concat([x_base.astype(float), size_dummies], axis=1)

    if design == "standard":
        Z_df = df[STANDARD_TREATMENT_COLUMNS].astype(float).copy()
        treatment_columns = list(STANDARD_TREATMENT_COLUMNS)
        design_metadata = {
            "design": "standard",
            "er_interaction_threshold": None,
            "z_definition": "Z = [chemo, hormon]",
        }

    else:
        low_er = (df["er"].astype(float) < ER_INTERACTION_THRESHOLD).astype(float)
        high_er = 1.0 - low_er

        Z_df = pd.DataFrame(
            {
                "hormon_low_er": df["hormon"].astype(float) * low_er,
                "hormon_high_er": df["hormon"].astype(float) * high_er,
                "chemo_low_er": df["chemo"].astype(float) * low_er,
                "chemo_high_er": df["chemo"].astype(float) * high_er,
            },
            index=df.index,
        )
        treatment_columns = list(ER_INTERACTION_TREATMENT_COLUMNS)
        design_metadata = {
            "design": "er_interaction",
            "er_interaction_threshold": float(ER_INTERACTION_THRESHOLD),
            "low_er_definition": "er < 61",
            "high_er_definition": "er >= 61",
            "z_definition": (
                "Z = [hormon * I(er < 61), hormon * I(er >= 61), "
                "chemo * I(er < 61), chemo * I(er >= 61)]"
            ),
        }

    T = df["T"].to_numpy(dtype=np.float64)
    delta = df["event"].to_numpy(dtype=np.int64)

    metadata = {
        "x_columns": list(X_df.columns),
        "z_columns": list(Z_df.columns),
        "treatment_columns": treatment_columns,
        "size_columns": list(size_dummies.columns),
        "n": int(df.shape[0]),
        "n_events": int(delta.sum()),
        "event_rate": float(delta.mean()),
        "censoring_rate": float(1.0 - delta.mean()),
        "max_followup": float(np.max(T)),
        "endpoint": "T = min(rtime, dtime); event = recur OR death",
        "excluded_columns": ["pid", "year", "rtime", "recur", "dtime", "death"],
        **design_metadata,
    }
    return X_df, Z_df, T, delta, metadata


def get_eval_times(T: np.ndarray, step: int = 50, max_time: float | None = None) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(-1)
    upper = float(np.nanmax(T)) if max_time is None else float(max_time)
    times = np.arange(float(step), upper, float(step), dtype=np.float64)
    if times.size == 0:
        times = np.asarray([max(1.0, upper * 0.5)], dtype=np.float64)
    return times


def make_outer_folds(
    n_samples: int,
    n_splits: int = 5,
    seed: int = 221100,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Create or load deterministic outer 5-fold definitions."""
    results_dir = ensure_dir(results_dir)
    folds_path = results_dir / "folds.csv"
    if folds_path.exists() and not overwrite:
        folds = pd.read_csv(folds_path)
        if folds.shape[0] == n_samples:
            return folds

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_id = np.full(n_samples, -1, dtype=int)
    for j, (_, test_idx) in enumerate(kf.split(np.arange(n_samples))):
        fold_id[test_idx] = j
    folds = pd.DataFrame({"row_id": np.arange(n_samples, dtype=int), "fold_id": fold_id})
    folds.to_csv(folds_path, index=False)
    return folds


def inner_train_val_indices(train_indices: np.ndarray, val_fraction: float = 0.20, seed: int = 221100) -> tuple[np.ndarray, np.ndarray]:
    train_indices = np.asarray(train_indices, dtype=int).reshape(-1)
    inner_train, inner_val = train_test_split(
        train_indices,
        test_size=val_fraction,
        random_state=seed,
        shuffle=True,
    )
    return np.asarray(inner_train, dtype=int), np.asarray(inner_val, dtype=int)


def standardize_from_train(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(np.asarray(X_train, dtype=np.float64))
    X_test_s = scaler.transform(np.asarray(X_test, dtype=np.float64))
    return X_train_s, X_test_s, scaler


def load_arrays_and_folds(
    data_path: str | Path = DEFAULT_DATA_PATH,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    n_splits: int = 5,
    seed: int = 221100,
    overwrite_folds: bool = False,
    design: str = "standard",
) -> dict[str, Any]:
    df = load_rotterdam_dataframe(data_path)
    X_df, Z_df, T, delta, metadata = make_design_dataframe(df, design=design)
    folds = make_outer_folds(
        n_samples=df.shape[0],
        n_splits=n_splits,
        seed=seed,
        results_dir=results_dir,
        overwrite=overwrite_folds,
    )
    eval_times = get_eval_times(T, step=50)
    save_json(Path(results_dir) / "data_metadata.json", {**metadata, "eval_times_step": 50, "n_eval_times": int(eval_times.size)})
    return {
        "df": df,
        "X_df": X_df,
        "Z_df": Z_df,
        "X": X_df.to_numpy(dtype=np.float64),
        "Z": Z_df.to_numpy(dtype=np.float64),
        "T": T,
        "delta": delta,
        "folds": folds,
        "eval_times": eval_times,
        "metadata": metadata,
    }


def model_output_dir(results_dir: str | Path, model_name: str) -> Path:
    sub = MODEL_DIRS.get(model_name, model_name)
    return ensure_dir(Path(results_dir) / sub)


def fold_file(results_dir: str | Path, model_name: str, fold_id: int, kind: str, suffix: str = "csv") -> Path:
    return model_output_dir(results_dir, model_name) / f"fold_{int(fold_id):02d}_{kind}.{suffix}"


def tuning_dir(results_dir: str | Path, fold_id: int) -> Path:
    return ensure_dir(Path(results_dir) / "tuning" / f"fold_{int(fold_id):02d}")


def best_params_path(results_dir: str | Path, fold_id: int, model_name: str) -> Path:
    return tuning_dir(results_dir, fold_id) / f"{model_name}_best_params.json"


def completed(results_dir: str | Path, model_name: str, fold_id: int) -> bool:
    log_path = fold_file(results_dir, model_name, fold_id, "log", suffix="json")
    if not log_path.exists():
        return False
    try:
        payload = load_json(log_path)
    except Exception:
        return False
    return payload.get("status") == "success"


def write_summary(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    pd.DataFrame([row]).to_csv(path, index=False)
