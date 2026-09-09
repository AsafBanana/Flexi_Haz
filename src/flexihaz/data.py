from __future__ import annotations

import numpy as np
from .exceptions import InputValidationError


def build_time_grid(T, tau: float, grid_size: int | str | None = None, include_observed_times: bool = True) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(-1)
    if grid_size is None:
        grid_size = len(T)
    elif grid_size == "n":
        grid_size = len(T)
    grid_size = int(grid_size)
    if grid_size < 2:
        raise InputValidationError("grid_size must be at least 2.")
    base = np.linspace(0.0, float(tau), grid_size)
    pieces = [base, np.array([0.0, float(tau)])]
    if include_observed_times:
        pieces.append(T[(T >= 0) & (T <= tau)])
    grid = np.unique(np.concatenate(pieces))
    grid.sort()
    if grid[0] != 0.0:
        grid = np.insert(grid, 0, 0.0)
    if grid[-1] != float(tau):
        grid = np.append(grid, float(tau))
    return grid.astype(np.float64)


def make_expanded_data(T, delta, X, Z, time_grid):
    """Build interval-level rows for the discretized full likelihood."""
    T = np.asarray(T, dtype=np.float64).reshape(-1)
    delta = np.asarray(delta, dtype=np.int64).reshape(-1)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    Z_arr = None if Z is None else np.asarray(Z, dtype=np.float64)
    if Z_arr is not None and Z_arr.ndim == 1:
        Z_arr = Z_arr.reshape(-1, 1)
    grid = np.asarray(time_grid, dtype=np.float64).reshape(-1)

    if X.ndim != 2:
        raise InputValidationError("X must be a 2D array.")
    if X.shape[0] != len(T):
        raise InputValidationError("X must have the same number of rows as T.")

    if Z_arr is not None:
        if Z_arr.ndim != 2:
            raise InputValidationError("Z must be a 2D array.")
        if Z_arr.shape[0] != len(T):
            raise InputValidationError("Z must have the same number of rows as T.")
    if grid.ndim != 1 or grid.size < 2:
        raise InputValidationError("time_grid must contain at least two points.")
    if np.any(np.diff(grid) <= 0):
        raise InputValidationError("time_grid must be strictly increasing.")

    x_rows, z_rows, t_rows, st_rows, event_rows, length_rows, ids = [], [], [], [], [], [], []
    for i in range(len(T)):
        # intervals (t_{j-1}, t_j] that start before observed time
        end_idx = min(np.searchsorted(grid, T[i], side="left") + 1, len(grid))
        for j in range(1, end_idx):
            start = grid[j - 1]
            end = grid[j]
            length = max(0.0, min(T[i], end) - start)
            if length <= 0:
                continue
            event = int(delta[i] == 1 and T[i] > start and T[i] <= end)
            x_rows.append(X[i])
            if Z_arr is not None:
                z_rows.append(Z_arr[i])
            t_rows.append(end)
            st_rows.append(start)
            event_rows.append(event)
            length_rows.append(length)
            ids.append(i)
    if len(t_rows) == 0:
        raise InputValidationError("Expanded dataset is empty. Check T, tau, and grid_size.")
    out = {
        "X": np.asarray(x_rows, dtype=np.float32),
        "t": np.asarray(t_rows, dtype=np.float32).reshape(-1, 1),
        "start": np.asarray(st_rows, dtype=np.float32).reshape(-1, 1),
        "event": np.asarray(event_rows, dtype=np.float32).reshape(-1, 1),
        "length": np.asarray(length_rows, dtype=np.float32).reshape(-1, 1),
        "subject_id": np.asarray(ids, dtype=np.int64).reshape(-1),
    }
    if Z_arr is not None:
        out["Z"] = np.asarray(z_rows, dtype=np.float32)
    return out
