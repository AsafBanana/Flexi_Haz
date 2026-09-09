"""Data-generating mechanism for the FLEXI-Haz simulation study.

This implements the simulation model used in the paper's simulation section:

    h(t | X, Z) = 0.1 * exp((0.1 + f(X)^2) * t + theta_0^T Z)

where

    f(X) = 0.2 * (X1 + X2) + 0.5 * X1 * X2 + X3^2
    theta_0 = (2, -1)

All coordinates of X and Z are sampled independently from Uniform[-1, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


THETA_TRUE = np.array([2.0, -1.0], dtype=np.float64)
TAU_DEFAULT = 10.0


@dataclass(frozen=True)
class DGMConfig:
    n: int = 8000
    tau: float = TAU_DEFAULT
    censoring_mean: float = 30.0
    theta_true: tuple[float, float] = (2.0, -1.0)


def f_x(X: np.ndarray) -> np.ndarray:
    """Compute f(X) = 0.2(X1+X2)+0.5X1X2+X3^2."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("X must have shape (n, 3).")
    return 0.2 * (X[:, 0] + X[:, 1]) + 0.5 * X[:, 0] * X[:, 1] + X[:, 2] ** 2


def linear_predictor_z(Z: np.ndarray, theta_true: np.ndarray | None = None) -> np.ndarray:
    """Compute theta_0^T Z."""
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2 or Z.shape[1] != 2:
        raise ValueError("Z must have shape (n, 2).")
    theta = THETA_TRUE if theta_true is None else np.asarray(theta_true, dtype=np.float64).reshape(2)
    return Z @ theta


def hazard_a(X: np.ndarray) -> np.ndarray:
    """Compute a(X) = 0.1 + f(X)^2."""
    fx = f_x(X)
    return 0.1 + fx**2


def true_cumulative_hazard(
    times: Iterable[float] | np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    theta_true: np.ndarray | None = None,
) -> np.ndarray:
    """Compute H_0(t | X, Z) on a time grid.

    Returns an array of shape (n, len(times)).
    """
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    if np.any(times < 0):
        raise ValueError("times must be nonnegative.")
    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    a = hazard_a(X).reshape(-1, 1)
    b = linear_predictor_z(Z, theta_true=theta_true).reshape(-1, 1)
    t = times.reshape(1, -1)

    # H(t | X, Z) = 0.1 exp(b) * (exp(a t) - 1) / a.
    # np.expm1 is more stable for small a*t.
    return 0.1 * np.exp(b) * np.expm1(a * t) / a


def true_survival(
    times: Iterable[float] | np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    theta_true: np.ndarray | None = None,
) -> np.ndarray:
    """Compute S_0(t | X, Z) = exp(-H_0(t | X, Z))."""
    H = true_cumulative_hazard(times, X, Z, theta_true=theta_true)
    return np.exp(-np.clip(H, 0.0, 745.0))


def sample_covariates(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample X=(X1,X2,X3) and Z=(Z1,Z2), all U[-1,1]."""
    X = rng.uniform(-1.0, 1.0, size=(int(n), 3)).astype(np.float64)
    Z = rng.uniform(-1.0, 1.0, size=(int(n), 2)).astype(np.float64)
    return X, Z


def sample_event_times(
    X: np.ndarray,
    Z: np.ndarray,
    rng: np.random.Generator,
    theta_true: np.ndarray | None = None,
) -> np.ndarray:
    """Sample exact event times by inverse transform sampling."""
    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    a = hazard_a(X)
    b = linear_predictor_z(Z, theta_true=theta_true)
    E = rng.exponential(scale=1.0, size=X.shape[0])

    # Solve H(U | X, Z) = E:
    # U = log(1 + a E / (0.1 exp(b))) / a.
    inside = 1.0 + (a * E) / (0.1 * np.exp(b))
    return np.log(inside) / a


def simulate_dataset(
    n: int,
    seed: int,
    tau: float = TAU_DEFAULT,
    censoring_mean: float = 30.0,
    theta_true: np.ndarray | None = None,
) -> dict[str, np.ndarray | float | int]:
    """Simulate one right-censored survival dataset.

    Parameters
    ----------
    n:
        Number of subjects.
    seed:
        Random seed for this dataset.
    tau:
        Administrative censoring horizon.
    censoring_mean:
        Mean of the independent exponential censoring time.
        The default 30.0 corresponds to censoring rate parameter 1/30.
        This is the practical interpretation that gives non-degenerate censoring
        with tau=30.
    theta_true:
        Optional true theta vector. Defaults to (2, -1).
    """
    rng = np.random.default_rng(int(seed))
    theta = THETA_TRUE if theta_true is None else np.asarray(theta_true, dtype=np.float64).reshape(2)

    X, Z = sample_covariates(n, rng)
    U = sample_event_times(X, Z, rng, theta_true=theta)
    C = rng.exponential(scale=float(censoring_mean), size=int(n))

    tau_arr = np.full(int(n), float(tau), dtype=np.float64)
    T = np.minimum(np.minimum(U, C), tau_arr)
    delta = ((U <= C) & (U <= float(tau))).astype(np.int64)

    random_censored = ((C < U) & (C <= float(tau))).astype(np.int64)
    admin_censored = ((U > float(tau)) & (C > float(tau))).astype(np.int64)

    return {
        "X": X,
        "Z": Z,
        "U_true": U.astype(np.float64),
        "C": C.astype(np.float64),
        "T": T.astype(np.float64),
        "delta": delta.astype(np.int64),
        "tau": float(tau),
        "seed": int(seed),
        "theta_true": theta.astype(np.float64),
        "event_rate": float(np.mean(delta)),
        "observed_censoring_rate": float(1.0 - np.mean(delta)),
        "random_censoring_rate": float(np.mean(random_censored)),
        "administrative_censoring_rate": float(np.mean(admin_censored)),
    }


def add_truth_on_grid(data: dict, eval_times: np.ndarray) -> dict:
    """Return a shallow copy with true H and S evaluated on eval_times."""
    out = dict(data)
    eval_times = np.asarray(eval_times, dtype=np.float64).reshape(-1)
    out["eval_times"] = eval_times
    out["true_cumulative_hazard"] = true_cumulative_hazard(eval_times, data["X"], data["Z"], data["theta_true"])
    out["true_survival"] = np.exp(-np.clip(out["true_cumulative_hazard"], 0.0, 745.0))
    return out


def save_npz(path: str | Path, data: dict) -> None:
    """Save a simulation dictionary as compressed NPZ."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def load_npz(path: str | Path) -> dict:
    """Load a compressed NPZ as a plain dictionary."""
    with np.load(Path(path), allow_pickle=False) as npz:
        return {key: npz[key] for key in npz.files}
