"""Data-generating mechanism for the "FLEXI-Haz: Additional Simulation
Settings" spec (Simulations_flexiHaz.pdf).

Covariates
----------
X1..X5 ~iid Uniform(0,1), X6 ~ Bernoulli(0.5).
s(X) = sqrt(12/5) * sum_{j=1}^5 (Xj - 1/2)          [E=0, Var=1]
Z = sqrt(rho)*s(X) + sqrt(1-rho)*eps, eps ~ N(0,1)   [Var(Z)=1, R^2(Z;X)=rho]

Hazard
------
a(X) = (X1^2 - X2^2 + X3^2 - X4^2 + X5^2 + X6 - 5) / 5
b(X) = (X1^2 + X2^2 + X3^3 + X4^4 + X5^5 + X6 - 8) / 5
theta0 = log(1.5)

lambda_gamma(t | X, Z) = 0.05 * exp{ t*exp(gamma*a(X)) + b(X) + theta0*Z }

so in FlexiHazardModel's h(t|X,Z) = exp{theta^T Z + g(t,X)} parameterization:
    g0_gamma(t, X) = log(0.05) + t*exp(gamma*a(X)) + b(X)

Event time (closed form, verified against the cumulative hazard):
    E ~ Exp(1)
    T = exp(-gamma*a(X)) * log{ 1 + 20*E*exp(gamma*a(X) - b(X) - theta0*Z) }

Censoring
---------
C ~ Uniform(0, c), Y = min(T, C), Delta = 1(T <= C).
c is calibrated per (gamma, rho) to hit a target censoring rate (e.g. 0.30 or
0.60), via root-finding on a large Monte Carlo sample of T: since
C ~ Uniform(0,c) is independent of T, P(T > C | c) = E_T[min(T,c)/c], which is
continuous and strictly decreasing in c, so a unique root exists.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

THETA0 = float(np.log(1.5))


def sample_covariates(n: int, rng: np.random.Generator) -> np.ndarray:
    """X = [X1..X5 ~ U(0,1), X6 ~ Bernoulli(0.5)], shape (n, 6)."""
    X_cont = rng.uniform(0.0, 1.0, size=(int(n), 5))
    X_bin = rng.binomial(1, 0.5, size=(int(n), 1)).astype(np.float64)
    return np.concatenate([X_cont, X_bin], axis=1)


def s_of_X(X: np.ndarray) -> np.ndarray:
    """s(X) = sqrt(12/5) * sum_{j=1}^5 (Xj - 1/2). Uses X1..X5 (columns 0-4)."""
    X = np.asarray(X, dtype=np.float64)
    return np.sqrt(12.0 / 5.0) * np.sum(X[:, :5] - 0.5, axis=1)


def sample_Z(X: np.ndarray, rho: float, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    eps = rng.normal(0.0, 1.0, size=n)
    return np.sqrt(rho) * s_of_X(X) + np.sqrt(1.0 - rho) * eps


def _h3_of_X(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    x1, x2, x3, x4, x5, x6 = (X[:, j] for j in range(6))
    return (x1**2 - x2**2 + x3**2 - x4**2 + x5**2 + x6 - 5.0) / 5.0


def _h4_of_X(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    x1, x2, x3, x4, x5, x6 = (X[:, j] for j in range(6))
    return (x1**2 + x2**2 + x3**3 + x4**4 + x5**5 + x6 - 8.0) / 5.0


def _I1_of_X(X: np.ndarray) -> np.ndarray:
    """Centered pairwise interactions: X~1*X~2 - X~2*X~3 + X~3*X~4 - X~4*X~5."""
    X = np.asarray(X, dtype=np.float64) - 0.5
    x1, x2, x3, x4, x5 = (X[:, j] for j in range(5))
    return x1 * x2 - x2 * x3 + x3 * x4 - x4 * x5


def _I2_of_X(X: np.ndarray) -> np.ndarray:
    """Centered pairwise interactions: X~1*X~3 + X~2*X~4 + X~3*X~5 + X~1*X~5."""
    X = np.asarray(X, dtype=np.float64) - 0.5
    x1, x2, x3, x4, x5 = (X[:, j] for j in range(5))
    return x1 * x3 + x2 * x4 + x3 * x5 + x1 * x5


def _h3_extreme_of_X(X: np.ndarray) -> np.ndarray:
    """k=2 ("extreme"): every squared term in h3 becomes cubed."""
    X = np.asarray(X, dtype=np.float64)
    x1, x2, x3, x4, x5, x6 = (X[:, j] for j in range(6))
    return (x1**3 - x2**3 + x3**3 - x4**3 + x5**3 + x6 - 5.0) / 5.0


def _h4_extreme_of_X(X: np.ndarray) -> np.ndarray:
    """k=2 ("extreme"): every squared term in h4 becomes cubed (X3^3/X4^4/X5^5 untouched)."""
    X = np.asarray(X, dtype=np.float64)
    x1, x2, x3, x4, x5, x6 = (X[:, j] for j in range(6))
    return (x1**3 + x2**3 + x3**3 + x4**4 + x5**5 + x6 - 8.0) / 5.0


def _I1_extreme_of_X(X: np.ndarray) -> np.ndarray:
    """k=2 ("extreme"): each X~i*X~j interaction becomes X~i^2*X~j."""
    X = np.asarray(X, dtype=np.float64) - 0.5
    x1, x2, x3, x4, x5 = (X[:, j] for j in range(5))
    return x1**2 * x2 - x2**2 * x3 + x3**2 * x4 - x4**2 * x5


def _I2_extreme_of_X(X: np.ndarray) -> np.ndarray:
    """k=2 ("extreme"): each X~i*X~j interaction becomes X~i^2*X~j."""
    X = np.asarray(X, dtype=np.float64) - 0.5
    x1, x2, x3, x4, x5 = (X[:, j] for j in range(5))
    return x1**2 * x3 + x2**2 * x4 + x3**2 * x5 + x1**2 * x5



# k=3 ("pure interaction"): a(X)/b(X) consist ONLY of Xi^2*Xj interaction terms
# -- no plain quadratic/cubic base terms at all -- one distinct set of index
# pairs per part (a(X) reuses I1's pairs, b(X) reuses I2's pairs, so the
# non-PH and PH parts don't share any term). Scale factors are calibrated
# (via a 200k-sample Monte Carlo check) so Var(a) and Var(b) each land close
# to Var(theta0*Z) = theta0^2 ~= 0.164 -- i.e., the non-PH, PH, and linear
# parts all contribute comparable, non-negligible variance to the log-hazard,
# rather than one term drowning out the others (the failure mode found when
# testing k=2 at nonlinear_scale=1: Var(theta0*Z) was 3-6x every nonlinear
# variant tried).
PURE_INTERACTION_SCALE_A = 6.0   # Var(I1_extreme) ~= 0.00419 -> Var(a) ~= 0.151
PURE_INTERACTION_SCALE_B = 5.5   # Var(I2_extreme) ~= 0.00531 -> Var(b) ~= 0.161


def a_of_X(X: np.ndarray, k: int = 0, nonlinear_scale: float = 1.0, nonph_scale: float = 1.0) -> np.ndarray:
    """a_k(X) -- the NON-PH part (interacts with t via gamma*a(X)):
    k=0 original additive setting; k=1 with interactions (I1);
    k=2 "extreme" -- squared base terms -> cubed, interactions Xi*Xj -> Xi^2*Xj;
    k=3 "pure interaction" -- ONLY Xi^2*Xj terms (I1's pairs), no base terms,
    scaled so Var(a) ~= Var(theta0*Z).
    nonlinear_scale multiplies BOTH a(X) and b(X) together (raises Var of the
    nonlinear parts relative to Var(theta0*Z)); nonph_scale multiplies ONLY
    a(X), for isolating/amplifying the non-PH pathway independent of b(X)."""
    if k == 0:
        out = _h3_of_X(X)
    elif k == 1:
        out = _h3_of_X(X) + _I1_of_X(X)
    elif k == 2:
        out = _h3_extreme_of_X(X) + _I1_extreme_of_X(X)
    elif k == 3:
        out = PURE_INTERACTION_SCALE_A * _I1_extreme_of_X(X)
    else:
        raise ValueError(f"unknown k={k!r}")
    return nonlinear_scale * nonph_scale * out


def b_of_X(X: np.ndarray, k: int = 0, nonlinear_scale: float = 1.0, ph_scale: float = 1.0) -> np.ndarray:
    """b_k(X) -- the PH part (no time interaction):
    k=0 original additive setting; k=1 with interactions (I2);
    k=2 "extreme" -- squared base terms -> cubed, interactions Xi*Xj -> Xi^2*Xj;
    k=3 "pure interaction" -- ONLY Xi^2*Xj terms (I2's pairs, distinct from
    a(X)'s), no base terms, scaled so Var(b) ~= Var(theta0*Z).
    nonlinear_scale multiplies BOTH a(X) and b(X) together; ph_scale multiplies
    ONLY b(X) (set to 0.0 to remove the PH part entirely, isolating the
    non-PH + linear-Z pathway)."""
    if k == 0:
        out = _h4_of_X(X)
    elif k == 1:
        out = _h4_of_X(X) + _I2_of_X(X)
    elif k == 2:
        out = _h4_extreme_of_X(X) + _I2_extreme_of_X(X)
    elif k == 3:
        out = PURE_INTERACTION_SCALE_B * _I2_extreme_of_X(X)
    else:
        raise ValueError(f"unknown k={k!r}")
    return nonlinear_scale * ph_scale * out


def sample_event_times(X: np.ndarray, Z: np.ndarray, gamma: float, rng: np.random.Generator,
                        k: int = 0, nonlinear_scale: float = 1.0,
                        ph_scale: float = 1.0, nonph_scale: float = 1.0) -> np.ndarray:
    a = a_of_X(X, k, nonlinear_scale, nonph_scale)
    b = b_of_X(X, k, nonlinear_scale, ph_scale)
    E = rng.exponential(1.0, size=X.shape[0])
    inside = 1.0 + 20.0 * E * np.exp(gamma * a - b - THETA0 * Z)
    return np.exp(-gamma * a) * np.log(inside)


def true_g0(t: np.ndarray, X: np.ndarray, gamma: float, k: int = 0, nonlinear_scale: float = 1.0,
            ph_scale: float = 1.0, nonph_scale: float = 1.0) -> np.ndarray:
    """g0_gamma(t, X) = log(0.05) + t*exp(gamma*a_k(X)) + b_k(X)."""
    a = a_of_X(X, k, nonlinear_scale, nonph_scale)
    b = b_of_X(X, k, nonlinear_scale, ph_scale)
    return np.log(0.05) + np.asarray(t).reshape(-1) * np.exp(gamma * a) + b


def _censoring_rate_given_c(c: float, T_mc: np.ndarray) -> float:
    """E_T[min(T,c)/c] = P(T > C) under C ~ Uniform(0,c) indep. of T."""
    return float(np.mean(np.minimum(T_mc, c) / c))


def calibrate_censoring_c(
    gamma: float,
    rho: float,
    target_rate: float,
    n_mc: int = 500_000,
    seed: int = 999_999,
    k: int = 0,
    nonlinear_scale: float = 1.0,
    ph_scale: float = 1.0,
    nonph_scale: float = 1.0,
) -> float:
    """Find c such that the expected censoring rate under C~Uniform(0,c) hits
    target_rate, for the given (gamma, rho, k, nonlinear_scale) DGM setting."""
    rng = np.random.default_rng(seed)
    X = sample_covariates(n_mc, rng)
    Z = sample_Z(X, rho, rng)
    T_mc = sample_event_times(X, Z, gamma, rng, k=k, nonlinear_scale=nonlinear_scale,
                               ph_scale=ph_scale, nonph_scale=nonph_scale)

    def f(c):
        return _censoring_rate_given_c(c, T_mc) - target_rate

    c_lo, c_hi = 1e-6, float(np.median(T_mc))
    # f is strictly decreasing in c; f(c_lo)->1-target>0, expand c_hi until f(c_hi)<0
    tries = 0
    while f(c_hi) > 0 and tries < 60:
        c_hi *= 2.0
        tries += 1
    if f(c_hi) > 0:
        raise RuntimeError("Could not bracket a root for the censoring calibration.")
    c_star = brentq(f, c_lo, c_hi, xtol=1e-8, rtol=1e-10, maxiter=200)
    return float(c_star)


# Setting-1 scale constants of the paper (Supplementary A.1.1): a(X) = 6 c_a (...), b(X) = 5.5 c_b (...),
# calibrated so the log-hazard variance splits PH / non-PH / linear as 70/0/30 (gamma = 0) and
# 35/35/30 (gamma = 0.5).  c_a is immaterial when gamma = 0.  Applied for k = 3 unless the
# scales are given explicitly.
PAPER_SCALES = {0.0: dict(nonph_scale=1.0, ph_scale=1.542799),
                0.5: dict(nonph_scale=1.574509, ph_scale=1.090924)}


def paper_scales(gamma: float, k: int) -> dict:
    """(nonph_scale = c_a, ph_scale = c_b) used in the paper for this gamma; 1.0 / 1.0 otherwise."""
    if k == 3 and float(gamma) in PAPER_SCALES:
        return dict(PAPER_SCALES[float(gamma)])
    return dict(nonph_scale=1.0, ph_scale=1.0)


@dataclass(frozen=True)
class DGMConfig:
    gamma: float
    rho: float
    censoring_rate: float
    n: int = 8000
    k: int = 3  # 3 = the paper's setting 1 (pure interactions, paper constants c_a, c_b); legacy variants:
                # 0 = original additive DeepEH-type setting, 1 = with interactions (I1/I2),
                # 2 = "extreme" (squared base terms -> cubed, interactions Xi*Xj -> Xi^2*Xj),
                # 3 = "pure interaction" (a/b are ONLY Xi^2*Xj terms, no base polynomial,
                #     scaled so Var(a)~=Var(b)~=Var(theta0*Z) -- PH/non-PH/linear all matter)
    nonlinear_scale: float = 1.0  # multiplies a(X)/b(X); >1 raises Var(nonlinear part)
                                   # relative to Var(theta0*Z), which is fixed at theta0^2
    ph_scale: float | None = None    # c_b: multiplies ONLY b(X) (the PH part); None -> paper constant
    nonph_scale: float | None = None  # c_a: multiplies ONLY a(X) (the non-PH part); None -> paper constant

    def __post_init__(self):
        defaults = paper_scales(self.gamma, self.k)
        if self.ph_scale is None:
            object.__setattr__(self, "ph_scale", float(defaults["ph_scale"]))
        if self.nonph_scale is None:
            object.__setattr__(self, "nonph_scale", float(defaults["nonph_scale"]))


def simulate_dataset(cfg: DGMConfig, seed: int, calibrated_c: float | None = None) -> dict:
    """Simulate one dataset under the additional-settings DGM.

    If calibrated_c is None, it is computed on the fly (slower; prefer passing
    a precomputed value shared across replications of the same (gamma, rho,
    censoring_rate, k) setting).
    """
    rng = np.random.default_rng(int(seed))
    X = sample_covariates(cfg.n, rng)
    Z = sample_Z(X, cfg.rho, rng)
    T = sample_event_times(X, Z, cfg.gamma, rng, k=cfg.k, nonlinear_scale=cfg.nonlinear_scale,
                            ph_scale=cfg.ph_scale, nonph_scale=cfg.nonph_scale)

    c = (calibrated_c if calibrated_c is not None
         else calibrate_censoring_c(cfg.gamma, cfg.rho, cfg.censoring_rate, k=cfg.k,
                                     nonlinear_scale=cfg.nonlinear_scale,
                                     ph_scale=cfg.ph_scale, nonph_scale=cfg.nonph_scale))
    C = rng.uniform(0.0, c, size=cfg.n)
    Y = np.minimum(T, C)
    delta = (T <= C).astype(np.int64)

    return {
        "X": X,
        "Z": Z.reshape(-1, 1),
        "T": Y.astype(np.float64),
        "T_true": T.astype(np.float64),
        "C": C.astype(np.float64),
        "delta": delta,
        "theta_true": np.array([THETA0], dtype=np.float64),
        "gamma": cfg.gamma,
        "rho": cfg.rho,
        "k": cfg.k,
        "nonlinear_scale": cfg.nonlinear_scale,
        "ph_scale": cfg.ph_scale,
        "nonph_scale": cfg.nonph_scale,
        "censoring_c": c,
        "target_censoring_rate": cfg.censoring_rate,
        "observed_censoring_rate": float(1.0 - delta.mean()),
        "event_rate": float(delta.mean()),
        "seed": int(seed),
    }
