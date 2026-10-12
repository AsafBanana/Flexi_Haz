# Survival-curve confidence intervals (finite-support X)

Pointwise confidence intervals for the FLEXI-Haz survival curve
`S(t | x0, z0) = exp{-H(t | x0, z0)}` of the partially linear hazard
`h(t | X, Z) = exp{theta'Z + g(t, X)}`, for covariate profiles `x0` on a
**finite support** of `X` (every `x0` is one of the observed profiles; `z0`
is free). The main package estimates `theta` and its covariance; this folder
adds inference on the curve itself. It is a self-contained TensorFlow
implementation and does not use the `flexihaz` (PyTorch) package.

## Files

``` text
flexihaz_ci.py              the estimator: fit_survival_ci(...) -> CrossFitSurvivalCI
rotterdam_survival_ci.py    Rotterdam breast-cancer example (binned covariates)
requirements.txt            tensorflow, lifelines, scikit-learn, pandas, numpy, scipy (+ matplotlib)
```

Outputs are written under `results/` (git-ignored).

## Method

A cross-fitted one-step estimator of the cumulative hazard:

1. the data are expanded to subject x interval rows on the grid of distinct
   observed times, and the network `g(t, X)` with the linear part `theta'Z`
   is trained on the piecewise-exponential likelihood (ReLU MLP on `(t, X)`,
   Adam, early stopping on an inner 67/33 subject split, Cox warm start of
   `theta`, no penalties);
2. `K` outer folds, stratified by the `X` profile. In fold `k` the model is
   trained on the other folds; the held-out fold supplies the influence
   function. The plug-in `H_hat(t | x0, z0) = int exp{theta'z0 + g(s, x0)} ds`
   (trapezoid rule on the grid) is averaged over folds;
3. with finite support, the nuisances `E[Z | X = x0, t]` and the stratum hazard
   `A(t | x0)` are stratum averages of the fitted hazard on the training folds.
   The one-step correction adds the mean efficient influence function
   (martingale residuals of the held-out subjects of stratum `x0`, plus the
   `theta` term through the inverse information), and the variance is the mean
   of its square, accumulated over folds;
4. the band is `H1 +- z_{1-alpha/2} se`, mapped to the survival scale.

The grid runs from the first event time to the smallest fold-wise maximum
follow-up, so every curve is reported where all folds have data; `H` is
measured from the first event time (no events occur before it). The one-step
estimate and the band are reported up to the last time at which profile `x0`
still has subjects at risk in every fold (`t_valid_max`; NaN afterwards),
because the correction weights `1 / {pi(x0) A(t | x0)}` are undefined beyond
it; the plug-in curve is always given. Time is divided by `time_scale`
(default `max(T)`) before entering the network.

## Running on your own data

Install the requirements (`pip install -r requirements.txt`; a GPU helps but
is not required). `X` must take finitely many values (bin continuous
covariates first), `Z` are the linear covariates, `T` the observed times and
`delta` the event indicators.

``` python
import numpy as np
from flexihaz_ci import fit_survival_ci

rng = np.random.default_rng(1)
n = 1500
X = rng.integers(0, 3, size=(n, 2)).astype(float)           # two 3-level covariates
Z = rng.integers(0, 2, size=(n, 1)).astype(float)           # one binary treatment
rate = 0.08 * np.exp(0.5 * X[:, 0] - 0.3 * X[:, 1] ** 2 - 0.7 * Z[:, 0])
event_time = rng.exponential(1.0 / rate)
censor = rng.uniform(5, 15, size=n)
T = np.minimum(event_time, censor)
delta = (event_time <= censor).astype(int)

ci = fit_survival_ci(
    X, Z, T, delta,
    nn_config=dict(hidden_layers_nodes=32, n_hidden_layers=2, learning_rate=1e-3,
                   batch_size=10000, patience=20, epochs=500),
    K=5, random_state=0,
)

x0 = np.array([1.0, 2.0])      # one of the observed X profiles (see ci.strata)
curve_treated = ci.survival(x0, z0=np.array([1.0]), alpha=0.05)
curve_control = ci.survival(x0, z0=np.array([0.0]), alpha=0.05)
print(curve_treated[["t", "S_onestep", "S_lo", "S_hi"]].iloc[::20])
print("theta by fold:", ci.theta_by_fold.ravel())
```

`ci.survival(x0, z0, alpha)` returns a DataFrame with `t`, the plug-in and
one-step cumulative hazards and survival (`H_plugin`, `H_onestep`, `S_plugin`,
`S_onestep`), the standard error of `H` (`se_H`) and the band (`S_lo`, `S_hi`).
`ci.cumulative_hazard(x0, z0, alpha)` gives the same on the hazard scale;
`ci.strata` lists the observed profiles; `ci.theta_by_fold` the per-fold
`theta`. The `nn_config` keys are the width and depth of the MLP, the learning
rate, the minibatch size in expanded rows, the early-stopping patience and the
maximum number of epochs.

## Rotterdam example

`rotterdam_survival_ci.py` applies the estimator to the Rotterdam
breast-cancer data of `experiments/rotterdam` (same first-event endpoint, time
in years). To get a finite support the clinical covariates are binned: age
(`<=45, 46-54, 55-65, >65`), nodes (`0, 1-3, >3`), ER (`<=3, 4-9, 10-49, >49`),
PgR (`<=9, 10-99, >99`), tumour size as recorded, menopausal status and grade
as recorded, giving 590 profiles; `Z = (hormon, chemo)`.

``` bash
python rotterdam_survival_ci.py                       # 5 folds, 64 x 4 MLP, seed 0
python rotterdam_survival_ci.py --smoke-test          # one-minute check
python rotterdam_survival_ci.py --profile 2,1,1,3,1,2,3 --profile 3,1,2,2,2,3,4
```

By default the five most frequent profiles are evaluated under no treatment,
hormonal therapy and chemotherapy. Outputs in `results/`:
`rotterdam_survival_curves.csv` (full curves and bands),
`rotterdam_survival_summary.csv` (5-year, 10-year and end-of-band survival
with CI per profile and treatment), `rotterdam_run.json` (configuration, per-fold
`theta`, timings) and `rotterdam_survival_profile0.png`.
