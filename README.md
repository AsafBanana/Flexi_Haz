# FLEXI-Haz

**FLEXI-Haz** is a Python package for flexible neural survival analysis
with support for both partially linear and fully nonparametric hazard
models.

The package builds on the partially linear FLEXI-Haz methodology of Ben
Arie and Gorfine (2025),

$$
\log h(t \mid X,Z)=\theta^\top Z+g(t,X).
$$

where selected covariates $Z$ retain interpretable, time-constant
log-hazard effects while a neural network $g(t,X)$ captures nonlinear
covariate effects, time dependence, and time-covariate interactions. The
package also provides a fully nonparametric variant,

$$
\log h(t \mid W)=g(t,W).
$$

in which all covariates are modeled flexibly by the neural network.

This repository contains the reusable Python implementation, tests, and
reproducible code for the two simulation settings and the Rotterdam
breast-cancer application of the accompanying paper, which gives the
theoretical background, the methodology and the full experimental design.

## Main features

-   **Partially linear FLEXI-Haz** with an interpretable linear
    component and a flexible neural nuisance component.
-   **Fully nonparametric FLEXI-Haz** for flexible survival prediction
    without a separate linear coefficient.
-   Direct full-likelihood estimation for right-censored survival data
    using a discretized time grid.
-   PyTorch neural-network implementation with configurable depth,
    width, activation, dropout, and optimization settings.
-   Subject-level train/validation splitting to avoid leakage between
    expanded interval-level observations.
-   Optional time scaling and covariate standardization.
-   Three-phase learning-rate schedule and validation-based early
    stopping.
-   Cox-based initialization and optional post-training refinement of
    the linear coefficient.
-   Variance estimation, confidence intervals, and coefficient summaries
    for the partially linear model.
-   Hazard, cumulative-hazard, and survival prediction.
-   Optuna-based hyperparameter tuning with single-split or K-fold
    validation.
-   Reproducible simulation and real-data experiments comparing FLEXI-Haz
    with the partially linear neural proportional-hazards model of Zhong
    et al. (2022), fitted with the same estimator (`model_type="zhong"`).

## Repository structure

```text
FLEXI-Haz/
├── README.md
├── pyproject.toml
├── requirements.txt
├── LICENSE
├── CITATION.cff
├── src/
│   └── flexihaz/
│       ├── __init__.py
│       ├── estimator.py
│       ├── networks.py
│       ├── loss.py
│       ├── data.py
│       ├── preprocessing.py
│       ├── inference.py
│       ├── tuning.py
│       ├── utils.py
│       └── exceptions.py
├── experiments/
│   ├── experiments_README.md
│   ├── setting1/     (setting1_README.md)
│   ├── setting2/     (setting2_README.md)
│   ├── rotterdam/    (rotterdam_README.md)
│   └── survival_ci/  (survival_ci_README.md)
├── tests/
└── docs/
    └── API.md
```

## Installation

From the repository root:

```bash
pip install -e .
```

For a regular local installation:

```bash
pip install .
```

The core package uses NumPy, SciPy, scikit-learn, PyTorch, pandas,
`lifelines` (Cox warm start of `theta`) and Optuna. The experiments
additionally need `scikit-survival` for the prediction metrics of setting 2
(`pip install -r requirements-experiments.txt`).

## Quick start

### Partially linear FLEXI-Haz

Supply the flexible covariates through `X` and the covariates whose
effects should remain linear through `Z`.

```python
import numpy as np
from flexihaz import FlexiHazardModel

model = FlexiHazardModel(
    tau=tau,
    hidden_layers=(64, 64),
    learning_rate=1e-3,
    three_phase_lr=True,
    batch_size=16**4,
    max_epochs=None,   # 500 for the default model, 5000 for model_type="zhong"
    patience=None,     # 30 for the default model, 35 for model_type="zhong"
    standardize=True,
    refine_theta=True,
    compute_theta_variance=True,
    scale_time=True,
    seed=1,
)

model.fit(T, delta, X, Z=Z)

times = np.linspace(0.0, tau, 100)
survival = model.predict_survival(
    X_new=X_test,
    times=times,
    Z_new=Z_test,
)

print(model.theta_summary(alpha=0.05))
```

The fitted log-hazard has the form

$$
\widehat{\eta}(t,X,Z)=
\widehat{\theta}^{\top}Z+\widehat{g}(t,X).
$$

### Fully nonparametric FLEXI-Haz

For the fully nonparametric model, combine all covariates and pass
`Z=None`.

```python
W = np.column_stack([X, Z])

model_np = FlexiHazardModel(
    tau=tau,
    hidden_layers=(64, 64),
    learning_rate=1e-3,
    max_epochs=None,   # 500 for the default model, 5000 for model_type="zhong"
    patience=None,     # 30 for the default model, 35 for model_type="zhong"
    standardize=True,
    scale_time=True,
    seed=1,
)

model_np.fit(T, delta, W, Z=None)

W_test = np.column_stack([X_test, Z_test])
survival_np = model_np.predict_survival(
    X_new=W_test,
    times=times,
    Z_new=None,
)
```

### Zhong et al. (2022) comparator

The partially linear Cox model of Zhong, Mueller and Wang (2022),
$h(t \mid X,Z)=h_0(t)\exp\{\theta^\top Z+g(X)\}$, is fitted with the same
estimator: pass `model_type="zhong"`. Training is full-batch on the Breslow
partial likelihood, the projection networks for the standard errors are
additive in `X` and `t`, and `theta` can be refined by profiling the partial
likelihood with `g` fixed.

```python
model_z = FlexiHazardModel(model_type="zhong", hidden_layers=(64, 64), learning_rate=1e-3,
                           three_phase_lr=False, refine_theta=False, seed=1)
model_z.fit(T, delta, X, Z=Z)
print(model_z.theta_summary(alpha=0.05))
```

### Hyperparameter tuning

```python
from flexihaz import tune_flexihaz

result = tune_flexihaz(
    T=T,
    delta=delta,
    X=X,
    Z=Z,
    tau=tau,
    n_trials=30,
    folds=5,
    max_epochs=120,
    seed=1,
)

print(result["best_params"])
print(result["best_value"])
```

Supplying `Z` tunes the partially linear model. Setting `Z=None` tunes
the fully nonparametric model.

### Confidence intervals for the survival curve

`experiments/survival_ci/flexihaz_ci.py` gives pointwise confidence intervals
for the survival curve `S(t | x0, z0)` when `X` has finite support (bin
continuous covariates first; `x0` must be one of the observed profiles, `z0`
is free). It is a self-contained TensorFlow implementation of a cross-fitted
one-step estimator and needs `tensorflow` and `lifelines`
(`pip install -r experiments/survival_ci/requirements.txt`).

```python
import sys
sys.path.insert(0, "experiments/survival_ci")
import numpy as np
from flexihaz_ci import fit_survival_ci

# X: (n, d) with finitely many distinct rows, Z: (n, p), T: (n,) times, delta: (n,) events
ci = fit_survival_ci(
    X, Z, T, delta,
    nn_config=dict(hidden_layers_nodes=64, n_hidden_layers=4, learning_rate=1e-3,
                   batch_size=10000, patience=50, epochs=2000),
    K=5,                 # outer cross-fitting folds
    random_state=0,
)

x0 = ci.strata[0]                       # an observed X profile
curve = ci.survival(x0, z0=np.array([1.0, 0.0]), alpha=0.05)
print(curve[["t", "S_onestep", "S_lo", "S_hi"]])
print(ci.theta_by_fold)                 # theta of each fold's network
```

`curve` holds the plug-in and one-step survival (`S_plugin`, `S_onestep`),
the band (`S_lo`, `S_hi`) and the cumulative-hazard standard error (`se_H`)
on the grid of observed times; the band is reported up to the last time at
which profile `x0` still has subjects at risk. See
[`experiments/survival_ci/survival_ci_README.md`](experiments/survival_ci/survival_ci_README.md)
for the method and a Rotterdam example.

## Package interface

The main public objects are:

```python
from flexihaz import FlexiHazardModel, tune_flexihaz
```

`FlexiHazardModel` coordinates preprocessing, time-grid construction,
expanded-data construction, neural-network training, optional
linear-coefficient refinement and inference, and prediction.

The package is internally divided into focused modules:

| Module | Purpose |
| --- | --- |
| `estimator.py` | Main estimator, fitting procedure, prediction, and linear-coefficient procedures |
| `networks.py` | Networks for the hazard component (FLEXI-Haz and Zhong) and the residual projections (joint and additive) |
| `data.py` | Time-grid and expanded interval-level dataset construction |
| `preprocessing.py` | Input validation, administrative truncation, and standardization |
| `loss.py` | Discretized negative full log-likelihood; Breslow negative partial log-likelihood |
| `inference.py` | Cross-fitted residual projections: covariance of `theta`, efficient score, confidence intervals |
| `tuning.py` | Optuna-based hyperparameter tuning |
| `utils.py` | Device and reproducibility utilities |
| `exceptions.py` | Package-specific exceptions |

## Experiments

Three reproducible evaluations under `experiments/`, all comparing partially
linear FLEXI-Haz with the proportional-hazards DNN of Zhong et al. (2022) for
treatment-effect estimation. See
[`experiments/experiments_README.md`](experiments/experiments_README.md).

- **Setting 1**: six data-generating settings (proportional or time-varying
  covariate effect x three levels of Z-X correlation), 30% censoring, 200
  replications at `n = 8000` and `n = 4000`, scalar `theta_0 = log 1.5`.
  [`experiments/setting1/setting1_README.md`](experiments/setting1/setting1_README.md)
- **Setting 2**: a nonlinear, non-proportional-hazards mechanism with
  `theta_0 = (1, -1)`, `n = 8000`, 200 replications.
  [`experiments/setting2/setting2_README.md`](experiments/setting2/setting2_README.md)
- **Rotterdam breast-cancer study**: hormonal-therapy and chemotherapy
  effects on the Rotterdam data (2,982 patients), overall and split by
  estrogen-receptor status.
  [`experiments/rotterdam/rotterdam_README.md`](experiments/rotterdam/rotterdam_README.md)

In addition, `experiments/survival_ci/` gives pointwise confidence intervals
for the FLEXI-Haz **survival curve** when `X` has finite support (cross-fitted
one-step estimator; a self-contained TensorFlow implementation), with a
Rotterdam example on binned covariates.
[`experiments/survival_ci/survival_ci_README.md`](experiments/survival_ci/survival_ci_README.md)

## Tests

Run the package test suite from the repository root with:

```bash
pytest
```

The tests cover input validation, time-grid and expanded-data
construction, likelihood calculations, network behavior, partially
linear and nonparametric fitting, prediction, subject-level validation
splitting, early stopping, time scaling, the three-phase learning-rate
schedule, inference, tuning, device handling, and reproducibility.

## Reproducibility

Each experiment folder contains the scripts for data generation, tuning,
model fitting and aggregation, and its README lists the exact commands and
seeds behind the reported tables. Generated datasets, tuning files and
results are written to git-ignored `outputs/` (or `results/`) folders and are
regenerated by those commands.

## References

-   Ben Arie, A. and Gorfine, M. (2025). *Flexible deep neural networks
    for partially linear survival data: Estimation and survival
    inference*. arXiv:2512.10570.
-   Cox, D. R. (1972). Regression models and life-tables. *Journal of
    the Royal Statistical Society: Series B*, 34(2), 187-220.
-   Zhong, Q., Mueller, J., and Wang, J.-L. (2022). Deep learning for
    the partially linear Cox model. *The Annals of Statistics*, 50(3),
    1348-1375.

## Citation

A `CITATION.cff` file is provided for citing the software. The citation
information can be updated when the accompanying FLEXI-Haz paper is
published.

## License

See `LICENSE` for the terms under which this software is distributed.
