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
│   └── rotterdam/    (rotterdam_README.md)
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

The core package uses Python scientific-computing and machine-learning
libraries including NumPy, SciPy, scikit-learn, PyTorch, and Optuna.
Some experiment baselines require additional packages such as
`lifelines`, `scikit-survival`, and TensorFlow.

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
| `networks.py` | Neural networks for the flexible hazard component and inference projections |
| `data.py` | Time-grid and expanded interval-level dataset construction |
| `preprocessing.py` | Input validation, administrative truncation, and standardization |
| `loss.py` | Discretized negative full log-likelihood |
| `inference.py` | Variance estimation and confidence intervals for the linear coefficient |
| `tuning.py` | Optuna-based hyperparameter tuning |
| `utils.py` | Device and reproducibility utilities |
| `exceptions.py` | Package-specific exceptions |

## Experiments

The repository contains three reproducible evaluations under `experiments/`,
all comparing partially linear FLEXI-Haz with the proportional-hazards DNN of
Zhong et al. (2022) for treatment-effect estimation. See
[`experiments/experiments_README.md`](experiments/experiments_README.md).

### Setting 1: six-setting simulation

Six data-generating settings (proportional or time-varying covariate effect,
three levels of Z-X correlation), 30% censoring, 200 replications at
`n = 8000` and `n = 4000`, scalar `theta_0 = log 1.5`. FLEXI-Haz uses
warm-started learning-rate phases (with a x20 learning rate for `theta` at
n = 4000), exact refinement of `theta` after each phase, five restarts
selected by the efficient score, and cross-fitted residual standard errors. See
[`experiments/setting1/setting1_README.md`](experiments/setting1/setting1_README.md).

### Setting 2: two-coefficient simulation

A nonlinear, non-proportional-hazards mechanism with `theta_0 = (1, -1)`,
`n = 8000`, 200 replications. Both models are fitted with `FlexiHazardModel`
(`model_type="flexihaz"` / `"zhong"`) after a grid search on an independent
tuning dataset with five common folds. See
[`experiments/setting2/setting2_README.md`](experiments/setting2/setting2_README.md).

### Rotterdam breast-cancer study

Hormonal-therapy and chemotherapy effects on the Rotterdam data (2,982
patients), overall and split by estrogen-receptor status at ER = 61, with five
shared outer folds, a per-fold exhaustive architecture grid at a fixed learning
rate, five random restarts, and residual-projection standard errors.
FLEXI-Haz uses `theta_lr_multiplier=20` for its linear branch. See
[`experiments/rotterdam/rotterdam_README.md`](experiments/rotterdam/rotterdam_README.md).

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

The experiment directories contain the scripts used for data generation,
tuning, model fitting, evaluation, and aggregation. Final human-readable
tuning outputs and aggregated experimental results are retained in the
repository. Large synthetic datasets, temporary logs, serialized Optuna
studies, and other regenerable artifacts do not need to be version
controlled.

## References

-   Ben Arie, A. and Gorfine, M. (2025). *Flexible deep neural networks
    for partially linear survival data: Estimation and survival
    inference*. arXiv:2512.10570.
-   Cox, D. R. (1972). Regression models and life-tables. *Journal of
    the Royal Statistical Society: Series B*, 34(2), 187-220.
-   Ishwaran, H., Kogalur, U. B., Blackstone, E. H., and Lauer, M. S.
    (2008). Random survival forests. *The Annals of Applied Statistics*,
    2(3), 841-860.
-   Zhong, Q., Mueller, J., and Wang, J.-L. (2022). Deep learning for
    the partially linear Cox model. *The Annals of Statistics*, 50(3),
    1348-1375.

## Citation

A `CITATION.cff` file is provided for citing the software. The citation
information can be updated when the accompanying FLEXI-Haz paper is
published.

## License

See `LICENSE` for the terms under which this software is distributed.
