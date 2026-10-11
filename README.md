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
reproducible code and outputs for the simulation study and Rotterdam
breast-cancer application described in the accompanying report. The report
also provides full explanations of the package, the theoretical background
and methodology behind FLEXI-Haz, implementation details, experimental
design, and additional discussion of the empirical results.

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
-   Reproducible simulation and real-data experiments with Cox PH,
    Random Survival Forest, and the partially linear neural
    proportional-hazards model of Zhong et al. (2022) as comparison
    methods.

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
├── examples/
├── experiments/
│   ├── experiments_README.md
│   ├── simulation/   (simulation_README.md)
│   └── rotterdam/    (rotterdam_README.md)
├── tests/
├── docs/
│   └── FLEXI_Haz_Report.pdf
└── .github/
    └── workflows/
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

The repository contains two reproducible empirical evaluations under
`experiments/`.

### Simulation study

The simulation study uses 200 independently generated training datasets
with $n=8000$, a fixed test set of 10,000 subjects, and administrative
horizon $\tau=10$. It compares:

-   partially linear FLEXI-Haz;
-   fully nonparametric FLEXI-Haz;
-   Zhong et al. (2022);
-   Cox proportional hazards;
-   Random Survival Forest.

The data-generating mechanism contains nonlinear and time-varying
covariate effects, so the global proportional-hazards assumption is
violated.

Across the 200 replications, partially linear FLEXI-Haz achieved an
integrated Brier score of **0.1139** and accurate estimation of the true
linear coefficients. Its empirical 95% confidence-interval coverage was
**0.945** for both coefficients.

See
[`experiments/simulation/simulation_README.md`](experiments/simulation/simulation_README.md)
for the complete reproduction workflow.

### Rotterdam breast-cancer study

The real-data experiment uses the Rotterdam breast-cancer dataset with
2,982 patients. The final analysis uses an ER-interaction design with
separate hormonal-therapy and chemotherapy effects for low- and
high-estrogen-receptor groups, using ER = 61 as the threshold.

Predictive evaluation is based on five shared outer folds for the neural
models and Random Survival Forest. The experiment also compares
ER-specific treatment-effect estimates and confidence intervals.

See [`experiments/rotterdam/rotterdam_README.md`](experiments/rotterdam/rotterdam_README.md)
for details.

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

## Report

The accompanying report is available at:

```text
docs/FLEXI_Haz_Report.pdf
```

It contains a complete explanation of the package, the theoretical
background and methodological formulation behind FLEXI-Haz, implementation
details, experimental design, full empirical results, and additional
discussion of the simulation and Rotterdam analyses.

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
