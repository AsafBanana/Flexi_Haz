# FLEXI-Haz Experiments

This directory contains the reproducible experiments accompanying the
FLEXI-Haz implementation and report.

Two experiments are included:

``` text
experiments/
├── simulation/
└── rotterdam/
```

## Simulation study

`simulation/` contains the controlled simulation used to evaluate both
prediction and inference under a nonlinear, non-proportional-hazards
data-generating mechanism.

The experiment compares:

-   partially linear FLEXI-Haz;
-   fully nonparametric FLEXI-Haz;
-   Zhong et al. (2022);
-   Cox proportional hazards;
-   Random Survival Forest.

The final study consists of 200 training replications with (n=8000) and
a common fixed test set of 10,000 subjects.

See [`simulation/README.md`](simulation/README.md) for the
data-generating mechanism, tuning protocol, execution commands, output
structure, and aggregation procedure.

## Rotterdam breast-cancer study

`rotterdam/` contains the real-data evaluation on the Rotterdam
breast-cancer dataset.

The final analysis uses the ER-interaction design from the report, with
separate hormonal-therapy and chemotherapy effects for patients with ER
below 61 and ER at least 61. Predictive evaluation uses shared five-fold
outer splits.

See [`rotterdam/README.md`](rotterdam/README.md) for the endpoint
definition, covariate construction, tuning and evaluation protocol,
execution commands, and result files.

## Installation

Install FLEXI-Haz from the repository root before running the
experiments:

``` bash
pip install -e .
```

The comparison models require additional experiment dependencies. In
particular, Cox PH uses `lifelines`, Random Survival Forest and
censoring-adjusted metrics use `scikit-survival`, and the Zhong et
al. implementation uses TensorFlow/Keras.

## Results and generated artifacts

The repository should retain compact, interpretable outputs needed to
inspect the reported results, such as:

-   selected hyperparameters;
-   tuning trial tables;
-   per-fold or per-replication summary tables where useful;
-   aggregated metrics;
-   treatment-effect summaries;
-   final figures.

Large or regenerable artifacts, including generated synthetic datasets,
temporary logs, model checkpoints, serialized Optuna studies, and
smoke-test output directories, should generally be regenerated rather
than committed.
