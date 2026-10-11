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

See [`simulation/simulation_README.md`](simulation/simulation_README.md) for the
data-generating mechanism, tuning protocol, execution commands, output
structure, and aggregation procedure.

## Rotterdam breast-cancer study

`rotterdam/` compares the treatment-effect estimates of partially linear
FLEXI-Haz and of the Zhong et al. (2022) proportional-hazards DNN on the
Rotterdam breast-cancer data (2,982 patients), for hormonal therapy and
chemotherapy overall (`standard` design) and split by estrogen-receptor
status at ER = 61 (`er_interaction` design).

Both models are fitted with `FlexiHazardModel` under one protocol: five
shared outer folds, a per-fold exhaustive architecture grid (depth 2-6,
width 25-500) at a fixed learning rate, five random restarts, and
cross-fitted residual-projection standard errors. FLEXI-Haz uses a x20
learning-rate multiplier for `theta`; Zhong is run with its own training
recipe (`--zhong-plain`).

See [`rotterdam/rotterdam_README.md`](rotterdam/rotterdam_README.md) for
the endpoint, covariate construction, protocol and commands.

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
