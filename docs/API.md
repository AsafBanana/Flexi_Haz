# FLEXI-Haz API

The current public package API exports:

```python
from flexihaz import FlexiHazardModel, tune_flexihaz
```

## `FlexiHazardModel`

`FlexiHazardModel` supports two model specifications.

### Partially linear model

```python
model.fit(T, delta, X, Z=Z)
```

fits

$$
\log h(t\mid X,Z)=\theta^\top Z+g(t,X).
$$

`X` contains covariates modeled through the neural component. `Z` contains covariates whose effects are represented by the separate linear component.

### Fully nonparametric model

```python
model.fit(T, delta, W, Z=None)
```

fits

$$
\log h(t\mid W)=g(t,W).
$$

All covariates to be modeled flexibly are passed through the `X` argument, named `W` above only for clarity.

## Constructor defaults

The current constructor is:

```python
FlexiHazardModel(
    hidden_layers=(64, 64),
    activation="relu",
    dropout=0.0,
    learning_rate=1e-3,
    three_phase_lr=True,
    batch_size=16**4,
    max_epochs=500,
    patience=30,
    tau=None,
    grid_size=None,
    include_observed_times=True,
    validation_fraction=0.33,
    standardize=True,
    refine_theta=True,
    compute_theta_variance=True,
    projection_max_epochs=300,
    projection_patience=30,
    projection_n_splits=5,
    theta_init="cox",
    cox_penalizer=1e-6,
    device="auto",
    seed=None,
    verbose=0,
    num_workers=0,
    pin_memory=None,
    persistent_workers=False,
    non_blocking=None,
    epoch_callback=None,
    scale_time=True,
    n_restarts=1,
    restart_selection="val_loss",
    theta_lr_multiplier=1.0,
    model_type="flexihaz",
)
```

## Fitting

```python
model.fit(T, delta, X, Z=None)
```

The estimator:

1. validates survival inputs;
2. applies administrative truncation at `tau`;
3. standardizes `X` and, when present, `Z`;
4. builds the time grid;
5. expands each subject into at-risk interval rows;
6. creates a subject-level train/validation split;
7. trains the neural hazard model;
8. restores the best validation state;
9. optionally refines the linear coefficient;
10. optionally estimates its covariance.

The validation split is performed by subject, not by expanded row.

### Multiple restarts

With `n_restarts > 1`, steps 7–9 are repeated for `n_restarts` networks initialized with seeds `seed, seed + 1, ...` on the same subject-level split, and one network is kept:

- `restart_selection="val_loss"` keeps the network with the lowest validation loss;
- `restart_selection="efficient_score"` keeps the network whose efficient score for `theta` is closest to zero (Euclidean norm over the columns of `Z`). The score uses residuals `Z - E[Z | X, t]` that are cross-fitted over the internal training subjects, so no residual is predicted by a projection network trained on that subject. It requires `Z`; without `Z` the estimator warns and selects by validation loss.

Restart 0 is identical to the single fit. After fitting, `model.restart_results_` lists each restart's seed, validation loss, `theta` (original scale) and, for score selection, the score and its norm; `model.selected_restart_` is the chosen index. `n_restarts=1` (the default) leaves both at their single-fit values (`None` and `0`).

### Learning-rate multiplier for `theta`

`theta_lr_multiplier` (default `1.0`) trains the linear branch with learning rate `learning_rate * theta_lr_multiplier`, as a separate Adam parameter group; the neural component keeps `learning_rate`. The multiplier applies in every phase of the three-phase schedule. With `1.0` the fit is identical to the default. It has no effect when `Z` is `None`.

### Zhong et al. (2022) proportional-hazards model

`model_type="zhong"` fits the partially linear Cox model of Zhong, Mueller and Wang (2022) with the same estimator interface:

$$
h(t\mid X,Z)=h_0(t)\exp\{\theta^\top Z+g(X)\}.
$$

Differences from the default `model_type="flexihaz"`:

- the network `g` takes only `X` (no time input), so the hazard is proportional;
- there is no interval expansion: the model is trained on one row per subject, full-batch, with the Breslow partial likelihood (no ties); `batch_size`, `grid_size` and `include_observed_times` do not affect training;
- `refine_theta` maximises the partial likelihood over `theta` with `g` fixed;
- the projection networks used for `var_theta_hat` and for `restart_selection="efficient_score"` are additive, `g_X(X) + g_T(t)`, with no `X`–`t` interaction, as in that paper; the efficient score is the partial-likelihood score with `Z` replaced by its cross-fitted residual;
- `predict_cumulative_hazard` and `predict_survival` use the Breslow baseline cumulative hazard of the fitted data, `predict_g` returns `g(X)` repeated over `times`, and `predict_hazard` raises an input-validation error (the baseline hazard is a step function).

Early stopping, the learning-rate phases, Cox initialization, `n_restarts`, `theta_lr_multiplier` and the covariance estimator work as for the default model. Zhong et al. train with `max_epochs=5000`, `patience=35` and one full-batch step per epoch, so consider raising `max_epochs` for this model.

When `theta_init="cox"`, the estimator attempts Cox initialization through `lifelines`. If that import or fit fails, the current implementation warns and falls back to zero initialization.

## Prediction methods

```python
model.predict_g(X_new, times)
model.predict_hazard(X_new, times, Z_new=None)
model.predict_cumulative_hazard(X_new, times, Z_new=None)
model.predict_survival(X_new, times, Z_new=None)
```

Prediction times must lie in `[0, tau]`.

If the fitted model contains a linear component, `Z_new` is required for hazard, cumulative-hazard, and survival prediction. If the model was fitted with `Z=None`, supplying `Z_new` raises an input-validation error.

## Linear coefficient output

After a partially linear fit:

```python
model.theta_hat
model.theta_hat_original_scale
```

If variance estimation was enabled:

```python
model.var_theta_hat
model.theta_ci_95
```

A user-facing summary is available through:

```python
model.theta_summary(alpha=0.05)
```

The returned dictionary contains `theta_hat` and, when covariance estimation is available, the variance matrix, standard errors, and the requested confidence interval.

## `tune_flexihaz`

```python
result = tune_flexihaz(
    T,
    delta,
    X,
    Z=None,
    tau=None,
    n_trials=30,
    timeout=None,
    seed=42,
    device="auto",
    verbose=1,
    search_space=None,
    folds=1,
    max_epochs=120,
    enable_pruning=True,
    pruner=None,
    model_kwargs=None,
    n_jobs=1,
)
```

The function returns:

```python
result["study"]
result["best_params"]
result["best_value"]
```

With `folds=1`, tuning uses the estimator's regular internal validation split. With `folds>1`, it evaluates each configuration using subject-level K-fold validation.

The current default search space includes hidden-layer configurations, learning rate, dropout, batch size, activation, and grid size.

Do not include `patience`, `max_epochs`, `compute_theta_variance`, `refine_theta`, `device`, `seed`, `verbose`, or `epoch_callback` in `model_kwargs` when they duplicate arguments that `tune_flexihaz` itself passes to the estimator.

## Package exceptions

The current package defines:

- `InputValidationError`
- `NotFittedError`
