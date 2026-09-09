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

\[
\log h(t\mid X,Z)=\theta^\top Z+g(t,X).
\]

`X` contains covariates modeled through the neural component. `Z` contains covariates whose effects are represented by the separate linear component.

### Fully nonparametric model

```python
model.fit(T, delta, W, Z=None)
```

fits

\[
\log h(t\mid W)=g(t,W).
\]

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
    weight_decay=0.0,
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
)
```

Important implementation detail: although `weight_decay` remains in the constructor and tuner interface, the current estimator explicitly fixes `self.weight_decay = 0.0`. Supplying another constructor value therefore does not currently enable weight decay.

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

The current default search space includes hidden-layer configurations, learning rate, dropout, batch size, activation, and grid size. A `weight_decay` search range can also be supplied, but note the current estimator behavior described above.

Do not include `patience`, `max_epochs`, `compute_theta_variance`, `refine_theta`, `device`, `seed`, `verbose`, or `epoch_callback` in `model_kwargs` when they duplicate arguments that `tune_flexihaz` itself passes to the estimator.

## Package exceptions

The current package defines:

- `InputValidationError`
- `NotFittedError`
