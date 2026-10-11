from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold

from .estimator import FlexiHazardModel
from .exceptions import InputValidationError
from .preprocessing import validate_survival_inputs, truncate_at_tau
from .data import make_expanded_data


def tune_flexihaz(
    T,
    delta,
    X,
    Z=None,
    *,
    tau=None,
    n_trials: int = 30,
    timeout=None,
    seed: int | None = 42,
    device="auto",
    verbose: int = 1,
    search_space: dict | None = None,
    folds: int = 1,
    max_epochs: int = 120,
    enable_pruning: bool = True,
    pruner=None,
    model_kwargs: dict | None = None,
    n_jobs: int = 1,
):
    """Tune FlexiHazardModel hyperparameters with Optuna.

    Parameters
    ----------
    T : array-like, shape (n,)
        Observed survival or censoring times.

    delta : array-like, shape (n,)
        Event indicators. 1 means observed event, 0 means censored.

    X : array-like, shape (n, d)
        Covariates entering the flexible neural-network component g(t, X).

    Z : array-like, shape (n, p), optional
        Covariates entering the linear component theta^T Z.
        If None, the fully nonparametric model is tuned.

    tau : float, optional
        Administrative time horizon. If None, max(T) is used.

    n_trials : int, default=30
        Number of Optuna trials.

    timeout : float, optional
        Maximum tuning time in seconds.

    seed : int or None, default=42
        Random seed for Optuna and model initialization.

    device : str, default="auto"
        "auto", "cpu", or a torch device string such as "cuda".

    verbose : int, default=0
        If positive, show Optuna progress bar.

    search_space : dict, optional
        Optional custom search space.If "weight_decay" is not provided,weight_decay is fixed at 0.0.
        If provided, it must be a positive tuple (low, high), and Optuna samples it on a log scale.

    folds : int, default=1
        Number of folds for outer validation during tuning.
        folds=1 uses the regular train-validation split inside FlexiHazardModel.
        folds>1 uses k-fold validation across subjects.

    max_epochs : int, default=120
        Maximum number of epochs used in each trial.

    enable_pruning : bool, default=True
        If True, allow Optuna to prune bad trials early.

    pruner : optuna.pruners.BasePruner, optional
        Custom Optuna pruner. If None and enable_pruning=True, MedianPruner is used.

    model_kwargs : dict, optional
        Extra keyword arguments passed to FlexiHazardModel.
        For example non_blocking or theta_init_subjects.

    Returns
    -------
    dict
        Dictionary with:
        - "study": Optuna study object
        - "best_params": best hyperparameters for FlexiHazardModel
        - "best_value": best validation loss
    """
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("tune_flexihaz requires optuna. Install with `pip install optuna`.") from exc

    n_trials = int(n_trials)
    folds = int(folds)
    max_epochs = int(max_epochs)
    n_jobs = int(n_jobs)
    extra_model_kwargs = {} if model_kwargs is None else dict(model_kwargs)

    if n_trials <= 0:
        raise InputValidationError("n_trials must be a positive integer.")
    if folds <= 0:
        raise InputValidationError("folds must be a positive integer.")
    if max_epochs <= 0:
        raise InputValidationError("max_epochs must be a positive integer.")
    if n_jobs == 0:
        raise InputValidationError("n_jobs cannot be 0. Use n_jobs=1 for sequential tuning or n_jobs=-1 for all available workers.")

    default_space = {
        "hidden_layers": [[32], [64], [128], [64, 64], [128, 64], [128, 128], [256, 128]],
        "learning_rate": (1e-4, 5e-2),
        "dropout": (0.0, 0.0),
        "batch_size": [32 ** 3, 16 ** 4, 24 ** 4, 32 ** 4],
        "activation": ["relu"],
        "grid_size": ["n"],
    }

    if search_space is not None:
        default_space.update(search_space)

    required_keys = {
        "hidden_layers",
        "learning_rate",
        "dropout",
        "batch_size",
        "activation",
        "grid_size",
    }
    missing = required_keys.difference(default_space)
    if missing:
        raise InputValidationError(f"search_space is missing required keys: {sorted(missing)}")
    
    if "weight_decay" in default_space:
        wd_space = default_space["weight_decay"]

        if not isinstance(wd_space, tuple) or len(wd_space) != 2:
            raise InputValidationError(
                "search_space['weight_decay'] must be a tuple like (1e-6, 1e-2)."
            )

        wd_low, wd_high = float(wd_space[0]), float(wd_space[1])

        if wd_low <= 0 or wd_high <= 0:
            raise InputValidationError(
                "search_space['weight_decay'] must be positive because it is sampled on a log scale."
            )

        if wd_low > wd_high:
            raise InputValidationError(
                "search_space['weight_decay'] must satisfy low <= high."
            )

    # Validate once at the start.
    T, delta, X, Z = validate_survival_inputs(T, delta, X, Z)

    if folds > T.shape[0]:
        raise InputValidationError("folds cannot be larger than the number of observations.")

    # Use a common tau across folds. This prevents each fold from using a different time horizon.
    tau_for_trials = float(np.max(T)) if tau is None else float(tau)

    def suggest_params(trial):
        hl_opts = default_space["hidden_layers"]
        hl_idx = trial.suggest_categorical("hidden_layers_idx", list(range(len(hl_opts))))

        lr_low, lr_high = map(float, default_space["learning_rate"])
        drop_low, drop_high = map(float, default_space["dropout"])

        params = {
            "hidden_layers": tuple(hl_opts[hl_idx]),
            "learning_rate": (
                lr_low
                if lr_low == lr_high
                else trial.suggest_float("learning_rate", lr_low, lr_high, log=True)
            ),
            "dropout": (
                drop_low
                if drop_low == drop_high
                else trial.suggest_float("dropout", drop_low, drop_high)
            ),
            "batch_size": trial.suggest_categorical(
                "batch_size",
                default_space["batch_size"],
            ),
            "activation": trial.suggest_categorical(
                "activation",
                default_space["activation"],
            ),
            "grid_size": trial.suggest_categorical(
                "grid_size",
                default_space["grid_size"],
            ),
        }

        if "weight_decay" in default_space:
            wd_low, wd_high = map(float, default_space["weight_decay"])

            if wd_low == wd_high:
                params["weight_decay"] = wd_low
            else:
                params["weight_decay"] = trial.suggest_float(
                    "weight_decay",
                    wd_low,
                    wd_high,
                    log=True,
                )
        else:
            params["weight_decay"] = 0.0

        return params
    
    def make_pruning_callback(trial, fold_id: int | None = None):
        if not enable_pruning:
            return None

        def pruning_callback(epoch, train_loss, val_loss, model):
            if fold_id is None:
                step = epoch
            else:
                step = fold_id * max_epochs + epoch

            trial.report(float(val_loss), step=step)

            if trial.should_prune():
                if verbose:
                    if fold_id is None:
                        print(
                            f"Trial {trial.number} pruned at epoch {epoch + 1} "
                            f"with val_loss={float(val_loss):.6f}"
                        )
                    else:
                        print(
                            f"Trial {trial.number} pruned at fold {fold_id + 1}, "
                            f"epoch {epoch + 1}, val_loss={float(val_loss):.6f}"
                        )

                raise optuna.TrialPruned()

            return False

        return pruning_callback

    def evaluate_on_holdout(model, T_val, delta_val, X_val, Z_val):
        """Compute validation negative log-likelihood on an outer holdout fold."""
        T_val, delta_val, _ = truncate_at_tau(T_val, delta_val, model.tau)

        if model.x_scaler_ is not None:
            X_val = model.x_scaler_.transform(X_val)
        else:
            X_val = np.asarray(X_val, dtype=np.float64)

        if Z_val is not None:
            if model.z_scaler_ is not None:
                Z_val = model.z_scaler_.transform(Z_val)
            else:
                Z_val = np.asarray(Z_val, dtype=np.float64)

        expanded_val = make_expanded_data(
            T_val,
            delta_val,
            X_val,
            Z_val,
            model.time_grid,
        )

        idx = np.arange(expanded_val["t"].shape[0])
        val_data = model._tensors_from_expanded(expanded_val, idx)
        return float(model._validation_loss(val_data))

    def objective_single_split(trial):
        params = suggest_params(trial)

        model = FlexiHazardModel(
            **params,
            tau=tau_for_trials,
            max_epochs=max_epochs,
            patience=15,
            compute_theta_variance=False,
            refine_theta=False,
            device=device,
            seed=None if seed is None else int(seed) + trial.number,
            verbose=0,
            epoch_callback=make_pruning_callback(trial),
            **extra_model_kwargs,
        )

        model.fit(T, delta, X, Z=Z)

        return float(np.min(model.train_history["val_loss"]))

    def objective_kfold(trial):
        params = suggest_params(trial)

        fold_losses = []

        kf = KFold(
            n_splits=folds,
            shuffle=True,
            random_state=None if seed is None else int(seed) + trial.number,
        )

        for fold_id, (train_idx, val_idx) in enumerate(kf.split(X)):
            T_train = T[train_idx]
            delta_train = delta[train_idx]
            X_train = X[train_idx]
            Z_train = None if Z is None else Z[train_idx]

            T_val = T[val_idx]
            delta_val = delta[val_idx]
            X_val = X[val_idx]
            Z_val = None if Z is None else Z[val_idx]

            model = FlexiHazardModel(
                **params,
                tau=tau_for_trials,
                max_epochs=max_epochs,
                patience=15,
                compute_theta_variance=False,
                refine_theta=False,
                device=device,
                seed=None if seed is None else int(seed) + trial.number * 1000 + fold_id,
                verbose=0,
                epoch_callback=make_pruning_callback(trial, fold_id=fold_id),
                **extra_model_kwargs,
            )

            model.fit(T_train, delta_train, X_train, Z=Z_train)

            fold_loss = evaluate_on_holdout(
                model,
                T_val,
                delta_val,
                X_val,
                Z_val,
            )
            fold_losses.append(fold_loss)

            if enable_pruning:
                mean_fold_loss = float(np.mean(fold_losses))
                step = (fold_id + 1) * max_epochs + fold_id
                trial.report(mean_fold_loss, step=step)

                if trial.should_prune():
                    if verbose:
                        print(
                            f"Trial {trial.number} pruned after fold {fold_id + 1} "
                            f"with mean_fold_loss={mean_fold_loss:.6f}"
                        )
                    raise optuna.TrialPruned()

        return float(np.mean(fold_losses))

    objective = objective_single_split if folds == 1 else objective_kfold

    sampler = optuna.samplers.TPESampler(seed=seed)

    if not enable_pruning:
        pruner = optuna.pruners.NopPruner()
    elif pruner is None:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
        )

    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        show_progress_bar=bool(verbose),
    )

    best = dict(study.best_trial.params)
    hl_opts = default_space["hidden_layers"]

    best_params = {
        "hidden_layers": tuple(hl_opts[best.pop("hidden_layers_idx")]),
        **best,
    }

    return {
        "study": study,
        "best_params": best_params,
        "best_value": study.best_value,
    }