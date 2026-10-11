"""
Robustness and regression tests for the flexihaz package.

Place this file in the package tests/ folder and run:

    pytest -q tests/test_flexihaz_package_robustness.py

The file intentionally includes tests for tricky edge cases and suspected flaws.
Some tests may fail on the current implementation. A failure is useful because it
points to a concrete issue that should be fixed before large simulations.
"""

from __future__ import annotations

import inspect
import math
import warnings

import numpy as np
import pytest
import torch

from flexihaz import FlexiHazardModel, tune_flexihaz
from flexihaz.data import build_time_grid, make_expanded_data
from flexihaz.exceptions import InputValidationError, NotFittedError
from flexihaz.inference import estimate_theta_variance, theta_confidence_interval
from flexihaz.loss import negative_log_likelihood
from flexihaz.networks import FlexiHazardNet, make_activation
from flexihaz.preprocessing import Standardizer, truncate_at_tau, validate_survival_inputs
from flexihaz.utils import resolve_device, set_seed


# -----------------------------------------------------------------------------
# Test data helpers
# -----------------------------------------------------------------------------


def small_survival_data(n: int = 24, *, linear: bool = False, seed: int = 123):
    """Create a small deterministic survival dataset for fast tests."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    base = 0.8 + 0.25 * X[:, 0] - 0.15 * X[:, 1]
    T = np.exp(base + 0.15 * rng.normal(size=n))
    T = np.clip(T, 0.15, 4.0)

    # Mixed events/censoring, with enough events in most splits.
    delta = np.zeros(n, dtype=int)
    delta[::2] = 1
    delta[1::5] = 1

    if not linear:
        return T.astype(float), delta, X.astype(float)

    Z = rng.normal(size=(n, 2))
    return T.astype(float), delta, X.astype(float), Z.astype(float)


def tiny_model_kwargs(**overrides):
    """Fast model settings used by estimator tests."""
    kwargs = dict(
        hidden_layers=(4,),
        activation="relu",
        dropout=0.0,
        learning_rate=1e-2,
        batch_size=128,
        max_epochs=3,
        patience=2,
        weight_decay=0.0,
        grid_size=4,
        include_observed_times=True,
        validation_fraction=0.25,
        standardize=True,
        refine_theta=False,
        compute_theta_variance=False,
        device="cpu",
        seed=123,
        verbose=0,
    )
    kwargs.update(overrides)
    return kwargs


# -----------------------------------------------------------------------------
# data.py tests
# -----------------------------------------------------------------------------


def test_build_time_grid_includes_boundaries_and_observed_times():
    T = np.array([0.7, 1.3, 2.0, 4.5])
    grid = build_time_grid(T, tau=3.0, grid_size=4, include_observed_times=True)

    assert grid[0] == 0.0
    assert grid[-1] == 3.0
    assert np.all(np.diff(grid) > 0)
    assert 0.7 in grid
    assert 1.3 in grid
    assert 2.0 in grid
    assert 4.5 not in grid


def test_build_time_grid_rejects_too_small_grid_size():
    with pytest.raises(InputValidationError):
        build_time_grid([1.0, 2.0], tau=2.0, grid_size=1)


def test_make_expanded_data_includes_partial_final_interval_when_T_not_on_grid():
    T = np.array([1.5])
    delta = np.array([0])
    X = np.array([[10.0, 20.0]])
    grid = np.array([0.0, 1.0, 2.0])

    expanded = make_expanded_data(T, delta, X, None, grid)

    assert expanded["t"].reshape(-1).tolist() == [1.0, 2.0]
    assert expanded["start"].reshape(-1).tolist() == [0.0, 1.0]
    assert np.allclose(expanded["length"].reshape(-1), [1.0, 0.5])
    assert expanded["event"].sum() == 0.0


def test_make_expanded_data_marks_event_in_correct_interval_when_event_inside_interval():
    T = np.array([1.5])
    delta = np.array([1])
    X = np.array([[1.0]])
    grid = np.array([0.0, 1.0, 2.0])

    expanded = make_expanded_data(T, delta, X, None, grid)

    assert np.allclose(expanded["length"].reshape(-1), [1.0, 0.5])
    assert expanded["event"].reshape(-1).tolist() == [0.0, 1.0]


def test_make_expanded_data_rejects_non_increasing_grid():
    with pytest.raises(InputValidationError):
        make_expanded_data(
            T=np.array([1.0, 2.0]),
            delta=np.array([1, 0]),
            X=np.ones((2, 1)),
            Z=None,
            time_grid=np.array([0.0, 1.0, 1.0, 2.0]),
        )


def test_make_expanded_data_1d_Z_should_be_column_vector():
    """Regression test for a common shape bug with one linear covariate."""
    T = np.array([1.0, 2.0])
    delta = np.array([1, 0])
    X = np.ones((2, 1))
    Z = np.array([0.5, -0.5])
    grid = np.array([0.0, 1.0, 2.0])

    expanded = make_expanded_data(T, delta, X, Z, grid)

    assert expanded["Z"].ndim == 2
    assert expanded["Z"].shape[1] == 1


def test_make_expanded_data_rejects_mismatched_lengths():
    """The function should not silently index arrays with inconsistent lengths."""
    with pytest.raises((InputValidationError, IndexError)):
        make_expanded_data(
            T=np.array([1.0, 2.0, 3.0]),
            delta=np.array([1, 0, 1]),
            X=np.ones((2, 1)),
            Z=None,
            time_grid=np.array([0.0, 1.0, 2.0, 3.0]),
        )


# -----------------------------------------------------------------------------
# preprocessing.py tests
# -----------------------------------------------------------------------------


def test_validate_survival_inputs_shapes_and_1d_covariates():
    T = [1.0, 2.0, 3.0]
    delta = [1, 0, 1]
    X = [0.1, 0.2, 0.3]
    Z = [1.0, 2.0, 3.0]

    T_out, delta_out, X_out, Z_out = validate_survival_inputs(T, delta, X, Z)

    assert T_out.shape == (3,)
    assert delta_out.shape == (3,)
    assert X_out.shape == (3, 1)
    assert Z_out.shape == (3, 1)


def test_validate_survival_inputs_rejects_nan_and_negative_times():
    with pytest.raises(InputValidationError):
        validate_survival_inputs([1.0, np.nan], [1, 0], [[1.0], [2.0]])

    with pytest.raises(InputValidationError):
        validate_survival_inputs([1.0, -0.5], [1, 0], [[1.0], [2.0]])


def test_validate_survival_inputs_rejects_no_events():
    with pytest.raises(InputValidationError):
        validate_survival_inputs([1.0, 2.0], [0, 0], [[1.0], [2.0]])


def test_validate_survival_inputs_rejects_fractional_delta_before_casting():
    """Regression test: delta=0.2 should not be silently cast to 0."""
    with pytest.raises(InputValidationError):
        validate_survival_inputs([1.0, 2.0, 3.0], [1.0, 0.2, 0.0], [[1.0], [2.0], [3.0]])


def test_truncate_at_tau_censors_events_after_tau():
    T = np.array([1.0, 3.0, 5.0])
    delta = np.array([1, 1, 0])

    T_trunc, delta_trunc, tau = truncate_at_tau(T, delta, tau=2.0)

    assert tau == 2.0
    assert np.allclose(T_trunc, [1.0, 2.0, 2.0])
    assert delta_trunc.tolist() == [1, 0, 0]


def test_truncate_at_tau_rejects_invalid_tau():
    with pytest.raises(InputValidationError):
        truncate_at_tau(np.array([1.0, 2.0]), np.array([1, 0]), tau=0.0)


def test_standardizer_constant_column_becomes_zero():
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    scaler = Standardizer()
    Xs = scaler.fit_transform(X)

    assert np.allclose(Xs[:, 1], 0.0)
    assert np.all(np.isfinite(Xs))


def test_standardizer_transform_before_fit_raises():
    with pytest.raises(Exception):
        Standardizer().transform(np.ones((2, 1)))


def test_standardizer_transform_rejects_wrong_number_of_columns():
    """Prediction-time shape mismatch should be caught clearly."""
    scaler = Standardizer().fit(np.ones((4, 2)))
    with pytest.raises(Exception):
        scaler.transform(np.ones((4, 3)))


# -----------------------------------------------------------------------------
# loss.py and networks.py tests
# -----------------------------------------------------------------------------


def test_negative_log_likelihood_matches_manual_formula():
    eta = torch.tensor([[0.0], [math.log(2.0)]])
    event = torch.tensor([[1.0], [0.0]])
    length = torch.tensor([[1.0], [0.5]])

    loss = negative_log_likelihood(eta, event, length, reduction="none")
    expected = torch.tensor([[1.0], [1.0]])

    assert torch.allclose(loss, expected, atol=1e-6)
    assert torch.allclose(negative_log_likelihood(eta, event, length, reduction="sum"), expected.sum())
    assert torch.allclose(negative_log_likelihood(eta, event, length), expected.mean())


def test_negative_log_likelihood_clips_eta_and_nonnegative_lengths():
    eta = torch.tensor([[1000.0], [-1000.0]])
    event = torch.tensor([[0.0], [1.0]])
    length = torch.tensor([[1.0], [-5.0]])

    loss = negative_log_likelihood(eta, event, length, reduction="none", clip=10.0)

    assert torch.all(torch.isfinite(loss))
    assert loss.shape == eta.shape


def test_make_activation_accepts_supported_and_rejects_unknown():
    for name in ["relu", "elu", "tanh", "gelu"]:
        assert make_activation(name) is not None

    with pytest.raises(ValueError):
        make_activation("not_an_activation")


def test_flexihaz_net_forward_shapes_without_Z():
    net = FlexiHazardNet(x_dim=2, z_dim=None, hidden_layers=(4,))
    t = torch.ones((5, 1))
    X = torch.ones((5, 2))

    eta, g = net(t, X, None)

    assert eta.shape == (5, 1)
    assert g.shape == (5, 1)


def test_flexihaz_net_forward_shapes_with_Z_and_missing_Z_error():
    net = FlexiHazardNet(x_dim=2, z_dim=1, hidden_layers=(4,))
    t = torch.ones((5, 1))
    X = torch.ones((5, 2))
    Z = torch.ones((5, 1))

    eta, g = net(t, X, Z)
    assert eta.shape == (5, 1)
    assert g.shape == (5, 1)
    assert net.theta().shape == (1,)

    with pytest.raises(ValueError):
        net(t, X, None)


# -----------------------------------------------------------------------------
# estimator.py tests
# -----------------------------------------------------------------------------


def test_predict_before_fit_raises_not_fitted_error():
    model = FlexiHazardModel(**tiny_model_kwargs())

    with pytest.raises(NotFittedError):
        model.predict_hazard([[0.0, 0.0]], [0.5, 1.0])

    # predict_g currently may raise NotFittedError through _predict_g_on_pairs.
    with pytest.raises(NotFittedError):
        model.predict_g([[0.0, 0.0]], [0.5, 1.0])


def test_fit_predict_nonlinear_shapes_and_survival_properties():
    T, delta, X = small_survival_data(n=20, linear=False)
    model = FlexiHazardModel(**tiny_model_kwargs())
    model.fit(T, delta, X)

    times = np.array([0.0, 0.5, 1.0])
    X_new = X[:3]
    g = model.predict_g(X_new, times)
    h = model.predict_hazard(X_new, times)
    H = model.predict_cumulative_hazard(X_new, times)
    S = model.predict_survival(X_new, times)

    assert model.is_fitted_
    assert g.shape == (3, 3)
    assert h.shape == (3, 3)
    assert H.shape == (3, 3)
    assert S.shape == (3, 3)
    assert np.all(np.isfinite(h))
    assert np.all(h > 0)
    assert np.all(H >= -1e-8)
    assert np.all((S >= -1e-8) & (S <= 1.0 + 1e-8))
    assert np.allclose(S, np.exp(-H))
    assert np.allclose(H[:, 0], 0.0, atol=1e-6)
    assert np.allclose(S[:, 0], 1.0, atol=1e-6)


def test_fit_predict_partially_linear_requires_Z_and_theta_summary():
    T, delta, X, Z = small_survival_data(n=24, linear=True)
    model = FlexiHazardModel(**tiny_model_kwargs())
    model.fit(T, delta, X, Z=Z)

    times = np.array([0.5, 1.0])
    h = model.predict_hazard(X[:4], times, Z_new=Z[:4])

    assert h.shape == (4, 2)
    assert model.theta_hat.shape == (Z.shape[1],)
    assert model.theta_hat_original_scale.shape == (Z.shape[1],)

    summary = model.theta_summary()
    assert "theta_hat" in summary
    assert summary["theta_hat"].shape == (Z.shape[1],)

    with pytest.raises(InputValidationError):
        model.predict_hazard(X[:4], times, Z_new=None)


def test_nonlinear_model_rejects_Z_at_prediction_time():
    T, delta, X = small_survival_data(n=20, linear=False)
    model = FlexiHazardModel(**tiny_model_kwargs())
    model.fit(T, delta, X)

    with pytest.raises(InputValidationError):
        model.predict_hazard(X[:2], [0.5], Z_new=np.ones((2, 1)))


def test_prediction_rejects_times_outside_tau():
    T, delta, X = small_survival_data(n=20, linear=False)
    model = FlexiHazardModel(**tiny_model_kwargs(tau=2.0))
    model.fit(T, delta, X)

    with pytest.raises(InputValidationError):
        model.predict_hazard(X[:2], [-0.1, 0.5])

    with pytest.raises(InputValidationError):
        model.predict_hazard(X[:2], [0.5, 2.1])


def test_unsorted_times_return_original_order_for_cumulative_hazard():
    T, delta, X = small_survival_data(n=20, linear=False)
    model = FlexiHazardModel(**tiny_model_kwargs())
    model.fit(T, delta, X)

    X_new = X[:2]
    unsorted_times = np.array([1.0, 0.0, 0.5])
    sorted_times = np.sort(unsorted_times)

    H_unsorted = model.predict_cumulative_hazard(X_new, unsorted_times)
    H_sorted = model.predict_cumulative_hazard(X_new, sorted_times)

    expected = H_sorted[:, np.argsort(np.argsort(unsorted_times))]
    assert np.allclose(H_unsorted, expected, atol=1e-6)


def test_train_val_split_is_grouped_by_subject_id_no_leakage():
    model = FlexiHazardModel(**tiny_model_kwargs(validation_fraction=0.3))
    subject_id = np.repeat(np.arange(10), 3)

    tr, va = model._train_val_indices(subject_id)

    train_subjects = set(subject_id[tr].tolist())
    val_subjects = set(subject_id[va].tolist())
    assert train_subjects.isdisjoint(val_subjects)
    assert len(train_subjects) > 0
    assert len(val_subjects) > 0


def test_training_early_stopping_stops_before_max_epochs_when_no_improvement():
    T, delta, X = small_survival_data(n=20, linear=False)
    model = FlexiHazardModel(**tiny_model_kwargs(max_epochs=20, patience=1, learning_rate=1e-12))
    model.fit(T, delta, X)

    assert len(model.train_history["val_loss"]) < 20
    assert model.train_history["best_epoch"] >= 0


def test_tau_that_removes_all_events_should_not_train_silently():
    """If all events occur after tau, training has no event signal after truncation."""
    T = np.array([5.0, 6.0, 7.0, 8.0])
    delta = np.array([1, 1, 1, 1])
    X = np.ones((4, 1))

    model = FlexiHazardModel(**tiny_model_kwargs(tau=1.0))

    with pytest.raises(Exception):
        model.fit(T, delta, X)


def test_constructor_rejects_invalid_hyperparameters():
    """These should fail at construction or fit, not fail deep inside PyTorch."""
    invalid_models = [
        dict(batch_size=0),
        dict(max_epochs=0),
        dict(patience=0),
        dict(validation_fraction=1.0),
        dict(dropout=-0.1),
    ]

    for override in invalid_models:
        with pytest.raises(Exception):
            FlexiHazardModel(**tiny_model_kwargs(**override))


def test_optional_dataloader_arguments_exist_if_supported_by_current_api():
    """Regression test for the planned DataLoader/device-transfer configurability."""
    sig = inspect.signature(FlexiHazardModel)
    expected = {"num_workers", "pin_memory", "persistent_workers", "non_blocking"}
    missing = expected.difference(sig.parameters)
    assert not missing, f"Missing FlexiHazardModel arguments: {sorted(missing)}"

def test_time_scaling_divides_network_time_by_tau():
    T, delta, X = small_survival_data(n=20, linear=False)

    model = FlexiHazardModel(
        **tiny_model_kwargs(
            tau=10.0,
            scale_time=True,
            max_epochs=2,
            patience=1,
            grid_size=5,
        )
    )
    model.fit(T, delta, X)

    t_net = model._prepare_time_for_network(np.array([0.0, 5.0, 10.0]))

    np.testing.assert_allclose(
        t_net.reshape(-1),
        np.array([0.0, 0.5, 1.0], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )

def test_prediction_times_remain_original_scale_when_time_scaled():
    T, delta, X = small_survival_data(n=20, linear=False)

    model = FlexiHazardModel(
        **tiny_model_kwargs(
            tau=10.0,
            scale_time=True,
            max_epochs=2,
            patience=1,
            grid_size=5,
        )
    )
    model.fit(T, delta, X)

    times = np.array([0.0, 2.5, 5.0, 10.0])
    S = model.predict_survival(X[:2], times)

    assert S.shape == (2, 4)
    assert np.all(np.isfinite(S))
    assert np.all((S >= 0.0) & (S <= 1.0))

def test_scale_time_false_keeps_old_time_behavior():
    model = FlexiHazardModel(
        scale_time=False,
        tau=10.0,
        max_epochs=2,
        patience=1,
        device="cpu",
    )

    T, delta, X = small_survival_data(n=20, linear=False)
    model.fit(T, delta, X)

    t_net = model._prepare_time_for_network(np.array([0.0, 5.0, 10.0]))

    np.testing.assert_allclose(
        t_net.reshape(-1),
        np.array([0.0, 5.0, 10.0], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )

def test_three_phase_lr_uses_base_lr_multipliers():
    T, delta, X = small_survival_data(n=20, linear=False)

    model = FlexiHazardModel(
        **tiny_model_kwargs(
            learning_rate=1e-3,
            three_phase_lr=True,
            max_epochs=2,
            patience=1,
        )
    )

    model.fit(T, delta, X)

    assert model.learning_rate_phases_used_ == (1e-3, 1e-4, 1e-5)
    assert len(model.phase_histories_) == 3
    assert len(model.train_history["phases"]) == 3

def test_three_phase_lr_false_uses_single_phase():
    T, delta, X = small_survival_data(n=20, linear=False)

    model = FlexiHazardModel(
        **tiny_model_kwargs(
            learning_rate=1e-3,
            three_phase_lr=False,
            max_epochs=2,
            patience=1,
        )
    )

    model.fit(T, delta, X)

    assert model.learning_rate_phases_used_ == (1e-3,)
    assert len(model.phase_histories_) == 1

# -----------------------------------------------------------------------------
# inference.py tests
# -----------------------------------------------------------------------------


def test_theta_confidence_interval_shape_and_values():
    theta = np.array([1.0, -2.0])
    var = np.diag([0.04, 0.09])

    ci = theta_confidence_interval(theta, var, alpha=0.05)

    assert ci.shape == (2, 2)
    assert ci[0, 0] < theta[0] < ci[0, 1]
    assert ci[1, 0] < theta[1] < ci[1, 1]


def test_estimate_theta_variance_fallback_for_few_events_returns_finite_matrices():
    T = np.array([1.0, 2.0, 3.0, 4.0])
    delta = np.array([1, 0, 0, 0])
    X = np.ones((4, 2))
    Z = np.array([[0.0, 1.0], [1.0, 0.5], [2.0, -0.5], [3.0, -1.0]])

    var, info = estimate_theta_variance(
        T,
        delta,
        X,
        Z,
        hidden_layers=(4,),
        max_epochs=1,
        patience=1,
        batch_size=4,
        device="cpu",
        seed=123,
    )

    assert var.shape == (2, 2)
    assert info.shape == (2, 2)
    assert np.all(np.isfinite(var))
    assert np.all(np.isfinite(info))


def test_estimate_theta_variance_accepts_1d_Z_as_single_covariate():
    T = np.array([1.0, 2.0, 3.0, 4.0])
    delta = np.array([1, 0, 1, 0])
    X = np.ones((4, 1))
    Z = np.array([0.0, 1.0, 2.0, 3.0])

    var, info = estimate_theta_variance(
        T,
        delta,
        X,
        Z,
        hidden_layers=(4,),
        max_epochs=1,
        patience=1,
        batch_size=4,
        device="cpu",
        seed=123,
    )

    assert var.shape == (1, 1)
    assert info.shape == (1, 1)


# -----------------------------------------------------------------------------
# tuning.py tests
# -----------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
def test_tune_flexihaz_single_split_returns_best_params_fast():
    pytest.importorskip("optuna")
    T, delta, X = small_survival_data(n=16, linear=False)

    result = tune_flexihaz(
        T,
        delta,
        X,
        Z=None,
        n_trials=1,
        folds=1,
        max_epochs=2,
        seed=123,
        device="cpu",
        verbose=0,
        search_space={
            "hidden_layers": [[4]],
            "learning_rate": (1e-2, 1e-2),
            "weight_decay": (1e-8, 1e-8),
            "dropout": (0.0, 0.0),
            "batch_size": [64],
            "activation": ["relu"],
            "grid_size": [3],
        },
    )

    assert "best_params" in result
    assert "best_value" in result
    assert np.isfinite(result["best_value"])
    assert result["best_params"]["hidden_layers"] == (4,)


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
def test_tune_flexihaz_kfold_returns_best_params_fast():
    pytest.importorskip("optuna")
    T, delta, X = small_survival_data(n=18, linear=False)

    result = tune_flexihaz(
        T,
        delta,
        X,
        Z=None,
        n_trials=1,
        folds=3,
        max_epochs=2,
        seed=123,
        device="cpu",
        verbose=0,
        search_space={
            "hidden_layers": [[4]],
            "learning_rate": (1e-2, 1e-2),
            "weight_decay": (1e-8, 1e-8),
            "dropout": (0.0, 0.0),
            "batch_size": [64],
            "activation": ["relu"],
            "grid_size": [3],
        },
    )

    assert np.isfinite(result["best_value"])
    assert result["best_params"]["hidden_layers"] == (4,)


def test_tune_flexihaz_rejects_invalid_folds_and_epochs():
    T, delta, X = small_survival_data(n=8, linear=False)

    with pytest.raises(InputValidationError):
        tune_flexihaz(T, delta, X, n_trials=1, folds=0, max_epochs=2)

    with pytest.raises(InputValidationError):
        tune_flexihaz(T, delta, X, n_trials=1, folds=20, max_epochs=2)

    with pytest.raises(InputValidationError):
        tune_flexihaz(T, delta, X, n_trials=1, folds=1, max_epochs=0)


def test_tune_flexihaz_parallel_and_pruning_arguments_exist_if_supported_by_current_api():
    """Regression test for planned tuning API: parallel trials and pruning."""
    sig = inspect.signature(tune_flexihaz)
    expected = {"n_jobs", "enable_pruning", "pruner", "model_kwargs"}
    missing = expected.difference(sig.parameters)
    assert not missing, f"Missing tune_flexihaz arguments: {sorted(missing)}"


# -----------------------------------------------------------------------------
# utils.py tests
# -----------------------------------------------------------------------------


def test_resolve_device_returns_torch_device():
    assert isinstance(resolve_device("cpu"), torch.device)
    assert isinstance(resolve_device("auto"), torch.device)


def test_set_seed_reproducibility_for_numpy_and_torch():
    set_seed(999)
    a_np = np.random.normal(size=3)
    a_torch = torch.randn(3)

    set_seed(999)
    b_np = np.random.normal(size=3)
    b_torch = torch.randn(3)

    assert np.allclose(a_np, b_np)
    assert torch.allclose(a_torch, b_torch)


# -----------------------------------------------------------------------------
# Multiple restarts (n_restarts / restart_selection)
# -----------------------------------------------------------------------------


def test_n_restarts_one_matches_default_fit():
    T, delta, X, Z = small_survival_data(n=40, linear=True)

    default = FlexiHazardModel(**tiny_model_kwargs()).fit(T, delta, X, Z=Z)
    single = FlexiHazardModel(**tiny_model_kwargs(n_restarts=1)).fit(T, delta, X, Z=Z)

    np.testing.assert_allclose(default.theta_hat, single.theta_hat, rtol=0, atol=0)
    assert single.selected_restart_ == 0
    assert single.restart_results_ is None


def test_restarts_val_loss_keeps_lowest_validation_loss():
    T, delta, X, Z = small_survival_data(n=40, linear=True)

    single = FlexiHazardModel(**tiny_model_kwargs()).fit(T, delta, X, Z=Z)
    model = FlexiHazardModel(**tiny_model_kwargs(n_restarts=3, restart_selection="val_loss")).fit(T, delta, X, Z=Z)

    results = model.restart_results_
    assert len(results) == 3
    assert [r["seed"] for r in results] == [123, 124, 125]
    best = int(np.argmin([r["val_loss"] for r in results]))
    assert model.selected_restart_ == best
    np.testing.assert_allclose(model.theta_hat_original_scale, results[best]["theta"], rtol=1e-6, atol=1e-6)
    # restart 0 is exactly the single fit
    np.testing.assert_allclose(results[0]["theta"], single.theta_hat_original_scale, rtol=1e-6, atol=1e-6)


def test_restarts_efficient_score_keeps_smallest_score_norm():
    T, delta, X, Z = small_survival_data(n=80, linear=True)

    model = FlexiHazardModel(
        **tiny_model_kwargs(
            n_restarts=3,
            restart_selection="efficient_score",
            projection_n_splits=2,
            projection_max_epochs=3,
            projection_patience=2,
        )
    ).fit(T, delta, X, Z=Z)

    results = model.restart_results_
    assert all(np.isfinite(r["score_norm"]) and r["score"].shape == (Z.shape[1],) for r in results)
    best = int(np.argmin([r["score_norm"] for r in results]))
    assert model.selected_restart_ == best
    np.testing.assert_allclose(model.theta_hat_original_scale, results[best]["theta"], rtol=1e-6, atol=1e-6)


def test_efficient_score_selection_without_z_falls_back_to_val_loss():
    T, delta, X = small_survival_data(n=40, linear=False)

    with pytest.warns(UserWarning, match="needs Z"):
        model = FlexiHazardModel(
            **tiny_model_kwargs(n_restarts=2, restart_selection="efficient_score")
        ).fit(T, delta, X)

    best = int(np.argmin([r["val_loss"] for r in model.restart_results_]))
    assert model.selected_restart_ == best


@pytest.mark.parametrize("kwargs", [dict(n_restarts=0), dict(restart_selection="score")])
def test_invalid_restart_arguments_raise(kwargs):
    with pytest.raises(InputValidationError):
        FlexiHazardModel(**tiny_model_kwargs(**kwargs))


# -----------------------------------------------------------------------------
# theta learning-rate multiplier
# -----------------------------------------------------------------------------


def test_theta_lr_multiplier_one_matches_default_fit():
    T, delta, X, Z = small_survival_data(n=40, linear=True)

    default = FlexiHazardModel(**tiny_model_kwargs()).fit(T, delta, X, Z=Z)
    same = FlexiHazardModel(**tiny_model_kwargs(theta_lr_multiplier=1.0)).fit(T, delta, X, Z=Z)

    np.testing.assert_allclose(default.theta_hat, same.theta_hat, rtol=0, atol=0)


def test_theta_lr_multiplier_changes_only_the_linear_branch_rate():
    T, delta, X, Z = small_survival_data(n=40, linear=True)

    model = FlexiHazardModel(**tiny_model_kwargs(theta_lr_multiplier=20.0, learning_rate=1e-3))
    model.fit(T, delta, X, Z=Z)
    assert np.all(np.isfinite(model.theta_hat))

    opt = model._make_optimizer(1e-3)
    assert len(opt.param_groups) == 2
    lin_ids = {id(p) for p in model.net_.linear.parameters()}
    net_group, lin_group = opt.param_groups
    assert net_group["lr"] == pytest.approx(1e-3)
    assert lin_group["lr"] == pytest.approx(20e-3)
    assert {id(p) for p in lin_group["params"]} == lin_ids
    assert not any(id(p) in lin_ids for p in net_group["params"])


def test_theta_lr_multiplier_without_z_uses_single_group():
    T, delta, X = small_survival_data(n=40, linear=False)
    model = FlexiHazardModel(**tiny_model_kwargs(theta_lr_multiplier=5.0)).fit(T, delta, X)
    assert len(model._make_optimizer(1e-3).param_groups) == 1


@pytest.mark.parametrize("mult", [0.0, -2.0, float("nan")])
def test_invalid_theta_lr_multiplier_raises(mult):
    with pytest.raises(InputValidationError):
        FlexiHazardModel(**tiny_model_kwargs(theta_lr_multiplier=mult))


# -----------------------------------------------------------------------------
# model_type="zhong" (Zhong et al. 2022 proportional-hazards model)
# -----------------------------------------------------------------------------


def zhong_kwargs(**overrides):
    kwargs = tiny_model_kwargs(model_type="zhong", max_epochs=5, patience=3)
    kwargs.update(overrides)
    return kwargs


def test_zhong_network_ignores_time_and_projection_is_additive():
    from flexihaz.networks import AdditiveProjectionNet, ZhongPHNet

    net = ZhongPHNet(x_dim=3, z_dim=2, hidden_layers=(4,))
    X = torch.randn(5, 3)
    Z = torch.randn(5, 2)
    eta_a, g_a = net(torch.zeros(5, 1), X, Z)
    eta_b, g_b = net(torch.ones(5, 1), X, Z)
    assert torch.allclose(eta_a, eta_b) and torch.allclose(g_a, g_b)
    assert eta_a.shape == (5, 1)

    proj = AdditiveProjectionNet(x_dim=3, hidden_layers=(4,))
    t = torch.randn(5, 1)
    out = proj(t, X)
    # no X-t interaction: changing X shifts the output by the same amount at every t
    X2 = X + 1.0
    diff = proj(t, X2) - out
    assert torch.allclose(diff, diff[:1].expand_as(diff) * 0 + diff, atol=1e-6)
    assert torch.allclose(proj(t, X2) - proj(t, X), proj(t + 1.0, X2) - proj(t + 1.0, X), atol=1e-5)


def test_cox_partial_likelihood_loss_matches_direct_formula():
    from flexihaz.loss import cox_partial_likelihood_loss

    rng = np.random.default_rng(0)
    T = rng.uniform(0.1, 3.0, size=12)
    d = rng.integers(0, 2, size=12).astype(np.float64)
    d[0] = 1.0
    eta = rng.normal(size=12)
    ll = 0.0
    for i in range(12):
        if d[i] == 1:
            ll += eta[i] - np.log(np.sum(np.exp(eta[T >= T[i]])))
    expected = -ll / d.sum()
    got = cox_partial_likelihood_loss(torch.as_tensor(eta).reshape(-1, 1), torch.as_tensor(T), torch.as_tensor(d))
    assert float(got) == pytest.approx(expected, rel=1e-6)


def test_zhong_fit_trains_on_subject_rows_and_predicts():
    T, delta, X, Z = small_survival_data(n=40, linear=True)
    model = FlexiHazardModel(**zhong_kwargs(compute_theta_variance=True, projection_n_splits=2,
                                            projection_max_epochs=3, projection_patience=2, refine_theta=True))
    model.fit(T, delta, X, Z=Z)

    assert model.model_type == "zhong"
    assert model.theta_hat_original_scale.shape == (2,)
    assert np.all(np.isfinite(model.theta_hat_original_scale))
    assert model.var_theta_hat.shape == (2, 2) and np.all(np.diag(model.var_theta_hat) > 0)

    times = np.linspace(0.0, model.tau, 5)
    g = model.predict_g(X[:3], times)
    assert g.shape == (3, 5)
    # g has no time input, so it is constant across the time axis
    np.testing.assert_allclose(g, np.repeat(g[:, :1], 5, axis=1), rtol=0, atol=0)

    H = model.predict_cumulative_hazard(X[:3], times, Z_new=Z[:3])
    S = model.predict_survival(X[:3], times, Z_new=Z[:3])
    assert H.shape == (3, 5) and S.shape == (3, 5)
    assert np.all(H[:, 0] == 0.0) and np.all(np.diff(H, axis=1) >= 0)
    assert np.all((S >= 0) & (S <= 1)) and np.all(S[:, 0] == 1.0)
    # Breslow: cumulative hazard is proportional across subjects
    pos = H[0] > 0
    assert pos.sum() >= 2
    ratio = H[1:, pos] / H[:1, pos]
    np.testing.assert_allclose(ratio, np.repeat(ratio[:, :1], ratio.shape[1], axis=1), rtol=1e-5)

    with pytest.raises(InputValidationError, match="predict_hazard"):
        model.predict_hazard(X[:3], times, Z_new=Z[:3])


def test_zhong_without_z_fits_nonparametric_ph():
    T, delta, X = small_survival_data(n=40, linear=False)
    model = FlexiHazardModel(**zhong_kwargs()).fit(T, delta, X)
    assert model.theta_hat is None
    S = model.predict_survival(X[:2], np.linspace(0.0, model.tau, 4))
    assert S.shape == (2, 4)


def test_zhong_restarts_with_efficient_score():
    T, delta, X, Z = small_survival_data(n=80, linear=True)
    model = FlexiHazardModel(
        **zhong_kwargs(n_restarts=3, restart_selection="efficient_score",
                       projection_n_splits=2, projection_max_epochs=3, projection_patience=2)
    ).fit(T, delta, X, Z=Z)
    results = model.restart_results_
    assert len(results) == 3
    assert all(np.isfinite(r["score_norm"]) and r["score"].shape == (Z.shape[1],) for r in results)
    best = int(np.argmin([r["score_norm"] for r in results]))
    assert model.selected_restart_ == best
    np.testing.assert_allclose(model.theta_hat_original_scale, results[best]["theta"], rtol=1e-6, atol=1e-6)


def test_zhong_refine_theta_maximises_profile_partial_likelihood():
    from flexihaz.loss import cox_partial_likelihood_loss

    T, delta, X, Z = small_survival_data(n=60, linear=True)
    model = FlexiHazardModel(**zhong_kwargs(refine_theta=True, three_phase_lr=False)).fit(T, delta, X, Z=Z)
    # the refinement is done on the internal training subjects only
    tr = model.internal_train_subject_ids_
    rows = model._subject_rows(T[tr], delta[tr], model.X_train_scaled_[tr], model.Z_train_scaled_[tr])
    g = model._predict_g_on_pairs(rows["X"], rows["t"], batch_size=model.batch_size).reshape(-1, 1)
    Zs = rows["Z"].astype(np.float64)
    T_t, d_t = torch.as_tensor(T[tr]), torch.as_tensor(delta[tr], dtype=torch.float64)

    def loss_at(theta):
        eta = torch.as_tensor(g + Zs @ theta.reshape(-1, 1))
        return float(cox_partial_likelihood_loss(eta, T_t, d_t))

    theta_hat = model.theta_hat.astype(np.float64)
    at_hat = loss_at(theta_hat)
    for shift in (0.2, -0.2):
        for j in range(theta_hat.size):
            other = theta_hat.copy()
            other[j] += shift
            assert loss_at(other) >= at_hat - 1e-6


def test_epoch_defaults_depend_on_model_type():
    base = FlexiHazardModel(device="cpu")
    assert (base.max_epochs, base.patience) == (500, 30)
    zh = FlexiHazardModel(device="cpu", model_type="zhong")
    assert (zh.max_epochs, zh.patience) == (5000, 35)
    explicit = FlexiHazardModel(device="cpu", model_type="zhong", max_epochs=7, patience=2)
    assert (explicit.max_epochs, explicit.patience) == (7, 2)


def test_invalid_model_type_raises():
    with pytest.raises(InputValidationError):
        FlexiHazardModel(**tiny_model_kwargs(model_type="cox"))


# -----------------------------------------------------------------------------
# Best validation loss carried across learning-rate phases; non-finite guard
# -----------------------------------------------------------------------------


def _state_equal(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


def test_fit_expanded_keeps_entry_weights_when_init_best_val_is_unbeatable():
    T, delta, X, Z = small_survival_data(n=24, linear=True)
    model = FlexiHazardModel(**tiny_model_kwargs(three_phase_lr=False, max_epochs=3, patience=5))
    model.fit(T, delta, X, Z=Z)

    expanded = make_expanded_data(T, delta, model.x_scaler_.transform(X), model.z_scaler_.transform(Z), model.time_grid)
    train_idx, val_idx = model._train_val_indices(expanded["subject_id"])
    entry_state = {k: v.clone() for k, v in model.net_.state_dict().items()}

    best = model._fit_expanded(expanded, train_idx, val_idx, learning_rate=1e-3, init_best_val=-1.0)

    assert best == -1.0
    assert model.train_history["best_epoch"] == -1
    assert _state_equal(model.net_.state_dict(), entry_state)


def test_fit_expanded_returns_best_validation_loss():
    T, delta, X, Z = small_survival_data(n=24, linear=True)
    model = FlexiHazardModel(**tiny_model_kwargs(three_phase_lr=False, max_epochs=4, patience=10))
    model.fit(T, delta, X, Z=Z)
    expanded = make_expanded_data(T, delta, model.x_scaler_.transform(X), model.z_scaler_.transform(Z), model.time_grid)
    train_idx, val_idx = model._train_val_indices(expanded["subject_id"])

    best = model._fit_expanded(expanded, train_idx, val_idx, learning_rate=1e-3)

    assert math.isfinite(best)
    assert best == pytest.approx(min(model.train_history["val_loss"]))


def test_three_phase_lr_never_ends_worse_than_an_earlier_phase():
    T, delta, X, Z = small_survival_data(n=30, linear=True, seed=7)
    model = FlexiHazardModel(**tiny_model_kwargs(three_phase_lr=True, max_epochs=6, patience=3))
    model.fit(T, delta, X, Z=Z)

    phases = model.train_history["phases"]
    assert len(phases) == 3
    # every phase that recorded an improvement must beat the best of all earlier phases
    best_so_far = math.inf
    for ph in phases:
        if ph["best_epoch"] >= 0:
            phase_best = min(ph["val_loss"])
            assert phase_best < best_so_far
            best_so_far = phase_best
        else:
            # no improvement in this phase: nothing in it beats the carried best
            assert min(ph["val_loss"]) >= best_so_far - 1e-8


def test_non_finite_validation_loss_is_ignored_and_warned(monkeypatch):
    T, delta, X, Z = small_survival_data(n=24, linear=True)
    model = FlexiHazardModel(**tiny_model_kwargs(three_phase_lr=False, max_epochs=3, patience=5))
    model.fit(T, delta, X, Z=Z)
    expanded = make_expanded_data(T, delta, model.x_scaler_.transform(X), model.z_scaler_.transform(Z), model.time_grid)
    train_idx, val_idx = model._train_val_indices(expanded["subject_id"])
    entry_state = {k: v.clone() for k, v in model.net_.state_dict().items()}

    monkeypatch.setattr(model, "_validation_loss", lambda data: float("nan"))
    with pytest.warns(UserWarning, match="non-finite validation loss"):
        best = model._fit_expanded(expanded, train_idx, val_idx, learning_rate=1e-3)

    assert best == math.inf
    assert model.train_history["best_epoch"] == -1
    assert _state_equal(model.net_.state_dict(), entry_state)


def test_cox_warm_start_uses_x_and_z_jointly():
    T, delta, X, Z = small_survival_data(n=40, linear=True, seed=3)
    model = FlexiHazardModel(**tiny_model_kwargs(three_phase_lr=False, max_epochs=1, patience=1, theta_init="cox"))
    model.fit(T, delta, X, Z=Z)
    joint = model.theta_init_original_scale_.copy()

    # Z-only fit for comparison
    model._initialize_theta_from_cox(T, delta, None, Z, model.z_scaler_.transform(Z))
    z_only = model.theta_init_original_scale_.copy()

    assert joint.shape == z_only.shape == (2,)
    assert not np.allclose(joint, z_only)


def test_refine_theta_uses_training_rows_only(monkeypatch):
    T, delta, X, Z = small_survival_data(n=30, linear=True, seed=11)
    seen = []
    for three_phase in (False, True):
        model = FlexiHazardModel(**tiny_model_kwargs(three_phase_lr=three_phase, max_epochs=2, patience=2, refine_theta=True))
        orig = model._refine_theta
        monkeypatch.setattr(model, "_refine_theta", lambda rows, _o=orig: (seen.append(rows["t"].shape[0]), _o(rows)))
        model.fit(T, delta, X, Z=Z)
        n_train_rows = int(np.isin(
            make_expanded_data(T, delta, model.x_scaler_.transform(X), model.z_scaler_.transform(Z), model.time_grid)["subject_id"],
            model.internal_train_subject_ids_).sum())
        assert seen and all(n == n_train_rows for n in seen)
        seen.clear()
