import numpy as np
from flexihaz import FlexiHazardModel


def make_data(n=60, seed=0, z=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    Z = rng.normal(size=(n, 1)) if z else None
    lin = 0.0 if Z is None else 0.5 * Z[:, 0]
    rate = np.exp(lin + 0.1 * X[:, 0])
    U = rng.exponential(1 / rate)
    C = rng.exponential(1.5, size=n)
    T = np.minimum(U, C)
    delta = (U <= C).astype(int)
    return T, delta, X, Z


def test_nonlinear_fit_predict_shapes():
    T, d, X, _ = make_data()
    tau = float(np.quantile(T, 0.8))
    model = FlexiHazardModel(hidden_layers=[8], max_epochs=2, patience=1, grid_size=20, tau=tau, compute_theta_variance=False)
    model.fit(T, d, X)
    times = np.linspace(0, tau, 5)
    assert model.predict_g(X[:3], times).shape == (3, 5)
    assert model.predict_survival(X[:3], times).shape == (3, 5)


def test_partially_linear_fit_predict_shapes():
    T, d, X, Z = make_data(z=True)
    tau = float(np.quantile(T, 0.8))
    model = FlexiHazardModel(hidden_layers=[8], max_epochs=2, patience=1, grid_size=20, tau=tau, compute_theta_variance=False)
    model.fit(T, d, X, Z=Z)
    times = np.linspace(0, tau, 5)
    assert model.theta_hat_original_scale.shape == (1,)
    assert model.predict_survival(X[:3], times, Z_new=Z[:3]).shape == (3, 5)
