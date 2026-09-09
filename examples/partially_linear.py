"""Minimal partially linear FLEXI-Haz example."""

import numpy as np

from flexihaz import FlexiHazardModel


rng = np.random.default_rng(1)
n = 200

X = rng.normal(size=(n, 2))
Z = rng.normal(size=(n, 1))

linear_predictor = 0.6 * Z[:, 0] + 0.2 * X[:, 0]
event_time = rng.exponential(scale=np.exp(-linear_predictor))
censor_time = rng.exponential(scale=2.0, size=n)

T = np.minimum(event_time, censor_time)
delta = (event_time <= censor_time).astype(int)
tau = float(np.max(T))

model = FlexiHazardModel(
    hidden_layers=(32, 32),
    learning_rate=1e-3,
    batch_size=1024,
    max_epochs=50,
    patience=10,
    tau=tau,
    theta_init="cox",
    refine_theta=True,
    compute_theta_variance=False,
    seed=1,
)

model.fit(T, delta, X, Z=Z)

times = np.linspace(0.0, tau, 50)
survival = model.predict_survival(X[:5], times, Z_new=Z[:5])

print("Survival array shape:", survival.shape)
print("Estimated theta:", model.theta_hat_original_scale)
