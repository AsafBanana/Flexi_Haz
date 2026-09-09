"""Minimal fully nonparametric FLEXI-Haz example."""

import numpy as np

from flexihaz import FlexiHazardModel


rng = np.random.default_rng(2)
n = 200

W = rng.normal(size=(n, 4))
linear_predictor = 0.2 * W[:, 0] - 0.15 * W[:, 1]
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
    refine_theta=False,
    compute_theta_variance=False,
    seed=2,
)

model.fit(T, delta, W, Z=None)

times = np.linspace(0.0, tau, 50)
survival = model.predict_survival(W[:5], times, Z_new=None)

print("Survival array shape:", survival.shape)
