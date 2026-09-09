"""Partially linear coefficient-inference example."""

import numpy as np

from flexihaz import FlexiHazardModel


rng = np.random.default_rng(3)
n = 300

X = rng.normal(size=(n, 2))
Z = rng.normal(size=(n, 1))

linear_predictor = 0.5 * Z[:, 0] + 0.15 * X[:, 0]
event_time = rng.exponential(scale=np.exp(-linear_predictor))
censor_time = rng.exponential(scale=2.0, size=n)

T = np.minimum(event_time, censor_time)
delta = (event_time <= censor_time).astype(int)

model = FlexiHazardModel(
    hidden_layers=(32, 32),
    max_epochs=60,
    patience=10,
    projection_max_epochs=60,
    projection_patience=10,
    projection_n_splits=3,
    theta_init="cox",
    refine_theta=True,
    compute_theta_variance=True,
    seed=3,
)

model.fit(T, delta, X, Z=Z)

summary = model.theta_summary(alpha=0.05)
print(summary)
