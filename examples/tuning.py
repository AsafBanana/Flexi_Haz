"""Small Optuna tuning example."""

import numpy as np

from flexihaz import tune_flexihaz


rng = np.random.default_rng(4)
n = 180

X = rng.normal(size=(n, 2))
Z = rng.normal(size=(n, 1))

linear_predictor = 0.4 * Z[:, 0] + 0.1 * X[:, 0]
event_time = rng.exponential(scale=np.exp(-linear_predictor))
censor_time = rng.exponential(scale=2.0, size=n)

T = np.minimum(event_time, censor_time)
delta = (event_time <= censor_time).astype(int)

result = tune_flexihaz(
    T=T,
    delta=delta,
    X=X,
    Z=Z,
    n_trials=3,
    folds=1,
    max_epochs=30,
    seed=4,
    search_space={
        "hidden_layers": [[16], [32, 32]],
        "learning_rate": (1e-3, 1e-2),
        "dropout": (0.0, 0.0),
        "batch_size": [512],
        "activation": ["relu"],
        "grid_size": ["n"],
    },
    model_kwargs={
        "refine_theta": False,
        "compute_theta_variance": False,
    },
)

print("Best parameters:", result["best_params"])
print("Best validation loss:", result["best_value"])
