"""Shared utilities for the Zhong et al. (2022) partially linear neural PH comparator.

This module intentionally keeps the model structure used in PH_model.ipynb:
    chi(X, Z) = g(X, 0) + beta^T Z
trained with a Cox partial-likelihood loss.  The et_input is kept for fidelity to
that notebook, but it is fed zeros, so the fitted model is proportional hazards.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


TRUE_THETA = np.array([2.0, -1.0], dtype=np.float64)
X_COLUMNS = ["X0", "X1", "X2"]
Z_COLUMNS = ["Z0", "Z1"]
ALL_COLUMNS = X_COLUMNS + Z_COLUMNS


@dataclass
class ZhongFitResult:
    model: Any
    beta_hat: np.ndarray
    beta_init: np.ndarray
    beta_cov: np.ndarray
    train_loss: float
    val_loss: float
    n_epochs_run: int
    best_epoch: int | None
    history: dict[str, list[float]]


def set_all_seeds(seed: int) -> None:
    """Set Python, NumPy, and TensorFlow seeds."""
    os.environ.setdefault("PYTHONHASHSEED", str(int(seed)))
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import tensorflow as tf

        tf.random.set_seed(int(seed))
    except Exception:
        pass


def cox_ph_loss_no_ties(y_true, y_pred, clip_pred: float = 80.0, eps: float = 1e-12):
    """Negative Cox partial log-likelihood.

    y_true columns are [T, start, event]. The start column is kept for alignment
    with PH_model.ipynb, but the PH Cox loss uses only T and event.

    y_pred has two columns [chi, g]. Only chi is used by the loss.
    """
    import tensorflow as tf

    T = tf.reshape(y_true[:, 0], [-1])
    d = tf.reshape(y_true[:, 2], [-1])
    y = tf.reshape(tf.clip_by_value(y_pred[:, 0], -clip_pred, clip_pred), [-1])

    order = tf.argsort(T, direction="DESCENDING")
    y_sorted = tf.gather(y, order)
    d_sorted = tf.gather(d, order)

    cumexp = tf.cumsum(tf.exp(y_sorted))
    log_denom = tf.math.log(tf.maximum(cumexp, eps))

    contrib = d_sorted * (y_sorted - log_denom)
    n_events = tf.reduce_sum(d_sorted)
    return -tf.reduce_sum(contrib) / tf.maximum(n_events, 1.0)


# class L1Var:
#     """L1 regularizer matching PH_model.ipynb."""

#     def __init__(self, coeff_var: float):
#         self.c = float(coeff_var)

#     def __call__(self, x):
#         import tensorflow as tf

#         return self.c * tf.reduce_sum(tf.abs(x))

#     def get_config(self):
#         return {"coeff_var": self.c}


def build_zhong_model(
    *,
    input_dim_X: int,
    input_dim_Z: int,
    hidden_layers: list[int] | tuple[int, ...],
    beta_init: np.ndarray,
    lmbd_l1: float = 0.0,
):
    """Build the neural PH model from PH_model.ipynb without changing its form."""
    import tensorflow as tf
    from tensorflow.keras import Input, Model
    from tensorflow.keras.initializers import Constant
    from tensorflow.keras.layers import Add, Concatenate, Dense
    from tensorflow.keras.regularizers import L1

    X_input = Input(shape=(input_dim_X,), name="X_input")
    Z_input = Input(shape=(input_dim_Z,), name="Z_input")
    et_input = Input(shape=(1,), name="et_input")

    reg = None if float(lmbd_l1) == 0.0 else L1(float(lmbd_l1))
    g_input = Concatenate(name="concat_xt")([et_input, X_input])
    x = g_input

    hidden_layers = [int(h) for h in hidden_layers]

    for width in hidden_layers:
        x = Dense(
            int(width),
            activation="relu",
            kernel_regularizer=reg,
            bias_regularizer=None,
        )(x)
        x = tf.keras.layers.Activation("relu")(x)

    g_out = Dense(1, activation=None, name="g_head", kernel_regularizer=reg, bias_regularizer=None)(x)
    beta_out = Dense(
        1,
        use_bias=False,
        activation=None,
        name="beta_layer",
        kernel_initializer=Constant(np.asarray(beta_init, dtype=np.float32).reshape(input_dim_Z, 1)),
    )(Z_input)

    chi = Add(name="chi")([g_out, beta_out])
    chi_g = Concatenate(axis=1, name="chi_g")([chi, g_out])
    return Model(inputs=[X_input, Z_input, et_input], outputs=chi_g)


def make_subject_dataframe(X: np.ndarray, Z: np.ndarray, T: np.ndarray | None = None, delta: np.ndarray | None = None) -> pd.DataFrame:
    df = pd.DataFrame(np.column_stack([X, Z]), columns=ALL_COLUMNS)
    if T is not None:
        df["T"] = np.asarray(T, dtype=np.float64).reshape(-1)
    if delta is not None:
        df["event"] = np.asarray(delta, dtype=np.int64).reshape(-1)
    return df


def cox_beta_init(X: np.ndarray, Z: np.ndarray, T: np.ndarray, delta: np.ndarray, penalizer: float = 0.0) -> np.ndarray:
    """Initialize beta from a standard Cox model fit on X and Z, as in the notebook."""
    from lifelines import CoxPHFitter

    df = make_subject_dataframe(X, Z, T, delta)
    cph = CoxPHFitter(penalizer=float(penalizer))
    cph.fit(df, duration_col="T", event_col="event", show_progress=False)
    return cph.params_[Z_COLUMNS].to_numpy(dtype=np.float64).reshape(-1, 1)


def make_model_inputs(X: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    Z = np.asarray(Z, dtype=np.float32)
    et_zero = np.zeros((X.shape[0], 1), dtype=np.float32)
    return X, Z, et_zero


def make_y(T: np.ndarray, delta: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float32).reshape(-1, 1)
    delta = np.asarray(delta, dtype=np.float32).reshape(-1, 1)
    start = np.zeros_like(T, dtype=np.float32)
    return np.concatenate([T, start, delta], axis=1).astype(np.float32)


def compile_zhong_model(model, learning_rate: float):
    from tensorflow.keras.optimizers import Adam

    model.compile(optimizer=Adam(learning_rate=float(learning_rate)), loss=cox_ph_loss_no_ties, jit_compile=False)
    return model


def fit_zhong_model(
    *,
    X: np.ndarray,
    Z: np.ndarray,
    T: np.ndarray,
    delta: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    hidden_layers: list[int] | tuple[int, ...],
    learning_rate: float,
    max_epochs: int,
    patience: int,
    batch_size: int,
    lmbd_l1: float,
    seed: int,
    cox_penalizer: float = 0.0,
    verbose: int = 0,
) -> ZhongFitResult:
    """Fit one Zhong-style neural PH model on a subject-level train/val split."""
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping

    set_all_seeds(seed)
    tf.keras.backend.clear_session()

    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64).reshape(-1)
    delta = np.asarray(delta, dtype=np.int64).reshape(-1)
    train_idx = np.asarray(train_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)

    beta_init = cox_beta_init(X[train_idx], Z[train_idx], T[train_idx], delta[train_idx], penalizer=cox_penalizer)

    model = build_zhong_model(
        input_dim_X=X.shape[1],
        input_dim_Z=Z.shape[1],
        hidden_layers=[int(h) for h in hidden_layers],
        beta_init=beta_init,
        lmbd_l1=float(lmbd_l1),
    )
    model = compile_zhong_model(model, learning_rate=float(learning_rate))

    train_inp = make_model_inputs(X[train_idx], Z[train_idx])
    val_inp = make_model_inputs(X[val_idx], Z[val_idx])
    y_train = make_y(T[train_idx], delta[train_idx])
    y_val = make_y(T[val_idx], delta[val_idx])

    early_stop = EarlyStopping(monitor="val_loss", patience=int(patience), restore_best_weights=True)
    history_obj = model.fit(
        train_inp,
        y_train,
        validation_data=(val_inp, y_val),
        batch_size=int(batch_size),
        epochs=int(max_epochs),
        callbacks=[early_stop],
        verbose=int(verbose),
        shuffle=True,
    )

    hist = {k: [float(x) for x in v] for k, v in history_obj.history.items()}
    beta_hat = model.get_layer("beta_layer").get_weights()[0].reshape(-1)
    train_loss = float(model.evaluate(train_inp, y_train, batch_size=int(batch_size), verbose=0))
    val_loss = float(model.evaluate(val_inp, y_val, batch_size=int(batch_size), verbose=0))
    val_losses = hist.get("val_loss", [])
    best_epoch = int(np.nanargmin(val_losses)) if len(val_losses) else None

    _, g_train = predict_risk_and_g(model, X[train_idx], Z[train_idx], batch_size=int(batch_size))
    beta_cov = cox_beta_covariance_with_offset(
        Z=Z[train_idx],
        T=T[train_idx],
        delta=delta[train_idx],
        beta=beta_hat,
        offset=g_train,
    )

    return ZhongFitResult(
        model=model,
        beta_hat=np.asarray(beta_hat, dtype=np.float64).reshape(-1),
        beta_init=np.asarray(beta_init, dtype=np.float64).reshape(-1),
        beta_cov=np.asarray(beta_cov, dtype=np.float64),
        train_loss=train_loss,
        val_loss=val_loss,
        n_epochs_run=len(hist.get("loss", [])),
        best_epoch=best_epoch,
        history=hist,
    )


def predict_risk_and_g(model, X: np.ndarray, Z: np.ndarray, batch_size: int = 100000) -> tuple[np.ndarray, np.ndarray]:
    pred = model.predict(make_model_inputs(X, Z), batch_size=int(batch_size), verbose=0)
    pred = np.asarray(pred, dtype=np.float64)
    return pred[:, 0].reshape(-1), pred[:, 1].reshape(-1)

def cox_beta_information_with_offset(
    *,
    Z: np.ndarray,
    T: np.ndarray,
    delta: np.ndarray,
    beta: np.ndarray,
    offset: np.ndarray,
) -> np.ndarray:
    """Observed Cox information matrix for beta, treating offset=g_hat(X) as fixed.

    This computes the beta-beta block of the Cox partial-likelihood information:

        I(beta) = sum_j d_j * [ E_w(ZZ^T) - E_w(Z)E_w(Z)^T ]

    where the risk-set weights are proportional to exp(offset_i + beta^T Z_i).

    Ties are handled by the Breslow form, which is also what we use for the
    baseline hazard prediction.
    """
    Z = np.asarray(Z, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64).reshape(-1)
    delta = np.asarray(delta, dtype=bool).reshape(-1)
    beta = np.asarray(beta, dtype=np.float64).reshape(-1)
    offset = np.asarray(offset, dtype=np.float64).reshape(-1)

    p = Z.shape[1]
    info = np.zeros((p, p), dtype=np.float64)

    eta = offset + Z @ beta
    risk = np.exp(np.clip(eta, -50.0, 50.0))

    event_times = np.sort(np.unique(T[delta]))
    eps = 1e-300

    for t in event_times:
        event_mask = delta & np.isclose(T, t)
        d_j = float(np.sum(event_mask))
        if d_j <= 0:
            continue

        risk_mask = T >= t
        Z_risk = Z[risk_mask]
        w = risk[risk_mask]

        s0 = float(np.sum(w))
        if s0 <= eps:
            continue

        s1 = np.sum(Z_risk * w[:, None], axis=0)
        s2 = Z_risk.T @ (Z_risk * w[:, None])

        mean_z = s1 / s0
        second_z = s2 / s0
        var_z = second_z - np.outer(mean_z, mean_z)

        info += d_j * var_z

    return info


def cox_beta_covariance_with_offset(
    *,
    Z: np.ndarray,
    T: np.ndarray,
    delta: np.ndarray,
    beta: np.ndarray,
    offset: np.ndarray,
    ridge: float = 1e-8,
) -> np.ndarray:
    """Approximate covariance matrix for beta using Cox Hessian with fixed offset."""
    info = cox_beta_information_with_offset(
        Z=Z,
        T=T,
        delta=delta,
        beta=beta,
        offset=offset,
    )

    p = info.shape[0]
    info_reg = info + float(ridge) * np.eye(p)

    try:
        cov = np.linalg.inv(info_reg)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(info_reg)

    cov = np.asarray(cov, dtype=np.float64)

    # Numerical safety: symmetrize.
    cov = 0.5 * (cov + cov.T)

    return cov


def breslow_baseline_cumulative_hazard(T: np.ndarray, delta: np.ndarray, eta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Breslow baseline cumulative hazard for a fitted Cox-type risk score."""
    T = np.asarray(T, dtype=np.float64).reshape(-1)
    delta = np.asarray(delta, dtype=bool).reshape(-1)
    eta = np.asarray(eta, dtype=np.float64).reshape(-1)

    event_times = np.sort(np.unique(T[delta]))
    if event_times.size == 0:
        return event_times, np.zeros(0, dtype=np.float64)

    risk = np.exp(np.clip(eta, -50.0, 50.0))
    increments = np.zeros(event_times.shape[0], dtype=np.float64)
    eps = 1e-300
    for j, t in enumerate(event_times):
        d_j = float(np.sum(delta & np.isclose(T, t)))
        denom = float(np.sum(risk[T >= t]))
        increments[j] = d_j / max(denom, eps)
    return event_times, np.cumsum(increments)


def evaluate_baseline_at_times(event_times: np.ndarray, cumhaz: np.ndarray, times: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    if len(event_times) == 0:
        return np.zeros(times.shape[0], dtype=np.float64)
    idx = np.searchsorted(event_times, times, side="right") - 1
    out = np.zeros(times.shape[0], dtype=np.float64)
    valid = idx >= 0
    out[valid] = cumhaz[idx[valid]]
    return out


def predict_survival_from_ph(
    *,
    model,
    train_X: np.ndarray,
    train_Z: np.ndarray,
    train_T: np.ndarray,
    train_delta: np.ndarray,
    test_X: np.ndarray,
    test_Z: np.ndarray,
    eval_times: np.ndarray,
    batch_size: int = 100000,
) -> np.ndarray:
    eta_train, _ = predict_risk_and_g(model, train_X, train_Z, batch_size=batch_size)
    eta_test, _ = predict_risk_and_g(model, test_X, test_Z, batch_size=batch_size)
    event_times, H0 = breslow_baseline_cumulative_hazard(train_T, train_delta, eta_train)
    H0_eval = evaluate_baseline_at_times(event_times, H0, eval_times)
    S_hat = np.exp(-np.outer(np.exp(np.clip(eta_test, -50.0, 50.0)), H0_eval))
    return np.clip(S_hat, 0.0, 1.0)


def make_random_train_val_split(n: int, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    idx = np.arange(int(n))
    train_idx, val_idx = train_test_split(idx, test_size=float(val_frac), random_state=int(seed), shuffle=True)
    return np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)


def write_empirical_theta_summary(theta_csv_paths: list, output_path, model_name: str = "zhong2022_ph") -> None:
    """Compute notebook-style empirical and estimated SE summaries from theta files.

    This mirrors the notebook quantities:

        np.std(beta_all, axis=0)
        np.mean([np.diag(x)**0.5 for x in cov_all], axis=0)

    Here cov_all is represented by the per-replication se columns.
    """
    rows_beta = []
    rows_se = []

    for path in theta_csv_paths:
        df = pd.read_csv(path)

        if {"theta_index", "theta_hat"}.issubset(df.columns):
            df = df.sort_values("theta_index")
            rows_beta.append(df["theta_hat"].to_numpy(dtype=np.float64))

            if "se" in df.columns:
                rows_se.append(pd.to_numeric(df["se"], errors="coerce").to_numpy(dtype=np.float64))

    if not rows_beta:
        return

    theta_hats = np.vstack(rows_beta)
    theta_true = TRUE_THETA[: theta_hats.shape[1]]

    emp_mean = np.mean(theta_hats, axis=0)
    emp_sd = np.std(theta_hats, axis=0, ddof=0)  # matches notebook np.std default

    if rows_se:
        se_mat = np.vstack(rows_se)
        mean_estimated_se = np.nanmean(se_mat, axis=0)
    else:
        mean_estimated_se = np.full(theta_hats.shape[1], np.nan)

    from scipy.stats import norm

    z90 = norm.ppf(0.95)
    z95 = norm.ppf(0.975)

    # Notebook-style empirical coverage using empirical SD.
    cover90_emp = np.mean(
        (theta_hats - z90 * emp_sd <= theta_true)
        & (theta_true <= theta_hats + z90 * emp_sd),
        axis=0,
    )
    cover95_emp = np.mean(
        (theta_hats - z95 * emp_sd <= theta_true)
        & (theta_true <= theta_hats + z95 * emp_sd),
        axis=0,
    )

    # Model-based coverage using per-replication estimated SE.
    if rows_se:
        se_mat = np.vstack(rows_se)
        cover90_est = np.nanmean(
            (theta_hats - z90 * se_mat <= theta_true)
            & (theta_true <= theta_hats + z90 * se_mat),
            axis=0,
        )
        cover95_est = np.nanmean(
            (theta_hats - z95 * se_mat <= theta_true)
            & (theta_true <= theta_hats + z95 * se_mat),
            axis=0,
        )
    else:
        cover90_est = np.full(theta_hats.shape[1], np.nan)
        cover95_est = np.full(theta_hats.shape[1], np.nan)

    out = pd.DataFrame(
        {
            "model": model_name,
            "theta_index": np.arange(theta_hats.shape[1]),
            "theta_true": theta_true,
            "mean_theta_hat": emp_mean,
            "bias": emp_mean - theta_true,
            "empirical_se_notebook": emp_sd,
            "mean_estimated_se_notebook": mean_estimated_se,
            "coverage_90_empirical_se": cover90_emp,
            "coverage_95_empirical_se": cover95_emp,
            "coverage_90_estimated_se": cover90_est,
            "coverage_95_estimated_se": cover95_est,
            "n_replications": theta_hats.shape[0],
        }
    )

    output_path = pd.io.common.stringify_path(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.to_csv(output_path, index=False)
