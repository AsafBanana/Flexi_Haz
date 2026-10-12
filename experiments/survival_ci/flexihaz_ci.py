"""Pointwise confidence intervals for FLEXI-Haz survival curves when X has finite support.

The main FLEXI-Haz package (PyTorch) estimates the linear coefficient theta of the partially
linear hazard h(t | X, Z) = exp{theta'Z + g(t, X)} and its covariance. This module adds
inference on the *curve*: for a covariate profile (x0, z0) with x0 one of finitely many
observed values of X, it estimates the cumulative hazard H(t | x0, z0) and returns a
pointwise Gaussian confidence band for it, hence for the survival curve S = exp(-H).

Method (one-step, cross-fitted estimator)
-----------------------------------------
* The data are expanded to subject x interval rows on the grid of distinct observed times
  (`expand_to_intervals`); the piecewise-exponential likelihood is used throughout.
* K outer folds (stratified by the X profile). In fold k the network g(t, X) and theta are
  trained on the other folds (with an inner 67/33 subject split for early stopping and a Cox
  warm start of theta); the held-out fold supplies the influence-function terms.
* Plug-in: H_hat(t | x0, z0) = int_0^t exp{theta'z0 + g(s, x0)} ds (trapezoid on the grid).
* One-step correction: because X has finite support, the nuisance E[Z | X = x0, t] and the
  stratum hazard A(t | x0) are estimated by stratum averages of the fitted hazard on the
  training folds; the correction adds the mean of the efficient influence function
  (martingale residuals of the held-out subjects in stratum x0, plus the theta part through
  the inverse information I^-1) and the variance is the mean of its square, both accumulated
  over folds.
* The band is H1 +- z_{1-alpha/2} * se, mapped to S by exp(-.).

The grid is restricted to [t_l, t_u], where t_l is the first event time (over the training
folds) and t_u the smallest fold-wise maximum follow-up, so every curve is reported where all
folds have data. Curves are relative to t_l (H(t_l) = 0; no events occur before it).

Implementation: TensorFlow / Keras (this module is self-contained and does not depend on the
`flexihaz` package). Public API: `fit_survival_ci` and the returned `CrossFitSurvivalCI`.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from lifelines import CoxPHFitter
from scipy.stats import norm
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Add, Concatenate, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

DEFAULT_NN_CONFIG = {
    "hidden_layers_nodes": 64,
    "n_hidden_layers": 4,
    "learning_rate": 1e-3,
    "batch_size": 10000,   # expanded rows per minibatch
    "patience": 50,
    "epochs": 2000,
}


def set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    tf.keras.utils.set_random_seed(int(seed))


# ----------------------------------------------------------------------------- data

def expand_to_intervals(X, Z, T, delta):
    """Expand subject-level data to one row per subject and grid interval (t_{j-1}, t_j].

    The grid is the sorted set of distinct observed times. Subject i contributes the intervals
    up to its observed time T_i; the event indicator E is delta_i on the last interval and 0
    elsewhere. Returns (rows, data): `rows` is a DataFrame with columns X0.., Z0.., E, st, et,
    id; `data` holds the subject-level arrays (covs_X, covs_Z, durations, events).
    """
    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64).reshape(-1)
    delta = np.asarray(delta).astype(np.int32).reshape(-1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    n = T.shape[0]
    if X.shape[0] != n or Z.shape[0] != n or delta.shape[0] != n:
        raise ValueError("X, Z, T and delta must have the same number of rows.")
    if np.any(T <= 0):
        raise ValueError("Observed times must be positive.")

    grid = np.unique(T)                               # sorted distinct observed times
    k = np.searchsorted(grid, T)                      # index of T_i in the grid = number of intervals
    subj = np.repeat(np.arange(n), k)
    j = np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k) + 1   # interval index 1..k_i
    st = grid[j - 1]
    et = grid[j]
    E = delta[subj] * (j == k[subj])

    rows = pd.DataFrame(
        np.concatenate([X[subj], Z[subj], E.reshape(-1, 1), st.reshape(-1, 1), et.reshape(-1, 1),
                        subj.reshape(-1, 1)], axis=1),
        columns=[f"X{i}" for i in range(X.shape[1])] + [f"Z{i}" for i in range(Z.shape[1])] + ["E", "st", "et", "id"],
    )
    data = dict(covs_X=X, covs_Z=Z, durations=T, events=delta)
    return rows, data


def split_subject_data(data, train_ids, test_ids):
    def sub(ids):
        ids = np.asarray(ids).astype(int)
        return {"ids": ids, "covs_X": data["covs_X"][ids], "covs_Z": data["covs_Z"][ids],
                "durations": data["durations"][ids], "events": data["events"][ids]}
    return sub(train_ids), sub(test_ids)


def build_strata_from_X(X_subj):
    """Finite-support strata: one stratum per distinct row of X."""
    keys = [tuple(row.tolist()) for row in np.asarray(X_subj, dtype=np.float64)]
    uniq = sorted(set(keys))
    key2j = {k: j for j, k in enumerate(uniq)}
    idx = np.array([key2j[k] for k in keys], dtype=int)
    return uniq, key2j, idx


def cumtrapz(y, x):
    out = np.zeros_like(y, dtype=np.float64)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


def cumtrapz_vec(Y, x):
    Y = np.asarray(Y, dtype=np.float64)
    out = np.zeros_like(Y)
    out[1:, :] = np.cumsum(0.5 * (Y[1:, :] + Y[:-1, :]) * np.diff(x).reshape(-1, 1), axis=0)
    return out


# ----------------------------------------------------------------------------- model

def piecewise_exponential_loss(y_true, y_pred, clip_pred=80.0, eps=1e-30, log_switch=50.0, dt_tiny=1e-8):
    """Negative piecewise-exponential log-likelihood per expanded row.

    y_true columns: [et, st, delta]; y_pred[:, 0] is chi = g(et, X) + theta'Z (log hazard on
    the interval, time measured in units of `time_scale`).
    """
    et, st, delta = y_true[:, 0:1], y_true[:, 1:2], y_true[:, 2:3]
    chi = tf.clip_by_value(y_pred[:, 0:1], -clip_pred, clip_pred)
    dt = tf.maximum(et - st, 0.0)
    use_log = tf.logical_or(chi > log_switch, dt < dt_tiny)
    integral_term = tf.where(use_log, tf.exp(tf.math.log(tf.maximum(dt, eps)) + chi), dt * tf.exp(chi))
    return tf.reduce_mean(integral_term - delta * chi)


def build_model(input_dim_X, input_dim_Z, nn_config, beta_init):
    """chi = g(t, X) + theta'Z with a ReLU MLP g on (t, X); outputs [chi, g] per row."""
    X_input = Input(shape=(input_dim_X,), name="X_input")
    Z_input = Input(shape=(input_dim_Z,), name="Z_input")
    t_input = Input(shape=(1,), name="et_input")
    x = Concatenate(name="concat_xt")([t_input, X_input])
    for _ in range(int(nn_config["n_hidden_layers"])):
        x = Dense(int(nn_config["hidden_layers_nodes"]), activation="relu")(x)
    g_out = Dense(1, activation=None, name="g_out")(x)
    beta_out = Dense(1, use_bias=False, activation=None, name="beta_layer",
                     kernel_initializer=Constant(np.asarray(beta_init, dtype=np.float32).reshape(-1, 1)))(Z_input)
    chi = Add(name="chi")([g_out, beta_out])
    model = Model(inputs=[X_input, Z_input, t_input], outputs=Concatenate(axis=1, name="chi_g")([chi, g_out]))
    model.compile(optimizer=Adam(learning_rate=float(nn_config["learning_rate"])), loss=piecewise_exponential_loss,
                  jit_compile=False)
    return model


def predict_chi(model, X, Z, t_scaled, batch=65536):
    X = np.asarray(X, dtype=np.float32)
    Z = np.asarray(Z, dtype=np.float32)
    t_scaled = np.asarray(t_scaled, dtype=np.float32).reshape(-1, 1)
    outs = []
    for s in range(0, X.shape[0], batch):
        e = min(X.shape[0], s + batch)
        out = model([tf.convert_to_tensor(X[s:e]), tf.convert_to_tensor(Z[s:e]), tf.convert_to_tensor(t_scaled[s:e])],
                    training=False)
        outs.append(out[:, 0].numpy())
    return np.concatenate(outs, axis=0).astype(np.float64)


def compute_cox_beta_init(data):
    """Cox (X + Z) warm start of theta (lifelines)."""
    X_dim, Z_dim = data["covs_X"].shape[1], data["covs_Z"].shape[1]
    X_cols = [f"X{i}" for i in range(X_dim)]
    Z_cols = [f"Z{i}" for i in range(Z_dim)]
    df = pd.DataFrame(np.concatenate([data["covs_X"], data["covs_Z"], data["durations"].reshape(-1, 1),
                                      data["events"].reshape(-1, 1)], axis=1),
                      columns=X_cols + Z_cols + ["durations", "events"])
    cph = CoxPHFitter(penalizer=1e-6)
    cph.fit(df, duration_col="durations", event_col="events", show_progress=False)
    return cph.params_[Z_cols].values.reshape(-1, 1)


def fit_main_model(rows, train_ids, val_ids, nn_config, beta_init, time_scale, t_l_bound, verbose=0):
    """Train g and theta on the subjects train_ids with early stopping on val_ids."""
    X_cols = [c for c in rows.columns if c.startswith("X")]
    Z_cols = [c for c in rows.columns if c.startswith("Z")]
    tr = rows[rows["id"].isin(np.asarray(train_ids).astype(int)) & (rows["st"] >= t_l_bound)]
    va = rows[rows["id"].isin(np.asarray(val_ids).astype(int)) & (rows["st"] >= t_l_bound)]

    def arrays(df):
        inp = tuple(np.asarray(a, dtype=np.float32) for a in
                    (df[X_cols].values, df[Z_cols].values, df["et"].values.reshape(-1, 1) / time_scale))
        y = np.concatenate([df["et"].values.reshape(-1, 1) / time_scale, df["st"].values.reshape(-1, 1) / time_scale,
                            df["E"].values.reshape(-1, 1)], axis=1).astype(np.float32)
        return inp, y

    train_inp, y_train = arrays(tr)
    val_inp, y_val = arrays(va)
    model = build_model(len(X_cols), len(Z_cols), nn_config, beta_init)
    es = EarlyStopping(monitor="val_loss", patience=int(nn_config["patience"]), restore_best_weights=True)
    hist = model.fit(train_inp, y_train, validation_data=(val_inp, y_val), batch_size=int(nn_config["batch_size"]),
                     epochs=int(nn_config["epochs"]), callbacks=[es], verbose=verbose)
    return model, hist.history


# ----------------------------------------------------------------------------- one-step pieces

def prepare_fold_for_one_step(rows_train_outer, data_train_outer, rows_eval, data_eval, model, t_grid, key2j,
                              time_scale, ridge=1e-8, eps=1e-18):
    """Stratum nuisances from the training folds and influence-function pieces of the held-out fold."""
    X_tr_subj = data_train_outer["covs_X"].astype(np.float64)
    Z_tr_subj = data_train_outer["covs_Z"].astype(np.float64)
    n_tr, p = X_tr_subj.shape[0], Z_tr_subj.shape[1]
    X_ev_subj = data_eval["covs_X"].astype(np.float64)
    n_ev = X_ev_subj.shape[0]

    subj_str_tr = np.array([key2j[tuple(x.tolist())] for x in X_tr_subj], dtype=int)
    subj_str_ev = np.array([key2j[tuple(x.tolist())] for x in X_ev_subj], dtype=int)
    J = len(key2j)
    pi_hat = np.bincount(subj_str_tr, minlength=J) / max(1, n_tr)       # stratum probabilities

    X_cols = [c for c in rows_train_outer.columns if c.startswith("X")]
    Z_cols = [c for c in rows_train_outer.columns if c.startswith("Z")]
    m = len(t_grid)

    # ---- training rows: stratum hazard A_hat(t | x) and E[Z | X = x, t] (g_star)
    tr_id2pos = {int(s): i for i, s in enumerate(np.asarray(data_train_outer["ids"]).astype(int))}
    sid_tr = np.array([tr_id2pos[int(s)] for s in rows_train_outer["id"].values.astype(int)], dtype=int)
    st_tr = rows_train_outer["st"].values.astype(np.float64)
    et_tr = rows_train_outer["et"].values.astype(np.float64)
    dt_tr = et_tr - st_tr
    Xr_tr = rows_train_outer[X_cols].values.astype(np.float32)
    Zr_tr = rows_train_outer[Z_cols].values.astype(np.float32)
    tidx_tr = np.clip(np.searchsorted(t_grid, et_tr), 0, m - 1)
    lam_tr = np.exp(predict_chi(model, Xr_tr, Zr_tr, et_tr / time_scale)) / time_scale
    row_str_tr = subj_str_tr[sid_tr]

    num_A = np.zeros((J, m))
    np.add.at(num_A, (row_str_tr, tidx_tr), lam_tr)
    A_hat = (num_A / max(1, n_tr)) / (pi_hat[:, None] + eps)

    num_g = np.zeros((J, m, p))
    den_g = np.zeros((J, m))
    for jj in range(p):
        np.add.at(num_g[:, :, jj], (row_str_tr, tidx_tr), lam_tr * Zr_tr[:, jj].astype(np.float64))
    np.add.at(den_g, (row_str_tr, tidx_tr), lam_tr)
    g_star = num_g / (den_g[:, :, None] + eps)

    diff_tr = Zr_tr.astype(np.float64) - g_star[row_str_tr, tidx_tr, :]
    I_hat = ((diff_tr * (dt_tr * lam_tr)[:, None]).T @ diff_tr) / max(1, n_tr)   # information for theta
    I_inv = np.linalg.inv(I_hat + ridge * np.eye(p))

    # ---- held-out rows: martingale increments and the theta influence function ell
    ev_id2pos = {int(s): i for i, s in enumerate(np.asarray(data_eval["ids"]).astype(int))}
    sid_ev = np.array([ev_id2pos[int(s)] for s in rows_eval["id"].values.astype(int)], dtype=int)
    st_ev = rows_eval["st"].values.astype(np.float64)
    et_ev = rows_eval["et"].values.astype(np.float64)
    E_ev = rows_eval["E"].values.astype(np.float64)
    Xr_ev = rows_eval[X_cols].values.astype(np.float32)
    Zr_ev = rows_eval[Z_cols].values.astype(np.float32)
    tidx_ev = np.clip(np.searchsorted(t_grid, et_ev), 0, m - 1)
    lam_ev = np.exp(predict_chi(model, Xr_ev, Zr_ev, et_ev / time_scale)) / time_scale
    dM_ev = E_ev - (et_ev - st_ev) * lam_ev
    row_str_ev = subj_str_ev[sid_ev]
    diff_ev = Zr_ev.astype(np.float64) - g_star[row_str_ev, tidx_ev, :]

    ell = np.zeros((n_ev, p))
    for jj in range(p):
        np.add.at(ell[:, jj], sid_ev, diff_ev[:, jj] * dM_ev)

    stratum_rows = {}
    for j in range(J):
        subj_idx = np.where(subj_str_ev == j)[0]
        if subj_idx.size == 0:
            continue
        pos_to_local = -np.ones(n_ev, dtype=int)
        pos_to_local[subj_idx] = np.arange(subj_idx.size)
        mask = row_str_ev == j
        stratum_rows[j] = dict(subj_idx=subj_idx, sid_local=pos_to_local[sid_ev[mask]].astype(int),
                               tidx=tidx_ev[mask].astype(int), dM=dM_ev[mask])

    return dict(J=J, m=m, p=p, pi_hat=pi_hat, A_hat=A_hat, g_star=g_star, I_inv=I_inv, ell=ell,
                ell_sum=ell.sum(axis=0), ell_outer=ell.T @ ell, stratum_rows=stratum_rows, n_eval=n_ev)


def _time_grid(rows, t_l_bound, t_eval):
    sub = rows[(rows["st"] >= t_l_bound) & (rows["et"] <= t_eval)]
    return np.sort(np.unique(sub["et"].values.astype(np.float64)))


class CrossFitSurvivalCI:
    """Result of `fit_survival_ci`: cross-fitted models and the pieces of the one-step estimator."""

    def __init__(self, folds, t_grid, key2j, uniq_keys, t_l_bound, t_eval, time_scale, n_total, beta_init, timings):
        self.folds = folds
        self.t_grid = np.asarray(t_grid, dtype=np.float64)
        self.key2j = key2j
        self.strata = [np.asarray(k, dtype=np.float64) for k in uniq_keys]
        self.t_l_bound = float(t_l_bound)
        self.t_eval = float(t_eval)
        self.time_scale = float(time_scale)
        self.n_total = int(n_total)
        self.beta_init = np.asarray(beta_init, dtype=np.float64).reshape(-1)
        self.timings = timings

    @property
    def theta_by_fold(self):
        """theta of each fold's network (original Z scale; Z is not standardized here)."""
        return np.vstack([f["model"].get_layer("beta_layer").get_weights()[0].reshape(-1) for f in self.folds])

    def cumulative_hazard(self, x0, z0, alpha=0.05):
        """Plug-in and one-step estimates of H(t | x0, z0) on the grid with a pointwise (1-alpha) band.

        x0 must equal one of the observed X profiles (finite support); z0 is any value of Z. The
        one-step estimate and the band are NaN after `t_valid_max`, the last grid time at which
        profile x0 still has subjects at risk in every fold (the plug-in H_hat is always given).
        """
        x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
        z0 = np.asarray(z0, dtype=np.float64).reshape(-1)
        key = tuple(x0.tolist())
        if key not in self.key2j:
            raise ValueError(f"x0={key} is not one of the {len(self.key2j)} observed X profiles; "
                             "the curve CI needs x0 on the (finite) support of X.")
        j0 = self.key2j[key]
        t_grid, m, n = self.t_grid, len(self.t_grid), self.n_total

        H_hat_cf = np.zeros(m)
        sum_phi = np.zeros(m)
        sum_phi2 = np.zeros(m)
        cutoff = m - 1          # last grid index at which the profile still has subjects at risk in every fold
        for fold in self.folds:
            model, n_ev = fold["model"], fold["n_eval"]
            X0 = np.repeat(x0.reshape(1, -1), m, axis=0)
            Z0 = np.repeat(z0.reshape(1, -1), m, axis=0)
            lam0 = np.exp(predict_chi(model, X0, Z0, t_grid / self.time_scale, batch=4096)) / self.time_scale
            H_hat_cf += (n_ev / n) * cumtrapz(lam0, t_grid)

            # the one-step weight 1 / (pi(x0) A(t | x0)) is only defined while stratum x0 has
            # subjects at risk (A_hat > 0); beyond that point the correction and the band are not reported
            A0 = fold["A_hat"][j0, :]
            valid = A0 > 0
            cutoff = min(cutoff, int(np.max(np.where(valid)[0])) if valid.any() else -1)
            w_time = np.where(valid, lam0 / (fold["pi_hat"][j0] * np.where(valid, A0, 1.0)), 0.0)
            integrand = lam0.reshape(-1, 1) * (z0.reshape(1, -1) - fold["g_star"][j0, :, :])
            U = fold["I_inv"] @ cumtrapz_vec(integrand, t_grid).T                 # (p, m)
            sum_phi_theta = (fold["ell_sum"].reshape(1, -1) @ U).reshape(-1)
            sum_phi_theta2 = np.einsum("km,kl,lm->m", U, fold["ell_outer"], U)

            if j0 in fold["stratum_rows"]:
                info = fold["stratum_rows"][j0]
                n0 = info["subj_idx"].size
                diff = np.zeros((n0, m + 1))
                if info["dM"].size > 0:
                    np.add.at(diff, (info["sid_local"], info["tidx"]), w_time[info["tidx"]] * info["dM"])
                phi_g = np.cumsum(diff[:, :m], axis=1)
                ell_sub = fold["ell"][info["subj_idx"], :]
                cross = np.sum(phi_g * (ell_sub @ U), axis=0)
                sum_phi += phi_g.sum(axis=0) + sum_phi_theta
                sum_phi2 += (phi_g ** 2).sum(axis=0) + sum_phi_theta2 + 2.0 * cross
            else:
                sum_phi += sum_phi_theta
                sum_phi2 += sum_phi_theta2

        H1 = H_hat_cf + sum_phi / n
        se = np.sqrt(sum_phi2) / n
        if cutoff < m - 1:
            H1[cutoff + 1:] = np.nan
            se[cutoff + 1:] = np.nan
        z = norm.ppf(1 - alpha / 2)
        return dict(t_grid=t_grid, H_hat=H_hat_cf, H1=H1, se=se, lo=H1 - z * se, hi=H1 + z * se,
                    t_l_bound=self.t_l_bound, t_eval=self.t_eval,
                    t_valid_max=float(t_grid[cutoff]) if cutoff >= 0 else np.nan)

    def survival(self, x0, z0, alpha=0.05):
        """DataFrame of the survival curve S(t | x0, z0): plug-in, one-step, and the (1-alpha) band
        (survival values clipped to [0, 1]; NaN after the profile's last at-risk time)."""
        out = self.cumulative_hazard(x0, z0, alpha=alpha)
        surv = lambda H: np.clip(np.exp(-H), 0.0, 1.0)
        return pd.DataFrame(dict(t=out["t_grid"], H_plugin=out["H_hat"], H_onestep=out["H1"], se_H=out["se"],
                                 S_plugin=surv(out["H_hat"]), S_onestep=surv(out["H1"]),
                                 S_lo=surv(out["hi"]), S_hi=surv(out["lo"])))


def fit_survival_ci(X, Z, T, delta, nn_config=None, K=5, time_scale=None, inner_val_frac=0.33, random_state=0,
                    verbose=True):
    """Fit the cross-fitted one-step estimator of the FLEXI-Haz survival curve with pointwise CIs.

    X: (n, d) covariates with FINITE support (every row a profile; the CI is available at observed
       profiles), Z: (n, p) linear covariates, T: (n,) observed times, delta: (n,) event indicators.
    nn_config: keys hidden_layers_nodes, n_hidden_layers, learning_rate, batch_size (expanded
       rows), patience, epochs (defaults in DEFAULT_NN_CONFIG). time_scale: time is divided by
       this before entering the network (default: max(T)). Returns a CrossFitSurvivalCI.
    """
    cfg = copy.deepcopy(DEFAULT_NN_CONFIG)
    if nn_config:
        cfg.update(nn_config)
    set_seed(random_state)
    t0 = time.perf_counter()
    rows, data = expand_to_intervals(X, Z, T, delta)
    if time_scale is None:
        time_scale = float(np.max(data["durations"]))
    ids = np.arange(data["covs_X"].shape[0])
    uniq_keys, key2j, strata = build_strata_from_X(data["covs_X"])

    try:
        splits = list(StratifiedKFold(n_splits=K, shuffle=True, random_state=random_state).split(ids, strata))
    except Exception as e:      # strata too small for stratification
        if verbose:
            print("StratifiedKFold failed, using KFold:", repr(e))
        splits = list(KFold(n_splits=K, shuffle=True, random_state=random_state).split(ids))

    # common grid: from the first event time to the smallest fold-wise maximum follow-up
    t_l_bounds, t_evals = [], []
    for tr_idx, _ in splits:
        rows_tr = rows[rows["id"].isin(ids[tr_idx])]
        t_l_bounds.append(float(rows_tr.loc[rows_tr["E"] == 1, "et"].min()))
        t_evals.append(float(data["durations"][ids[tr_idx]].max()))
    t_l_bound, t_eval = float(np.max(t_l_bounds)), float(np.min(t_evals))
    t_grid = _time_grid(rows, t_l_bound, t_eval)
    beta_init = compute_cox_beta_init(data)
    if verbose:
        print(f"n={len(ids)} subjects, {len(rows)} expanded rows, {len(uniq_keys)} X profiles, "
              f"grid {len(t_grid)} points on [{t_l_bound:.4g}, {t_eval:.4g}], Cox warm start theta={beta_init.ravel()}")

    folds, timings = [], {"expand_sec": time.perf_counter() - t0}
    for k, (tr_idx, te_idx) in enumerate(splits, start=1):
        t1 = time.perf_counter()
        outer_train_ids, outer_eval_ids = ids[tr_idx], ids[te_idx]
        try:
            tr_in, va_in = train_test_split(outer_train_ids, test_size=inner_val_frac, random_state=random_state + k,
                                            shuffle=True, stratify=strata[tr_idx])
        except Exception:
            tr_in, va_in = train_test_split(outer_train_ids, test_size=inner_val_frac, random_state=random_state + k,
                                            shuffle=True)
        model, hist = fit_main_model(rows, tr_in, va_in, cfg, beta_init, time_scale, t_l_bound)
        data_train_outer, data_eval = split_subject_data(data, outer_train_ids, outer_eval_ids)
        in_window = (rows["st"] >= t_l_bound) & (rows["et"] <= t_eval)
        rows_train_outer = rows[rows["id"].isin(outer_train_ids) & in_window]
        rows_eval = rows[rows["id"].isin(outer_eval_ids) & in_window]
        fold = prepare_fold_for_one_step(rows_train_outer, data_train_outer, rows_eval, data_eval, model, t_grid,
                                         key2j, time_scale)
        fold["model"] = model
        fold["outer_eval_ids"] = outer_eval_ids
        fold["epochs_run"] = len(hist["loss"])
        folds.append(fold)
        timings[f"fold_{k}_sec"] = time.perf_counter() - t1
        if verbose:
            theta_k = model.get_layer("beta_layer").get_weights()[0].reshape(-1)
            print(f"fold {k}/{K}: train={len(tr_in)} val={len(va_in)} eval={len(outer_eval_ids)} "
                  f"epochs={fold['epochs_run']} theta={np.round(theta_k, 4)} ({timings[f'fold_{k}_sec']:.0f}s)")

    return CrossFitSurvivalCI(folds, t_grid, key2j, uniq_keys, t_l_bound, t_eval, time_scale, len(ids), beta_init, timings)
