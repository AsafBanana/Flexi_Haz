from __future__ import annotations

import copy
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import KFold
from scipy.stats import norm

from .exceptions import InputValidationError

from .networks import ProjectionNet
from .utils import resolve_device


def _iter_batches(tensors: dict, batch_size: int, shuffle: bool):
    """Yield batches of a dict of same-length tensors via direct indexing.

    Avoids torch's default DataLoader, whose per-sample __getitem__/collate
    loop is orders of magnitude slower than a single vectorized index_select
    when the whole dataset already lives in memory as tensors.
    """
    n = next(iter(tensors.values())).shape[0]
    batch_size = max(1, int(batch_size))
    order = torch.randperm(n) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        yield {k: v[idx] for k, v in tensors.items()}


def _fit_projection_net(model, X_tr, t_tr, y_tr, X_va, t_va, y_va, *, device, lr, weight_decay, batch_size, max_epochs, patience):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    crit = nn.MSELoss()
    train_tensors = {
        "X": torch.as_tensor(X_tr, dtype=torch.float32),
        "t": torch.as_tensor(t_tr, dtype=torch.float32),
        "y": torch.as_tensor(y_tr.reshape(-1, 1), dtype=torch.float32),
    }
    Xv = torch.as_tensor(X_va, dtype=torch.float32, device=device)
    tv = torch.as_tensor(t_va, dtype=torch.float32, device=device)
    yv = torch.as_tensor(y_va.reshape(-1, 1), dtype=torch.float32, device=device)
    best_state = copy.deepcopy(model.state_dict())
    best = float("inf")
    wait = 0
    for _ in range(int(max_epochs)):
        model.train()
        for batch in _iter_batches(train_tensors, batch_size, shuffle=True):
            xb, tb, yb = batch["X"].to(device), batch["t"].to(device), batch["y"].to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(tb, xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = float(crit(model(tv, Xv), yv).detach().cpu())
        if val < best:
            best = val
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= int(patience):
                break
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def _predict_projection(model, X, t, *, device, batch_size):
    model.to(device)
    model.eval()
    X_t = torch.as_tensor(X, dtype=torch.float32)
    t_t = torch.as_tensor(t, dtype=torch.float32)
    batch_size = max(1, int(batch_size))
    n = X_t.shape[0]
    out = []
    for start in range(0, n, batch_size):
        xb = X_t[start:start + batch_size].to(device)
        tb = t_t[start:start + batch_size].to(device)
        out.append(model(tb, xb).cpu().numpy())
    return np.vstack(out).reshape(-1)


def estimate_theta_variance(
    T,
    delta,
    X,
    Z,
    *,
    hidden_layers=(64, 64),
    activation="relu",
    dropout=0.0,
    learning_rate=1e-3,
    weight_decay=1e-4,
    batch_size=256,
    max_epochs=300,
    patience=30,
    n_splits=5,
    ridge=1e-8,
    device="auto",
    seed=123
):
    """Estimate Var(theta_hat) by the event-weighted residual projection described in FLEXI-Haz."""
    device = resolve_device(device)
    T = np.asarray(T, dtype=np.float32).reshape(-1, 1)
    delta = np.asarray(delta, dtype=np.int64).reshape(-1)
    X = np.asarray(X, dtype=np.float32)
    Z = np.asarray(Z, dtype=np.float32)

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    if X.ndim != 2:
        raise InputValidationError("X must be a 2D array.")

    if Z.ndim != 2:
        raise InputValidationError("Z must be a 2D array.")

    if T.shape[0] != delta.shape[0] or T.shape[0] != X.shape[0] or T.shape[0] != Z.shape[0]:
        raise InputValidationError("T, delta, X, and Z must have the same number of rows.")

    n, p = Z.shape
    event_idx = np.where(delta == 1)[0]
    if event_idx.size <= max(2, p):
        I_hat = np.cov(Z.T) if n > 1 else np.eye(p)
        I_hat = np.atleast_2d(I_hat) + ridge * np.eye(p)
        return np.linalg.pinv(I_hat) / n, I_hat

    k = min(int(n_splits), event_idx.size)
    residuals_event = np.zeros((event_idx.size, p), dtype=np.float32)
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    X_ev, T_ev, Z_ev = X[event_idx], T[event_idx], Z[event_idx]

    for tr, va in kf.split(event_idx):
        for j in range(p):
            proj = ProjectionNet(X.shape[1], hidden_layers=hidden_layers, activation=activation, dropout=dropout)
            _fit_projection_net(
                proj,
                X_ev[tr],
                T_ev[tr],
                Z_ev[tr, j],
                X_ev[va],
                T_ev[va],
                Z_ev[va, j],
                device=device,
                lr=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
                max_epochs=max_epochs,
                patience=patience,
            )
            pred = _predict_projection(proj, X_ev[va], T_ev[va], device=device, batch_size=batch_size)
            residuals_event[va, j] = Z_ev[va, j] - pred

    R = np.zeros((n, p), dtype=np.float64)
    R[event_idx] = residuals_event.astype(np.float64)
    I_hat = (R.T @ R) / n
    I_hat = I_hat + float(ridge) * np.eye(p)
    asymptotic_cov = np.linalg.pinv(I_hat)
    var_theta_hat = asymptotic_cov / n
    return var_theta_hat, I_hat


def crossfitted_score_residuals(
    X_subj,
    T_subj,
    Z_subj,
    delta_subj,
    row_subject_pos,
    X_rows,
    t_rows,
    *,
    hidden_layers=(64, 64),
    activation="relu",
    dropout=0.0,
    learning_rate=1e-3,
    weight_decay=0.0,
    batch_size=256,
    max_epochs=300,
    patience=30,
    n_splits=5,
    inner_val_fraction=0.2,
    device="auto",
    seed=123,
):
    """Cross-fitted residuals Z - E[Z | X, t] at every expanded row, for the efficient score.

    Subjects (event and censored) are split into n_splits folds. For each fold, one
    projection net per Z column is trained on the EVENT subjects of the other folds
    (target Z at (X, T)), with an inner split of those subjects for early stopping, and
    predicts every expanded row of the subjects in the held-out fold. So no residual is
    ever predicted by a net that saw that subject.

    X_subj, T_subj, Z_subj, delta_subj: subject-level arrays (T already on the network time scale).
    row_subject_pos: for each expanded row, the position of its subject in the subject arrays.
    X_rows, t_rows: expanded-row covariates and times (network time scale).
    Returns R with shape (n_rows, p).
    """
    device = resolve_device(device)
    X_subj = np.asarray(X_subj, dtype=np.float32)
    T_subj = np.asarray(T_subj, dtype=np.float32).reshape(-1, 1)
    Z_subj = np.asarray(Z_subj, dtype=np.float32)
    if Z_subj.ndim == 1:
        Z_subj = Z_subj.reshape(-1, 1)
    delta_subj = np.asarray(delta_subj).reshape(-1)
    row_subject_pos = np.asarray(row_subject_pos).reshape(-1)
    X_rows = np.asarray(X_rows, dtype=np.float32)
    t_rows = np.asarray(t_rows, dtype=np.float32).reshape(-1, 1)
    Z_rows = Z_subj[row_subject_pos]

    n_subj, p = Z_subj.shape
    rng = np.random.default_rng(seed)
    subj_fold = rng.integers(0, int(n_splits), size=n_subj)
    row_fold = subj_fold[row_subject_pos]
    R = np.zeros((X_rows.shape[0], p), dtype=np.float64)

    for k in range(int(n_splits)):
        fit_subj = np.where((subj_fold != k) & (delta_subj == 1))[0]
        row_mask = row_fold == k
        if fit_subj.size < 10 or not np.any(row_mask):
            continue
        perm = rng.permutation(fit_subj)
        n_inner_val = max(1, int(inner_val_fraction * perm.size))
        inner_va, inner_tr = perm[:n_inner_val], perm[n_inner_val:]
        for j in range(p):
            proj = ProjectionNet(X_subj.shape[1], hidden_layers=hidden_layers, activation=activation, dropout=dropout)
            _fit_projection_net(
                proj,
                X_subj[inner_tr],
                T_subj[inner_tr],
                Z_subj[inner_tr, j],
                X_subj[inner_va],
                T_subj[inner_va],
                Z_subj[inner_va, j],
                device=device,
                lr=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
                max_epochs=max_epochs,
                patience=patience,
            )
            pred = _predict_projection(proj, X_rows[row_mask], t_rows[row_mask], device=device, batch_size=max(batch_size, 2048))
            R[row_mask, j] = Z_rows[row_mask, j] - pred
    return R


def theta_confidence_interval(theta_hat, var_theta_hat, alpha=0.05):
    theta_hat = np.asarray(theta_hat, dtype=np.float64).reshape(-1)
    var_theta_hat = np.asarray(var_theta_hat, dtype=np.float64)
    se = np.sqrt(np.maximum(np.diag(var_theta_hat), 0.0))
    z = norm.ppf(1 - alpha / 2)
    return np.column_stack([theta_hat - z * se, theta_hat + z * se])
