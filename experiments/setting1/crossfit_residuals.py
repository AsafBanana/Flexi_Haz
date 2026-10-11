"""K-fold cross-fitted Z-vs-(X,t) residuals for FlexiHaz, computed ONCE per
replication (independent of any trained main model -- it's purely a
regression of Z on (X,t) using train-subject event data) and reused for BOTH:

  1. the K=3 multi-init trial-selection score (needs a residual estimate at
     EVERY expanded train row -- event AND at-risk/censored intervals, since
     FlexiHaz's orthogonalized score sums over the whole expanded dataset,
     unlike Zhong's Cox partial-likelihood score which only needs event rows).
  2. the selected trial's theta variance, using the EXACT SAME formula as
     flexihaz.inference.estimate_theta_variance (R^T R / n at event rows,
     censored rows contribute zero) -- so the SE is computed from the same
     residuals as the score, not a separately re-cross-fitted set.

This replaces the previous (buggy) approach in multi_init_best_of_3_tuned.py,
which used a SINGLE 80/20 split per trial for the score (not cross-fitted --
predictions were made on the same rows used to train the projection net for
~80% of event subjects) and a SEPARATE, independently cross-fitted call to
estimate_theta_variance for the SE. Both used real cross-fitting for the SE;
only the selection score was uncross-fitted, which broke its orthogonality
property and made it a noisier signal than intended.

Cross-fitting design: EVERY train subject (event and censored) is assigned to
one of K folds. Only EVENT subjects ever contribute TRAINING data to a fold's
projection net (a censored subject's censoring time isn't a valid Z~T
regression target). For prediction:
  - an event subject's rows are predicted by the model that held THEM out
    (genuine cross-fitting, zero leakage).
  - a censored subject's rows are predicted by the model of whichever fold
    they were assigned to -- valid because that model never trained on ANY
    censored subject's data in the first place (it only ever saw the OTHER
    folds' event subjects), so there is no leakage for censored subjects
    regardless of which fold's model predicts them.
"""

from __future__ import annotations

import numpy as np

from flexihaz.networks import ProjectionNet
from flexihaz.inference import _fit_projection_net, _predict_projection
from flexihaz.utils import set_seed

PROJECTION_BATCH = 2048  # prediction batch only (memory), does not affect the fit
PROJECTION_TRAIN_BATCH = 2048  # minibatch size for fitting the projection nets
# if set, overrides PROJECTION_TRAIN_BATCH: batch = ceil(fold inner-train size / this), i.e. a fixed
# number of minibatches per epoch regardless of dataset size
PROJECTION_BATCHES_PER_EPOCH = None
PROJECTION_MAX_EPOCHS = 300
PROJECTION_PATIENCE = 30
INNER_VAL_FRAC = 0.20  # internal split within each fold's training subjects, early-stopping only


def compute_crossfitted_residuals(model, expanded, train_idx, T, delta, X_scaled, Z_scaled,
                                   proj_hidden_layers, projection_lr, seed, device, n_splits=5, return_projs=False):
    """Returns:
      resid_all: (n_expanded_train_rows,) cross-fitted (Z_i - ghat*(X_i,t_i))
        at every expanded row belonging to a train subject.
      var_theta_hat_scaled: scalar Var(theta_hat) on the STANDARDIZED-Z scale,
        computed via the SAME R^T R / n formula as estimate_theta_variance,
        using residuals restricted to event rows (censored rows contribute 0).
    """
    train_subject_ids = np.unique(expanded["subject_id"][train_idx])
    n_train_subj = len(train_subject_ids)
    delta_tr = delta[train_subject_ids]
    event_idx_local = np.where(delta_tr == 1)[0]

    X_ev = X_scaled[train_subject_ids][event_idx_local].astype(np.float32)
    T_ev = model._prepare_time_for_network(T[train_subject_ids][event_idx_local]).astype(np.float32)
    Z_ev = Z_scaled[train_subject_ids][event_idx_local].reshape(-1).astype(np.float32)
    n_ev = len(X_ev)

    rng = np.random.default_rng(seed)
    subj_fold = rng.integers(0, n_splits, size=n_train_subj)
    event_fold = subj_fold[event_idx_local]

    projs = [None] * n_splits
    ev_resid = np.zeros(n_ev, dtype=np.float64)
    for k in range(n_splits):
        tr_mask = event_fold != k
        va_mask = event_fold == k
        if va_mask.sum() == 0 or tr_mask.sum() < 10:
            continue
        tr_idx_ev = np.where(tr_mask)[0]
        inner_rng = np.random.default_rng(seed + 1000 + k)
        perm = inner_rng.permutation(len(tr_idx_ev))
        n_inner_val = max(1, int(INNER_VAL_FRAC * len(tr_idx_ev)))
        inner_va = tr_idx_ev[perm[:n_inner_val]]
        inner_tr = tr_idx_ev[perm[n_inner_val:]]

        fit_bs = (PROJECTION_TRAIN_BATCH if PROJECTION_BATCHES_PER_EPOCH is None
                  else max(1, int(np.ceil(len(inner_tr) / PROJECTION_BATCHES_PER_EPOCH))))
        set_seed(seed + k)
        proj = ProjectionNet(X_ev.shape[1], hidden_layers=proj_hidden_layers, activation="relu", dropout=0.0)
        proj = _fit_projection_net(
            proj, X_ev[inner_tr], T_ev[inner_tr], Z_ev[inner_tr],
            X_ev[inner_va], T_ev[inner_va], Z_ev[inner_va],
            device=device, lr=projection_lr, weight_decay=0.0, batch_size=fit_bs,
            max_epochs=PROJECTION_MAX_EPOCHS, patience=PROJECTION_PATIENCE,
        )
        pred_va = _predict_projection(proj, X_ev[va_mask], T_ev[va_mask], device=device,
                                       batch_size=PROJECTION_BATCH)
        ev_resid[va_mask] = Z_ev[va_mask] - pred_va.reshape(-1)
        projs[k] = proj

    # ---- SE: same R^T R / n formula as estimate_theta_variance, event rows only ----
    R = np.zeros(n_train_subj, dtype=np.float64)
    R[event_idx_local] = ev_resid
    I_hat = float(np.sum(R ** 2)) / n_train_subj
    I_hat_ridge = I_hat + 1e-8
    var_theta_hat_scaled = (1.0 / I_hat_ridge) / n_train_subj

    # ---- score: cross-fitted residual at EVERY expanded train row ----
    X_all = expanded["X"][train_idx].astype(np.float32)
    t_raw_all = expanded["t"][train_idx].reshape(-1).astype(np.float64)
    t_scaled_all = model._prepare_time_for_network(t_raw_all).astype(np.float32)
    Z_all = expanded["Z"][train_idx].reshape(-1).astype(np.float64)
    row_subject_id = expanded["subject_id"][train_idx].reshape(-1)

    max_subject_id = int(row_subject_id.max()) + 1
    fold_by_subject_id = np.full(max_subject_id, -1, dtype=np.int64)
    fold_by_subject_id[train_subject_ids] = subj_fold
    row_fold = fold_by_subject_id[row_subject_id]

    resid_all = np.zeros(len(X_all), dtype=np.float64)
    for k in range(n_splits):
        if projs[k] is None:
            continue
        row_mask = row_fold == k
        if not np.any(row_mask):
            continue
        pred = _predict_projection(projs[k], X_all[row_mask], t_scaled_all[row_mask],
                                    device=device, batch_size=PROJECTION_BATCH)
        resid_all[row_mask] = Z_all[row_mask] - pred.reshape(-1)

    if return_projs:
        return resid_all, var_theta_hat_scaled, projs
    return resid_all, var_theta_hat_scaled


def residuals_on_heldout_rows(projs, model, expanded, row_idx):
    """Residual Z - ghat*(X,t) on rows of subjects that NO projection net trained on
    (e.g. the main net's validation subjects): average of all fold models' predictions."""
    X = expanded["X"][row_idx].astype(np.float32)
    t_scaled = model._prepare_time_for_network(expanded["t"][row_idx].reshape(-1).astype(np.float64)).astype(np.float32)
    Z = expanded["Z"][row_idx].reshape(-1).astype(np.float64)
    preds = [_predict_projection(p, X, t_scaled, device=next(p.parameters()).device, batch_size=PROJECTION_BATCH).reshape(-1)
             for p in projs if p is not None]
    return Z - np.mean(preds, axis=0)
