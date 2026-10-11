from __future__ import annotations

import copy
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from scipy.optimize import minimize
import warnings

from .data import build_time_grid, make_expanded_data
from .exceptions import InputValidationError, NotFittedError
from .inference import crossfitted_score_residuals, estimate_theta_variance, theta_confidence_interval
from .loss import cox_partial_likelihood_loss, negative_log_likelihood
from .networks import FlexiHazardNet, ZhongPHNet
from .preprocessing import Standardizer, validate_survival_inputs, truncate_at_tau
from .utils import resolve_device, set_seed


class FlexiHazardModel:
    """PyTorch implementation of FLEXI-Haz with optional linear component.

    If Z is None, fits h(t | X) = exp{g(t, X)}.
    If Z is provided, fits h(t | X, Z) = exp{theta^T Z + g(t, X)}.

    n_restarts > 1 trains that many networks (seeds seed, seed+1, ...) on the same
    train/validation split and keeps one, chosen by restart_selection:
      "val_loss"        -- lowest validation loss of the final weights;
      "efficient_score" -- efficient score for theta closest to zero (L2 norm over Z
                           columns), using residuals Z - E[Z | X, t] cross-fitted over
                           the internal training subjects (requires Z).
    Per-restart diagnostics are stored in restart_results_ and the chosen index in
    selected_restart_.

    theta_lr_multiplier > 1 trains the linear (theta) branch with learning rate
    learning_rate * theta_lr_multiplier (its own Adam parameter group), in every LR
    phase; the network keeps learning_rate.

    model_type="zhong" fits the proportional-hazards model of Zhong, Mueller and Wang
    (2022) instead: h(t | X, Z) = h0(t) exp{theta^T Z + g(X)}, g without time input,
    trained full-batch on the Breslow partial likelihood (no interval expansion;
    batch_size and grid_size are ignored). theta is refined by profiling the partial
    likelihood with g fixed, and the projection networks for the covariance and the
    efficient score are additive, g_X(X) + g_T(t), as in that paper. Cumulative hazard
    and survival use the Breslow baseline; predict_hazard is unavailable. Unless given,
    max_epochs / patience default to 5000 / 35 for this model (500 / 30 otherwise).
    """

    def __init__(
        self,
        hidden_layers=(64, 64),
        activation="relu",
        dropout=0.0,
        learning_rate=1e-3,
        three_phase_lr=True,
        batch_size=16 ** 4,
        max_epochs=None,
        patience=None,
        weight_decay=0.0,
        tau=None,
        grid_size=None,
        include_observed_times=True,
        validation_fraction=0.33,
        standardize=True,
        refine_theta=True,
        compute_theta_variance=True,
        projection_max_epochs=300,
        projection_patience=30,
        projection_n_splits=5,
        projection_batch_size=None,
        theta_init="cox",
        cox_penalizer=1e-6,
        device="auto",
        seed=None,
        verbose=0,
        num_workers=0,
        pin_memory=None,
        persistent_workers=False,
        non_blocking=None,
        epoch_callback=None,
        scale_time=True,
        n_restarts=1,
        restart_selection="val_loss",
        theta_lr_multiplier=1.0,
        model_type="flexihaz",
        theta_init_subjects="all",
    ):
        self.hidden_layers = tuple(hidden_layers)
        self.activation = activation
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.three_phase_lr = bool(three_phase_lr)
        self.batch_size = int(batch_size)
        # defaults depend on model_type (resolved below): 500 / 30 for "flexihaz",
        # 5000 / 35 for "zhong" (one full-batch step per epoch, as in Zhong et al. 2022)
        self.model_type = str(model_type).lower()
        if self.model_type not in {"flexihaz", "zhong"}:
            raise InputValidationError("model_type must be 'flexihaz' or 'zhong'.")
        default_max_epochs, default_patience = (5000, 35) if self.model_type == "zhong" else (500, 30)
        self.max_epochs = default_max_epochs if max_epochs is None else int(max_epochs)
        self.patience = default_patience if patience is None else int(patience)
        self.weight_decay = 0.0 # If you want to enable weight decay in future, define as float(weight_decay)
        self.tau = tau
        self.grid_size = grid_size
        self.include_observed_times = bool(include_observed_times)
        self.validation_fraction = float(validation_fraction)
        self.standardize = bool(standardize)
        self.refine_theta = bool(refine_theta)
        self.compute_theta_variance = bool(compute_theta_variance)
        self.projection_max_epochs = int(projection_max_epochs)
        self.projection_patience = int(projection_patience)
        # minibatch of the projection (residual) networks; None -> min(batch_size, 2048)
        self.projection_batch_size = None if projection_batch_size is None else int(projection_batch_size)
        if self.projection_batch_size is not None and self.projection_batch_size <= 0:
            raise InputValidationError("projection_batch_size must be a positive integer or None.")
        self.projection_n_splits = int(projection_n_splits)
        self.theta_init = theta_init
        self.cox_penalizer = float(cox_penalizer)
        self.theta_init_ = None
        self.theta_init_original_scale_ = None
        self.device = resolve_device(device)
        self.seed = seed
        self.verbose = int(verbose)

        if self.batch_size <= 0:
            raise InputValidationError("batch_size must be a positive integer.")
        if self.max_epochs <= 0:
            raise InputValidationError("max_epochs must be a positive integer.")
        if self.patience <= 0:
            raise InputValidationError("patience must be a positive integer.")
        if not (0.0 <= self.validation_fraction < 1.0):
            raise InputValidationError("validation_fraction must be in [0, 1).")
        if not (0.0 <= self.dropout < 1.0):
            raise InputValidationError("dropout must be in [0, 1).")
        if self.learning_rate <= 0:
            raise InputValidationError("learning_rate must be positive.")
        if self.weight_decay < 0:
            raise InputValidationError("weight_decay must be nonnegative.")
        
        self.num_workers = int(num_workers)

        if self.num_workers < 0:
            raise InputValidationError("num_workers must be nonnegative.")
        
        if pin_memory is None:
            self.pin_memory = self.device.type == "cuda"
        else:
            self.pin_memory = bool(pin_memory)

        self.persistent_workers = bool(persistent_workers) and self.num_workers > 0

        if non_blocking is None:
            self.non_blocking = self.device.type == "cuda"
        else:
            self.non_blocking = bool(non_blocking)

        self.epoch_callback = epoch_callback

        self.scale_time = bool(scale_time)
        self.time_scale_: float | None = None

        self.theta_init_subjects = str(theta_init_subjects).lower()
        if self.theta_init_subjects not in {"all", "train"}:
            raise InputValidationError("theta_init_subjects must be 'all' or 'train'.")
        self.n_restarts = int(n_restarts)
        self.restart_selection = str(restart_selection)
        if self.n_restarts < 1:
            raise InputValidationError("n_restarts must be a positive integer.")
        if self.restart_selection not in {"val_loss", "efficient_score"}:
            raise InputValidationError("restart_selection must be 'val_loss' or 'efficient_score'.")
        self.restart_results_: list[dict] | None = None
        self.selected_restart_: int | None = None

        self.theta_lr_multiplier = float(theta_lr_multiplier)
        if not np.isfinite(self.theta_lr_multiplier) or self.theta_lr_multiplier <= 0:
            raise InputValidationError("theta_lr_multiplier must be a positive number.")
        self.baseline_cumhaz_: tuple[np.ndarray, np.ndarray] | None = None

        self.net_: FlexiHazardNet | None = None
        self.g_network = None
        self.x_scaler_: Standardizer | None = None
        self.z_scaler_: Standardizer | None = None
        self.time_grid: np.ndarray | None = None
        self.train_history: dict[str, list[float] | int] | None = None
        self.phase_histories_: list[dict] | None = None
        self.learning_rate_phases_used_: tuple[float, ...] | None = None
        self.theta_hat: np.ndarray | None = None
        self.theta_hat_original_scale: np.ndarray | None = None
        self.var_theta_hat: np.ndarray | None = None
        self.information_matrix: np.ndarray | None = None
        self.theta_ci_95: np.ndarray | None = None
        self.is_fitted_: bool = False
        self.has_linear_: bool = False
        self.x_dim_: int | None = None
        self.z_dim_: int | None = None
        self.T_train_: np.ndarray | None = None
        self.delta_train_: np.ndarray | None = None
        self.X_train_scaled_: np.ndarray | None = None
        self.Z_train_scaled_: np.ndarray | None = None
        self.internal_train_subject_ids_: np.ndarray | None = None
        self.internal_val_subject_ids_: np.ndarray | None = None

    def _prepare_X(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.x_scaler_ is not None:
            X = self.x_scaler_.transform(X)
        return X.astype(np.float32)

    def _prepare_Z(self, Z):
        if not self.has_linear_:
            if Z is not None:
                raise InputValidationError("This model was fitted with Z=None, so Z_new must not be supplied.")
            return None
        if Z is None:
            raise InputValidationError("This model was fitted with Z, so Z_new is required.")
        Z = np.asarray(Z, dtype=np.float64)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        if self.z_scaler_ is not None:
            Z = self.z_scaler_.transform(Z)
        return Z.astype(np.float32)

    def fit(self, T, delta, X, Z=None):
        set_seed(self.seed)
        T, delta, X, Z = validate_survival_inputs(T, delta, X, Z)
        T, delta, tau = truncate_at_tau(T, delta, self.tau)
        self.tau = tau
        self.time_scale_ = float(tau) if self.scale_time else 1.0

        if np.sum(delta) == 0:
            raise InputValidationError(
                "No observed events remain after truncation at tau. "
                "Use a larger tau or check the event indicators."
            )
        self.has_linear_ = Z is not None
        self.x_dim_ = X.shape[1]
        self.z_dim_ = None if Z is None else Z.shape[1]

        self.x_scaler_ = Standardizer() if self.standardize else None
        X_scaled = self.x_scaler_.fit_transform(X) if self.standardize else X.copy()
        Z_scaled = None
        self.z_scaler_ = None
        if Z is not None:
            self.z_scaler_ = Standardizer() if self.standardize else None
            Z_scaled = self.z_scaler_.fit_transform(Z) if self.standardize else Z.copy()

        self.time_grid = build_time_grid(T, tau, self.grid_size, self.include_observed_times)
        if self.model_type == "zhong":
            # partial likelihood works on subjects: one row per subject, no interval expansion
            expanded = self._subject_rows(T, delta, X_scaled, Z_scaled)
        else:
            expanded = make_expanded_data(T, delta, X_scaled, Z_scaled, self.time_grid)
        self.baseline_cumhaz_ = None

        # The split does not touch the torch RNG, so doing it before building the
        # network leaves the single-restart fit identical to before.
        train_idx, val_idx = self._train_val_indices(expanded["subject_id"])

        train_subject_ids = np.unique(expanded["subject_id"][train_idx]).astype(int)
        val_subject_ids = np.unique(expanded["subject_id"][val_idx]).astype(int)

        self.internal_train_subject_ids_ = train_subject_ids
        self.internal_val_subject_ids_ = val_subject_ids

        # subjects used for the Cox warm start of theta ("all" or the internal training subjects)
        init_subj = train_subject_ids if self.theta_init_subjects == "train" else np.arange(len(T))
        T_init, delta_init, X_init = T[init_subj], delta[init_subj], X[init_subj]
        Z_init = None if Z is None else Z[init_subj]
        Z_scaled_init = None if Z_scaled is None else Z_scaled[init_subj]

        if self.n_restarts == 1:
            self._build_net(X_scaled, Z_scaled)
            if self.has_linear_:
                self._initialize_theta_from_cox(T_init, delta_init, X_init, Z_init, Z_scaled_init)
            self._train_current_net(expanded, train_idx, val_idx)
            self.restart_results_ = None
            self.selected_restart_ = 0
        else:
            self._fit_restarts(expanded, train_idx, val_idx, T_init, delta_init, X_init, X_scaled, Z_init, Z_scaled_init)

        self.theta_hat = None if not self.has_linear_ else self.net_.theta()
        if self.theta_hat is not None and self.z_scaler_ is not None:
            self.theta_hat_original_scale = self.theta_hat / self.z_scaler_.scale_.reshape(-1)
        else:
            self.theta_hat_original_scale = self.theta_hat

        if self.has_linear_ and self.compute_theta_variance:
            theta_var_idx = self.internal_train_subject_ids_

            if theta_var_idx is None or theta_var_idx.size == 0:
                raise InputValidationError(
                    "Internal training split is empty, so theta variance cannot be computed."
                )

            if np.sum(delta[theta_var_idx]) == 0:
                raise InputValidationError(
                    "No observed events in the internal training split, "
                    "so theta variance cannot be computed."
                )

            T_for_projection = self._prepare_time_for_network(T[theta_var_idx])

            self.var_theta_hat, self.information_matrix = estimate_theta_variance(
                T_for_projection,
                delta[theta_var_idx],
                X_scaled[theta_var_idx],
                Z_scaled[theta_var_idx],
                hidden_layers=self.hidden_layers,
                activation=self.activation,
                dropout=self.dropout,
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                batch_size=self._projection_batch(),
                max_epochs=self.projection_max_epochs,
                patience=self.projection_patience,
                n_splits=self.projection_n_splits,
                device=self.device,
                seed=self.seed,
                projection_net=self._projection_net_kind(),
            )
            # Convert variance to original Z scale when standardization was used.
            if self.z_scaler_ is not None:
                scale = self.z_scaler_.scale_.reshape(-1)
                D = np.diag(1.0 / scale)
                self.var_theta_hat = D @ self.var_theta_hat @ D
            self.theta_ci_95 = theta_confidence_interval(self.theta_hat_original_scale, self.var_theta_hat, alpha=0.05)

        self.g_network = self.net_.g_net
        self.T_train_ = T
        self.delta_train_ = delta
        self.X_train_scaled_ = X_scaled.astype(np.float32)
        self.Z_train_scaled_ = None if Z_scaled is None else Z_scaled.astype(np.float32)
        self.is_fitted_ = True
        return self

    def _projection_batch(self):
        return self.projection_batch_size if self.projection_batch_size is not None else min(self.batch_size, 2048)

    def _projection_net_kind(self):
        return "additive" if self.model_type == "zhong" else "joint"

    @staticmethod
    def _subject_rows(T, delta, X_scaled, Z_scaled):
        """Subject-level rows in the same layout as make_expanded_data (used by model_type='zhong')."""
        n = len(T)
        out = {
            "X": np.asarray(X_scaled, dtype=np.float32),
            "t": np.asarray(T, dtype=np.float32).reshape(-1, 1),
            "event": np.asarray(delta, dtype=np.float32).reshape(-1, 1),
            "subject_id": np.arange(n, dtype=np.int64),
        }
        if Z_scaled is not None:
            out["Z"] = np.asarray(Z_scaled, dtype=np.float32)
        return out

    def _build_net(self, X_scaled, Z_scaled):
        net_cls = ZhongPHNet if self.model_type == "zhong" else FlexiHazardNet
        self.net_ = net_cls(
            x_dim=X_scaled.shape[1],
            z_dim=None if Z_scaled is None else Z_scaled.shape[1],
            hidden_layers=self.hidden_layers,
            activation=self.activation,
            dropout=self.dropout,
        ).to(self.device)

    @staticmethod
    def _rows_subset(expanded, idx):
        """The rows idx of an expanded / subject-level dict (validation rows must never
        enter the theta refinement; they are used for early stopping only)."""
        return {k: (v[idx] if isinstance(v, np.ndarray) and v.shape[0] == expanded["t"].shape[0] else v)
                for k, v in expanded.items()}

    def _train_current_net(self, expanded, train_idx, val_idx):
        self._fit_expanded_phases(expanded, train_idx, val_idx)

        if self.has_linear_ and self.refine_theta and not self.three_phase_lr:
            self._refine_theta(self._rows_subset(expanded, train_idx))

    def _fit_restarts(self, expanded, train_idx, val_idx, T, delta, X, X_scaled, Z, Z_scaled):
        """Train n_restarts networks on the same split; keep the one picked by restart_selection."""
        val_data = self._tensors_from_expanded(expanded, val_idx)
        results, states, histories = [], [], []
        for k in range(self.n_restarts):
            # restart 0 continues the RNG stream seeded at the top of fit(), so it is
            # exactly the single-restart fit; later restarts reseed with seed + k.
            if k > 0 and self.seed is not None:
                set_seed(self.seed + k)
            self._build_net(X_scaled, Z_scaled)
            if self.has_linear_:
                if k == 0:
                    self._initialize_theta_from_cox(T, delta, X, Z, Z_scaled)
                else:
                    with torch.no_grad():
                        self.net_.linear.weight.copy_(
                            torch.as_tensor(self.theta_init_.reshape(1, -1), dtype=torch.float32, device=self.device)
                        )
            self._train_current_net(expanded, train_idx, val_idx)

            rec = {
                "restart": k,
                "seed": None if self.seed is None else self.seed + k,
                "val_loss": float(self._validation_loss(val_data)),
            }
            if self.has_linear_:
                theta = self.net_.theta()
                if self.z_scaler_ is not None:
                    theta = theta / self.z_scaler_.scale_.reshape(-1)
                rec["theta"] = np.asarray(theta, dtype=np.float64)
            results.append(rec)
            states.append(copy.deepcopy(self.net_.state_dict()))
            histories.append((self.train_history, self.phase_histories_, self.learning_rate_phases_used_))

        use_score = self.restart_selection == "efficient_score" and self.has_linear_
        if self.restart_selection == "efficient_score" and not self.has_linear_:
            warnings.warn("restart_selection='efficient_score' needs Z; selecting by validation loss instead.")
        if use_score:
            # residuals are computed after all restarts are trained so that restart 0 is
            # unaffected by the extra RNG draws of the projection nets
            R_rows = self._score_residuals(expanded, train_idx, T, delta, X_scaled, Z_scaled)
            for rec, state in zip(results, states):
                self.net_.load_state_dict(state)
                score = self._efficient_score(expanded, train_idx, R_rows)
                rec["score"] = score
                rec["score_norm"] = float(np.linalg.norm(score))
            best = int(np.argmin([r["score_norm"] for r in results]))
        else:
            best = int(np.argmin([r["val_loss"] for r in results]))

        self.net_.load_state_dict(states[best])
        self.train_history, self.phase_histories_, self.learning_rate_phases_used_ = histories[best]
        self.restart_results_ = results
        self.selected_restart_ = best

    def _score_residuals(self, expanded, train_idx, T, delta, X_scaled, Z_scaled):
        subj = self.internal_train_subject_ids_
        pos = np.full(int(np.max(expanded["subject_id"])) + 1, -1, dtype=np.int64)
        pos[subj] = np.arange(subj.size)
        row_pos = pos[expanded["subject_id"][train_idx]]
        return crossfitted_score_residuals(
            X_scaled[subj],
            self._prepare_time_for_network(T[subj]),
            Z_scaled[subj],
            delta[subj],
            row_pos,
            expanded["X"][train_idx],
            self._prepare_time_for_network(expanded["t"][train_idx]),
            hidden_layers=self.hidden_layers,
            activation=self.activation,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            batch_size=self._projection_batch(),
            max_epochs=self.projection_max_epochs,
            patience=self.projection_patience,
            n_splits=self.projection_n_splits,
            device=self.device,
            seed=0 if self.seed is None else self.seed,
            projection_net=self._projection_net_kind(),
        )

    def _efficient_score(self, expanded, train_idx, R_rows):
        """Efficient score for theta of the current net, using cross-fitted residuals R.

        model_type="flexihaz": sum over training rows of R_i * (length_i * exp(eta_i) - event_i).
        model_type="zhong": partial-likelihood score with Z replaced by its residual,
        sum over training events of R_i - (risk-set weighted mean of R), weights exp(eta).
        """
        X = expanded["X"][train_idx]
        t = expanded["t"][train_idx]
        Z = expanded["Z"][train_idx].astype(np.float64)
        event = expanded["event"][train_idx].astype(np.float64).reshape(-1, 1)
        g = self._predict_g_on_pairs(X, t, batch_size=self.batch_size).reshape(-1, 1)
        theta = self.net_.theta().astype(np.float64).reshape(-1, 1)
        eta = np.clip(g + Z @ theta, -80.0, 80.0)
        if self.model_type == "zhong":
            order = np.argsort(-t.reshape(-1), kind="stable")
            w = np.exp(eta.reshape(-1)[order])
            R_s = np.asarray(R_rows, dtype=np.float64)[order]
            cum_w = np.cumsum(w).reshape(-1, 1)
            cum_Rw = np.cumsum(R_s * w.reshape(-1, 1), axis=0)
            return np.sum(event[order] * (R_s - cum_Rw / np.maximum(cum_w, 1e-300)), axis=0)
        length = expanded["length"][train_idx].astype(np.float64).reshape(-1, 1)
        return np.sum(R_rows * (length * np.exp(eta) - event), axis=0)

    def _train_val_indices(self, subject_id):
        if self.validation_fraction <= 0:
            idx = np.arange(len(subject_id))
            return idx, idx
        splitter = GroupShuffleSplit(n_splits=1, test_size=self.validation_fraction, random_state=self.seed)
        rows = np.arange(len(subject_id))
        tr, va = next(splitter.split(rows, groups=subject_id))
        return tr, va
    
    @staticmethod
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

    def _to_device(self, tensor):
        return tensor.to(self.device, non_blocking=self.non_blocking)
    
    def _prepare_time_for_network(self, t):
        if self.time_scale_ is None:
            raise NotFittedError("Call fit() before using time-dependent model inputs.")

        t_arr = np.asarray(t, dtype=np.float32).reshape(-1, 1)
        return t_arr / float(self.time_scale_)
    
    def _initialize_theta_from_cox(self, T, delta, X_original, Z_original, Z_scaled):
        """Initialize the linear coefficient theta from a Cox PH fit on (X, Z).

        The Cox model is fitted jointly on the flexible covariates X and the
        linear covariates Z (X entering linearly), and the Z coefficients are
        used as the warm start. Fitting on Z alone would let theta absorb the
        part of X's effect that leaks through the X-Z correlation. Pass
        X_original=None to fit on Z only.

        The neural network is trained on standardized Z when standardize=True.
        Therefore, the value copied into self.net_.linear.weight must be on the
        standardized-Z scale.

        If Z_scaled = (Z_original - mean) / scale, then:
            theta_scaled = theta_original * scale

        because:
            theta_original^T Z_original
            = theta_original^T mean + (theta_original * scale)^T Z_scaled

        The constant theta_original^T mean can be absorbed by g(t, X), so only
        theta_scaled is copied into the linear branch.
        """
        if not self.has_linear_ or self.net_ is None or self.net_.linear is None:
            return

        if self.theta_init is None or str(self.theta_init).lower() in {"none", "random"}:
            return

        init = str(self.theta_init).lower()

        if init == "zero":
            theta_scaled = np.zeros(self.z_dim_, dtype=np.float64)

        elif init == "cox":
            try:
                import pandas as pd
                from lifelines import CoxPHFitter
            except Exception as exc:
                warnings.warn(
                    "theta_init='cox' was requested, but lifelines is not installed. "
                    "Falling back to zero initialization. "
                    f"Original error: {exc}"
                )
                theta_scaled = np.zeros(self.z_dim_, dtype=np.float64)
            else:
                try:
                    z_cols = [f"Z{j}" for j in range(self.z_dim_)]
                    df = pd.DataFrame(np.asarray(Z_original, dtype=np.float64).reshape(len(T), -1), columns=z_cols)
                    if X_original is not None:
                        X_arr = np.asarray(X_original, dtype=np.float64).reshape(len(T), -1)
                        for j in range(X_arr.shape[1]):
                            df[f"X{j}"] = X_arr[:, j]
                    df["T"] = np.asarray(T, dtype=np.float64)
                    df["delta"] = np.asarray(delta, dtype=int)

                    cph = CoxPHFitter(penalizer=self.cox_penalizer)
                    cph.fit(df, duration_col="T", event_col="delta", show_progress=False)

                    theta_original = cph.params_.loc[z_cols].to_numpy(dtype=np.float64)

                    if self.z_scaler_ is not None:
                        theta_scaled = theta_original * self.z_scaler_.scale_.reshape(-1)
                    else:
                        theta_scaled = theta_original

                except Exception as exc:
                    warnings.warn(
                        "Cox PH theta initialization failed. "
                        "Falling back to zero initialization. "
                        f"Original error: {exc}"
                    )
                    theta_scaled = np.zeros(self.z_dim_, dtype=np.float64)
        else:
            raise ValueError("theta_init must be one of None, 'random', 'zero', or 'cox'.")

        self.theta_init_ = theta_scaled.copy()

        if self.z_scaler_ is not None:
            self.theta_init_original_scale_ = theta_scaled / self.z_scaler_.scale_.reshape(-1)
        else:
            self.theta_init_original_scale_ = theta_scaled.copy()

        with torch.no_grad():
            self.net_.linear.weight.copy_(
                torch.as_tensor(theta_scaled.reshape(1, -1), dtype=torch.float32, device=self.device)
            )

    def _fit_expanded_phases(self, expanded, train_idx, val_idx):
        base_lr = float(self.learning_rate)

        if self.three_phase_lr:
            phase_lrs = (
                base_lr,
                base_lr * 0.1,
                base_lr * 0.01,
            )
        else:
            phase_lrs = (base_lr,)

        self.learning_rate_phases_used_ = tuple(float(lr) for lr in phase_lrs)

        all_train_losses = []
        all_val_losses = []
        phase_histories = []

        epoch_offset = 0
        global_best_epoch = -1
        global_best_val = float("inf")

        for phase_idx, lr in enumerate(self.learning_rate_phases_used_):
            if self.verbose and len(self.learning_rate_phases_used_) > 1:
                print(
                    f"Starting LR phase {phase_idx + 1}/{len(self.learning_rate_phases_used_)} "
                    f"with learning_rate={lr:g}"
                )

            # the best validation loss is carried into the next phase, so a lower-LR
            # phase only replaces the weights when it beats the whole run's best so far
            global_best_val = self._fit_expanded(
                expanded,
                train_idx,
                val_idx,
                learning_rate=lr,
                epoch_offset=epoch_offset,
                init_best_val=None if phase_idx == 0 else global_best_val,
            )

            phase_train_losses = list(self.train_history["train_loss"])
            phase_val_losses = list(self.train_history["val_loss"])
            phase_best_epoch = int(self.train_history["best_epoch"])
            phase_epochs_run = len(phase_train_losses)

            if phase_best_epoch >= 0:
                global_best_epoch = epoch_offset + phase_best_epoch

            phase_history = {
                "phase": phase_idx + 1,
                "learning_rate": float(lr),
                "train_loss": phase_train_losses,
                "val_loss": phase_val_losses,
                "best_epoch": phase_best_epoch,
                "best_epoch_global": epoch_offset + phase_best_epoch if phase_best_epoch >= 0 else -1,
                "epochs_run": phase_epochs_run,
            }

            phase_histories.append(phase_history)
            all_train_losses.extend(phase_train_losses)
            all_val_losses.extend(phase_val_losses)

            epoch_offset += phase_epochs_run

            if self.three_phase_lr and self.has_linear_ and self.refine_theta:
                # exact re-solve of theta with g fixed, on the TRAINING rows only
                self._refine_theta(self._rows_subset(expanded, train_idx))

        self.phase_histories_ = phase_histories

        # the weights in use are those of the best epoch over ALL phases (the best
        # validation loss is carried across phases), so report that epoch
        self.train_history = {
            "train_loss": all_train_losses,
            "val_loss": all_val_losses,
            "best_epoch": global_best_epoch,
            "global_best_epoch": global_best_epoch,
            "phases": phase_histories,
        }

    def _fit_expanded(self, expanded, train_idx, val_idx, learning_rate=None, epoch_offset=0, init_best_val=None):
        """One early-stopped training run; returns the best validation loss reached.

        init_best_val: best validation loss of an earlier phase. Early stopping then
        only accepts weights that improve on the whole run so far, and if nothing
        improves the weights at entry (the earlier phase's best) are kept. None
        resets to +inf (single-phase behaviour). Non-finite validation losses never
        count as an improvement.
        """
        net = self.net_
        lr = self.learning_rate if learning_rate is None else float(learning_rate)

        if lr <= 0:
            raise InputValidationError("learning_rate must be positive.")
        
        opt = self._make_optimizer(lr)
        train_data = self._tensors_from_expanded(expanded, train_idx, device=torch.device("cpu"))
        val_data = self._tensors_from_expanded(expanded, val_idx)
        # the partial likelihood needs the whole risk set, so model_type="zhong" is full-batch
        batch_size = self.batch_size if self.model_type == "flexihaz" else len(train_idx)

        best_state = copy.deepcopy(net.state_dict())
        best_val = float("inf") if init_best_val is None else float(init_best_val)
        best_epoch = -1
        wait = 0
        n_nonfinite = 0
        train_losses, val_losses = [], []
        for epoch in range(self.max_epochs):
            net.train()
            losses = []
            for batch in self._iter_batches(train_data, batch_size, shuffle=True):
                xb = self._to_device(batch["X"])
                tb = self._to_device(batch["t"])
                eb = self._to_device(batch["event"])
                if self.has_linear_:
                    zb = self._to_device(batch["Z"])
                    eta, _ = net(tb, xb, zb)
                else:
                    eta, _ = net(tb, xb, None)
                if self.model_type == "zhong":
                    loss = cox_partial_likelihood_loss(eta, tb, eb)
                else:
                    loss = negative_log_likelihood(eta, eb, self._to_device(batch["length"]))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(float(loss.detach().cpu()))

            val_loss = self._validation_loss(val_data)
            train_losses.append(float(np.mean(losses)))
            val_losses.append(float(val_loss))
            
            if self.verbose:
                print(f"epoch={epoch+1}, train_loss={train_losses[-1]:.6f}, val_loss={val_loss:.6f}")

            if self.epoch_callback is not None:
                should_stop = self.epoch_callback(
                    epoch=epoch_offset + epoch,
                    train_loss=train_losses[-1],
                    val_loss=float(val_loss),
                    model=self,
                )
                if should_stop:
                    break

            if not np.isfinite(val_loss):
                n_nonfinite += 1
            if np.isfinite(val_loss) and val_loss < best_val - 1e-8:
                best_val = float(val_loss)
                best_state = copy.deepcopy(net.state_dict())
                best_epoch = epoch
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        if n_nonfinite:
            warnings.warn(
                f"{n_nonfinite} of {len(val_losses)} epochs had a non-finite validation loss; "
                "those epochs were ignored by early stopping."
            )
        net.load_state_dict(best_state)
        self.train_history = {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "best_epoch": best_epoch,
            "learning_rate": lr,
        }
        return best_val

    def _make_optimizer(self, lr):
        """Adam over the network; with theta_lr_multiplier != 1 the linear branch gets its own group."""
        net = self.net_
        linear = getattr(net, "linear", None)
        if self.has_linear_ and linear is not None and self.theta_lr_multiplier != 1.0:
            lin_ids = {id(p) for p in linear.parameters()}
            groups = [
                {"params": [p for p in net.parameters() if id(p) not in lin_ids]},
                {"params": list(linear.parameters()), "lr": lr * self.theta_lr_multiplier},
            ]
            return torch.optim.Adam(groups, lr=lr, weight_decay=self.weight_decay)
        return torch.optim.Adam(net.parameters(), lr=lr, weight_decay=self.weight_decay)

    def _tensors_from_expanded(self, expanded, idx, device=None):
        device = self.device if device is None else device
        out = {
            "X": torch.as_tensor(expanded["X"][idx], dtype=torch.float32, device=device),
            "t": torch.as_tensor(
                self._prepare_time_for_network(expanded["t"][idx]),
                dtype=torch.float32,
                device=device,
                ),
            "event": torch.as_tensor(expanded["event"][idx], dtype=torch.float32, device=device),
        }
        if "length" in expanded:
            out["length"] = torch.as_tensor(expanded["length"][idx], dtype=torch.float32, device=device)
        if self.has_linear_:
            out["Z"] = torch.as_tensor(expanded["Z"][idx], dtype=torch.float32, device=device)
        return out

    @torch.no_grad()
    def _validation_loss(self, data):
        self.net_.eval()
        if self.model_type == "zhong":
            # the partial likelihood needs the whole risk set: one full pass over the (subject-level) rows
            if self.has_linear_:
                eta, _ = self.net_(data["t"], data["X"], data["Z"])
            else:
                eta, _ = self.net_(data["t"], data["X"], None)
            return float(cox_partial_likelihood_loss(eta, data["t"], data["event"]).detach().cpu())
        # mean NLL accumulated in minibatches: identical value to a single pass, but bounded GPU memory
        # (a single pass over all expanded validation rows can need several GB of activations)
        n = data["t"].shape[0]
        total = 0.0
        for batch in self._iter_batches(data, self.batch_size, shuffle=False):
            if self.has_linear_:
                eta, _ = self.net_(batch["t"], batch["X"], batch["Z"])
            else:
                eta, _ = self.net_(batch["t"], batch["X"], None)
            total += float(negative_log_likelihood(eta, batch["event"], batch["length"], reduction="sum").detach().cpu())
        return total / n

    @torch.no_grad()
    def _predict_g_on_pairs(self, X_pairs, t_pairs, batch_size=None):
        if self.net_ is None:
            raise NotFittedError("Model network has not been initialized.")
        batch_size = max(1, self.batch_size if batch_size is None else int(batch_size))
        t_pairs_net = self._prepare_time_for_network(t_pairs)
        X_t = torch.as_tensor(X_pairs, dtype=torch.float32)
        t_t = torch.as_tensor(t_pairs_net, dtype=torch.float32)
        n = X_t.shape[0]
        out = []
        self.net_.eval()
        for start in range(0, n, batch_size):
            xb = self._to_device(X_t[start:start + batch_size])
            tb = self._to_device(t_t[start:start + batch_size])
            zb = None if not self.has_linear_ else torch.zeros((xb.shape[0], self.z_dim_), dtype=torch.float32, device=self.device)
            _, g = self.net_(tb, xb, zb)
            out.append(g.cpu().numpy())
        return np.vstack(out).reshape(-1)

    def _refine_theta(self, expanded):
        if not self.has_linear_:
            return
        if self.model_type == "zhong":
            self._refine_theta_partial_likelihood(expanded)
            return
        X = expanded["X"]
        Z = expanded["Z"].astype(np.float64)
        t = expanded["t"]
        event = expanded["event"].astype(np.float64)
        length = expanded["length"].astype(np.float64)
        g = self._predict_g_on_pairs(X, t, batch_size=self.batch_size).reshape(-1, 1)
        theta0 = self.net_.theta().astype(np.float64)

        def obj(theta):
            eta = g + Z @ theta.reshape(-1, 1)
            eta = np.clip(eta, -80.0, 80.0)
            return float(np.sum(length * np.exp(eta) - event * eta) / len(event))

        def grad(theta):
            eta = g + Z @ theta.reshape(-1, 1)
            eta = np.clip(eta, -80.0, 80.0)
            gr = Z * (length * np.exp(eta) - event)
            return np.mean(gr, axis=0)

        res = minimize(obj, theta0, jac=grad, method="L-BFGS-B")
        if res.success:
            with torch.no_grad():
                self.net_.linear.weight.copy_(torch.as_tensor(res.x.reshape(1, -1), dtype=torch.float32, device=self.device))

    def _refine_theta_partial_likelihood(self, rows):
        """model_type='zhong': set theta to the maximiser of the Breslow partial likelihood with g fixed."""
        g = self._predict_g_on_pairs(rows["X"], rows["t"], batch_size=self.batch_size).reshape(-1)
        order = np.argsort(-rows["t"].reshape(-1), kind="stable")
        g_s = g[order].astype(np.float64)
        Z_s = rows["Z"][order].astype(np.float64)
        d_s = rows["event"].reshape(-1)[order].astype(np.float64)
        n_events = max(float(d_s.sum()), 1.0)
        theta0 = self.net_.theta().astype(np.float64)

        def obj_grad(theta):
            eta = np.clip(g_s + Z_s @ theta, -80.0, 80.0)
            w = np.exp(eta)
            cum_w = np.maximum(np.cumsum(w), 1e-300)
            cum_Zw = np.cumsum(Z_s * w.reshape(-1, 1), axis=0)
            ll = np.sum(d_s * (eta - np.log(cum_w)))
            grad = np.sum(d_s.reshape(-1, 1) * (Z_s - cum_Zw / cum_w.reshape(-1, 1)), axis=0)
            return -ll / n_events, -grad / n_events

        res = minimize(obj_grad, theta0, jac=True, method="L-BFGS-B")
        if res.success:
            with torch.no_grad():
                self.net_.linear.weight.copy_(torch.as_tensor(res.x.reshape(1, -1), dtype=torch.float32, device=self.device))

    @torch.no_grad()
    def _predict_eta_subjects(self, X_scaled, Z_scaled):
        """Linear predictor eta = g(X) + theta^T Z on subject rows (model_type='zhong')."""
        self.net_.eval()
        X_t = torch.as_tensor(X_scaled, dtype=torch.float32)
        Z_t = None if Z_scaled is None else torch.as_tensor(Z_scaled, dtype=torch.float32)
        n = X_t.shape[0]
        batch_size = max(1, self.batch_size)
        out = []
        for start in range(0, n, batch_size):
            xb = self._to_device(X_t[start:start + batch_size])
            tb = torch.zeros((xb.shape[0], 1), dtype=torch.float32, device=self.device)
            zb = None if Z_t is None else self._to_device(Z_t[start:start + batch_size])
            eta, _ = self.net_(tb, xb, zb)
            out.append(eta.cpu().numpy())
        return np.vstack(out).reshape(-1).astype(np.float64)

    def _breslow_baseline(self):
        """Breslow baseline cumulative hazard H0 at the distinct event times of the fitted data."""
        if self.baseline_cumhaz_ is None:
            T = self.T_train_
            delta = self.delta_train_.astype(bool)
            eta = self._predict_eta_subjects(self.X_train_scaled_, self.Z_train_scaled_)
            risk = np.exp(np.clip(eta, -80.0, 80.0))
            event_times = np.sort(np.unique(T[delta]))
            increments = np.zeros(event_times.shape[0], dtype=np.float64)
            for j, s in enumerate(event_times):
                d_j = float(np.sum(delta & (T == s)))
                increments[j] = d_j / max(float(np.sum(risk[T >= s])), 1e-300)
            self.baseline_cumhaz_ = (event_times, np.cumsum(increments))
        return self.baseline_cumhaz_

    def _check_fitted(self):
        if not self.is_fitted_ or self.net_ is None:
            raise NotFittedError("Call fit() before prediction.")

    def _make_prediction_pairs(self, X_new, times):
        X_new = self._prepare_X(X_new)
        times = np.asarray(times, dtype=np.float64).reshape(-1)
        if np.any(times < 0) or np.any(times > self.tau + 1e-12):
            raise InputValidationError("times must lie within [0, tau].")
        n, m = X_new.shape[0], times.size
        X_pairs = np.repeat(X_new, m, axis=0)
        t_pairs = np.tile(times, n).reshape(-1, 1).astype(np.float32)
        return X_new, times, X_pairs, t_pairs, n, m

    def predict_g(self, X_new, times):
        self._check_fitted()
        _, _, X_pairs, t_pairs, n, m = self._make_prediction_pairs(X_new, times)
        vals = self._predict_g_on_pairs(X_pairs, t_pairs, batch_size=self.batch_size)
        return vals.reshape(n, m)

    def predict_hazard(self, X_new, times, Z_new=None):
        self._check_fitted()
        if self.model_type == "zhong":
            raise InputValidationError(
                "predict_hazard is not available for model_type='zhong' (the Breslow baseline hazard is a step "
                "function); use predict_cumulative_hazard or predict_survival."
            )
        X_scaled, times, X_pairs, t_pairs, n, m = self._make_prediction_pairs(X_new, times)
        Z_scaled = self._prepare_Z(Z_new)
        if self.has_linear_:
            if Z_scaled.shape[0] != n:
                raise InputValidationError("X_new and Z_new must have the same number of rows.")
            Z_pairs = np.repeat(Z_scaled, m, axis=0)
        else:
            Z_pairs = None
        t_pairs_net = self._prepare_time_for_network(t_pairs)
        X_t = torch.as_tensor(X_pairs, dtype=torch.float32)
        t_t = torch.as_tensor(t_pairs_net, dtype=torch.float32)
        Z_t = torch.as_tensor(Z_pairs, dtype=torch.float32) if self.has_linear_ else None
        n_rows = X_t.shape[0]
        batch_size = max(1, self.batch_size)
        out = []
        self.net_.eval()
        with torch.no_grad():
            for start in range(0, n_rows, batch_size):
                xb = self._to_device(X_t[start:start + batch_size])
                tb = self._to_device(t_t[start:start + batch_size])
                if self.has_linear_:
                    zb = self._to_device(Z_t[start:start + batch_size])
                    eta, _ = self.net_(tb, xb, zb)
                else:
                    eta, _ = self.net_(tb, xb, None)
                out.append(torch.exp(torch.clamp(eta, -80, 80)).cpu().numpy())
        return np.vstack(out).reshape(n, m)

    def predict_cumulative_hazard(self, X_new, times, Z_new=None):
        self._check_fitted()
        times = np.asarray(times, dtype=np.float64).reshape(-1)
        if self.model_type == "zhong":
            if np.any(times < 0) or np.any(times > self.tau + 1e-12):
                raise InputValidationError("times must lie within [0, tau].")
            X_scaled = self._prepare_X(X_new)
            Z_scaled = self._prepare_Z(Z_new)
            if self.has_linear_ and Z_scaled.shape[0] != X_scaled.shape[0]:
                raise InputValidationError("X_new and Z_new must have the same number of rows.")
            event_times, H0 = self._breslow_baseline()
            idx = np.searchsorted(event_times, times, side="right") - 1
            H0_at = np.where(idx >= 0, H0[np.maximum(idx, 0)], 0.0) if event_times.size else np.zeros(times.size)
            eta = self._predict_eta_subjects(X_scaled, Z_scaled)
            return np.outer(np.exp(np.clip(eta, -80.0, 80.0)), H0_at)
        if np.any(np.diff(times) < 0):
            order = np.argsort(times)
            inv = np.argsort(order)
            return self.predict_cumulative_hazard(X_new, times[order], Z_new)[:, inv]
        grid = self.time_grid
        eval_grid = np.unique(np.concatenate([grid[grid <= np.max(times)], times, np.array([0.0])]))
        eval_grid.sort()
        haz_grid = self.predict_hazard(X_new, eval_grid, Z_new=Z_new)
        dt = np.diff(eval_grid, prepend=0.0)
        dt[0] = 0.0
        cum = np.cumsum(haz_grid * dt.reshape(1, -1), axis=1)
        return np.vstack([np.interp(times, eval_grid, row) for row in cum])

    def predict_survival(self, X_new, times, Z_new=None):
        H = self.predict_cumulative_hazard(X_new, times, Z_new=Z_new)
        return np.exp(-H)

    def theta_summary(self, alpha: float = 0.05):
        if not self.has_linear_:
            raise InputValidationError("theta_summary is available only when the model was fitted with Z.")
        theta = np.asarray(self.theta_hat_original_scale, dtype=np.float64).reshape(-1)
        out = {"theta_hat": theta}
        if self.var_theta_hat is not None:
            ci = theta_confidence_interval(theta, self.var_theta_hat, alpha=alpha)
            out.update(
                {
                    "var_theta_hat": self.var_theta_hat,
                    "se_theta_hat": np.sqrt(np.maximum(np.diag(self.var_theta_hat), 0.0)),
                    f"ci_{int((1-alpha)*100)}": ci,
                }
            )
        return out
