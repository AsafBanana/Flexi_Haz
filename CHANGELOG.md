# Changelog

All notable changes to FLEXI-Haz will be documented in this file.

The format follows the principles of Keep a Changelog. Version numbers follow semantic versioning where practical.

## [Unreleased]

### Added

- `n_restarts` and `restart_selection` options for `FlexiHazardModel`: fit several networks and keep the one with the lowest validation loss or the efficient score closest to zero, with per-restart diagnostics in `restart_results_`.
- `--n-restarts` / `--restart-selection` options in the partially linear simulation runner.
- `theta_lr_multiplier` option for `FlexiHazardModel`: the linear (`theta`) branch is trained with `learning_rate * theta_lr_multiplier` in its own Adam parameter group, in every learning-rate phase.
- `model_type="zhong"` option for `FlexiHazardModel`: the proportional-hazards model of Zhong, Mueller and Wang (2022), `h(t | X, Z) = h0(t) exp{theta^T Z + g(X)}`, fitted with the same estimator. The network has no time input, training is full-batch on the Breslow partial likelihood, `theta` is refined by profiling the partial likelihood with `g` fixed, and the projection networks for the covariance and the efficient score are additive (`g_X(X) + g_T(t)`). Cumulative hazard and survival use the Breslow baseline. New `ZhongPHNet`, `AdditiveProjectionNet`, `cox_partial_likelihood_loss`, and a `projection_net` argument for `estimate_theta_variance` / `crossfitted_score_residuals`.
- `--theta-lr-mult` and `--model-type {flexihaz,zhong}` options in the partially linear simulation runner.

### Planned

- Public package release.
- PyPI publication.
- Citation update when the accompanying paper becomes available.

## [0.1.0] - 2026-09-09

### Added

- Partially linear neural hazard model with a separate linear covariate component.
- Fully nonparametric neural hazard model.
- Administrative truncation and time-grid construction.
- Expanded interval-level full-likelihood training.
- Subject-level train/validation splitting.
- Three-phase learning-rate schedule and early stopping.
- Optional Cox initialization of the linear coefficient.
- Post-training coefficient refinement.
- Variance estimation and confidence intervals for the partially linear coefficient.
- Hazard, cumulative-hazard, and survival prediction.
- Optuna-based hyperparameter tuning.
- Simulation and Rotterdam breast-cancer reproducibility experiments.
- Package robustness and smoke tests.
