# Changelog

All notable changes to FLEXI-Haz will be documented in this file.

The format follows the principles of Keep a Changelog. Version numbers follow semantic versioning where practical.

## [Unreleased]

### Added

- `n_restarts` and `restart_selection` options for `FlexiHazardModel`: fit several networks and keep the one with the lowest validation loss or the efficient score closest to zero, with per-restart diagnostics in `restart_results_`.
- `--n-restarts` / `--restart-selection` options in the partially linear simulation runner.

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
