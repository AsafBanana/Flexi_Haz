# FLEXI-Haz Methodology

This document summarizes the methodology implemented by the current package. The full project report contains the detailed derivations, experiments, and discussion.

## Partially linear hazard model

FLEXI-Haz models

\[
h(t\mid X,Z)=\exp\{\theta^\top Z+g(t,X)\},
\]

where `Z` enters through a separate linear term and `X` enters through a neural function of time and covariates.

This preserves interpretable effects for selected covariates while allowing the remaining hazard structure to be nonlinear and time-varying.

## Fully nonparametric model

When `Z=None`, the package fits

\[
h(t\mid X)=\exp\{g(t,X)\}.
\]

This variant removes the separate linear component and is intended for flexible hazard and survival modeling.

## Time discretization and expanded data

The estimator constructs a grid

\[
0=t_0<t_1<\cdots<t_m=\tau.
\]

A subject contributes one expanded row for each at-risk interval. For interval \(j\),

\[
w_{ij}=\min(T_i,t_j)-t_{j-1}
\]

is the exposure length, and the interval event indicator identifies whether the observed event occurred in that interval.

The neural network is evaluated at the interval endpoint \(t_j\).

## Full-likelihood objective

For the partially linear model, the interval-level negative log-likelihood contribution is

\[
w_{ij}\exp\{\theta^\top Z_i+g(t_j,X_i)\}
-
\Delta_{ij}\{\theta^\top Z_i+g(t_j,X_i)\}.
\]

The fully nonparametric model uses the same likelihood without the linear term.

## Neural network

`FlexiHazardNet` contains an MLP for the flexible component. Time is concatenated with the flexible covariates. When `scale_time=True`, the network receives time divided by the fitted administrative horizon.

When `Z` is present, a separate bias-free linear layer represents \(\theta^\top Z\).

## Training

The model is trained with Adam on mini-batches of expanded rows.

Validation splitting is performed at the subject level before the expanded rows are partitioned, preventing rows from the same subject from appearing in both training and validation data.

When `three_phase_lr=True`, training proceeds through three learning-rate phases based on

\[
\alpha,\quad 0.1\alpha,\quad 0.01\alpha.
\]

Each phase uses validation-based early stopping. The estimator retains the best model state.

## Linear coefficient refinement

For a partially linear model, the package can freeze the learned neural nuisance component and refine the linear coefficient by direct numerical optimization of the discretized likelihood.

## Inference

When requested, FLEXI-Haz estimates uncertainty for the partially linear coefficient using the projection/cross-fitting procedure implemented in `inference.py`. The resulting covariance matrix is used to form standard errors and confidence intervals.

## Prediction

The package exposes:

- the learned flexible component \(g(t,X)\);
- the hazard;
- cumulative hazard;
- survival probability.

Cumulative hazard is numerically accumulated on a grid combining the fitted time grid and requested prediction times, and survival is computed as

\[
\widehat S(t)=\exp\{-\widehat H(t)\}.
\]

For the complete methodological description and empirical evaluation, see `FLEXI_Haz_Report.pdf`.
