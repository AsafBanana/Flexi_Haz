# Setting 2: two-coefficient simulation, theta_0 = (1, -1)

Treatment-effect estimation with partially linear **FLEXI-Haz** versus the
proportional-hazards DNN of **Zhong, Mueller & Wang (2022)** under a nonlinear,
non-proportional-hazards mechanism with a two-dimensional linear part. Both
models are fitted with `flexihaz.FlexiHazardModel` (`--model-type flexihaz` /
`zhong`): same data, same folds, same tuning grid, same restart machinery,
same residual-projection standard errors.

## Files

``` text
simulation_dgm.py                 data-generating mechanism, true survival, dataset I/O
simulation_io.py                  paths, JSON/CSV helpers, dataset loading, completion checks
metrics_survival.py               theta inference metrics and IPCW prediction metrics
generate_simulation_datasets.py   the 200 training datasets and the fixed test set
tune_grid_cv.py                   FLEXI-Haz architecture grid (27 archs, fixed lr) on an independent tuning dataset
tune_zhong_optuna.py              Zhong comparator tuning: 100 Optuna TPE trials over the 27 archs x lr in [1e-3, 5e-2]
run_replications.py               fits either model on the saved replications
aggregate_results.py              per-model and cross-model summary tables
```

Outputs go to `outputs/` (git-ignored): `datasets/`, `tuning/`, `results/`.

## Data-generating mechanism

`X = (X_1, X_2, X_3)`, `Z = (Z_1, Z_2)`, all independent `U(-1, 1)`;

``` text
h(t | X, Z) = 0.1 exp{ (0.1 + f(X)^2) t + theta_0' Z },   f(X) = 0.2 (X_1 + X_2) + 0.5 X_1 X_2 + X_3^2
```

`theta_0 = (1, -1)`, administrative horizon `tau = 10`, exponential censoring with
mean 30 (about 21% censoring overall). The X effect multiplies `t`, so the
hazard is non-proportional in X; the Z effect is time-constant, so FLEXI-Haz is
correctly specified and Zhong's PH model is not. `n = 8000` training subjects,
200 replications (seeds `221100 + r`), one fixed test set of 10,000 subjects.

## Tuning

Both models are tuned on an independent tuning dataset of the same size as the
replications (seed 500000), separately for each sample size. Weight decay 0,
dropout 0.

- **FLEXI-Haz** (`tune_grid_cv.py`): five common subject-level folds (`KFold`,
  seed 221100). Each of the 27 architectures of Table 7 (depth 1-5, widths
  16-256; 135 fits) is trained on an outer training fold with the estimator's
  67/33 subject split for early stopping and scored by the negative
  log-likelihood on the outer held-out fold; the architecture with the smallest
  mean outer-fold NLL is selected. Learning rate fixed at `1e-3` (three-phase
  schedule inside each fit), max 2000 epochs, patience 15.
- **Zhong** (`tune_zhong_optuna.py`): 100 Optuna TPE trials over the same 27
  architectures and a learning rate sampled log-uniformly from `[1e-3, 5e-2]`.
  Each trial runs five-fold subject-level CV with the folds reshuffled between
  trials; the held-out fold is the early-stopping set and gives the fold's Cox
  negative partial log-likelihood; the trial value is the mean over folds, and
  all five fits must succeed. Fits use the comparator's recipe below.

## Replications (`run_replications.py`)

- **FLEXI-Haz**: tuned architecture, 67/33 subject split, batch 100,000
  expanded rows, learning-rate phases `1e-3 -> 1e-4 -> 1e-5` (best validation
  loss carried across phases, best weights restored), max 2000 epochs per phase,
  patience 30. Integration grid: n equally spaced points on [0, 10] plus all
  observed event and censoring times; time rescaled by 10. Covariate scaling
  and the Cox (X+Z) warm start of `theta` are computed from all subjects
  supplied to the fit. At each phase boundary `theta` is re-solved exactly with
  L-BFGS-B, the network held fixed, on the training rows. One initialization
  per replication (`--n-restarts` for more). Covariance from the internal training
  subjects (`n_tr = 5360` at n = 8000): per coordinate of Z a projection net
  with the main net's hidden widths regresses `Z_j` on `(T/10, X)` among
  events, 5 folds, Adam `1e-3`, batch 128, max 2000 epochs, patience 30, each
  held-out fold used for early stopping and prediction; with
  `R_i = Delta_i (Z_i - m(T_i, X_i))` on the standardized scale,
  `V_std = (1/n_tr) ((1/n_tr) sum R_i R_i' + 1e-8 I)^-1`, reported as
  `D^-1 V_std D^-1`.
- **Zhong** (`--zhong-plain`): tuned architecture and learning rate, one
  initialization, raw (unstandardized) covariates, Cox (X+Z) warm start on the
  training subjects, full-batch Adam at the constant tuned rate, max 2000
  epochs, patience 30, best validation weights restored, no `theta`
  refinement, no penalties. The SE uses the same residual-projection
  construction as FLEXI-Haz with the additive network `g_X(X) + g_T(t)` of
  Zhong et al. (no X-t interaction).

Each replication writes `theta_hat`, SE and 90% / 95% CI indicators per
coefficient, the restarts' diagnostics, and the prediction metrics on the fixed
test set (IBS, time-dependent AUC, C-index, survival bias).

## Running

``` bash
# 1. data (theta_0 = (1, -1) is the default; --n for other sample sizes)
python generate_simulation_datasets.py --output-dir outputs --n 8000

# 2. tuning (FLEXI-Haz grid can be split over machines with --arch-ids, then --combine)
python tune_grid_cv.py --output-dir outputs
python tune_grid_cv.py --output-dir outputs --combine
python tune_zhong_optuna.py --output-dir outputs

# 3. replications (split with --start-rep / --end-rep)
python run_replications.py --output-dir outputs --model-type flexihaz
python run_replications.py --output-dir outputs --model-type zhong --zhong-plain

# 4. tables
python aggregate_results.py --output-dir outputs
```

`aggregate_results.py` writes `outputs/results/combined/summary_theta_all_models.csv`
(mean estimate, bias, empirical SD, mean estimated SE, 90% / 95% coverage per
coefficient) and `summary_prediction_metrics_all_models.csv`.

## Reference

Zhong, Q., Mueller, J., and Wang, J.-L. (2022). Deep learning for the
partially linear Cox model. *The Annals of Statistics*, 50(3), 1348-1375.
