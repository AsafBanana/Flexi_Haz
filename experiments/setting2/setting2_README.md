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
tune_grid_cv.py                   architecture (and, for Zhong, learning-rate) grid on an independent tuning dataset
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

## Tuning (`tune_grid_cv.py`)

One independent tuning dataset of size 8000 (seed 500000) and five common
subject-level folds (`KFold`, seed 221100). Each candidate is trained on an
outer training fold, with the estimator's 67/33 subject split for early
stopping, and scored by the negative log-likelihood on the outer held-out
fold; the configuration minimising the mean outer-fold NLL is selected.
Weight decay 0, dropout 0, no Optuna.

- **FLEXI-Haz**: the 27 architectures of Table 7 (depth 1-5, widths 16-256),
  learning rate fixed at `1e-3`, 135 fits, 3-phase schedule, max 2000 epochs,
  patience 15.
- **Zhong**: the same 27 architectures x Zhong's learning-rate grid
  `{1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1}`, 945 fits, trained as Zhong does
  (one constant-rate phase, full batch, max 5000 epochs, patience 35).

## Replications (`run_replications.py`)

- **FLEXI-Haz**: tuned architecture, learning-rate phases `1e-3 -> 1e-4 -> 1e-5`
  (best validation loss carried across phases), max 2000 epochs, patience 30,
  Cox (X+Z) warm start, exact `theta` refinement after training, 5 restarts
  (lowest validation loss kept), SE from 5-fold cross-fitted residual nets
  `Z - E[Z | X, t]` with one joint `(t, X)` network.
- **Zhong** (`--zhong-plain`): tuned architecture **and learning rate**, one
  constant-rate phase, no `theta` refinement, full-batch Breslow partial
  likelihood, 5 restarts, SE from the same residual construction with the
  additive network `g_X(X) + g_T(t)` (no X-t interaction), as in the paper.

Each replication writes `theta_hat`, SE and 90% / 95% CI indicators per
coefficient, the restarts' diagnostics, and the prediction metrics on the fixed
test set (IBS, time-dependent AUC, C-index, survival bias).

## Running

``` bash
# 1. data (theta_0 = (1, -1) is the default)
python generate_simulation_datasets.py --output-dir outputs

# 2. tuning (split over machines with --arch-ids / --arch-start --arch-end; then combine)
python tune_grid_cv.py --output-dir outputs --model-type flexihaz
python tune_grid_cv.py --output-dir outputs --model-type flexihaz --combine
python tune_grid_cv.py --output-dir outputs --model-type zhong
python tune_grid_cv.py --output-dir outputs --model-type zhong --combine

# 3. replications (split with --start-rep / --end-rep)
python run_replications.py --output-dir outputs --model-type flexihaz --n-restarts 5
python run_replications.py --output-dir outputs --model-type zhong --zhong-plain --n-restarts 5

# 4. tables
python aggregate_results.py --output-dir outputs
```

`aggregate_results.py` writes `outputs/results/combined/summary_theta_all_models.csv`
(mean estimate, bias, empirical SD, mean estimated SE, 90% / 95% coverage per
coefficient) and `summary_prediction_metrics_all_models.csv`.

## Reference

Zhong, Q., Mueller, J., and Wang, J.-L. (2022). Deep learning for the
partially linear Cox model. *The Annals of Statistics*, 50(3), 1348-1375.
