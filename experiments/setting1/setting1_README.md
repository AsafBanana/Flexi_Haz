# Setting 1: six-setting simulation (PH / non-PH x three Z-X correlations)

Treatment-effect estimation with partially linear **FLEXI-Haz** versus the
proportional-hazards DNN of **Zhong, Mueller & Wang (2022)** under six
data-generating settings, `gamma in {0, 0.5}` (PH / non-PH) times
`rho in {0, 0.5, 0.75}` (R^2 of Z given X), 30% censoring, 200 replications
per setting, at `n = 8000` and `n = 4000`. The target is the scalar coefficient
`theta_0 = log 1.5 = 0.4055`.

## Files

``` text
dgm_additional.py        data-generating mechanism and censoring calibration
common.py                shared constants (seeds, lr phases) and FLEXI-Haz's exact theta refinement
crossfit_residuals.py    cross-fitted Z - E[Z | X, t] residuals (FLEXI-Haz score and SE)
tune_flexihaz_grid.py    FLEXI-Haz main-net and projection-net grids (35 archs each, fixed lr) on the tuning dataset
run_flexihaz_reps.py     FLEXI-Haz replications: warm-started lr phases, 5 restarts, efficient-score selection
zhong_ph_prod.py         Zhong model pieces: CoxPHDNN, Breslow loss, additive residual net, profile-likelihood refinement
tune_zhong_grid.py       Zhong architecture x learning-rate grid (35 x 7) on the tuning dataset
run_zhong_reps.py        Zhong replications: 3 lr phases, 5 restarts, val-loss selection, refinement
aggregate_setting1.py    one table: theta, bias, empirical SD, model SE, 90/95% coverage
```

Outputs go to `outputs/` (git-ignored).

## Data-generating mechanism

`X_1..X_5 ~ U(0,1)`, `X_6 ~ Bernoulli(1/2)`; with `s(X) = sqrt(12/5) sum_j (X_j - 1/2)`,
`Z = sqrt(rho) s(X) + sqrt(1 - rho) eps`, `eps ~ N(0,1)`, so `Var(Z) = 1` and
`R^2(Z | X) = rho`. Hazard

``` text
lambda(t | X, Z) = 0.05 exp{ t exp(gamma a(X)) + b(X) + theta_0 Z }
```

with `a(X)`, `b(X)` pure interaction terms of the centred covariates (`--k 3`),
scaled so the log-hazard variance splits PH / non-PH / linear as 70/0/30 when
`gamma = 0` and 35/35/30 when `gamma = 0.5`. `gamma = 0` is proportional hazards;
`gamma = 0.5` makes the X effect time-varying. Censoring: `C = c U`, with `c`
calibrated by Monte Carlo per setting to 30% censoring (computed by the tuners
and stored in their JSON output, reused by the replication runners).
Replication `r` uses seed `221100 + r`; the tuning dataset uses seed 500000.

## Models and training

**FLEXI-Haz** (`run_flexihaz_reps.py`): `h(t | X, Z) = exp{theta Z + g(t, X)}`,
piecewise-exponential likelihood on the expanded data (grid of all observed
times), ReLU net, weight decay 0, dropout 0, standardized inputs, Cox (X+Z)
warm start of `theta`, validation fraction 0.33 by subject.

- warm-started learning-rate phases, at most 500 epochs each, patience 30 on
  the validation NLL, best validation loss carried across phases:
  `n = 8000`: three phases `1e-3 -> 1e-4 -> 1e-5`, `theta` trained at the
  network rate; `n = 4000`: four phases `1e-3 -> 1e-4 -> 1e-5 -> 1e-6` with the
  **theta learning rate x20** the phase rate (`--theta-lr-mult 20`), so that the
  linear branch is not frozen by the small network rate on the smaller sample;
- after each phase `theta` is re-solved exactly (L-BFGS on the training rows with
  `g` fixed, `common.refine_theta`);
- minibatch of 100,000 expanded rows at `n = 8000`; at `n = 4000`
  ceil(#expanded training rows / 213) (`--main-batches-per-epoch 213`), which
  keeps the same number of optimizer steps per epoch;
- 5 random restarts (seeds 910000-910004); the restart whose **efficient score**
  (orthogonalised score built from cross-fitted residuals `Z - E[Z | X, t]`) is
  closest to zero is reported;
- SE from the same cross-fitted residuals: `Var(theta) = (R'R)^-1` over event rows,
  5 subject folds, projection nets Adam 1e-3, at most 300 epochs, patience 30.

**Zhong et al. (2022)** (`run_zhong_reps.py`): `h(t | X, Z) = h_0(t) exp{theta Z + g(X)}`,
Breslow partial likelihood, full batch, at most 5000 epochs, patience 35,
standardized inputs, Cox (X+Z) warm start.

- learning-rate phases `lr -> lr/10 -> lr/100` with the tuned `lr`;
- 5 random restarts; the one with the lowest validation partial likelihood is kept;
- `theta` is refined by maximising the Breslow profile likelihood with `g` fixed
  (`theta_hat_refined`; the jointly trained coefficient is kept as
  `theta_hat_unrefined`);
- SE from 5-fold cross-fitted residuals of the **additive** residual network
  `g_X(X) + g_T(t)` (no X-t interaction), `Var(theta) = (R'R)^-1` over event
  subjects, as in the paper.

## Tuning

One tuning dataset of size 8000 per setting (seed 500000), 67/33 subject split
for early stopping.

- FLEXI-Haz main net: depth `{2,3,4,5,6}` x width `{25,50,100,150,200,300,500}`
  (35 architectures) with the 3-phase schedule, learning rate not searched,
  scored by validation NLL. The landscape is flat, so the rule is: among
  configurations within 1% of the minimum, the one with the fewest layers
  (`--prefer-fewer-layers`).
- FLEXI-Haz projection net: its own search over the same 35 architectures,
  scored by validation MSE of the regression of Z on (X, t) at the training
  subjects' event times. It does not reuse the main-net architecture.
- Zhong: the same 35 architectures x learning rates
  `{1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1}`, argmin of the validation
  partial likelihood; the residual net reuses the selected architecture, as
  in the paper.

The tuners write their selections (and the calibrated censoring constant) to
`outputs/tuning_flexihaz_<tag>.json` / `outputs/tuning_zhong_<tag>.json`, which
the replication runners take through `--config`.

## Running

Install the package (`pip install -e .` from the repository root; or set
`FLEXIHAZ_SOURCE_DIR=<repo>/src`). All commands from this directory; one call
per setting, `gamma in {0.0, 0.5} x rho in {0.0, 0.5, 0.75}` (shown for
`gamma = 0.5, rho = 0.5`). Replications can be split with `--rep-start` /
`--n-reps`.

``` bash
# tuning (one tuning dataset of 8000 per setting)
python tune_flexihaz_grid.py --gamma 0.5 --rho 0.5 --censoring-rate 0.3 --k 3 --n 8000 --prefer-fewer-layers
python tune_zhong_grid.py    --gamma 0.5 --rho 0.5 --censoring-rate 0.3 --k 3 --n 8000

# n = 8000: FLEXI-Haz with three lr phases (1e-3, 1e-4, 1e-5), theta at the network rate
python run_flexihaz_reps.py --gamma 0.5 --rho 0.5 --censoring-rate 0.3 --k 3 --n 8000 \
    --n-reps 200 --n-init 5 --config outputs/tuning_flexihaz_gamma0.5_rho0.5_cens30_k3_n8000.json

# n = 4000: FLEXI-Haz with four lr phases and theta learning rate x20 (same tuning file)
python run_flexihaz_reps.py --gamma 0.5 --rho 0.5 --censoring-rate 0.3 --k 3 --n 4000 \
    --n-reps 200 --n-init 5 --theta-lr-mult 20 --lr-phases 1e-3,1e-4,1e-5,1e-6 \
    --main-batches-per-epoch 213 --config outputs/tuning_flexihaz_gamma0.5_rho0.5_cens30_k3_n8000.json

# Zhong, both sample sizes (same recipe)
python run_zhong_reps.py --gamma 0.5 --rho 0.5 --censoring-rate 0.3 --k 3 --n 8000 \
    --n-reps 200 --config outputs/tuning_zhong_gamma0.5_rho0.5_cens30_k3_n8000.json
python run_zhong_reps.py --gamma 0.5 --rho 0.5 --censoring-rate 0.3 --k 3 --n 4000 \
    --n-reps 200 --config outputs/tuning_zhong_gamma0.5_rho0.5_cens30_k3_n8000.json

python aggregate_setting1.py          # -> outputs/setting1_summary.csv (one row per model, setting and n)
```

Per-replication CSVs record the selected restart, all restarts' estimates,
scores and validation losses, the one-step and root efficient-score
corrections (FLEXI-Haz), refined and unrefined `theta` (Zhong), the SE and the
fit time. `aggregate_setting1.py` reports, per model and setting, the number of
replications, mean `theta_hat`, bias in %, empirical SD, mean model SE and
90% / 95% coverage of `theta_0` (normal CIs with the model SE).

## Reference

Zhong, Q., Mueller, J., and Wang, J.-L. (2022). Deep learning for the
partially linear Cox model. *The Annals of Statistics*, 50(3), 1348-1375.
