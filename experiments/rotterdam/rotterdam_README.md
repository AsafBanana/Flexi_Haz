# Rotterdam Breast-Cancer Experiment

This directory contains the real-data experiment used to evaluate
FLEXI-Haz on the Rotterdam breast-cancer dataset.

The final reported analysis uses an **ER-interaction design** to
estimate separate hormonal-therapy and chemotherapy effects for patients
with low and high estrogen-receptor (ER) levels while allowing the
remaining clinical covariates to enter flexibly.

## Dataset and endpoint

The dataset contains **2,982 patients**.

The survival outcome is defined as the time from primary surgery to the
earlier of breast-cancer recurrence or death:

``` python
T = np.minimum(rtime, dtime)
delta = ((recur == 1) | (death == 1)).astype(int)
```

The `year` variable is excluded from the modeling design.

## Final ER-interaction design

Patients are divided using the threshold:

``` text
ER = 61
```

The partially linear treatment component is

\[ Z=
```{=tex}
\begin{pmatrix}
\text{hormon}\,I(\text{ER}<61)\\
\text{hormon}\,I(\text{ER}\ge61)\\
\text{chemo}\,I(\text{ER}<61)\\
\text{chemo}\,I(\text{ER}\ge61)
\end{pmatrix}
```
. \]

In the implementation these columns are named:

``` text
hormon_low_er
hormon_high_er
chemo_low_er
chemo_high_er
```

The flexible covariates contain:

``` text
age
nodes
pgr
er
meno
grade
tumor size
```

Tumor size is one-hot encoded. The remaining variables retain their
numerical representations.

For the fully nonparametric FLEXI-Haz model, the treatment and clinical
covariates are combined and modeled through the flexible neural
component.

## Compared models

The experiment evaluates:

1.  partially linear FLEXI-Haz;
2.  fully nonparametric FLEXI-Haz;
3.  Zhong et al. (2022);
4.  Cox proportional hazards;
5.  Random Survival Forest.

For partially linear FLEXI-Haz and Zhong et al. (2022), the four
ER-specific treatment terms form the linear component.

## Evaluation protocol

For FLEXI-Haz, Zhong et al. (2022), and Random Survival Forest:

-   five outer folds are generated once and shared across methods;
-   within each outer training fold, hyperparameters are selected using
    an internal 80/20 train-validation split;
-   the selected configuration is fitted on the outer training data and
    evaluated on the held-out outer fold.

Evaluation times are generated every 50 days over the observed follow-up
range.

The Cox PH results reported in the project use the full dataset with the
same ER-interaction design rather than the five-fold predictive
protocol.

## Files

``` text
rotterdam_breast_cancer.csv
    Rotterdam breast-cancer data used by the experiment.

rotterdam_io.py
    Dataset loading, endpoint construction, ER-interaction design,
    outer folds, inner splits, and shared result paths.

rotterdam_metrics.py
    IPCW predictive metrics, FLEXI-Haz/Cox treatment-effect
    extraction, and RSF survival prediction utilities.

tune_flexihaz_rotterdam.py
    Per-fold tuning for partially linear and fully nonparametric FLEXI-Haz.

tune_zhong2022_rotterdam.py
    Per-fold tuning for Zhong et al. (2022).

tune_rsf_rotterdam.py
    Per-fold Random Survival Forest tuning.

zhong2022_ph_common.py
    Shared Zhong-model implementation and survival-prediction utilities.

run_flexihaz_pl_rotterdam.py
    Partially linear FLEXI-Haz outer-fold evaluation.

run_flexihaz_nonlinear_rotterdam.py
    Fully nonparametric FLEXI-Haz outer-fold evaluation.

run_zhong2022_rotterdam.py
    Zhong et al. outer-fold evaluation.

run_rsf_rotterdam.py
    Random Survival Forest outer-fold evaluation.

run_cox_ph_rotterdam.py
    Cox PH fitting and ER-specific treatment-effect estimation.

run_all_rotterdam.py
    Orchestrates the complete experiment.

aggregate_rotterdam_results.py
    Aggregates fold-level metrics and treatment effects.

analyze_results_er_interaction.ipynb
    Analysis used to summarize and visualize the final ER-interaction results.
```

Development-only or obsolete files such as old Cox runners, placeholder
Zhong scripts, old non-interaction analyses, cluster logs, and
smoke-test result artifacts are not part of the final reported
experiment and should not be included in the polished public version.

## Installation

Install FLEXI-Haz from the repository root:

``` bash
pip install -e .
```

The comparison models additionally require packages including:

``` bash
pip install lifelines scikit-survival tensorflow
```

## Running the final experiment

The final analysis should use:

``` text
design = er_interaction
```

and write to the final Rotterdam result directory.

A complete run can be orchestrated with `run_all_rotterdam.py`. Use the
script's command-line options to set the final tuning and training
budgets used in the report:

``` text
FLEXI-Haz / Zhong tuning trials = 100 per outer fold
RSF tuning                   = per outer fold
final neural max epochs      = 2000
projection max epochs        = 2000
outer folds                  = 5
```

The neural-model search space consists of one- to five-hidden-layer ReLU
networks with widths between 16 and 256 and learning rate in
`[1e-3, 5e-2]`. FLEXI-Haz uses `grid_size="n"` and zero dropout in the
reported experiment.

Because the scripts expose command-line arguments, inspect `--help`
before launching a long run:

``` bash
python run_all_rotterdam.py --help
```

Individual folds can also be distributed across jobs using the
`--start-fold` and `--end-fold` arguments supported by the model
runners.

## Predictive metrics

The experiment reports:

-   integrated Brier score;
-   mean Brier score;
-   time-dependent AUC;
-   mean AUC;
-   concordance index.

For the four cross-validated models, aggregate values are reported as
mean and standard deviation across the five outer folds.

## Treatment-effect inference

For partially linear FLEXI-Haz, Zhong et al. (2022), and Cox PH, the
experiment reports ER-specific log-hazard coefficients, standard errors,
95% confidence intervals, and corresponding hazard ratios.

The final reported coefficient estimates are:

  Treatment      ER group      FLEXI-Haz   Zhong et al. (2022)       Cox PH
  -------------- ---------- ------------ --------------------- ------------
  Hormonal       Low              -0.190                 0.093        0.083
  Hormonal       High         **-0.496**                -0.159   **-0.306**
  Chemotherapy   Low          **-0.309**                -0.029       -0.074
  Chemotherapy   High         **-0.352**                -0.132       -0.098

Bold FLEXI-Haz/Cox entries correspond to reported 95% confidence
intervals that exclude zero.

## Reported predictive results

  Model                                 IBS   Mean Brier   Mean AUC   C-index
  -------------------------------- -------- ------------ ---------- ---------
  FLEXI-Haz, partially linear        0.1959       0.1956     0.7258    0.6574
  FLEXI-Haz, fully nonparametric     0.2013       0.2011     0.7151    0.6559
  Zhong et al. (2022)                0.2141       0.2138     0.6122    0.5825
  Random Survival Forest             0.1922       0.1919     0.7308    0.6604
  Cox PH                             0.1861       0.1854     0.7181    0.6680

The first four rows are out-of-sample means across the five outer folds.
The Cox PH row corresponds to the full-data Cox analysis used in the
report and therefore should not be interpreted as directly comparable
cross-validated predictive performance.

## Output structure

The final result directory contains:

``` text
data_metadata.json
folds.csv
aggregated/
analysis_outputs/
tuning/
flexihaz_partial_linear/
flexihaz_nonlinear/
zhong2022_ph/
random_survival_forest/
cox_ph/
```

Important aggregated files include:

``` text
summary_by_fold.csv
summary_by_model.csv
brier_by_time.csv
brier_by_time_summary.csv
auc_by_time.csv
auc_by_time_summary.csv
treatment_effects_by_fold.csv
treatment_effects_summary.csv
```

Human-readable CSV/JSON outputs and final figures are suitable for
version control. Cluster logs, temporary smoke-test outputs, serialized
Optuna `.pkl` studies, and obsolete result directories should be
excluded.

## Exhaustive architecture grid CV (`run_rotterdam_grid_cv.py`)

`run_rotterdam_grid_cv.py` runs the treatment-effect analysis with the
same tuning protocol for FLEXI-Haz and for the Zhong et al. (2022)
proportional-hazards DNN (both via `FlexiHazardModel`, `--model-type
flexihaz|zhong`):

- outer folds from `results/folds.csv`, fit on each fold's training rows;
- per fold, an exhaustive grid over depth L in {2, 3, 4, 5, 6} and width
  w in {25, 50, 100, 150, 200, 300, 500} (35 architectures,
  `hidden_layers = (w,) * L`) at a fixed learning rate (default 1e-3),
  weight decay 0, dropout 0; each candidate is scored by its validation
  loss on the estimator's internal 67%/33% subject split and the argmin
  is selected;
- the selected architecture is refitted with `--n-restarts` (default 5)
  random restarts, keeping the lowest-validation-loss restart; theta, SE
  (joint-(t, X) residual nets for FLEXI-Haz, additive g_X(X) + g_T(t)
  nets for Zhong), 90% and 95% CIs are written per fold and averaged
  across folds.

``` bash
python run_rotterdam_grid_cv.py --model-type flexihaz --design er_interaction --theta-lr-mult 20
python run_rotterdam_grid_cv.py --model-type zhong --design er_interaction --zhong-plain
```

`--zhong-plain` reproduces Zhong's own training recipe (one constant-lr
phase, no profile-likelihood refinement of theta); without it the
3-phase schedule and the refinement are used for both models.  Outputs
go to `results/<model_type>_gridcv[_thetalr<M>][_plain]/<design>/`
(`fold_XX_grid.csv`, `fold_XX_treatment_effects.csv`,
`fold_XX_log.json`, `summary_treatment_effects.csv`).

## Analysis notebook

`analyze_results_er_interaction.ipynb` summarizes the final
ER-interaction experiment and generates report-level tables and figures.
The older non-interaction analysis notebook is not part of the final
reported analysis.

## Reference

Zhong, Q., Mueller, J., and Wang, J.-L. (2022). Deep learning for the
partially linear Cox model. *The Annals of Statistics*, 50(3),
1348-1375.
