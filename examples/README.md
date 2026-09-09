# FLEXI-Haz Examples

These scripts provide small executable demonstrations of the public package API. They are intentionally separate from the full reproducibility experiments under `experiments/`.

Run them from the repository root after installing the package:

```bash
pip install -e .
```

Available examples:

- `partially_linear.py`: fit a model with flexible covariates `X` and linear covariates `Z`.
- `fully_nonparametric.py`: fit a model with all covariates in the flexible neural component.
- `inference.py`: estimate and summarize the partially linear coefficient.
- `tuning.py`: run a small Optuna hyperparameter search.

The synthetic datasets in these scripts are only for demonstrating the API. They are not the simulation study used in the report.
