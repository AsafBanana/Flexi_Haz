# Contributing to FLEXI-Haz

FLEXI-Haz is research software. Contributions that improve correctness, numerical stability, testing, documentation, reproducibility, or usability are welcome.

## Development setup

Clone the repository and install the package in editable mode with development dependencies:

```bash
pip install -e ".[dev]"
```

To run the full test suite:

```bash
pytest
```

The current test suite includes both fast smoke tests and broader regression/robustness tests.

## Repository layout

Reusable package code belongs in:

```text
src/flexihaz/
```

Research-specific code belongs under:

```text
experiments/
```

Small user-facing demonstrations belong under:

```text
examples/
```

Avoid adding generated datasets, checkpoints, temporary logs, Optuna databases, or other large reproducible artifacts to version control.

## Pull requests

Before submitting a pull request:

1. Run `pytest`.
2. Add or update tests for behavioral changes.
3. Update documentation for public API changes.
4. Keep experiment-specific logic out of the reusable package unless it is genuinely part of the FLEXI-Haz API.
5. Do not silently change reported experimental configurations or results.

For changes that affect the statistical methodology or inference procedure, describe the mathematical and computational consequences explicitly in the pull request.
