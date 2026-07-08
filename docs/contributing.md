# Contributing

Issues and pull requests are welcome at
[PhylaTech/mlflow-autogluon](https://github.com/PhylaTech/mlflow-autogluon).

## Development environment

The repo ships an `environment.yml` for a conda/mamba based dev environment:

```bash
mamba env create -f environment.yml
mamba run -n mlflow-autogluon pip install -e .
```

Plain virtualenv works too:

```bash
pip install -e .[dev]
```

## Running the checks

```bash
# Tests (fast: the suite trains tiny DUMMY-model predictors)
mamba run -n mlflow-autogluon pytest

# Coverage (CI enforces 90 percent)
mamba run -n mlflow-autogluon pytest --cov=mlflow_autogluon --cov-fail-under=90

# Lint
mamba run -n mlflow-autogluon ruff check mlflow_autogluon tests
```

CI runs the suite on Python 3.10 to 3.12 against the latest MLflow, plus a job pinned
to `mlflow<3` to guard the oldest supported line.

## Documentation

Docs are MkDocs Material, deployed to GitHub Pages on every push to `main`:

```bash
mamba run -n mlflow-autogluon mkdocs serve
```

## Roadmap

- Model signature inference during autologging
- ClearML integration (tracked separately)
