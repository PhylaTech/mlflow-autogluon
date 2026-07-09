# mlflow-autogluon

[![CI](https://github.com/PhylaTech/mlflow-autogluon/actions/workflows/ci.yml/badge.svg)](https://github.com/PhylaTech/mlflow-autogluon/actions/workflows/ci.yml)
[![Coverage](https://phylatech.github.io/mlflow-autogluon/badges/coverage.svg)](https://github.com/PhylaTech/mlflow-autogluon/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/mlflow-autogluon.svg)](https://pypi.org/project/mlflow-autogluon/)
[![Docs](https://github.com/PhylaTech/mlflow-autogluon/actions/workflows/docs.yml/badge.svg)](https://phylatech.github.io/mlflow-autogluon/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://github.com/PhylaTech/mlflow-autogluon)

MLflow community model flavor and autologging for [AutoGluon](https://auto.gluon.ai) predictors.

**Documentation: [phylatech.github.io/mlflow-autogluon](https://phylatech.github.io/mlflow-autogluon/)**

AutoGluon has no built-in MLflow flavor, and the MLflow maintainers have asked for this
integration to live as a [community flavor](https://mlflow.org/docs/latest/ml/community-model-flavors/)
(see [mlflow/mlflow#13214](https://github.com/mlflow/mlflow/issues/13214) and
[autogluon/autogluon#1404](https://github.com/autogluon/autogluon/issues/1404)).
This package provides that integration:

- **Model flavor**: `save_model`, `log_model`, and `load_model` for
  `TabularPredictor`, `TimeSeriesPredictor`, and `MultiModalPredictor`, with full
  round-trip fidelity.
- **PyFunc support**: logged models load with `mlflow.pyfunc.load_model` and serve with
  `mlflow models serve`, including `predict_proba` via inference params and long-format
  DataFrame forecasting for timeseries models.
- **Autologging**: one call to `mlflow_autogluon.autolog()` records params, leaderboard
  metrics, artifacts, and the fitted predictor for every `fit` call, across all
  installed predictor types.

## Installation

```bash
pip install mlflow-autogluon[tabular]
```

## Autologging quickstart

```python
import mlflow_autogluon
from autogluon.tabular import TabularPredictor

mlflow_autogluon.autolog()

predictor = TabularPredictor(label="target").fit(train_data, presets="medium_quality")
```

That single `autolog()` call gives you, per `fit`:

| What | Examples |
| --- | --- |
| Params | `label`, `problem_type`, `eval_metric`, `presets`, `time_limit`, `hyperparameters`, `train_rows` |
| Metrics | `best_model_score_val`, `fit_time_seconds`, per-model `score_val_*` / `fit_time_*` |
| Tags | `estimator_name`, `autogluon_version`, `best_model`, `problem_type` |
| Artifacts | `leaderboard.csv`, optional `fit_summary.json`, the fitted model |

Because this is a community flavor, `mlflow.autolog()` does not enable it automatically;
call `mlflow_autogluon.autolog()` explicitly. Options:

```python
mlflow_autogluon.autolog(
    log_models=True,             # log the fitted predictor as an MLflow model
    log_leaderboard=True,        # leaderboard CSV artifact + per-model metrics
    log_fit_summary=False,       # fit_summary() as a JSON artifact
    registered_model_name=None,  # also register logged models under this name
    extra_tags=None,             # extra tags applied to autologged runs
    disable=False,               # turn the integration off
)
```

## Manual logging

```python
import mlflow
import mlflow_autogluon

with mlflow.start_run():
    model_info = mlflow_autogluon.log_model(
        ag_model=predictor,
        artifact_path="model",
        input_example=train_data.drop(columns=["target"]).head(),
    )

# Load back the native predictor
predictor = mlflow_autogluon.load_model(model_info.model_uri)

# Or load as a generic pyfunc for serving
pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
predictions = pyfunc_model.predict(test_data)
probabilities = pyfunc_model.predict(test_data, params={"predict_method": "predict_proba"})
```

Serving works out of the box:

```bash
mlflow models serve -m "models:/<name>/<version>" -p 5001
```

## Compatibility

- Python >= 3.9 (CI: 3.10, 3.11, 3.12)
- MLflow >= 2.15 (CI: latest 3.x and the 2.x line)
- AutoGluon >= 1.1: `TabularPredictor` (`[tabular]` extra), `TimeSeriesPredictor`
  (`[timeseries]`), and `MultiModalPredictor` (`[multimodal]`)

See [examples/autolog_quickstart.py](https://github.com/PhylaTech/mlflow-autogluon/blob/main/examples/autolog_quickstart.py)
for a complete runnable walkthrough on a real dataset.

## Roadmap

- Model signature inference during autologging
- ClearML integration (tracked separately)

## Development

With [pixi](https://pixi.sh) (recommended):

```bash
pixi install
pixi run test           # tabular test suite
pixi run lint           # ruff
pixi run docs           # live documentation server
pixi run -e full coverage   # full suite with the 100 percent gate
```

Or with conda/mamba via the shipped `environment.yml`:

```bash
mamba env create -f environment.yml
mamba run -n mlflow-autogluon pip install -e .
mamba run -n mlflow-autogluon pytest
```

See the [contributing guide](https://phylatech.github.io/mlflow-autogluon/contributing/)
for the full task list and release process.

## License

Apache License 2.0. See [LICENSE](https://github.com/PhylaTech/mlflow-autogluon/blob/main/LICENSE).
