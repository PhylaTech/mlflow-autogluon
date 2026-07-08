# mlflow-autogluon

MLflow community model flavor and autologging for [AutoGluon](https://auto.gluon.ai) predictors.

AutoGluon has no built-in MLflow flavor, and the MLflow maintainers have asked for this
integration to live as a
[community flavor](https://mlflow.org/docs/latest/ml/community-model-flavors/)
(see [mlflow/mlflow#13214](https://github.com/mlflow/mlflow/issues/13214) and
[autogluon/autogluon#1404](https://github.com/autogluon/autogluon/issues/1404)).
This package provides that integration.

## Features

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

## At a glance

```python
import mlflow_autogluon
from autogluon.tabular import TabularPredictor

mlflow_autogluon.autolog()

predictor = TabularPredictor(label="target").fit(train_data, presets="medium_quality")
```

Every `fit` call now produces a fully populated MLflow run. See the
[Quickstart](quickstart.md) for a complete walkthrough.

## Compatibility

| Dependency | Supported versions | Verified in CI |
| --- | --- | --- |
| Python | >= 3.9 | 3.10, 3.11, 3.12 |
| MLflow | >= 2.15 | 2.22.x and 3.x |
| AutoGluon | >= 1.1 (tabular, timeseries, multimodal) | 1.5.x |

Install the extra matching your predictor type:

```bash
pip install mlflow-autogluon[tabular]      # TabularPredictor
pip install mlflow-autogluon[timeseries]   # TimeSeriesPredictor
pip install mlflow-autogluon[multimodal]   # MultiModalPredictor
```

## License

Apache License 2.0.
