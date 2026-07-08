# mlflow-autogluon

MLflow community model flavor and autologging for [AutoGluon](https://auto.gluon.ai) predictors.

AutoGluon has no built-in MLflow flavor, and the MLflow maintainers have asked for this
integration to live as a [community flavor](https://mlflow.org/docs/latest/ml/community-model-flavors/)
(see [mlflow/mlflow#13214](https://github.com/mlflow/mlflow/issues/13214) and
[autogluon/autogluon#1404](https://github.com/autogluon/autogluon/issues/1404)).
This package provides that integration:

- **Model flavor**: `save_model`, `log_model`, and `load_model` for
  `autogluon.tabular.TabularPredictor`, with full round-trip fidelity.
- **PyFunc support**: logged models load with `mlflow.pyfunc.load_model` and serve with
  `mlflow models serve`, including `predict_proba` via inference params.
- **Autologging**: one call to `mlflow_autogluon.autolog()` records params, leaderboard
  metrics, artifacts, and the fitted predictor for every `fit` call.

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

- Python >= 3.9
- MLflow >= 2.15 (including MLflow 3.x)
- AutoGluon >= 1.1 (`TabularPredictor`; other predictor types are on the roadmap)

## Roadmap

- `TimeSeriesPredictor` and `MultiModalPredictor` support
- Model signature inference during autologging
- ClearML integration (tracked separately)

## Development

The repo ships an `environment.yml` for a conda/mamba based dev environment:

```bash
mamba env create -f environment.yml
mamba run -n mlflow-autogluon pip install -e .
mamba run -n mlflow-autogluon pytest
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
