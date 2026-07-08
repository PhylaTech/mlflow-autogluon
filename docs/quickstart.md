# Quickstart

## Install

```bash
pip install mlflow-autogluon[tabular]
```

The `tabular` extra pulls in `autogluon.tabular`. If AutoGluon is already installed,
plain `pip install mlflow-autogluon` is enough.

## Autologging in three lines

```python
import mlflow_autogluon
from autogluon.tabular import TabularPredictor

mlflow_autogluon.autolog()

predictor = TabularPredictor(label="target").fit(train_data)
```

That is the whole integration. Every `fit` call creates (or reuses) an MLflow run and
records:

| What | Examples |
| --- | --- |
| Params | `label`, `problem_type`, `eval_metric`, `presets`, `time_limit`, `hyperparameters`, `train_rows` |
| Metrics | `best_model_score_val`, `fit_time_seconds`, per-model `score_val_*` / `fit_time_*` |
| Tags | `estimator_name`, `autogluon_version`, `best_model`, `problem_type` |
| Artifacts | `leaderboard.csv`, optional `fit_summary.json`, the fitted model |

!!! note
    Because this is a community flavor, `mlflow.autolog()` does not enable it
    automatically. Call `mlflow_autogluon.autolog()` explicitly.

## Load the model back

```python
import mlflow
import mlflow_autogluon

run_id = mlflow.last_active_run().info.run_id

# As the native AutoGluon predictor
predictor = mlflow_autogluon.load_model(f"runs:/{run_id}/model")

# Or as a generic pyfunc
pyfunc_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
predictions = pyfunc_model.predict(test_data)
```

## Where to go next

- [Autologging](autologging.md) for all configuration options
- [Model flavor](model-flavor.md) for manual `log_model` / `save_model` workflows
- [Serving](serving.md) for REST scoring with `mlflow models serve`
