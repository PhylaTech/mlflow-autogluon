# Autologging

`mlflow_autogluon.autolog()` patches the `fit` method of every installed AutoGluon
predictor type (`TabularPredictor`, `TimeSeriesPredictor`, `MultiModalPredictor`)
using MLflow's own autologging machinery (`safe_patch`), so it behaves like the
built-in integrations: runs are created automatically when none is active,
user-created runs are reused, and a logging failure never breaks your training call.

## Options

```python
mlflow_autogluon.autolog(
    log_models=True,             # log the fitted predictor as an MLflow model
    log_leaderboard=True,        # leaderboard CSV artifact + per-model metrics
    log_fit_summary=False,       # fit_summary() as a JSON artifact
    registered_model_name=None,  # also register logged models under this name
    extra_tags=None,             # extra tags applied to autologged runs
    disable=False,               # turn the integration off
    silent=False,                # suppress autologging warnings
)
```

## What gets logged

### Parameters

All non-data arguments of the `fit` call, plus predictor configuration:

- `label`, `problem_type`, `eval_metric` from the predictor (`target` and
  `prediction_length` for timeseries predictors)
- `presets`, `time_limit`, `hyperparameters`, `num_bag_folds`, `num_stack_levels`,
  and any other keyword arguments passed to `fit`
- `train_rows` and `train_columns` when the training data has a shape

Dict and list values (such as `hyperparameters`) are JSON-encoded and truncated to
500 characters to stay within MLflow param limits.

### Metrics

- `fit_time_seconds`: wall-clock duration of the `fit` call
- `best_model_score_val`: validation score of the best model
- every numeric leaderboard column per model, e.g. `score_val_<model>`,
  `fit_time_<model>`, `pred_time_val_<model>` (column sets vary by predictor type)

`MultiModalPredictor` has no leaderboard, so leaderboard metrics and the CSV
artifact are skipped for it.

### Tags

- `estimator_name` (`TabularPredictor`)
- `autogluon_version`
- `best_model` and `problem_type` after training completes
- anything passed via `extra_tags`

### Artifacts

- `leaderboard.csv`: the full leaderboard
- `fit_summary.json` when `log_fit_summary=True`
- `model/`: the fitted predictor logged with the `autogluon` flavor
  (disable with `log_models=False`)

## Registering models automatically

```python
mlflow_autogluon.autolog(registered_model_name="churn-classifier")
```

Every subsequent `fit` logs the predictor and creates a new version of
`churn-classifier` in the model registry.

## Using your own runs

Autologging respects an already-active run:

```python
import mlflow

with mlflow.start_run(run_name="experiment-42"):
    predictor.fit(train_data)          # logs into experiment-42
    mlflow.log_param("my_param", 1)    # your own logging works alongside
```

!!! warning
    Calling `fit` twice inside the same run logs parameters twice. MLflow rejects
    conflicting param values, and mlflow-autogluon downgrades that rejection to a
    warning so training is never interrupted; the second fit's params are simply not
    recorded. Use one run per fit when you need full param capture.

## Disabling

```python
mlflow_autogluon.autolog(disable=True)
```
