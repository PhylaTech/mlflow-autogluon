# Model flavor

The `autogluon` flavor stores the predictor via `TabularPredictor.clone()` inside the
MLflow model directory, together with pinned requirements and a `python_function`
flavor for generic inference.

## Logging a model manually

```python
import mlflow
import mlflow_autogluon

with mlflow.start_run():
    model_info = mlflow_autogluon.log_model(
        ag_model=predictor,
        artifact_path="model",
        input_example=train_data.drop(columns=["target"]).head(),
        registered_model_name="my-model",   # optional
    )
```

On MLflow 3.x you can use the newer naming convention instead:

```python
model_info = mlflow_autogluon.log_model(ag_model=predictor, name="model")
```

## Saving to a local path

```python
mlflow_autogluon.save_model(ag_model=predictor, path="my_model")
```

The resulting directory is a standard MLflow model:

```text
my_model/
    MLmodel
    ag_model/           # cloned AutoGluon predictor
    conda.yaml
    python_env.yaml
    requirements.txt
```

`requirements.txt` pins the exact `autogluon.tabular` version used for training, so
serving environments reproduce the training environment.

## Loading

```python
# Native predictor: full AutoGluon API (leaderboard, feature_importance, ...)
predictor = mlflow_autogluon.load_model("models:/my-model/1")

# Generic pyfunc: uniform predict() interface
pyfunc_model = mlflow.pyfunc.load_model("models:/my-model/1")
```

## PyFunc semantics

- `pyfunc_model.predict(df)` returns a numpy array of predictions, matching the
  behavior of built-in flavors such as `mlflow.sklearn`.
- `pyfunc_model.predict(df, params={"predict_method": "predict_proba"})` returns the
  class-probability DataFrame for classifiers.

The `predict_method` inference param is declared in the model signature at save time,
so it also works through REST serving. Signatures you pass explicitly are preserved;
the params schema is added only when missing.

## Signatures and input examples

```python
from mlflow.models import infer_signature

signature = infer_signature(features, predictor.predict(features))
mlflow_autogluon.log_model(
    ag_model=predictor,
    artifact_path="model",
    signature=signature,
    input_example=features.head(),
)
```
