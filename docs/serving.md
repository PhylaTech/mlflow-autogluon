# Serving

Models logged with the `autogluon` flavor include a `python_function` flavor, so all
standard MLflow deployment targets work.

## Local REST server

```bash
mlflow models serve -m "models:/my-model/1" -p 5001
```

Score with the standard `/invocations` payloads:

```bash
curl -s http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{
        "dataframe_split": {
          "columns": ["num_a", "num_b", "cat_a"],
          "data": [[0.12, -1.3, "x"], [0.5, 0.7, "z"]]
        }
      }'
```

```json
{"predictions": [1, 0]}
```

## Class probabilities over REST

Pass the `predict_method` inference param in the payload:

```bash
curl -s http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{
        "dataframe_split": {
          "columns": ["num_a", "num_b", "cat_a"],
          "data": [[0.12, -1.3, "x"]]
        },
        "params": {"predict_method": "predict_proba"}
      }'
```

```json
{"predictions": [{"0": 0.31, "1": 0.69}]}
```

## Batch scoring from the CLI

```bash
mlflow models predict \
  -m runs:/<run_id>/model \
  -i input.json \
  -o predictions.json \
  --content-type json \
  --env-manager local
```

## Environment reproduction

The logged model pins `autogluon.tabular` to the training version in
`requirements.txt`. With the default `--env-manager virtualenv`, MLflow rebuilds that
environment before serving; `--env-manager local` reuses the current one.

!!! tip
    The test suite exercises this exact path: `tests/test_serving.py` runs
    `mlflow models predict` against a saved model, including the
    `predict_proba` param route, on every CI build.
