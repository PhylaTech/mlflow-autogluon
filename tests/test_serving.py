"""End-to-end scoring tests through MLflow's serving machinery.

These run the same code path as ``mlflow models serve`` / ``mlflow models
predict`` (scoring payload parsing, pyfunc dispatch, JSON serialization),
so they catch integration breakage that unit tests miss.
"""

import json
import subprocess
import sys

import pytest

import mlflow_autogluon


@pytest.fixture(scope="module")
def saved_model_path(fitted_predictor, tmp_path_factory):
    path = tmp_path_factory.mktemp("serving") / "model"
    mlflow_autogluon.save_model(ag_model=fitted_predictor, path=str(path))
    return path


def _cli_predict(model_path, payload, tmp_path):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text(json.dumps(payload))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mlflow",
            "models",
            "predict",
            "-m",
            str(model_path),
            "-i",
            str(input_path),
            "-o",
            str(output_path),
            "--content-type",
            "json",
            "--env-manager",
            "local",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return json.loads(output_path.read_text())


def test_cli_predict(saved_model_path, test_data, fitted_predictor, tmp_path):
    payload = {"dataframe_split": test_data.to_dict(orient="split")}
    result = _cli_predict(saved_model_path, payload, tmp_path)

    predictions = result["predictions"]
    assert len(predictions) == len(test_data)
    assert predictions == fitted_predictor.predict(test_data).tolist()


def test_cli_predict_proba_via_params(saved_model_path, test_data, fitted_predictor, tmp_path):
    payload = {
        "dataframe_split": test_data.to_dict(orient="split"),
        "params": {"predict_method": "predict_proba"},
    }
    result = _cli_predict(saved_model_path, payload, tmp_path)

    predictions = result["predictions"]
    assert len(predictions) == len(test_data)
    expected = fitted_predictor.predict_proba(test_data)
    first_row = predictions[0]
    assert pytest.approx(list(first_row.values())) == expected.iloc[0].tolist()
