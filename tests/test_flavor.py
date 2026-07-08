import os

import mlflow
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException

import mlflow_autogluon


def test_save_and_load_roundtrip(fitted_predictor, test_data, tmp_path):
    model_path = tmp_path / "model"
    mlflow_autogluon.save_model(ag_model=fitted_predictor, path=str(model_path))

    assert (model_path / "MLmodel").exists()
    assert (model_path / "requirements.txt").exists()
    assert (model_path / "conda.yaml").exists()

    loaded = mlflow_autogluon.load_model(f"file://{model_path}")
    pd.testing.assert_series_equal(
        loaded.predict(test_data), fitted_predictor.predict(test_data)
    )


def test_mlmodel_flavor_configuration(fitted_predictor, tmp_path):
    model_path = tmp_path / "model"
    mlflow_autogluon.save_model(ag_model=fitted_predictor, path=str(model_path))

    model = mlflow.models.Model.load(str(model_path / "MLmodel"))
    assert mlflow_autogluon.FLAVOR_NAME in model.flavors
    flavor_conf = model.flavors[mlflow_autogluon.FLAVOR_NAME]
    assert flavor_conf["model_type"] == "tabular"
    assert "autogluon_version" in flavor_conf
    assert "python_function" in model.flavors


def test_default_requirements_pin_autogluon():
    reqs = mlflow_autogluon.get_default_pip_requirements()
    assert any(r.startswith("autogluon.tabular==") for r in reqs)


def test_save_rejects_non_predictor(tmp_path):
    with pytest.raises(MlflowException, match="TabularPredictor"):
        mlflow_autogluon.save_model(ag_model="not a predictor", path=str(tmp_path / "m"))


def test_log_model_and_load_from_run(fitted_predictor, test_data, tracking_uri):
    with mlflow.start_run() as run:
        model_info = mlflow_autogluon.log_model(
            ag_model=fitted_predictor, artifact_path="model"
        )

    assert model_info.model_uri is not None
    loaded = mlflow_autogluon.load_model(f"runs:/{run.info.run_id}/model")
    pd.testing.assert_series_equal(
        loaded.predict(test_data), fitted_predictor.predict(test_data)
    )


def test_pyfunc_predict_and_predict_proba(fitted_predictor, test_data, tmp_path):
    model_path = tmp_path / "model"
    mlflow_autogluon.save_model(ag_model=fitted_predictor, path=str(model_path))

    pyfunc_model = mlflow.pyfunc.load_model(f"file://{model_path}")

    predictions = pyfunc_model.predict(test_data)
    assert len(predictions) == len(test_data)
    assert list(predictions) == fitted_predictor.predict(test_data).tolist()

    proba = pyfunc_model.predict(test_data, params={"predict_method": "predict_proba"})
    expected_proba = fitted_predictor.predict_proba(test_data)
    assert proba.shape == expected_proba.shape
    pd.testing.assert_frame_equal(pd.DataFrame(proba), pd.DataFrame(expected_proba))


def test_pyfunc_rejects_unknown_predict_method(fitted_predictor, test_data, tmp_path):
    model_path = tmp_path / "model"
    mlflow_autogluon.save_model(ag_model=fitted_predictor, path=str(model_path))
    pyfunc_model = mlflow.pyfunc.load_model(f"file://{model_path}")

    with pytest.raises(MlflowException, match="predict_method"):
        pyfunc_model.predict(test_data, params={"predict_method": "decision_function"})


def test_save_with_input_example(fitted_predictor, test_data, tmp_path):
    model_path = tmp_path / "model"
    mlflow_autogluon.save_model(
        ag_model=fitted_predictor,
        path=str(model_path),
        input_example=test_data.head(3),
    )
    assert os.path.exists(model_path / "input_example.json")
