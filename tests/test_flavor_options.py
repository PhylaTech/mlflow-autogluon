import inspect

import mlflow
import pandas as pd
from mlflow.models import Model, ModelSignature, infer_signature

import mlflow_autogluon


def test_load_model_with_dst_path(fitted_predictor, test_data, tmp_path):
    model_path = tmp_path / "model"
    mlflow_autogluon.save_model(ag_model=fitted_predictor, path=str(model_path))

    dst = tmp_path / "downloaded"
    dst.mkdir()  # MLflow requires the destination to exist
    loaded = mlflow_autogluon.load_model(f"file://{model_path}", dst_path=str(dst))
    pd.testing.assert_series_equal(
        loaded.predict(test_data), fitted_predictor.predict(test_data)
    )
    assert any(dst.iterdir())


def test_explicit_signature_keeps_io_and_gains_params_schema(
    fitted_predictor, test_data, tmp_path
):
    signature = infer_signature(test_data, fitted_predictor.predict(test_data))
    assert signature.params is None

    model_path = tmp_path / "model"
    mlflow_autogluon.save_model(
        ag_model=fitted_predictor, path=str(model_path), signature=signature
    )

    saved = Model.load(str(model_path / "MLmodel"))
    assert saved.signature.inputs == signature.inputs
    assert saved.signature.outputs == signature.outputs
    assert saved.signature.params is not None
    param_names = [p.name for p in saved.signature.params.params]
    assert "predict_method" in param_names


def test_signature_with_params_schema_is_untouched(fitted_predictor, test_data, tmp_path):
    from mlflow.types import ParamSchema, ParamSpec

    signature = ModelSignature(
        params=ParamSchema([ParamSpec("predict_method", "string", "predict_proba")])
    )
    model_path = tmp_path / "model"
    mlflow_autogluon.save_model(
        ag_model=fitted_predictor, path=str(model_path), signature=signature
    )
    saved = Model.load(str(model_path / "MLmodel"))
    assert saved.signature.params.params[0].default == "predict_proba"


def test_log_model_with_name_kwarg(fitted_predictor, test_data, tracking_uri):
    """Exercise the MLflow 3.x name= path when available."""
    supports_name = "name" in inspect.signature(Model.log).parameters
    with mlflow.start_run() as run:
        model_info = mlflow_autogluon.log_model(ag_model=fitted_predictor, name="ag_model")

    assert model_info.model_uri is not None
    if supports_name:
        loaded = mlflow_autogluon.load_model(model_info.model_uri)
    else:
        loaded = mlflow_autogluon.load_model(f"runs:/{run.info.run_id}/ag_model")
    pd.testing.assert_series_equal(
        loaded.predict(test_data), fitted_predictor.predict(test_data)
    )


def test_extra_pip_requirements(fitted_predictor, tmp_path):
    model_path = tmp_path / "model"
    mlflow_autogluon.save_model(
        ag_model=fitted_predictor,
        path=str(model_path),
        extra_pip_requirements=["scikit-learn"],
    )
    requirements = (model_path / "requirements.txt").read_text().splitlines()
    assert any(r.startswith("autogluon.tabular==") for r in requirements)
    assert "scikit-learn" in requirements
