import pytest
import yaml
from autogluon.tabular import TabularPredictor
from mlflow.exceptions import MlflowException
from mlflow.models import Model

import mlflow_autogluon
from mlflow_autogluon.autolog import _stringify_param, _try_log
from tests.conftest import LABEL


def test_get_default_conda_env_pins_autogluon():
    env = mlflow_autogluon.get_default_conda_env()
    pip_deps = next(d for d in env["dependencies"] if isinstance(d, dict))["pip"]
    assert any(str(d).startswith("autogluon.tabular==") for d in pip_deps)


def test_save_rejects_unfit_predictor(tmp_path):
    predictor = TabularPredictor(label=LABEL, path=str(tmp_path / "p"), verbosity=0)
    with pytest.raises(MlflowException, match="must be fit"):
        mlflow_autogluon.save_model(ag_model=predictor, path=str(tmp_path / "model"))


def test_save_with_metadata(fitted_predictor, tmp_path):
    model_path = tmp_path / "model"
    mlflow_autogluon.save_model(
        ag_model=fitted_predictor, path=str(model_path), metadata={"owner": "mlops"}
    )
    saved = Model.load(str(model_path / "MLmodel"))
    assert saved.metadata == {"owner": "mlops"}


def test_save_with_conda_env_dict(fitted_predictor, tmp_path):
    model_path = tmp_path / "model"
    mlflow_autogluon.save_model(
        ag_model=fitted_predictor,
        path=str(model_path),
        conda_env=mlflow_autogluon.get_default_conda_env(),
    )
    written = yaml.safe_load((model_path / "conda.yaml").read_text())
    assert "dependencies" in written


def test_load_rejects_unknown_model_type(fitted_predictor, tmp_path):
    model_path = tmp_path / "model"
    mlflow_autogluon.save_model(ag_model=fitted_predictor, path=str(model_path))

    mlmodel_path = model_path / "MLmodel"
    conf = yaml.safe_load(mlmodel_path.read_text())
    conf["flavors"][mlflow_autogluon.FLAVOR_NAME]["model_type"] = "not_a_real_type"
    mlmodel_path.write_text(yaml.safe_dump(conf))

    with pytest.raises(MlflowException, match="model_type"):
        mlflow_autogluon.load_model(f"file://{model_path}")


def test_stringify_param_scalars():
    assert _stringify_param(None) == "None"
    assert _stringify_param(True) == "True"
    assert _stringify_param(3) == "3"
    assert _stringify_param("presets") == "presets"


def test_stringify_param_truncates_long_values():
    value = {f"key_{i}": "x" * 50 for i in range(100)}
    text = _stringify_param(value)
    assert len(text) <= 500
    assert text.endswith("...")


def test_stringify_param_handles_unserializable_objects():
    class Opaque:
        __slots__ = ()

    assert "Opaque" in _stringify_param(Opaque())


def test_try_log_swallows_exceptions():
    def boom():
        raise RuntimeError("logging exploded")

    _try_log(boom)  # must not raise
