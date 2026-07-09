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


# ---------------------------------------------------------------------------
# Error and fallback branches
# ---------------------------------------------------------------------------


def test_registry_rejects_unknown_model_type():
    with pytest.raises(MlflowException, match="Unsupported autogluon model_type"):
        mlflow_autogluon.get_default_pip_requirements("not_a_type")


def test_stringify_param_falls_back_to_str_on_circular_reference():
    circular = {}
    circular["self"] = circular
    text = _stringify_param(circular)
    assert "self" in text


def test_extract_fit_params_falls_back_to_kwargs():
    from mlflow_autogluon.autolog import _extract_fit_params

    # an uninspectable original triggers the kwargs-only fallback
    params = _extract_fit_params(None, object(), (), {"time_limit": 5, "train_data": "df"})
    assert params == {"time_limit": 5}


def test_get_leaderboard_legacy_silent_fallback():
    from mlflow_autogluon.autolog import _get_leaderboard

    class LegacyPredictor:
        def leaderboard(self, display=None, silent=None):
            if display is not None:
                raise TypeError("unexpected keyword argument 'display'")
            return "legacy-leaderboard"

    assert _get_leaderboard(LegacyPredictor()) == "legacy-leaderboard"


def test_log_leaderboard_derives_best_model_without_model_best(tracking_uri):
    import mlflow
    import pandas as pd

    from mlflow_autogluon.autolog import _log_leaderboard

    class LeaderboardOnly:
        def leaderboard(self, display=False):
            return pd.DataFrame(
                {"model": ["Best", "Worst"], "score_val": [0.9, 0.1]}
            )

    with mlflow.start_run() as run:
        _log_leaderboard(LeaderboardOnly(), best_model=None)

    data = mlflow.get_run(run.info.run_id).data
    assert data.tags["best_model"] == "Best"
    assert data.metrics["best_model_score_val"] == 0.9


def test_persist_predictor_multimodal_save_without_standalone(tmp_path):
    from mlflow_autogluon.flavor import _persist_predictor

    class LegacySave:
        def save(self, path):
            (tmp_path / "saved").mkdir()

    _persist_predictor(LegacySave(), "multimodal", str(tmp_path / "dst"))
    assert (tmp_path / "saved").exists()


def test_persist_predictor_copytree_fallback(tmp_path):
    from mlflow_autogluon.flavor import _persist_predictor

    src = tmp_path / "src"
    src.mkdir()
    (src / "artifact.txt").write_text("weights")

    class NoClone:
        path = str(src)

        def save(self):
            pass

    dst = tmp_path / "dst"
    _persist_predictor(NoClone(), "timeseries", str(dst))
    assert (dst / "artifact.txt").read_text() == "weights"


def test_save_with_pip_constraints(fitted_predictor, tmp_path):
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("numpy<3\n")
    model_path = tmp_path / "model"
    mlflow_autogluon.save_model(
        ag_model=fitted_predictor,
        path=str(model_path),
        pip_requirements=["autogluon.tabular", f"-c {constraints}"],
    )
    assert (model_path / "constraints.txt").read_text().strip() == "numpy<3"


def test_load_model_missing_package_message(monkeypatch):
    import types

    from mlflow_autogluon import flavor

    def raising_import(name):
        raise ImportError(f"No module named {name!r}")

    monkeypatch.setattr(
        flavor, "importlib", types.SimpleNamespace(import_module=raising_import)
    )
    with pytest.raises(MlflowException, match="pip install autogluon.tabular"):
        flavor._load_model_from_data_path("/nonexistent", "tabular")


def test_autolog_skips_missing_predictor_packages(monkeypatch):
    import sys
    import types

    autolog_module = sys.modules["mlflow_autogluon.autolog"]

    real_import = __import__("importlib").import_module

    def tabular_only(name):
        if name != "autogluon.tabular":
            raise ImportError(name)
        return real_import(name)

    monkeypatch.setattr(
        autolog_module, "importlib", types.SimpleNamespace(import_module=tabular_only)
    )
    mlflow_autogluon.autolog(disable=True)
    mlflow_autogluon.autolog()  # must not raise: tabular is available
    mlflow_autogluon.autolog(disable=True)


def test_autolog_raises_without_any_predictor_package(monkeypatch):
    import sys
    import types

    autolog_module = sys.modules["mlflow_autogluon.autolog"]

    def nothing_available(name):
        raise ImportError(name)

    monkeypatch.setattr(
        autolog_module, "importlib", types.SimpleNamespace(import_module=nothing_available)
    )
    mlflow_autogluon.autolog(disable=True)
    with pytest.raises(ImportError, match="No AutoGluon predictor packages"):
        mlflow_autogluon.autolog()


def test_wrapper_get_raw_model():
    from mlflow_autogluon.flavor import _AutoGluonModelWrapper

    marker = object()
    assert _AutoGluonModelWrapper(marker).get_raw_model() is marker


def test_load_pyfunc_directly_on_predictor_directory(fitted_predictor, test_data, tmp_path):
    from mlflow_autogluon import flavor

    model_path = tmp_path / "model"
    mlflow_autogluon.save_model(ag_model=fitted_predictor, path=str(model_path))

    # older layouts pointed pyfunc data at the predictor directory itself
    wrapper = flavor._load_pyfunc(str(model_path / "ag_model"))
    assert len(wrapper.predict(test_data)) == len(test_data)


def test_log_dataset_ignores_non_dataframe_inputs():
    from mlflow_autogluon.autolog import _log_dataset

    _log_dataset("s3://bucket/train.csv")  # must not raise or need a run


def test_infer_signature_skips_non_dataframe_train_data():
    from mlflow_autogluon.autolog import _infer_signature_and_example

    assert _infer_signature_and_example(object(), None) == (None, None)


def test_infer_signature_swallows_prediction_errors():
    import pandas as pd

    from mlflow_autogluon.autolog import _infer_signature_and_example

    # an object of unknown type makes _model_type_of raise inside the guard
    result = _infer_signature_and_example(object(), pd.DataFrame({"a": [1]}))
    assert result == (None, None)


def test_sample_features_without_label_column():
    import pandas as pd

    from mlflow_autogluon.autolog import _sample_features

    class Unlabeled:
        label = None

    frame = pd.DataFrame({"a": range(10)})
    sample = _sample_features(Unlabeled(), frame)
    assert list(sample.columns) == ["a"]
    assert len(sample) == 5
