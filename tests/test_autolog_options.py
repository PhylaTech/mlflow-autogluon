import mlflow
import numpy as np
import pandas as pd
import pytest
from autogluon.tabular import TabularPredictor
from mlflow.tracking import MlflowClient

import mlflow_autogluon
from tests.conftest import FAST_HYPERPARAMETERS, LABEL


@pytest.fixture
def disable_autolog_after():
    yield
    mlflow_autogluon.autolog(disable=True)


def _fit_predictor(train_data, path, **fit_kwargs):
    predictor = TabularPredictor(label=LABEL, path=str(path), verbosity=0)
    predictor.fit(train_data, hyperparameters=FAST_HYPERPARAMETERS, **fit_kwargs)
    return predictor


def _single_run():
    runs = mlflow.search_runs(output_format="list")
    assert len(runs) == 1
    return runs[0]


def test_autolog_regression_problem_type(tracking_uri, tmp_path, disable_autolog_after):
    rng = np.random.default_rng(0)
    train_data = pd.DataFrame(
        {
            "num_a": rng.normal(size=60),
            "num_b": rng.normal(size=60),
        }
    )
    train_data[LABEL] = train_data["num_a"] * 2.0 + rng.normal(scale=0.1, size=60)

    mlflow_autogluon.autolog(log_models=False)
    _fit_predictor(train_data, tmp_path / "predictor")

    run = _single_run()
    assert run.data.tags["problem_type"] == "regression"
    assert "best_model_score_val" in run.data.metrics


def test_autolog_fit_summary_artifact(train_data, tracking_uri, tmp_path, disable_autolog_after):
    mlflow_autogluon.autolog(log_models=False, log_fit_summary=True)
    _fit_predictor(train_data, tmp_path / "predictor")

    run = _single_run()
    artifact_paths = [f.path for f in MlflowClient().list_artifacts(run.info.run_id)]
    assert "fit_summary.json" in artifact_paths


def test_autolog_extra_tags(train_data, tracking_uri, tmp_path, disable_autolog_after):
    mlflow_autogluon.autolog(log_models=False, extra_tags={"team": "mlops", "env": "ci"})
    _fit_predictor(train_data, tmp_path / "predictor")

    run = _single_run()
    assert run.data.tags["team"] == "mlops"
    assert run.data.tags["env"] == "ci"


def test_autolog_registered_model_name(train_data, tracking_uri, tmp_path, disable_autolog_after):
    mlflow_autogluon.autolog(registered_model_name="ag-classifier")
    _fit_predictor(train_data, tmp_path / "predictor")

    versions = MlflowClient().search_model_versions("name='ag-classifier'")
    assert len(versions) == 1


def test_autolog_second_fit_never_breaks_training(
    train_data, tracking_uri, tmp_path, disable_autolog_after
):
    """Param collisions on a second fit in the same run must not raise."""
    mlflow_autogluon.autolog(log_models=False)
    with mlflow.start_run():
        _fit_predictor(train_data, tmp_path / "p1", time_limit=60)
        predictor = _fit_predictor(train_data, tmp_path / "p2", time_limit=30)
    assert predictor.model_best is not None


def test_autolog_leaderboard_disabled(train_data, tracking_uri, tmp_path, disable_autolog_after):
    mlflow_autogluon.autolog(log_models=False, log_leaderboard=False)
    _fit_predictor(train_data, tmp_path / "predictor")

    run = _single_run()
    assert "best_model_score_val" not in run.data.metrics
    artifact_paths = [f.path for f in MlflowClient().list_artifacts(run.info.run_id)]
    assert "leaderboard.csv" not in artifact_paths
