import mlflow
import pandas as pd
import pytest
from autogluon.tabular import TabularPredictor
from mlflow.tracking import MlflowClient

import mlflow_autogluon
from tests.conftest import FAST_HYPERPARAMETERS, LABEL


def _fit_predictor(train_data, path):
    predictor = TabularPredictor(label=LABEL, path=str(path), verbosity=0)
    predictor.fit(train_data, hyperparameters=FAST_HYPERPARAMETERS, time_limit=60)
    return predictor


@pytest.fixture
def autolog_run(train_data, tracking_uri, tmp_path):
    mlflow_autogluon.autolog()
    try:
        predictor = _fit_predictor(train_data, tmp_path / "predictor")
        runs = mlflow.search_runs(output_format="list")
        assert len(runs) == 1
        yield predictor, runs[0]
    finally:
        mlflow_autogluon.autolog(disable=True)


def test_autolog_logs_params_and_tags(autolog_run):
    _, run = autolog_run

    assert run.data.params["label"] == LABEL
    assert "hyperparameters" in run.data.params
    assert run.data.params["time_limit"] == "60"
    assert run.data.params["train_rows"] == "80"

    assert run.data.tags["estimator_name"] == "TabularPredictor"
    assert "autogluon_version" in run.data.tags
    assert "best_model" in run.data.tags


def test_autolog_logs_leaderboard_metrics_and_artifact(autolog_run):
    _, run = autolog_run

    assert "best_model_score_val" in run.data.metrics
    assert "fit_time_seconds" in run.data.metrics
    assert any(key.startswith("score_val_") for key in run.data.metrics)

    artifact_paths = [f.path for f in MlflowClient().list_artifacts(run.info.run_id)]
    assert "leaderboard.csv" in artifact_paths


def test_autolog_logs_loadable_model(autolog_run, test_data):
    predictor, run = autolog_run

    loaded = mlflow_autogluon.load_model(f"runs:/{run.info.run_id}/model")
    pd.testing.assert_series_equal(loaded.predict(test_data), predictor.predict(test_data))


def test_autolog_respects_log_models_false(train_data, tracking_uri, tmp_path):
    mlflow_autogluon.autolog(log_models=False)
    try:
        _fit_predictor(train_data, tmp_path / "predictor")
        runs = mlflow.search_runs(output_format="list")
        assert len(runs) == 1
        artifact_paths = [
            f.path for f in MlflowClient().list_artifacts(runs[0].info.run_id)
        ]
        assert "model" not in artifact_paths
        assert "leaderboard.csv" in artifact_paths
    finally:
        mlflow_autogluon.autolog(disable=True)


def test_autolog_disable(train_data, tracking_uri, tmp_path):
    mlflow_autogluon.autolog(disable=True)
    _fit_predictor(train_data, tmp_path / "predictor")
    assert mlflow.search_runs(output_format="list") == []


def test_autolog_uses_active_run(train_data, tracking_uri, tmp_path):
    mlflow_autogluon.autolog()
    try:
        with mlflow.start_run() as run:
            _fit_predictor(train_data, tmp_path / "predictor")
        runs = mlflow.search_runs(output_format="list")
        assert len(runs) == 1
        assert runs[0].info.run_id == run.info.run_id
    finally:
        mlflow_autogluon.autolog(disable=True)
