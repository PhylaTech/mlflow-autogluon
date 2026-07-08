import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

import mlflow_autogluon

ag_timeseries = pytest.importorskip("autogluon.timeseries")

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor  # noqa: E402

PREDICTION_LENGTH = 3
TS_HYPERPARAMETERS = {"Naive": {}}


def _make_long_frame(n_items=3, n_steps=30):
    rng = np.random.default_rng(0)
    frames = []
    for item in range(n_items):
        frames.append(
            pd.DataFrame(
                {
                    "item_id": f"item_{item}",
                    "timestamp": pd.date_range("2024-01-01", periods=n_steps, freq="D"),
                    "target": rng.normal(loc=10 * (item + 1), scale=1.0, size=n_steps),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def train_frame():
    return _make_long_frame()


@pytest.fixture(scope="module")
def fitted_ts_predictor(train_frame, tmp_path_factory):
    path = tmp_path_factory.mktemp("ag_ts") / "predictor"
    predictor = TimeSeriesPredictor(
        target="target",
        prediction_length=PREDICTION_LENGTH,
        path=str(path),
        verbosity=0,
    )
    predictor.fit(
        TimeSeriesDataFrame.from_data_frame(train_frame),
        hyperparameters=TS_HYPERPARAMETERS,
    )
    return predictor


def test_save_and_load_roundtrip(fitted_ts_predictor, train_frame, tmp_path):
    model_path = tmp_path / "model"
    mlflow_autogluon.save_model(ag_model=fitted_ts_predictor, path=str(model_path))

    loaded = mlflow_autogluon.load_model(f"file://{model_path}")
    assert isinstance(loaded, TimeSeriesPredictor)

    data = TimeSeriesDataFrame.from_data_frame(train_frame)
    expected = fitted_ts_predictor.predict(data)
    actual = loaded.predict(data)
    pd.testing.assert_frame_equal(pd.DataFrame(actual), pd.DataFrame(expected))


def test_mlmodel_configuration(fitted_ts_predictor, tmp_path):
    model_path = tmp_path / "model"
    mlflow_autogluon.save_model(ag_model=fitted_ts_predictor, path=str(model_path))

    model = mlflow.models.Model.load(str(model_path / "MLmodel"))
    flavor_conf = model.flavors[mlflow_autogluon.FLAVOR_NAME]
    assert flavor_conf["model_type"] == "timeseries"

    requirements = (model_path / "requirements.txt").read_text().splitlines()
    assert any(r.startswith("autogluon.timeseries==") for r in requirements)


def test_pyfunc_predict_accepts_long_dataframe(fitted_ts_predictor, train_frame, tmp_path):
    model_path = tmp_path / "model"
    mlflow_autogluon.save_model(ag_model=fitted_ts_predictor, path=str(model_path))

    pyfunc_model = mlflow.pyfunc.load_model(f"file://{model_path}")
    forecast = pyfunc_model.predict(train_frame)

    assert isinstance(forecast, pd.DataFrame)
    assert {"item_id", "timestamp", "mean"} <= set(forecast.columns)
    assert len(forecast) == train_frame["item_id"].nunique() * PREDICTION_LENGTH

    expected = fitted_ts_predictor.predict(
        TimeSeriesDataFrame.from_data_frame(train_frame)
    ).reset_index()
    pd.testing.assert_frame_equal(forecast, expected)


def test_pyfunc_rejects_predict_proba(fitted_ts_predictor, train_frame, tmp_path):
    model_path = tmp_path / "model"
    mlflow_autogluon.save_model(ag_model=fitted_ts_predictor, path=str(model_path))
    pyfunc_model = mlflow.pyfunc.load_model(f"file://{model_path}")

    with pytest.raises(Exception, match="predict"):
        pyfunc_model.predict(train_frame, params={"predict_method": "predict_proba"})


def test_autolog_timeseries(train_frame, tracking_uri, tmp_path):
    mlflow_autogluon.autolog()
    try:
        predictor = TimeSeriesPredictor(
            target="target",
            prediction_length=PREDICTION_LENGTH,
            path=str(tmp_path / "predictor"),
            verbosity=0,
        )
        predictor.fit(
            TimeSeriesDataFrame.from_data_frame(train_frame),
            hyperparameters=TS_HYPERPARAMETERS,
        )

        runs = mlflow.search_runs(output_format="list")
        assert len(runs) == 1
        run = runs[0]

        assert run.data.params["target"] == "target"
        assert run.data.params["prediction_length"] == str(PREDICTION_LENGTH)
        assert run.data.tags["estimator_name"] == "TimeSeriesPredictor"
        assert run.data.tags["best_model"] == "Naive"
        assert "best_model_score_val" in run.data.metrics
        assert any(k.startswith("score_val_") for k in run.data.metrics)

        artifact_paths = [f.path for f in MlflowClient().list_artifacts(run.info.run_id)]
        assert "leaderboard.csv" in artifact_paths

        loaded = mlflow_autogluon.load_model(f"runs:/{run.info.run_id}/model")
        assert isinstance(loaded, TimeSeriesPredictor)
    finally:
        mlflow_autogluon.autolog(disable=True)
