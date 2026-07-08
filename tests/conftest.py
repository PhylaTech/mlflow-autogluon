import numpy as np
import pandas as pd
import pytest
from autogluon.tabular import TabularPredictor

# Cheap, deterministic model configuration so tests run in seconds.
FAST_HYPERPARAMETERS = {"DUMMY": {}}
LABEL = "target"


def _make_data(n_rows, seed):
    rng = np.random.default_rng(seed)
    features = pd.DataFrame(
        {
            "num_a": rng.normal(size=n_rows),
            "num_b": rng.normal(size=n_rows),
            "cat_a": rng.choice(["x", "y", "z"], size=n_rows),
        }
    )
    target = (features["num_a"] + rng.normal(scale=0.1, size=n_rows) > 0).astype(int)
    return features.assign(**{LABEL: target})


@pytest.fixture(scope="session")
def train_data():
    return _make_data(n_rows=80, seed=0)


@pytest.fixture(scope="session")
def test_data():
    return _make_data(n_rows=20, seed=1).drop(columns=[LABEL])


@pytest.fixture(scope="session")
def fitted_predictor(train_data, tmp_path_factory):
    path = tmp_path_factory.mktemp("ag") / "predictor"
    predictor = TabularPredictor(label=LABEL, path=str(path), verbosity=0)
    predictor.fit(train_data, hyperparameters=FAST_HYPERPARAMETERS)
    return predictor


@pytest.fixture
def tracking_uri(tmp_path, monkeypatch):
    import mlflow

    # chdir so the default artifact root (./mlruns) lands in tmp_path too
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    yield uri
    if mlflow.active_run() is not None:
        mlflow.end_run()
