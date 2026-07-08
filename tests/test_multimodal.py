import mlflow
import numpy as np
import pandas as pd
import pytest

import mlflow_autogluon

ag_multimodal = pytest.importorskip("autogluon.multimodal")

from autogluon.multimodal import MultiModalPredictor  # noqa: E402

# AutoGluon's GPU banner probes every torch-visible device through NVML,
# which crashes on non-NVIDIA accelerators such as Apple MPS (upstream bug:
# the try/except only guards the import, not the probe). Neutralize it so
# the tests run on any hardware.
from autogluon.multimodal.learners import base as _ag_mm_base  # noqa: E402

_ag_mm_base.BaseLearner.log_gpu_info = staticmethod(lambda num_gpus, config: None)

# smallest practical text backbone so the test runs on CPU in a couple of
# minutes; downloaded once into the HF cache
MM_HYPERPARAMETERS = {
    "model.names": ["hf_text"],
    "model.hf_text.checkpoint_name": "prajjwal1/bert-tiny",
    "optim.max_epochs": 1,
    "env.num_workers": 0,
    # force CPU: keeps the test deterministic and avoids AutoGluon probing
    # non-NVIDIA accelerators (e.g. Apple MPS) through pynvml
    "env.accelerator": "cpu",
    "env.num_gpus": 0,
}
LABEL = "label"


def _make_text_frame(n_rows=32, seed=0):
    """Varied full sentences so AutoGluon infers the column as text, not categorical."""
    rng = np.random.default_rng(seed)
    subjects = ["the product", "this gadget", "the delivery", "my purchase", "the device"]
    positive = ["exceeded all my expectations", "worked flawlessly from day one",
                "was absolutely worth every penny", "made my daily routine much easier"]
    negative = ["broke within the first week", "was a complete waste of money",
                "never worked the way it was advertised", "arrived damaged and unusable"]
    fillers = ["honestly speaking", "after a month of use", "to my surprise",
               "without any doubt", "compared to alternatives"]
    rows = []
    for i in range(n_rows):
        label = int(rng.random() > 0.5)
        phrase = rng.choice(positive) if label else rng.choice(negative)
        text = (
            f"{rng.choice(fillers)}, {rng.choice(subjects)} {phrase} "
            f"and I have mentioned this in review number {i}."
        )
        rows.append({"text": text, LABEL: label})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def train_frame():
    return _make_text_frame()


@pytest.fixture(scope="module")
def test_frame():
    return _make_text_frame(n_rows=8, seed=1).drop(columns=[LABEL])


@pytest.fixture(scope="module")
def fitted_mm_predictor(train_frame, tmp_path_factory):
    path = tmp_path_factory.mktemp("ag_mm") / "predictor"
    predictor = MultiModalPredictor(label=LABEL, path=str(path), verbosity=0)
    predictor.fit(
        train_frame,
        hyperparameters=MM_HYPERPARAMETERS,
        column_types={"text": "text"},
        time_limit=300,
    )
    return predictor


def test_save_and_load_roundtrip(fitted_mm_predictor, test_frame, tmp_path):
    model_path = tmp_path / "model"
    mlflow_autogluon.save_model(ag_model=fitted_mm_predictor, path=str(model_path))

    loaded = mlflow_autogluon.load_model(f"file://{model_path}")
    assert isinstance(loaded, MultiModalPredictor)

    expected = fitted_mm_predictor.predict(test_frame)
    actual = loaded.predict(test_frame)
    assert list(actual) == list(expected)


def test_mlmodel_configuration(fitted_mm_predictor, tmp_path):
    model_path = tmp_path / "model"
    mlflow_autogluon.save_model(ag_model=fitted_mm_predictor, path=str(model_path))

    model = mlflow.models.Model.load(str(model_path / "MLmodel"))
    flavor_conf = model.flavors[mlflow_autogluon.FLAVOR_NAME]
    assert flavor_conf["model_type"] == "multimodal"

    requirements = (model_path / "requirements.txt").read_text().splitlines()
    assert any(r.startswith("autogluon.multimodal==") for r in requirements)


def test_pyfunc_predict_and_predict_proba(fitted_mm_predictor, test_frame, tmp_path):
    model_path = tmp_path / "model"
    mlflow_autogluon.save_model(ag_model=fitted_mm_predictor, path=str(model_path))

    pyfunc_model = mlflow.pyfunc.load_model(f"file://{model_path}")

    predictions = pyfunc_model.predict(test_frame)
    assert len(predictions) == len(test_frame)
    assert list(predictions) == list(fitted_mm_predictor.predict(test_frame))

    proba = pyfunc_model.predict(test_frame, params={"predict_method": "predict_proba"})
    assert len(proba) == len(test_frame)


def test_autolog_multimodal(train_frame, tracking_uri, tmp_path):
    mlflow_autogluon.autolog(log_models=False)
    try:
        predictor = MultiModalPredictor(
            label=LABEL, path=str(tmp_path / "predictor"), verbosity=0
        )
        predictor.fit(
            train_frame,
            hyperparameters=MM_HYPERPARAMETERS,
            column_types={"text": "text"},
            time_limit=300,
        )

        runs = mlflow.search_runs(output_format="list")
        assert len(runs) == 1
        run = runs[0]

        assert run.data.params["label"] == LABEL
        assert run.data.tags["estimator_name"] == "MultiModalPredictor"
        assert "fit_time_seconds" in run.data.metrics
    finally:
        mlflow_autogluon.autolog(disable=True)
