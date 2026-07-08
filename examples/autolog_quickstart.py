"""End-to-end mlflow-autogluon example.

Trains an AutoGluon TabularPredictor on the breast cancer dataset with
autologging enabled, then loads the logged model back (native and pyfunc)
and scores it.

Run:
    python examples/autolog_quickstart.py [--tracking-uri sqlite:///mlflow.db]
"""

import argparse
import tempfile

import mlflow
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

import mlflow_autogluon

LABEL = "target"


def load_data():
    data = load_breast_cancer(as_frame=True)
    frame = data.frame
    train_frame, test_frame = train_test_split(
        frame, test_size=0.2, random_state=42, stratify=frame[LABEL]
    )
    return train_frame, test_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default=None, help="MLflow tracking URI")
    parser.add_argument("--time-limit", type=int, default=60, help="fit budget in seconds")
    args = parser.parse_args()

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("autogluon-quickstart")

    mlflow_autogluon.autolog(log_fit_summary=True)

    train_frame, test_frame = load_data()

    with tempfile.TemporaryDirectory() as model_dir:
        predictor = TabularPredictor(label=LABEL, path=model_dir, verbosity=1)
        predictor.fit(
            train_frame,
            time_limit=args.time_limit,
            # sklearn-backed models so the example runs on a base install;
            # remove this argument to use the full default model zoo
            hyperparameters={"RF": {}, "XT": {}, "LR": {}},
        )

        run = mlflow.last_active_run()
        print(f"\nAutologged run: {run.info.run_id}")
        print(f"  best model:   {run.data.tags['best_model']}")
        print(f"  score_val:    {run.data.metrics['best_model_score_val']:.4f}")

        model_uri = f"runs:/{run.info.run_id}/model"
        features = test_frame.drop(columns=[LABEL])

        native = mlflow_autogluon.load_model(model_uri)
        accuracy = (native.predict(features) == test_frame[LABEL]).mean()
        print(f"  native reload test accuracy: {accuracy:.4f}")

        pyfunc_model = mlflow.pyfunc.load_model(model_uri)
        predictions = pyfunc_model.predict(features)
        proba = pyfunc_model.predict(features, params={"predict_method": "predict_proba"})
        print(f"  pyfunc predictions: {pd.Series(predictions).value_counts().to_dict()}")
        print(f"  pyfunc proba columns: {list(proba.columns)}")


if __name__ == "__main__":
    main()
