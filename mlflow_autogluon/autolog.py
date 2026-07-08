"""Autologging support for AutoGluon.

Patches ``TabularPredictor.fit`` so that a single call to
:func:`mlflow_autogluon.autolog` records parameters, leaderboard metrics,
artifacts, and (optionally) the fitted predictor itself to MLflow, mirroring
the behavior of MLflow's built-in autologging integrations.

Because this is a community flavor, ``mlflow.autolog()`` does not enable it;
call ``mlflow_autogluon.autolog()`` explicitly before fitting.
"""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import numbers
import os
import tempfile
import time
from typing import Any

import mlflow
from mlflow.utils.autologging_utils import (
    autologging_integration,
    get_autologging_config,
    safe_patch,
)

from mlflow_autogluon.flavor import (
    _PREDICTOR_REGISTRY,
    FLAVOR_NAME,
    _model_type_of,
    log_model,
)

_logger = logging.getLogger(__name__)

_MAX_PARAM_LENGTH = 500

# fit() arguments that are data payloads rather than hyperparameters
_NON_PARAM_FIT_ARGS = {"self", "train_data", "tuning_data", "test_data", "kwargs"}


@autologging_integration(FLAVOR_NAME)
def autolog(
    log_models: bool = True,
    log_leaderboard: bool = True,
    log_fit_summary: bool = False,
    registered_model_name: str | None = None,
    extra_tags: dict[str, Any] | None = None,
    disable: bool = False,
    exclusive: bool = False,
    disable_for_unsupported_versions: bool = False,
    silent: bool = False,
) -> None:  # pylint: disable=unused-argument
    """Enable automatic logging for AutoGluon ``TabularPredictor.fit`` calls.

    After calling this function, every ``fit`` call logs to the active MLflow
    run (a run is created automatically when none is active):

    - Parameters: predictor configuration (label, problem type, eval metric)
      and fit arguments (presets, time_limit, hyperparameters, bagging and
      stacking settings, and so on).
    - Metrics: validation score and fit time per trained model from the
      leaderboard, plus the best model's validation score and total fit time.
    - Tags: AutoGluon version and best model name.
    - Artifacts: the leaderboard as CSV, optionally the fit summary as JSON,
      and the fitted predictor logged with the ``autogluon`` flavor.

    Args:
        log_models: If ``True``, log the fitted predictor as an MLflow model.
        log_leaderboard: If ``True``, log the leaderboard as a CSV artifact
            and per-model metrics.
        log_fit_summary: If ``True``, log ``predictor.fit_summary()`` as a
            JSON artifact.
        registered_model_name: If given, logged models are also registered
            under this name in the model registry.
        extra_tags: Dict of extra tags to set on autologged runs.
        disable: If ``True``, disable the integration.
        exclusive: If ``True``, autologged content is not logged to
            user-created fluent runs.
        disable_for_unsupported_versions: If ``True``, disable autologging for
            untested AutoGluon versions.
        silent: If ``True``, suppress all MLflow event logs and warnings from
            autologging.
    """
    patched_any = False
    for module_name, class_name in _PREDICTOR_REGISTRY.values():
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        predictor_class = getattr(module, class_name)
        safe_patch(FLAVOR_NAME, predictor_class, "fit", _patched_fit, manage_run=True)
        patched_any = True
    if not patched_any:
        raise ImportError(
            "No AutoGluon predictor packages found. Install at least one of: "
            + ", ".join(m for m, _ in _PREDICTOR_REGISTRY.values())
        )


def _patched_fit(original, self, *args, **kwargs):
    _try_log(_log_pretraining, self, original, args, kwargs)
    start_time = time.time()
    result = original(self, *args, **kwargs)
    fit_duration = time.time() - start_time
    _try_log(_log_posttraining, self, fit_duration)
    return result


def _try_log(fn, *args):
    """Never let a logging failure break the user's training run."""
    try:
        fn(*args)
    except Exception as e:
        if not get_autologging_config(FLAVOR_NAME, "silent", False):
            _logger.warning("mlflow-autogluon autologging failed: %s", e)


def _stringify_param(value):
    if value is None or isinstance(value, (str, bool, numbers.Number)):
        return str(value)
    try:
        text = json.dumps(value, default=str)
    except (TypeError, ValueError):
        text = str(value)
    if len(text) > _MAX_PARAM_LENGTH:
        text = text[: _MAX_PARAM_LENGTH - 3] + "..."
    return text


def _extract_fit_params(original, self, args, kwargs):
    """Map fit() positional and keyword arguments to parameter names."""
    try:
        bound = inspect.signature(original).bind_partial(self, *args, **kwargs)
        arguments = dict(bound.arguments)
        arguments.update(arguments.pop("kwargs", {}))
    except (TypeError, ValueError):
        arguments = dict(kwargs)
    return {
        name: value for name, value in arguments.items() if name not in _NON_PARAM_FIT_ARGS
    }


# predictor attributes worth capturing as params, when present
_PREDICTOR_ATTR_PARAMS = ("label", "target", "problem_type", "prediction_length")


def _log_pretraining(self, original, args, kwargs):
    params = {}
    for attr in _PREDICTOR_ATTR_PARAMS:
        value = getattr(self, attr, None)
        if value is not None:
            params[attr] = value
    eval_metric = getattr(self, "eval_metric", None)
    if eval_metric is not None:
        params["eval_metric"] = getattr(eval_metric, "name", str(eval_metric))

    for name, value in _extract_fit_params(original, self, args, kwargs).items():
        if value is not None:
            params[name] = value

    train_data = kwargs.get("train_data", args[0] if args else None)
    if hasattr(train_data, "shape"):
        params["train_rows"] = train_data.shape[0]
        params["train_columns"] = train_data.shape[1]

    mlflow.log_params({k: _stringify_param(v) for k, v in params.items()})

    model_type = _model_type_of(self)
    module_name, _ = _PREDICTOR_REGISTRY[model_type]
    tags = {
        "estimator_name": type(self).__name__,
        "autogluon_version": importlib.import_module(module_name).__version__,
    }
    extra_tags = get_autologging_config(FLAVOR_NAME, "extra_tags", None)
    if extra_tags:
        tags.update(extra_tags)
    mlflow.set_tags(tags)


def _get_leaderboard(self):
    try:
        return self.leaderboard(display=False)
    except TypeError:
        return self.leaderboard(silent=True)


def _log_posttraining(self, fit_duration):
    mlflow.log_metric("fit_time_seconds", fit_duration)

    problem_type = getattr(self, "problem_type", None)
    if problem_type is not None:
        mlflow.set_tag("problem_type", problem_type)

    best_model = getattr(self, "model_best", None)
    if best_model is not None:
        mlflow.set_tag("best_model", best_model)

    if get_autologging_config(FLAVOR_NAME, "log_leaderboard", True) and hasattr(
        self, "leaderboard"
    ):
        _try_log(_log_leaderboard, self, best_model)

    if get_autologging_config(FLAVOR_NAME, "log_fit_summary", False) and hasattr(
        self, "fit_summary"
    ):
        _try_log(_log_fit_summary, self)

    if get_autologging_config(FLAVOR_NAME, "log_models", True):
        registered_model_name = get_autologging_config(
            FLAVOR_NAME, "registered_model_name", None
        )
        # name= resolves to the MLflow 3 convention when available and falls
        # back to artifact_path on MLflow 2.x, avoiding deprecation warnings.
        log_model(
            ag_model=self,
            name="model",
            registered_model_name=registered_model_name,
        )


def _log_leaderboard(self, best_model):
    leaderboard = _get_leaderboard(self)

    metrics = {}
    for _, row in leaderboard.iterrows():
        model_name = row.get("model")
        # column sets differ per predictor type (e.g. fit_time_marginal for
        # timeseries), so log every numeric leaderboard column
        for column, value in row.items():
            if column == "model":
                continue
            if isinstance(value, numbers.Number) and value == value:  # skip NaN
                metrics[f"{column}_{model_name}"] = float(value)

    if best_model is None and not leaderboard.empty and "score_val" in leaderboard:
        # TimeSeriesPredictor has no model_best attribute; the leaderboard is
        # sorted by validation score, best first
        best_model = leaderboard.iloc[0]["model"]
        mlflow.set_tag("best_model", best_model)

    if best_model is not None:
        best_rows = leaderboard[leaderboard["model"] == best_model]
        if not best_rows.empty:
            score = best_rows.iloc[0].get("score_val")
            if isinstance(score, numbers.Number) and score == score:
                metrics["best_model_score_val"] = float(score)
    if metrics:
        mlflow.log_metrics(metrics)

    with tempfile.TemporaryDirectory() as tmp_dir:
        leaderboard_path = os.path.join(tmp_dir, "leaderboard.csv")
        leaderboard.to_csv(leaderboard_path, index=False)
        mlflow.log_artifact(leaderboard_path)


def _log_fit_summary(self):
    summary = self.fit_summary(verbosity=0)
    with tempfile.TemporaryDirectory() as tmp_dir:
        summary_path = os.path.join(tmp_dir, "fit_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        mlflow.log_artifact(summary_path)
