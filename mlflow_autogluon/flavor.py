"""MLflow model flavor for AutoGluon predictors.

Implements the standard community-flavor surface described in
https://mlflow.org/docs/latest/ml/community-model-flavors/:

- :func:`save_model` / :func:`log_model` persist a fitted AutoGluon predictor
  as an MLflow model, including a ``python_function`` flavor for generic
  inference and serving.
- :func:`load_model` restores the native AutoGluon predictor from a model URI.
- :func:`_load_pyfunc` is the pyfunc loader entry point used by
  ``mlflow.pyfunc.load_model`` and ``mlflow models serve``.
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import TYPE_CHECKING, Any

import yaml
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types import ParamSchema, ParamSpec
from mlflow.utils.environment import (
    _CONDA_ENV_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _mlflow_conda_env,
    _process_conda_env,
    _process_pip_requirements,
    _PythonEnv,
    _validate_env_arguments,
)
from mlflow.utils.file_utils import write_to
from mlflow.utils.model_utils import (
    _add_code_from_conf_to_system_path,
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)

if TYPE_CHECKING:
    from autogluon.tabular import TabularPredictor
    from mlflow.models.model import ModelInfo

FLAVOR_NAME = "autogluon"

_MODEL_DATA_SUBPATH = "ag_model"
_MODEL_TYPE_TABULAR = "tabular"

_logger = logging.getLogger(__name__)


def _get_autogluon_version() -> str:
    import autogluon.tabular

    return autogluon.tabular.__version__


def get_default_pip_requirements() -> list[str]:
    """Return the default pip requirements for models produced by this flavor."""
    return [f"autogluon.tabular=={_get_autogluon_version()}"]


def get_default_conda_env() -> dict[str, Any]:
    """Return the default conda environment for models produced by this flavor."""
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


def _default_params_schema():
    return ParamSchema([ParamSpec("predict_method", "string", "predict")])


def _with_default_params_schema(signature):
    """Ensure the signature declares the ``predict_method`` inference param.

    MLflow's pyfunc layer drops inference ``params`` unless the model signature
    defines a params schema, which would silently disable ``predict_proba``.
    """
    if signature is None:
        return ModelSignature(params=_default_params_schema())
    if signature.params is None:
        return ModelSignature(
            inputs=signature.inputs,
            outputs=signature.outputs,
            params=_default_params_schema(),
        )
    return signature


def _validate_ag_model(ag_model):
    cls_name = type(ag_model).__name__
    if cls_name != "TabularPredictor":
        raise MlflowException(
            f"The autogluon flavor currently supports TabularPredictor only, got {cls_name}. "
            "Support for other predictor types is planned.",
            INVALID_PARAMETER_VALUE,
        )
    if not getattr(ag_model, "is_fit", True):
        raise MlflowException(
            "The predictor must be fit before it can be saved.", INVALID_PARAMETER_VALUE
        )


def save_model(
    ag_model: TabularPredictor,
    path: str,
    conda_env: dict[str, Any] | str | None = None,
    code_paths: list[str] | None = None,
    mlflow_model: Model | None = None,
    signature: ModelSignature | None = None,
    input_example: Any | None = None,
    pip_requirements: list[str] | str | None = None,
    extra_pip_requirements: list[str] | str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a fitted AutoGluon predictor to a local path in MLflow model format.

    Args:
        ag_model: Fitted ``autogluon.tabular.TabularPredictor`` instance.
        path: Local filesystem destination for the MLflow model.
        conda_env: Conda environment dict or path to a conda YAML file.
        code_paths: Local code paths to package with the model.
        mlflow_model: Existing :class:`mlflow.models.Model` to add flavors to.
        signature: :class:`mlflow.models.ModelSignature` describing input/output.
        input_example: Example model input, saved alongside the model.
        pip_requirements: Override for the default pip requirements.
        extra_pip_requirements: Additional pip requirements.
        metadata: Custom metadata dict stored in the MLmodel file.
    """
    _validate_ag_model(ag_model)
    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    path = os.path.abspath(path)
    _validate_and_prepare_target_save_path(path)
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    if mlflow_model is None:
        mlflow_model = Model()
    mlflow_model.signature = _with_default_params_schema(signature)
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)
    if metadata is not None:
        mlflow_model.metadata = metadata

    model_data_path = os.path.join(path, _MODEL_DATA_SUBPATH)
    ag_model.clone(path=model_data_path)

    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow_autogluon.flavor",
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        code=code_dir_subpath,
    )
    mlflow_model.add_flavor(
        FLAVOR_NAME,
        autogluon_version=_get_autogluon_version(),
        model_type=_MODEL_TYPE_TABULAR,
        data=_MODEL_DATA_SUBPATH,
        code=code_dir_subpath,
    )
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    if conda_env is None:
        default_reqs = get_default_pip_requirements() if pip_requirements is None else None
        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(
            default_reqs, pip_requirements, extra_pip_requirements
        )
    else:
        conda_env, pip_requirements, pip_constraints = _process_conda_env(conda_env)

    with open(os.path.join(path, _CONDA_ENV_FILE_NAME), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)
    if pip_constraints:
        write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))
    write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))
    _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))


def log_model(
    ag_model: TabularPredictor,
    artifact_path: str | None = None,
    conda_env: dict[str, Any] | str | None = None,
    code_paths: list[str] | None = None,
    registered_model_name: str | None = None,
    signature: ModelSignature | None = None,
    input_example: Any | None = None,
    await_registration_for: int = DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements: list[str] | str | None = None,
    extra_pip_requirements: list[str] | str | None = None,
    metadata: dict[str, Any] | None = None,
    name: str | None = None,
    **kwargs: Any,
) -> ModelInfo:
    """Log a fitted AutoGluon predictor as an MLflow artifact for the current run.

    Args:
        ag_model: Fitted ``autogluon.tabular.TabularPredictor`` instance.
        artifact_path: Run-relative artifact path (MLflow 2.x convention).
        name: Model name (MLflow 3.x convention). Falls back to ``artifact_path``
            on MLflow versions that do not support it.
        registered_model_name: If given, register the model under this name.
        kwargs: Remaining arguments are forwarded to :func:`save_model`.

    Returns:
        A :class:`mlflow.models.model.ModelInfo` describing the logged model.
    """
    log_params = inspect.signature(Model.log).parameters
    if name is not None and "name" in log_params:
        # MLflow 3.x: artifact_path is still a required positional, pass None
        # alongside name to opt into the new naming convention.
        log_kwargs = {"artifact_path": None, "name": name}
    else:
        log_kwargs = {"artifact_path": artifact_path or name or "model"}

    return Model.log(
        flavor=inspect.getmodule(save_model),
        ag_model=ag_model,
        conda_env=conda_env,
        code_paths=code_paths,
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        metadata=metadata,
        **log_kwargs,
        **kwargs,
    )


def _load_model_from_data_path(ag_model_path, model_type=_MODEL_TYPE_TABULAR):
    if model_type == _MODEL_TYPE_TABULAR:
        from autogluon.tabular import TabularPredictor

        return TabularPredictor.load(ag_model_path)
    raise MlflowException(
        f"Unsupported autogluon model_type in flavor configuration: {model_type}",
        INVALID_PARAMETER_VALUE,
    )


def load_model(model_uri: str, dst_path: str | None = None) -> TabularPredictor:
    """Load a native AutoGluon predictor from an MLflow model URI.

    Args:
        model_uri: URI of the MLflow model, e.g. ``runs:/<run_id>/model`` or
            ``models:/<name>/<version>``.
        dst_path: Optional local destination for downloaded artifacts.

    Returns:
        The restored AutoGluon predictor (``TabularPredictor``).
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    flavor_conf = _get_flavor_configuration(
        model_path=local_model_path, flavor_name=FLAVOR_NAME
    )
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
    ag_model_path = os.path.join(local_model_path, flavor_conf.get("data", _MODEL_DATA_SUBPATH))
    return _load_model_from_data_path(
        ag_model_path, flavor_conf.get("model_type", _MODEL_TYPE_TABULAR)
    )


class _AutoGluonModelWrapper:
    """Pyfunc-compatible wrapper around an AutoGluon predictor."""

    def __init__(self, ag_model):
        self.ag_model = ag_model

    def get_raw_model(self):
        return self.ag_model

    def predict(self, dataframe, params=None):
        """Run inference on a pandas DataFrame.

        Args:
            dataframe: Input feature frame.
            params: Optional dict of inference parameters. Supported key:
                ``predict_method`` with value ``"predict"`` (default) or
                ``"predict_proba"``.
        """
        params = params or {}
        predict_method = params.get("predict_method", "predict")
        if predict_method == "predict":
            # ndarray so REST serving returns plain scalars, matching the
            # behavior of built-in flavors such as mlflow.sklearn.
            return self.ag_model.predict(dataframe).to_numpy()
        if predict_method == "predict_proba":
            return self.ag_model.predict_proba(dataframe)
        raise MlflowException(
            f"Unsupported predict_method: {predict_method!r}. "
            "Expected 'predict' or 'predict_proba'.",
            INVALID_PARAMETER_VALUE,
        )


def _load_pyfunc(path):
    """Load the model as a pyfunc wrapper. Called by ``mlflow.pyfunc.load_model``."""
    try:
        flavor_conf = _get_flavor_configuration(model_path=path, flavor_name=FLAVOR_NAME)
        data_subpath = flavor_conf.get("data", _MODEL_DATA_SUBPATH)
        model_type = flavor_conf.get("model_type", _MODEL_TYPE_TABULAR)
        ag_model_path = os.path.join(path, data_subpath)
    except MlflowException:
        # ``path`` already points at the predictor directory (older layouts).
        ag_model_path = path
        model_type = _MODEL_TYPE_TABULAR
    return _AutoGluonModelWrapper(_load_model_from_data_path(ag_model_path, model_type))
