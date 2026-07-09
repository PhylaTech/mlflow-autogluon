"""MLflow community flavor and autologging for AutoGluon predictors."""

from mlflow_autogluon.autolog import autolog
from mlflow_autogluon.flavor import (
    FLAVOR_NAME,
    get_default_conda_env,
    get_default_pip_requirements,
    load_model,
    log_model,
    save_model,
)

__version__ = "0.1.1"  # x-release-please-version

__all__ = [
    "FLAVOR_NAME",
    "autolog",
    "get_default_conda_env",
    "get_default_pip_requirements",
    "load_model",
    "log_model",
    "save_model",
]
