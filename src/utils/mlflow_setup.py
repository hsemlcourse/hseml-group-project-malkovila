from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from src.config import PROJECT_ROOT

MLRUNS_DIR = PROJECT_ROOT / "mlruns"
_TRACKING_URI = f"file:{MLRUNS_DIR.as_posix()}"


def setup_mlflow(experiment_name: str = "cp2") -> None:
    import mlflow

    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", _TRACKING_URI))
    mlflow.set_experiment(experiment_name)


@contextmanager
def mlflow_run(run_name: str, tags: dict[str, str] | None = None, nested: bool = False) -> Iterator[Any]:
    import mlflow

    setup_mlflow()
    with mlflow.start_run(run_name=run_name, nested=nested, tags=tags or {}) as run:
        yield run


def log_model_sklearn(model: Any, artifact_path: str = "model") -> None:
    import mlflow.sklearn

    mlflow.sklearn.log_model(model, artifact_path)
