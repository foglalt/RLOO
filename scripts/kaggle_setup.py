"""Utility helpers to bootstrap Kaggle notebook environment for this project."""

from __future__ import annotations

import sys
from pathlib import Path


DEFAULT_DATASET_PATH = Path("/kaggle/input/rloo")
REPO_SRC_PATH = DEFAULT_DATASET_PATH / "src"


def setup_pythonpath(dataset_path: Path | None = None) -> None:
    """Append the project src directory to the active Python path."""

    project_path = (dataset_path or DEFAULT_DATASET_PATH) / "src"
    if project_path.exists() and str(project_path) not in sys.path:
        sys.path.append(str(project_path))


def ensure_output_dirs() -> None:
    """Create working directories for checkpoints and logs."""

    working_root = Path("/kaggle/working/rloo_runs")
    working_root.mkdir(parents=True, exist_ok=True)


__all__ = ["setup_pythonpath", "ensure_output_dirs"]
