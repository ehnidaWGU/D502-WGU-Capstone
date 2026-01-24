"""
paths.py

Centralized path utilities for project
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

def find_repo_root(start: Path | None = None) -> Path:
    """
    Find repo root by searching upwards till a directory with marker file/folder is discovered

    Markers used:
    - .git
    - pyproject.toml
    - requirements.txt
    """
    current = (start or Path.cwd()).resolve()

    markers = (".git", "pyproject.toml", "requirements.txt")
    for parent in (current, *current.parents):
        if any((parent / m).exists() for m in markers):
            return parent

    raise FileNotFoundError(
        "Could not locate repo root. Expected one of: "
        f"{', '.join(markers)} in a parent directory of {current}"
    )



@dataclass(frozen=True)
class ProjectPaths:
    """
    Convenience container for common project paths.
    """
    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def raw_dir(self) -> Path:
        # Your raw dataset lives under data/raw/raw_dataset/
        return self.data_dir / "raw" / "raw_dataset"

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def features_dir(self) -> Path:
        return self.data_dir / "features"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def notebooks_dir(self) -> Path:
        return self.root / "notebooks"

    @property
    def dashboard_dir(self) -> Path:
        return self.root / "dashboard"

    @property
    def sqlite_db_path(self) -> Path:
        # Store the DB in interim; do NOT commit to git
        return self.interim_dir / "home_credit.db"

    def ensure_dirs(self) -> None:
        """
        Create commonly-used output directories if they do not exist.
        (Safe to call repeatedly.)
        """
        for p in (
            self.interim_dir,
            self.processed_dir,
            self.features_dir,
            self.models_dir,
            self.reports_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


def get_paths(start: Path | None = None) -> ProjectPaths:
    """
    Public helper to get a ProjectPaths instance based on the detected repo root.
    """
    root = find_repo_root(start=start)
    return ProjectPaths(root=root)