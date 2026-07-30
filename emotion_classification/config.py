"""Canonical filesystem paths for the project.

Import these instead of hard-coding paths so code works regardless of the
current working directory (scripts, tests).
"""

from pathlib import Path

# Repo root = parent of the package directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

MODELS_DIR: Path = PROJECT_ROOT / "models"


def ensure_dirs() -> None:
    """Create the standard data/model directories if they don't exist."""
    for d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
