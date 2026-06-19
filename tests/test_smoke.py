"""Smoke tests — verify the package imports and paths resolve.

These intentionally avoid importing torch so they pass quickly on any Python.
"""

import emotion_classification
from emotion_classification import config


def test_version():
    assert emotion_classification.__version__


def test_project_root_exists():
    assert config.PROJECT_ROOT.is_dir()


def test_data_paths_under_root():
    assert config.RAW_DATA_DIR.parent == config.DATA_DIR
    assert config.DATA_DIR.parent == config.PROJECT_ROOT
