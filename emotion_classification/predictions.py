"""Prediction cache: the per-example probabilities of one model on one evaluation set.

An experiment used to keep only its scorecard row and throw the probabilities
away, so any new analysis (a tuned threshold, a calibration fit, a paired
significance test) needed a retrain. A :class:`PredictionRecord` keeps them:
``y_true`` and ``y_prob`` for one (model, seed, evaluation set), plus a manifest
(``meta``) saying where they came from and what the run cost. Each record is one
compressed ``.npz`` file, so analyses can be re-run from disk on any machine.

numpy + stdlib only (no torch), so this imports and tests anywhere.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Any run of characters outside [A-Za-z0-9._] becomes a single "-". The dash is in
# the run on purpose, so "a->b" gives "a-b" rather than "a--b".
_UNSAFE = re.compile(r"[^A-Za-z0-9._]+")


@dataclass
class PredictionRecord:
    """Probabilities and ground truth for one (model, seed, evaluation set)."""

    y_true: np.ndarray   # (n_samples, n_labels) multi-hot {0, 1}
    y_prob: np.ndarray   # (n_samples, n_labels) probabilities, dtype as the model produced
    label_names: list
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.y_true = np.asarray(self.y_true)
        self.y_prob = np.asarray(self.y_prob)
        self.label_names = [str(name) for name in self.label_names]
        if self.y_true.shape != self.y_prob.shape:
            raise ValueError(
                f"y_true shape {self.y_true.shape} != y_prob shape {self.y_prob.shape}"
            )
        if self.y_prob.ndim != 2 or self.y_prob.shape[1] != len(self.label_names):
            raise ValueError(
                f"expected (n_samples, {len(self.label_names)}) arrays, "
                f"got shape {self.y_prob.shape}"
            )

    def __len__(self) -> int:
        return int(self.y_prob.shape[0])

    @property
    def n_labels(self) -> int:
        return len(self.label_names)


def slug(text: str) -> str:
    """Make ``text`` safe as a file-name part (``go_emotions:test`` -> ``go_emotions-test``)."""
    return _UNSAFE.sub("-", str(text)).strip("-")


def prediction_path(root, run_name: str, model: str, seed: int, eval_set: str) -> Path:
    """Where the record for (run, model, seed, evaluation set) lives under ``root``.

    The key has no hyperparameter component: two different configurations of one
    model in the same run would share a file. The pipeline refuses such a run
    before training; give each configuration its own run name.
    """
    parts = {"run name": run_name, "model": model, "evaluation set": eval_set}
    for what, value in parts.items():
        if not slug(value):
            raise ValueError(f"{what} {value!r} has no file-safe characters")
    name = f"{slug(model)}__seed{int(seed)}__{slug(eval_set)}.npz"
    return Path(root) / slug(run_name) / name


def save_prediction(record: PredictionRecord, path) -> Path:
    """Write ``record`` to ``path`` (a ``.npz`` file) and return the path.

    The file is written under a temporary name and renamed into place, so an
    interrupted batch never leaves a half-written record behind. ``y_prob`` keeps
    the dtype the model produced, which makes a cached score identical to the
    in-process one. ``y_true`` is stored as 0/1 bytes, so it must be multi-hot,
    and ``meta`` must be JSON-serialisable; both are checked before writing.
    """
    # Everything that can fail is checked before a file is opened.
    meta_json = json.dumps(record.meta)
    if not np.isin(record.y_true, (0, 1)).all():
        raise ValueError("y_true must be multi-hot (only 0 and 1) to be cached")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as fh:
            np.savez_compressed(
                fh,
                y_true=record.y_true.astype(np.uint8),
                y_prob=record.y_prob,
                label_names=np.asarray(record.label_names, dtype=str),
                meta=np.asarray(meta_json),
            )
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def read_meta(path) -> dict:
    """Read only the manifest of a cached record (the arrays stay on disk)."""
    with np.load(Path(path), allow_pickle=False) as archive:
        return json.loads(archive["meta"].item())


def load_prediction(path) -> PredictionRecord:
    """Read back a record written by :func:`save_prediction`."""
    with np.load(Path(path), allow_pickle=False) as archive:
        return PredictionRecord(
            y_true=archive["y_true"].astype(np.float32),
            y_prob=archive["y_prob"],
            label_names=[str(name) for name in archive["label_names"]],
            meta=json.loads(archive["meta"].item()),
        )


def list_predictions(root) -> list:
    """Manifest of every cached record under ``root``, each with its ``path``."""
    out = []
    for path in sorted(Path(root).rglob("*.npz")):
        meta = read_meta(path)
        meta["path"] = str(path)
        out.append(meta)
    return out
