"""Run a model through training + evaluation and emit a scorecard row.

This is the glue that makes the scorecard fair: it measures *the same things the
same way* for every tier. Given a model and prepared train/eval splits, it

1. seeds all RNGs for reproducibility,
2. times training (``train_seconds``),
3. times inference and normalizes to per-sample latency (``predict_latency_ms``),
4. measures the model footprint (``model_size_mb``),
5. scores predictions with :func:`metrics.evaluate` (predictive + calibration),

and packs everything into a :class:`~emotion_classification.scorecard.ScorecardRow`.
"""

from __future__ import annotations

import importlib
import os
import random
import time
from typing import Optional

import numpy as np

from .metrics import evaluate
from .models.base import EmotionModel
from .preprocessing import PreparedSplit
from .scorecard import ScorecardRow

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed Python, NumPy, and (if importable) PyTorch for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # Torch is optional — seed it only if the deep/transformer tiers are present.
    if importlib.util.find_spec("torch") is not None:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def run_experiment(
    model: EmotionModel,
    train: PreparedSplit,
    test: PreparedSplit,
    *,
    dataset_name: str,
    threshold: float = 0.5,
    seed: int = DEFAULT_SEED,
    cost_usd: Optional[float] = None,
    extra_meta: Optional[dict] = None,
) -> ScorecardRow:
    """Train ``model`` on ``train``, evaluate on ``test``, return a scorecard row."""
    set_seed(seed)

    t0 = time.perf_counter()
    model.fit(train.texts, train.Y)
    train_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_prob = model.predict_proba(test.texts)
    predict_seconds = time.perf_counter() - t1
    latency_ms = (predict_seconds / max(len(test), 1)) * 1000.0

    scores = evaluate(test.Y, y_prob, threshold=threshold)

    meta = {
        "n_train": len(train),
        "n_test": len(test),
        "n_labels": test.n_labels,
        "labels": test.label_names,
        "threshold": threshold,
        "seed": seed,
    }
    if extra_meta:
        meta.update(extra_meta)

    return ScorecardRow(
        model=getattr(model, "name", model.__class__.__name__),
        dataset=dataset_name,
        macro_f1=scores["macro_f1"],
        micro_f1=scores["micro_f1"],
        subset_accuracy=scores["subset_accuracy"],
        ece=scores["ece"],
        train_seconds=train_seconds,
        predict_latency_ms=latency_ms,
        model_size_mb=model.size_mb(),
        cost_usd=cost_usd,
        meta=meta,
    )
