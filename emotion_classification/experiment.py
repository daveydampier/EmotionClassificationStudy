"""Train a model once, predict on many evaluation sets, and score the results.

This is the glue that makes the scorecard fair: it measures *the same things the
same way* for every tier. The work is split in two so that nothing has to be
retrained to ask a new question of a run:

* :func:`fit_and_predict` seeds all RNGs, times training, then predicts on every
  named evaluation set. It returns one :class:`~emotion_classification.
  predictions.PredictionRecord` per set: probabilities, ground truth, and a
  manifest holding the timings, the model footprint and the device.
* :func:`score` turns one record into a :class:`~emotion_classification.
  scorecard.ScorecardRow` (predictive and calibration metrics, the per-emotion
  breakdown, optional bootstrap intervals). It gives the same row for a record
  just produced and for one loaded back from the prediction cache.

:func:`run_experiment` is the original one-call form (train, evaluate on one set,
return the row), kept as a thin wrapper over the two.
"""

from __future__ import annotations

import importlib
import os
import random
import time
from collections.abc import Mapping
from typing import Optional

import numpy as np

from .metrics import (
    bootstrap_f1_ci, bootstrap_per_label_f1_ci, evaluate, per_label_metrics,
)
from .models.base import EmotionModel
from .predictions import PredictionRecord
from .preprocessing import PreparedSplit
from .scorecard import ScorecardRow

DEFAULT_SEED = 42


def _detect_device(model) -> str:
    """Report the hardware a model ran on ('cpu' or e.g. 'cuda:0').

    Classical (scikit-learn) models have no torch module and run on CPU. Torch
    models expose their fitted network as ``module_`` (deep tier) or ``model_``
    (transformer tier); we read the device of its parameters. Recorded per run so
    the training-time / latency fairness caveat is visible in the results.
    """
    if importlib.util.find_spec("torch") is None:
        return "cpu"
    import torch

    for attr in ("module_", "model_"):
        module = getattr(model, attr, None)
        if isinstance(module, torch.nn.Module):
            try:
                return str(next(module.parameters()).device)
            except StopIteration:
                pass
    return "cpu"


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


_NOT_A_SCALAR = object()


def _json_scalar(value):
    """``value`` if JSON can store it as a scalar or a flat list of scalars."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)) and all(
        item is None or isinstance(item, (str, bool, int, float)) for item in value
    ):
        return list(value)
    return _NOT_A_SCALAR


def model_params(model) -> dict:
    """The hyperparameters ``model`` was built with, as a flat JSON-safe dict.

    Public scalar attributes are taken as they are (``epochs``, ``lr``,
    ``features``). An attribute that is a scikit-learn object, or wraps one as
    ``.estimator``, contributes the scalars of its ``get_params()`` under a dotted
    key: ``vectorizer.max_features``, ``head.C``. Private attributes and fitted
    state (names ending in ``_``) are left out, so the result is the same before
    and after training. Written into every prediction manifest, so a cached record
    says exactly which configuration produced it.
    """
    out: dict = {}
    for key, value in getattr(model, "__dict__", {}).items():
        if key.startswith("_") or key.endswith("_"):
            continue
        scalar = _json_scalar(value)
        if scalar is not _NOT_A_SCALAR:
            out[key] = scalar
            continue
        source = value if hasattr(value, "get_params") else getattr(value, "estimator", None)
        if hasattr(source, "get_params"):
            for name, param in sorted(source.get_params(deep=True).items()):
                scalar = _json_scalar(param)
                if scalar is not _NOT_A_SCALAR:
                    out[f"{key}.{name}"] = scalar
    return out


def fit_and_predict(
    model: EmotionModel,
    train: PreparedSplit,
    eval_sets: "Mapping[str, PreparedSplit]",
    *,
    seed: int = DEFAULT_SEED,
    meta: Optional[dict] = None,
) -> "dict[str, PredictionRecord]":
    """Train ``model`` once on ``train``, then predict on every evaluation set.

    ``eval_sets`` maps a name (by convention ``"<corpus>:<split>"``) to a prepared
    split over the same labels as ``train``. Returns one record per set, in the
    order given. Every record carries the caller's ``meta`` plus what was measured:
    ``model``, ``params`` (see :func:`model_params`), ``seed``, ``n_train``,
    ``train_seconds``, ``model_size_mb``, ``device``, and per set ``eval_set``,
    ``n_eval``, ``predict_seconds``. Measured values win over caller keys of the
    same name.

    Sets are predicted in the order given, each timed on its own. The first one
    includes any one-off warm-up cost, so list the set whose latency you report
    first.
    """
    if not eval_sets:
        raise ValueError("eval_sets must name at least one evaluation set")
    for name, split in eval_sets.items():
        if list(split.label_names) != list(train.label_names):
            raise ValueError(
                f"evaluation set {name!r} has labels {list(split.label_names)} but "
                f"the model is trained on {list(train.label_names)}"
            )

    set_seed(seed)

    t0 = time.perf_counter()
    model.fit(train.texts, train.Y)
    train_seconds = time.perf_counter() - t0

    predicted: dict = {}
    for name, split in eval_sets.items():
        t1 = time.perf_counter()
        y_prob = model.predict_proba(split.texts)
        predicted[name] = (y_prob, time.perf_counter() - t1)

    run_meta = dict(meta or {})
    run_meta.update({
        "model": getattr(model, "name", model.__class__.__name__),
        "params": model_params(model),
        "seed": int(seed),
        "n_train": len(train),
        "train_seconds": float(train_seconds),
        "model_size_mb": float(model.size_mb()),
        "device": _detect_device(model),
    })

    records: dict = {}
    for name, split in eval_sets.items():
        y_prob, predict_seconds = predicted[name]
        records[name] = PredictionRecord(
            y_true=split.Y,
            y_prob=y_prob,
            label_names=list(split.label_names),
            meta={**run_meta, "eval_set": name, "n_eval": len(split),
                  "predict_seconds": float(predict_seconds)},
        )
    return records


def score(
    record: PredictionRecord,
    *,
    dataset_name: Optional[str] = None,
    threshold: float = 0.5,
    bootstrap: int = 0,
    cost_usd: Optional[float] = None,
    extra_meta: Optional[dict] = None,
) -> ScorecardRow:
    """Turn one prediction record into a scorecard row.

    Timings, model size and device come from the record's manifest, so a record
    loaded from the cache scores exactly as it did in process. ``dataset_name``
    defaults to the record's evaluation-set name. ``bootstrap`` (default 0 = off)
    is the number of resamples for confidence intervals on macro/micro-F1 and
    per-label F1.
    """
    m = record.meta
    n = len(record)
    seed = int(m.get("seed", DEFAULT_SEED))
    predict_seconds = float(m.get("predict_seconds", 0.0))
    latency_ms = (predict_seconds / max(n, 1)) * 1000.0
    throughput = (n / predict_seconds) if predict_seconds > 0 else 0.0

    scores = evaluate(record.y_true, record.y_prob, threshold=threshold)
    per_label = per_label_metrics(record.y_true, record.y_prob, record.label_names, threshold)
    per_label_f1 = {name: v["f1"] for name, v in per_label.items()}
    per_label_support = {name: v["support"] for name, v in per_label.items()}

    macro_f1_ci = micro_f1_ci = None
    per_label_f1_ci: dict = {}
    if bootstrap:
        agg = bootstrap_f1_ci(record.y_true, record.y_prob, threshold,
                              n_boot=bootstrap, seed=seed)
        macro_f1_ci = list(agg["macro"])
        micro_f1_ci = list(agg["micro"])
        per_label_f1_ci = {
            name: list(bounds)
            for name, bounds in bootstrap_per_label_f1_ci(
                record.y_true, record.y_prob, record.label_names, threshold,
                n_boot=bootstrap, seed=seed,
            ).items()
        }

    meta = {
        "n_train": m.get("n_train"),
        "n_test": n,
        "n_labels": record.n_labels,
        "labels": list(record.label_names),
        "threshold": threshold,
        "seed": seed,
    }
    if extra_meta:
        meta.update(extra_meta)

    return ScorecardRow(
        model=m.get("model", "unknown"),
        dataset=dataset_name if dataset_name is not None else m.get("eval_set", "unknown"),
        macro_f1=scores["macro_f1"],
        micro_f1=scores["micro_f1"],
        subset_accuracy=scores["subset_accuracy"],
        ece=scores["ece"],
        train_seconds=float(m.get("train_seconds", 0.0)),
        predict_latency_ms=latency_ms,
        throughput=throughput,
        model_size_mb=float(m.get("model_size_mb", 0.0)),
        cost_usd=cost_usd if cost_usd is not None else m.get("cost_usd"),
        device=m.get("device"),
        per_label_f1=per_label_f1,
        per_label_support=per_label_support,
        macro_f1_ci=macro_f1_ci,
        micro_f1_ci=micro_f1_ci,
        per_label_f1_ci=per_label_f1_ci,
        meta=meta,
    )


def run_experiment(
    model: EmotionModel,
    train: PreparedSplit,
    test: PreparedSplit,
    *,
    dataset_name: str,
    threshold: float = 0.5,
    seed: int = DEFAULT_SEED,
    cost_usd: Optional[float] = None,
    bootstrap: int = 0,
    extra_meta: Optional[dict] = None,
) -> ScorecardRow:
    """Train ``model`` on ``train``, evaluate on ``test``, return a scorecard row.

    The one-call form: :func:`fit_and_predict` on a single evaluation set followed
    by :func:`score`. ``bootstrap`` (default 0 = off) is the number of bootstrap
    resamples for test-set confidence intervals.
    """
    records = fit_and_predict(model, train, {"test": test}, seed=seed)
    return score(records["test"], dataset_name=dataset_name, threshold=threshold,
                 bootstrap=bootstrap, cost_usd=cost_usd, extra_meta=extra_meta)
