"""The experiment pipeline: train each model once and fan it out to many evaluation sets.

:func:`run_pipeline` takes a :class:`RunConfig`, loads the training corpus and
every corpus an evaluation set names, and for each (model, seed) trains once and
predicts on all evaluation sets. Each prediction is scored into a scorecard row
and, when a predictions directory is given, written to the prediction cache so
it can be re-scored later (tuned thresholds, calibration, significance tests)
without a retrain. :func:`score_record` rebuilds a scorecard row from a cached
file alone.

An evaluation set is named ``"<corpus>:<split>"``, for example
``go_emotions:test`` or ``sem_eval_2018_task_1:test``.
"""

from __future__ import annotations

import json
import platform
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Optional

import numpy as np

from .corpora import get_corpus, resolve_schema
from .errors import ConfigError
from .experiment import fit_and_predict, score
from .models.registry import build_model, get_model_spec
from .predictions import PredictionRecord, prediction_path, save_prediction, slug
from .preprocessing import PreparedSplit, prepare_dataset
from .scorecard import Scorecard, ScorecardRow

DEFAULT_VAL_SPLIT = "validation"

# Manifest fields that are copied into a scorecard row's ``meta``.
ROW_META_KEYS = ("schema", "features", "epochs", "subsample_train", "test_dataset", "eval_set")


@dataclass
class RunConfig:
    """Everything that defines one pipeline run."""

    models: list
    dataset: str = "go_emotions"            # the training corpus
    schema: str = "native"
    features: str = "tfidf"
    train_split: str = "train"
    test_split: str = "test"
    # Evaluation sets to score, as "<corpus>:<split>". None means the training
    # corpus's test split.
    eval_sets: Optional[list] = None
    # Training-corpus split predicted and cached (not scored) for later tuning.
    # Only used when predictions_dir is set; None turns it off.
    val_split: Optional[str] = DEFAULT_VAL_SPLIT
    seeds: list = field(default_factory=lambda: [42])
    epochs: Optional[int] = None
    limit_train: Optional[int] = None       # first N training rows (smoke runs)
    limit_eval: Optional[int] = None        # first N rows of every evaluation set
    subsample_train: Optional[int] = None   # random N training rows, redrawn per seed
    bootstrap: int = 0
    predictions_dir: Optional[Path] = None  # where to cache predictions; None = don't
    run_name: str = "run"


@dataclass
class RunResult:
    """What a pipeline run produced."""

    card: Scorecard
    prediction_paths: list = field(default_factory=list)


def parse_eval_set(name: str) -> tuple:
    """Split ``"<corpus>:<split>"`` into ``(corpus, split)``."""
    corpus, sep, split = name.partition(":")
    if not sep or not corpus or not split:
        raise ConfigError(f"an evaluation set is named '<corpus>:<split>', got {name!r}")
    return corpus, split


def head(split: PreparedSplit, n: int) -> PreparedSplit:
    """The first ``n`` rows of ``split`` as a new object; the input is not modified."""
    return PreparedSplit(split.texts[:n], split.Y[:n], list(split.label_names))


def subsample(split: PreparedSplit, n: int, seed: int) -> PreparedSplit:
    """A random ``n``-row subset of ``split``, drawn with ``seed``.

    The draw is ``np.random.RandomState(seed).choice`` without replacement, rows
    kept in their original order: the same draw the matched-size control runs
    used. If ``n`` is not smaller than the split, the split is returned as-is.
    The input is never modified.
    """
    if n >= len(split):
        return split
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(len(split), size=n, replace=False))
    return PreparedSplit([split.texts[i] for i in idx], split.Y[idx], list(split.label_names))


def dataset_label(cfg: RunConfig, eval_set: str) -> str:
    """The scorecard ``dataset`` value for one evaluation set.

    The training corpus's test split is just the corpus name; another corpus is
    ``"<train>-><corpus>"``; any split other than the test split is appended as
    ``":<split>"``.
    """
    corpus, split = parse_eval_set(eval_set)
    label = cfg.dataset if corpus == cfg.dataset else f"{cfg.dataset}->{corpus}"
    return label if split == cfg.test_split else f"{label}:{split}"


def score_record(record: PredictionRecord, *, threshold: float = 0.5,
                 bootstrap: int = 0) -> ScorecardRow:
    """Rebuild the scorecard row for a record written by :func:`run_pipeline`.

    The dataset label and the row's descriptive fields come from the record's own
    manifest, so a cached file is enough: nothing has to be passed back in. The
    pipeline builds its own rows through this function, which is what keeps a
    cached row and the original identical. ``bootstrap`` is a property of the
    scoring, not of the predictions, so it is not stored; pass the value the
    original run used to reproduce its intervals.
    """
    m = record.meta
    missing = [key for key in ("dataset", *ROW_META_KEYS) if key not in m]
    if missing:
        raise ValueError(
            f"this record's manifest lacks {', '.join(missing)}; "
            f"it was not written by run_pipeline"
        )
    return score(record, dataset_name=m["dataset"], threshold=threshold,
                 bootstrap=bootstrap, extra_meta={key: m[key] for key in ROW_META_KEYS})


def _versions() -> dict:
    """Interpreter and library versions, recorded in every manifest."""
    out = {"python": platform.python_version()}
    for package in ("numpy", "scikit-learn", "torch", "transformers"):
        try:
            out[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    return out


def _run_meta(cfg: RunConfig, model_key: str) -> dict:
    """The part of a manifest that is known before a model is trained."""
    spec = get_model_spec(model_key)
    return {
        "run_name": cfg.run_name,
        "train_corpus": cfg.dataset,
        "train_split": cfg.train_split,
        "schema": cfg.schema,
        "features": cfg.features,
        "epochs": cfg.epochs,
        "subsample_train": cfg.subsample_train,
        "limit_train": cfg.limit_train,
        "limit_eval": cfg.limit_eval,
        "model_key": spec.name,     # the registry name; "model" is the fitted model's own
        "route": spec.route,
    }


def _check_cache_targets(cfg: RunConfig) -> None:
    """Refuse a run in which two fits would write the same prediction file.

    A cache file is keyed by (run name, model name, seed, evaluation set). The
    model name is the fitted model's own, so two registry names for one model, two
    checkpoints with the same basename, a repeated model, or a repeated seed all
    collide. Building a model is cheap and trains nothing.
    """
    seen: dict = {}
    for model_key in cfg.models:
        model = build_model(model_key, features=cfg.features, epochs=cfg.epochs)
        cache_name = slug(getattr(model, "name", model.__class__.__name__))
        for seed in cfg.seeds:
            target = (cache_name, int(seed))
            if target in seen:
                raise ConfigError(
                    f"{seen[target]!r} and {model_key!r} at seed {int(seed)} would write "
                    f"the same prediction file ({cache_name}__seed{int(seed)}); drop the "
                    f"repeat, or run them under separate run names"
                )
            seen[target] = model_key


def _preflight(cfg: RunConfig) -> tuple:
    """Check everything that can be checked without loading a corpus.

    Returns ``(label_names, report_names)``. Raises :class:`ConfigError` for an
    unknown model, corpus or schema, a malformed or repeated evaluation set, a
    corpus that does not share the training label space, two fits that would share
    a cache file, or a setting the manifest cannot store.
    """
    if not cfg.models:
        raise ConfigError("RunConfig.models must name at least one model")
    if not cfg.seeds:
        raise ConfigError("RunConfig.seeds must hold at least one seed")
    for model_key in cfg.models:
        get_model_spec(model_key)

    label_names, _ = resolve_schema(cfg.dataset, cfg.schema)
    report_names = list(cfg.eval_sets) if cfg.eval_sets else [f"{cfg.dataset}:{cfg.test_split}"]
    for name in report_names:
        parse_eval_set(name)
    if len(set(report_names)) != len(report_names):
        raise ConfigError(f"duplicate evaluation sets: {', '.join(report_names)}")
    for name in report_names:
        corpus, _ = parse_eval_set(name)
        names, _ = resolve_schema(corpus, cfg.schema)
        if names != label_names:
            raise ConfigError(
                f"corpus {corpus!r} does not share a label space with "
                f"{cfg.dataset!r} under the {cfg.schema!r} schema; use a shared "
                f"schema such as ekman6"
            )

    if cfg.predictions_dir:
        _check_cache_targets(cfg)
        try:
            for model_key in cfg.models:
                json.dumps(_run_meta(cfg, model_key))
        except TypeError as err:
            raise ConfigError(
                f"a run setting cannot be stored in the prediction manifest: {err}"
            ) from None
    return label_names, report_names


def run_pipeline(cfg: RunConfig, *, log: Optional[Callable] = None) -> RunResult:
    """Run every model in ``cfg`` over every seed and evaluation set.

    ``log``, if given, is called with one progress line at a time (the command
    line passes ``print``). Configuration problems raise :class:`ConfigError`
    before any model is trained: most of them before a corpus is even loaded, and
    a missing split or a loader that disagrees with the registry right after.
    """
    say = log if log is not None else (lambda message: None)
    label_names, report_names = _preflight(cfg)
    prepared_cache: dict = {}

    def prepared(corpus: str) -> dict:
        if corpus not in prepared_cache:
            spec = get_corpus(corpus)
            names, projector = resolve_schema(corpus, cfg.schema)
            dataset = spec.loader()
            if list(dataset.label_names) != list(spec.native_labels):
                raise ConfigError(
                    f"corpus {corpus!r}: its loader returned labels "
                    f"{list(dataset.label_names)} but the registry lists "
                    f"{list(spec.native_labels)}"
                )
            prepared_cache[corpus] = prepare_dataset(dataset, names, projector)
        return prepared_cache[corpus]

    def take(corpus: str, split: str) -> PreparedSplit:
        splits = prepared(corpus)
        if split not in splits:
            raise ConfigError(
                f"corpus {corpus!r} has no split {split!r}; it has: {', '.join(splits)}"
            )
        return splits[split]

    def capped(split: PreparedSplit) -> PreparedSplit:
        return head(split, cfg.limit_eval) if cfg.limit_eval else split

    train = take(cfg.dataset, cfg.train_split)
    if cfg.limit_train:
        train = head(train, cfg.limit_train)
    eval_sets = {name: capped(take(*parse_eval_set(name))) for name in report_names}

    # The training corpus's validation split is predicted and cached but not scored:
    # it is there so thresholds and calibration can be tuned later without a retrain.
    # It goes last, so it never warms the model up for a set whose latency is reported.
    val_name = None
    if cfg.predictions_dir and cfg.val_split:
        candidate = f"{cfg.dataset}:{cfg.val_split}"
        if candidate not in eval_sets:
            if cfg.val_split in prepared(cfg.dataset):
                val_name = candidate
                eval_sets[val_name] = capped(take(cfg.dataset, cfg.val_split))
            elif cfg.val_split == DEFAULT_VAL_SPLIT:
                say(f"  note: {cfg.dataset} has no '{cfg.val_split}' split, so no "
                    f"predictions are cached for tuning")
            else:
                take(cfg.dataset, cfg.val_split)    # asked for by name: say it is missing

    n_fit = min(cfg.subsample_train, len(train)) if cfg.subsample_train else len(train)
    drawn = f" (random subsample of {len(train)}, redrawn per seed)" if n_fit < len(train) else ""
    sizes = "  ".join(f"{name}={len(split)}" for name, split in eval_sets.items())
    say(f"  train={n_fit}{drawn}  n_labels={len(label_names)}  {sizes}")

    result = RunResult(Scorecard())
    versions = _versions()
    multiseed = len(cfg.seeds) > 1
    for model_key in cfg.models:
        say(f"\n=== {model_key} ===")
        run_meta = _run_meta(cfg, model_key)
        for seed in cfg.seeds:
            fit_train = train
            if cfg.subsample_train:
                fit_train = subsample(train, cfg.subsample_train, seed)
            model = build_model(model_key, features=cfg.features, epochs=cfg.epochs)
            records = fit_and_predict(model, fit_train, eval_sets, seed=seed, meta={
                **run_meta,
                "versions": versions,
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
            for eval_name, record in records.items():
                # Everything a scorecard row needs goes into the manifest first, and
                # the row is then built from the manifest alone (see score_record).
                record.meta.update({
                    "dataset": dataset_label(cfg, eval_name),
                    "test_dataset": parse_eval_set(eval_name)[0],
                    "scored": eval_name != val_name,
                })
                if cfg.predictions_dir:
                    path = prediction_path(cfg.predictions_dir, cfg.run_name,
                                           record.meta["model"], seed, eval_name)
                    result.prediction_paths.append(save_prediction(record, path))
                if eval_name == val_name:
                    continue
                row = score_record(record, bootstrap=cfg.bootstrap)
                result.card.add(row)
                tag = f"[seed {seed}] " if multiseed else ""
                if len(report_names) > 1:
                    tag += f"[{row.dataset}] "
                say(f"  {tag}macro_f1={row.macro_f1:.3f}  micro_f1={row.micro_f1:.3f}  "
                    f"ece={row.ece:.3f}  train={row.train_seconds:.1f}s  "
                    f"latency={row.predict_latency_ms:.2f}ms  thrpt={row.throughput:.0f}/s  "
                    f"size={row.model_size_mb:.1f}MB  device={row.device}")
    return result
