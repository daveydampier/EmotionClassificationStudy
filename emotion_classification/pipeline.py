"""The experiment pipeline: train each model once and fan it out to many evaluation sets.

:func:`run_pipeline` takes a :class:`RunConfig`, loads the training corpus and
every corpus an evaluation set names, and for each (model, seed) trains once and
predicts on all evaluation sets. Each prediction is scored into a scorecard row
and, when a predictions directory is given, written to the prediction cache so
it can be re-scored later (tuned thresholds, calibration, significance tests)
without a retrain.

An evaluation set is named ``"<corpus>:<split>"``, for example
``go_emotions:test`` or ``sem_eval_2018_task_1:test``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .corpora import get_corpus, resolve_schema
from .errors import ConfigError
from .experiment import fit_and_predict, score
from .models.registry import build_model
from .predictions import prediction_path, save_prediction
from .preprocessing import PreparedSplit, prepare_dataset
from .scorecard import Scorecard


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
    val_split: Optional[str] = "validation"
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


def run_pipeline(cfg: RunConfig, *, log: Optional[Callable] = None) -> RunResult:
    """Run every model in ``cfg`` over every seed and evaluation set.

    ``log``, if given, is called with one progress line at a time (the command
    line passes ``print``). Configuration problems raise :class:`ConfigError`
    before any model is trained.
    """
    say = log if log is not None else (lambda message: None)
    if not cfg.models:
        raise ConfigError("RunConfig.models must name at least one model")
    if not cfg.seeds:
        raise ConfigError("RunConfig.seeds must hold at least one seed")

    label_names, _ = resolve_schema(cfg.dataset, cfg.schema)
    prepared_cache: dict = {}

    def prepared(corpus: str) -> dict:
        if corpus not in prepared_cache:
            names, projector = resolve_schema(corpus, cfg.schema)
            if names != label_names:
                raise ConfigError(
                    f"corpus {corpus!r} does not share a label space with "
                    f"{cfg.dataset!r} under the {cfg.schema!r} schema; use a shared "
                    f"schema such as ekman6"
                )
            dataset = get_corpus(corpus).loader()
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

    report_names = list(cfg.eval_sets) if cfg.eval_sets else [f"{cfg.dataset}:{cfg.test_split}"]
    if len(set(report_names)) != len(report_names):
        raise ConfigError(f"duplicate evaluation sets: {', '.join(report_names)}")
    eval_sets = {name: capped(take(*parse_eval_set(name))) for name in report_names}

    # The training corpus's validation split is predicted and cached but not scored:
    # it is there so thresholds and calibration can be tuned later without a retrain.
    val_name = None
    if cfg.predictions_dir and cfg.val_split:
        candidate = f"{cfg.dataset}:{cfg.val_split}"
        if candidate not in eval_sets and cfg.val_split in prepared(cfg.dataset):
            val_name = candidate
            eval_sets[val_name] = capped(take(cfg.dataset, cfg.val_split))

    sizes = "  ".join(f"{name}={len(split)}" for name, split in eval_sets.items())
    say(f"  train={len(train)}  n_labels={len(label_names)}  {sizes}")
    if cfg.subsample_train and cfg.subsample_train < len(train):
        say(f"  each seed trains on a random {cfg.subsample_train}-row subsample")

    result = RunResult(Scorecard())
    multiseed = len(cfg.seeds) > 1
    for model_name in cfg.models:
        say(f"\n=== {model_name} ===")
        for seed in cfg.seeds:
            fit_train = train
            if cfg.subsample_train:
                fit_train = subsample(train, cfg.subsample_train, seed)
            model = build_model(model_name, features=cfg.features, epochs=cfg.epochs)
            records = fit_and_predict(model, fit_train, eval_sets, seed=seed, meta={
                "run_name": cfg.run_name,
                "train_corpus": cfg.dataset,
                "train_split": cfg.train_split,
                "schema": cfg.schema,
                "features": cfg.features,
                "epochs": cfg.epochs,
                "subsample_train": cfg.subsample_train,
            })
            for eval_name, record in records.items():
                if cfg.predictions_dir:
                    path = prediction_path(cfg.predictions_dir, cfg.run_name,
                                           record.meta["model"], seed, eval_name)
                    result.prediction_paths.append(save_prediction(record, path))
                if eval_name == val_name:
                    continue
                row = score(record, dataset_name=dataset_label(cfg, eval_name),
                            bootstrap=cfg.bootstrap, extra_meta={
                                "schema": cfg.schema,
                                "features": cfg.features,
                                "epochs": cfg.epochs,
                                "subsample_train": cfg.subsample_train,
                                "test_dataset": parse_eval_set(eval_name)[0],
                                "eval_set": eval_name,
                            })
                result.card.add(row)
                tag = f"[seed {seed}] " if multiseed else ""
                if len(report_names) > 1:
                    tag += f"[{row.dataset}] "
                say(f"  {tag}macro_f1={row.macro_f1:.3f}  micro_f1={row.micro_f1:.3f}  "
                    f"ece={row.ece:.3f}  train={row.train_seconds:.1f}s  "
                    f"latency={row.predict_latency_ms:.2f}ms  thrpt={row.throughput:.0f}/s  "
                    f"size={row.model_size_mb:.1f}MB  device={row.device}")
    return result
