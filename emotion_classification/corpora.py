"""Corpus registry: every dataset the study can train on or evaluate against.

One :class:`CorpusSpec` per corpus records how to load it, its native label set,
and how those labels project into the shared schemas. The pipeline and the
command line look corpora up here by name, so adding a corpus is one loader plus
one :func:`register` call, with no change to the runner.

Loaders import the HuggingFace ``datasets`` package lazily, so importing this
module needs neither that package nor the network.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Optional

from . import labels as L
from .errors import ConfigError
from .loaders import EmotionDataset, load_goemotions, load_semeval2018
from .preprocessing import Projector

SCHEMAS: tuple = ("native", "ekman6", "sentiment3")


@dataclass(frozen=True)
class CorpusSpec:
    """How to load one corpus and project its labels."""

    name: str                 # registry key, e.g. "go_emotions"
    prefix: str               # results-file stem prefix, e.g. "go"
    loader: Callable[[], EmotionDataset]
    native_labels: tuple
    ekman_map: Optional[Mapping] = None      # native label -> Ekman label (None drops it)
    sentiment_map: Optional[Mapping] = None  # native label -> sentiment label
    multi_label: bool = True


CORPORA: dict = {}


def register(spec: CorpusSpec) -> CorpusSpec:
    """Add ``spec`` to the registry. A name can be registered only once."""
    if spec.name in CORPORA:
        raise ValueError(f"corpus {spec.name!r} is already registered")
    CORPORA[spec.name] = spec
    return spec


def get_corpus(name: str) -> CorpusSpec:
    """Look up a corpus by name."""
    try:
        return CORPORA[name]
    except KeyError:
        known = ", ".join(sorted(CORPORA))
        raise ConfigError(f"unknown corpus {name!r}; known corpora: {known}") from None


def corpus_names() -> list:
    """Names of every registered corpus, sorted."""
    return sorted(CORPORA)


def _projector(mapping: Mapping) -> Projector:
    def project(native_labels):
        return L.map_labels(native_labels, mapping)

    return project


def resolve_schema(corpus: str, schema: str):
    """Return ``(label_names, projector)`` for ``corpus`` under ``schema``.

    ``projector`` maps a list of native labels into the target schema; it is
    ``None`` for the ``native`` schema, meaning "use the corpus's labels as-is".
    """
    spec = get_corpus(corpus)
    if schema == "native":
        return list(spec.native_labels), None
    if schema == "ekman6":
        if spec.ekman_map is None:
            raise ConfigError(f"corpus {corpus!r} has no Ekman map")
        return [*L.EKMAN, L.NEUTRAL], _projector(spec.ekman_map)
    if schema == "sentiment3":
        if spec.sentiment_map is None:
            raise ConfigError(f"the sentiment3 schema is not defined for corpus {corpus!r}")
        return [*L.SENTIMENT, L.NEUTRAL], _projector(spec.sentiment_map)
    raise ConfigError(f"unknown schema {schema!r}; choose from {', '.join(SCHEMAS)}")


register(CorpusSpec(
    name="go_emotions", prefix="go", loader=load_goemotions,
    native_labels=tuple(L.GOEMOTIONS),
    ekman_map=L.GOEMOTIONS_TO_EKMAN, sentiment_map=L.GOEMOTIONS_TO_SENTIMENT,
))
register(CorpusSpec(
    name="sem_eval_2018_task_1", prefix="sem", loader=load_semeval2018,
    native_labels=tuple(L.SEMEVAL2018),
    ekman_map=L.SEMEVAL2018_TO_EKMAN,
))
