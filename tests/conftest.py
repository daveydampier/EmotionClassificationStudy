"""Shared fixtures: two in-memory toy corpora (no network, no torch)."""

import numpy as np
import pytest

from emotion_classification import corpora
from emotion_classification.corpora import CorpusSpec
from emotion_classification.loaders import EmotionDataset, Example
from tests.synthetic import LABELS, synthetic_split


def _toy_dataset(name: str, seed: int) -> EmotionDataset:
    def to_examples(split):
        return [Example(text, [LABELS[j] for j in np.flatnonzero(row)])
                for text, row in zip(split.texts, split.Y)]

    return EmotionDataset(name, list(LABELS), {
        "train": to_examples(synthetic_split(160, seed)),
        "validation": to_examples(synthetic_split(40, seed + 1)),
        "test": to_examples(synthetic_split(60, seed + 2)),
    })


@pytest.fixture
def toy_corpora(monkeypatch):
    """Register corpora ``toy`` and ``toy_b``; return how often each was loaded."""
    loads = {"toy": 0, "toy_b": 0}

    def loader(name, seed):
        def load():
            loads[name] += 1
            return _toy_dataset(name, seed)

        return load

    for name, seed in (("toy", 100), ("toy_b", 200)):
        spec = CorpusSpec(name=name, prefix=name, loader=loader(name, seed),
                          native_labels=tuple(LABELS))
        monkeypatch.setitem(corpora.CORPORA, name, spec)
    return loads
