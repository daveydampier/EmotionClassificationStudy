"""Shared fixtures: two in-memory toy corpora (no network, no torch)."""

import pytest

from emotion_classification import corpora
from emotion_classification.corpora import CorpusSpec
from tests.synthetic import LABELS, toy_dataset


@pytest.fixture
def toy_corpora(monkeypatch):
    """Register corpora ``toy`` and ``toy_b``; return how often each was loaded."""
    loads = {"toy": 0, "toy_b": 0}

    def loader(name, seed):
        def load():
            loads[name] += 1
            return toy_dataset(name, seed)

        return load

    for name, seed in (("toy", 100), ("toy_b", 200)):
        spec = CorpusSpec(name=name, prefix=name, loader=loader(name, seed),
                          native_labels=tuple(LABELS))
        monkeypatch.setitem(corpora.CORPORA, name, spec)
    return loads
