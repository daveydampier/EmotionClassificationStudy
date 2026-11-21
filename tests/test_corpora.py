"""Tests for the corpus registry and schema resolution (pure Python; no network)."""

import pytest

from emotion_classification import corpora
from emotion_classification import labels as L
from emotion_classification.corpora import (
    CorpusSpec, get_corpus, register, resolve_schema,
)
from emotion_classification.errors import ConfigError
from emotion_classification.loaders import EmotionDataset


def test_builtin_corpora_are_registered():
    assert corpora.corpus_names() == ["go_emotions", "sem_eval_2018_task_1"]
    go = get_corpus("go_emotions")
    assert go.prefix == "go" and go.multi_label
    assert list(go.native_labels) == L.GOEMOTIONS
    assert get_corpus("sem_eval_2018_task_1").prefix == "sem"


def test_unknown_corpus_lists_known_ones():
    with pytest.raises(ConfigError, match="go_emotions"):
        get_corpus("nope")


def test_register_refuses_duplicates():
    with pytest.raises(ValueError, match="already registered"):
        register(get_corpus("go_emotions"))


def test_native_schema_uses_native_labels_and_no_projector():
    names, projector = resolve_schema("go_emotions", "native")
    assert names == L.GOEMOTIONS
    assert projector is None


def test_ekman_schema_matches_the_label_module_projection():
    names, projector = resolve_schema("go_emotions", "ekman6")
    assert names == [*L.EKMAN, L.NEUTRAL]
    sample = ["annoyance", "grief", "neutral"]
    assert projector(sample) == L.goemotions_to_ekman(sample) == ["anger", "neutral", "sadness"]


def test_semeval_ekman_projection_drops_unmapped_labels():
    names, projector = resolve_schema("sem_eval_2018_task_1", "ekman6")
    assert names == [*L.EKMAN, L.NEUTRAL]
    assert projector(["trust", "love", "pessimism"]) == ["joy", "sadness"]


def test_sentiment_schema_only_where_defined():
    names, projector = resolve_schema("go_emotions", "sentiment3")
    assert names == [*L.SENTIMENT, L.NEUTRAL]
    assert projector(["joy", "anger"]) == ["negative", "positive"]
    with pytest.raises(ConfigError, match="sentiment3"):
        resolve_schema("sem_eval_2018_task_1", "sentiment3")


def test_unknown_schema_rejected():
    with pytest.raises(ConfigError, match="unknown schema"):
        resolve_schema("go_emotions", "plutchik8")


def test_corpus_without_ekman_map_rejects_ekman_schema(monkeypatch):
    spec = CorpusSpec(name="toy", prefix="toy",
                      loader=lambda: EmotionDataset("toy", ["a"], {}),
                      native_labels=("a",))
    monkeypatch.setitem(corpora.CORPORA, "toy", spec)
    with pytest.raises(ConfigError, match="Ekman"):
        resolve_schema("toy", "ekman6")


def test_config_error_is_a_value_error():
    assert issubclass(ConfigError, ValueError)
