"""Tests for the label-harmonization map. No third-party deps required."""

from emotion_classification import labels as L


def test_goemotions_has_28_with_neutral_last():
    assert len(L.GOEMOTIONS) == 28
    assert len(set(L.GOEMOTIONS)) == 28  # no duplicates
    assert L.GOEMOTIONS[-1] == "neutral"


def test_ekman_groups_cover_all_27_exactly_once():
    grouped = [e for labs in L.EKMAN_GROUPS.values() for e in labs]
    assert sorted(grouped) == sorted(e for e in L.GOEMOTIONS if e != "neutral")
    assert len(grouped) == len(set(grouped))  # each emotion in exactly one group
    assert set(L.EKMAN_GROUPS) == set(L.EKMAN)


def test_sentiment_groups_cover_all_27_exactly_once():
    grouped = [e for labs in L.SENTIMENT_GROUPS.values() for e in labs]
    assert sorted(grouped) == sorted(e for e in L.GOEMOTIONS if e != "neutral")
    assert len(grouped) == len(set(grouped))
    assert set(L.SENTIMENT_GROUPS) == set(L.SENTIMENT)


def test_goemotions_to_ekman_covers_every_label():
    assert set(L.GOEMOTIONS_TO_EKMAN) == set(L.GOEMOTIONS)
    assert L.GOEMOTIONS_TO_EKMAN["neutral"] == "neutral"
    # every non-neutral target is a valid Ekman emotion
    targets = {v for k, v in L.GOEMOTIONS_TO_EKMAN.items() if k != "neutral"}
    assert targets <= set(L.EKMAN)


def test_goemotions_to_ekman_examples():
    assert L.goemotions_to_ekman(["annoyance", "grief"]) == ["anger", "sadness"]
    assert L.goemotions_to_ekman(["love", "joy"]) == ["joy"]  # both collapse
    assert L.goemotions_to_ekman(["neutral"]) == ["neutral"]


def test_semeval2018_to_ekman_drops_unmapped():
    # anticipation + trust have no Ekman equivalent and are dropped
    assert L.semeval2018_to_ekman(["anticipation", "trust"]) == []
    assert L.semeval2018_to_ekman(["love", "optimism"]) == ["joy"]
    assert L.semeval2018_to_ekman(["pessimism"]) == ["sadness"]
    assert L.semeval2018_to_ekman(["anger", "anticipation"]) == ["anger"]


def test_unknown_label_raises():
    import pytest

    with pytest.raises(KeyError):
        L.goemotions_to_ekman(["not_an_emotion"])
