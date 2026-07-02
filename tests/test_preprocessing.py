"""Tests for label projection / multi-hot preparation."""

import numpy as np

from emotion_classification.loaders import Example
from emotion_classification.preprocessing import (
    PreparedSplit, label_support, prepare_split, to_multihot,
)


def test_label_support_counts_positives_per_label():
    Y = np.array([[1, 0, 1], [0, 0, 1]], dtype=np.float32)
    split = PreparedSplit(["a", "b"], Y, ["x", "y", "z"])
    assert label_support(split) == {"x": 1, "y": 0, "z": 2}


def test_to_multihot_basic():
    Y = to_multihot([["a", "c"], ["b"]], ["a", "b", "c"])
    assert Y.tolist() == [[1, 0, 1], [0, 1, 0]]


def test_to_multihot_ignores_unknown_labels():
    Y = to_multihot([["a", "zzz"]], ["a", "b"])
    assert Y.tolist() == [[1, 0]]


def test_prepare_split_with_projector():
    examples = [
        Example(text="furious", labels=["annoyance"]),
        Example(text="weeping", labels=["grief"]),
    ]
    # goemotions -> ekman: annoyance->anger, grief->sadness
    from emotion_classification import labels as L

    split = prepare_split(examples, [*L.EKMAN, "neutral"], L.goemotions_to_ekman)
    assert split.texts == ["furious", "weeping"]
    anger_col = split.label_names.index("anger")
    sadness_col = split.label_names.index("sadness")
    assert split.Y[0, anger_col] == 1.0
    assert split.Y[1, sadness_col] == 1.0


def test_prepare_split_drops_empty_projection():
    # anticipation/trust drop out of the Ekman schema -> example removed
    from emotion_classification import labels as L

    examples = [
        Example(text="kept", labels=["anger"]),
        Example(text="dropped", labels=["anticipation"]),
    ]
    split = prepare_split(examples, [*L.EKMAN, "neutral"], L.semeval2018_to_ekman)
    assert split.texts == ["kept"]
    assert split.Y.shape == (1, len(L.EKMAN) + 1)


def test_prepare_split_keep_empty_when_drop_disabled():
    from emotion_classification import labels as L

    examples = [Example(text="dropped", labels=["anticipation"])]
    split = prepare_split(examples, [*L.EKMAN, "neutral"], L.semeval2018_to_ekman,
                          drop_empty=False)
    assert split.texts == ["dropped"]
    assert np.all(split.Y == 0)
