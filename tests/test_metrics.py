"""Tests for the metrics contract (numpy + sklearn only)."""

import numpy as np
import pytest

from emotion_classification import metrics as M


def test_perfect_predictions_score_one():
    y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=float)
    y_prob = y_true.copy()  # perfectly confident & correct
    s = M.classification_metrics(y_true, y_prob)
    assert s["macro_f1"] == 1.0
    assert s["micro_f1"] == 1.0
    assert s["subset_accuracy"] == 1.0


def test_subset_accuracy_is_exact_match():
    y_true = np.array([[1, 0], [1, 1]], dtype=float)
    # first row correct, second row one label wrong -> exact match = 0.5
    y_prob = np.array([[0.9, 0.1], [0.9, 0.2]])
    s = M.classification_metrics(y_true, y_prob, threshold=0.5)
    assert s["subset_accuracy"] == 0.5


def test_binarize_threshold():
    probs = np.array([[0.4, 0.6], [0.5, 0.49]])
    out = M.binarize(probs, threshold=0.5)
    assert out.tolist() == [[0, 1], [1, 0]]


def test_ece_zero_when_confident_and_correct():
    y_true = np.array([[1, 0], [0, 1]], dtype=float)
    y_prob = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert M.expected_calibration_error(y_true, y_prob) == 0.0


def test_ece_high_when_confident_and_wrong():
    y_true = np.array([[1, 1], [1, 1]], dtype=float)
    y_prob = np.array([[0.0, 0.0], [0.0, 0.0]])  # confidently wrong
    # confidence ~1.0, accuracy 0 -> ECE near 1.0
    assert M.expected_calibration_error(y_true, y_prob) > 0.9


def test_evaluate_bundles_all_axes():
    y_true = np.array([[1, 0, 1]], dtype=float)
    y_prob = np.array([[0.8, 0.2, 0.7]])
    s = M.evaluate(y_true, y_prob)
    assert {"macro_f1", "micro_f1", "subset_accuracy", "ece"} <= set(s)


def test_per_label_f1_keys_and_perfect():
    y_true = np.array([[1, 0], [0, 1]], dtype=float)
    y_prob = y_true.copy()
    d = M.per_label_f1(y_true, y_prob, ["joy", "anger"])
    assert set(d) == {"joy", "anger"}
    assert d["joy"] == 1.0 and d["anger"] == 1.0


def test_per_label_metrics_support_and_zero_f1():
    # column 'a' has 2 positives, 'b' has 2; predictions are all-zero -> F1 = 0.
    y_true = np.array([[1, 0], [1, 1], [0, 1]], dtype=float)
    y_prob = np.zeros((3, 2))
    m = M.per_label_metrics(y_true, y_prob, ["a", "b"])
    assert m["a"]["support"] == 2
    assert m["b"]["support"] == 2
    assert m["a"]["f1"] == 0.0 and m["b"]["f1"] == 0.0


def test_per_label_metrics_length_mismatch_raises():
    y_true = np.array([[1, 0, 1]], dtype=float)
    y_prob = np.array([[0.9, 0.1, 0.8]])
    with pytest.raises(ValueError):
        M.per_label_metrics(y_true, y_prob, ["only", "two"])


def test_bootstrap_f1_ci_valid_intervals():
    rng = np.random.default_rng(0)
    y_true = (rng.random((200, 3)) > 0.6).astype(float)
    y_prob = rng.random((200, 3))
    ci = M.bootstrap_f1_ci(y_true, y_prob, n_boot=100, seed=0)
    for key in ("macro", "micro"):
        lo, hi = ci[key]
        assert 0.0 <= lo <= hi <= 1.0


def test_bootstrap_reproducible_with_seed():
    rng = np.random.default_rng(1)
    y_true = (rng.random((150, 2)) > 0.5).astype(float)
    y_prob = rng.random((150, 2))
    assert M.bootstrap_f1_ci(y_true, y_prob, n_boot=80, seed=7) == \
        M.bootstrap_f1_ci(y_true, y_prob, n_boot=80, seed=7)


def test_bootstrap_per_label_ci_structure_and_rare_is_wider():
    # label 0 frequent + stable; label 1 rare + a couple of errors -> wider CI
    n = 300
    y_true = np.zeros((n, 2))
    y_true[:150, 0] = 1
    y_true[:6, 1] = 1
    y_prob = y_true * 0.9 + 0.05
    y_prob[0, 1] = 0.1     # one rare-label error to create instability
    y_prob[200, 1] = 0.9
    ci = M.bootstrap_per_label_f1_ci(y_true, y_prob, ["freq", "rare"], n_boot=200, seed=0)
    assert set(ci) == {"freq", "rare"}
    for lo, hi in ci.values():
        assert 0.0 <= lo <= hi <= 1.0
    rare_width = ci["rare"][1] - ci["rare"][0]
    freq_width = ci["freq"][1] - ci["freq"][0]
    assert rare_width > 0 and rare_width >= freq_width


def test_bootstrap_per_label_ci_length_mismatch_raises():
    y_true = np.array([[1, 0, 1]], dtype=float)
    y_prob = np.array([[0.9, 0.1, 0.8]])
    with pytest.raises(ValueError):
        M.bootstrap_per_label_f1_ci(y_true, y_prob, ["only", "two"], n_boot=10)
