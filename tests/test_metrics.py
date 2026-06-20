"""Tests for the metrics contract (numpy + sklearn only)."""

import numpy as np

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
