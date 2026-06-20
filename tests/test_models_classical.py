"""End-to-end test of the classical tier through the experiment runner.

Uses tiny synthetic data (no network, no torch) so it runs in CI and exercises
the whole spine: model interface -> metrics -> scorecard row.
"""

import numpy as np
import pytest

from emotion_classification.experiment import run_experiment
from emotion_classification.preprocessing import PreparedSplit

sklearn = pytest.importorskip("sklearn")

from emotion_classification.models.classical import TfidfLogReg  # noqa: E402


def _toy_data():
    # Two cleanly separable labels keyed off distinctive tokens.
    pos = ["happy joyful glad delighted", "joyful and glad today"]
    neg = ["angry furious mad rage", "furious and mad now"]
    texts = pos + neg
    label_names = ["joy", "anger"]
    Y = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)
    return PreparedSplit(texts, Y, label_names)


def test_tfidf_logreg_fit_predict_shapes():
    data = _toy_data()
    model = TfidfLogReg().fit(data.texts, data.Y)
    proba = model.predict_proba(data.texts)
    assert proba.shape == (4, 2)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert model.size_mb() > 0


def test_constant_label_column_handled():
    # 'anger' is never positive -> _BinaryRelevance must not raise and predicts 0.
    texts = ["happy glad", "joyful glad", "glad happy"]
    Y = np.array([[1, 0], [1, 0], [1, 0]], dtype=np.float32)
    model = TfidfLogReg().fit(texts, Y)
    proba = model.predict_proba(texts)
    assert proba.shape == (3, 2)
    assert np.allclose(proba[:, 1], 0.0)  # constant-zero column


def test_run_experiment_emits_scorecard_row():
    data = _toy_data()
    row = run_experiment(TfidfLogReg(), data, data, dataset_name="toy")
    assert row.model == "tfidf_logreg"
    assert row.dataset == "toy"
    assert 0.0 <= row.macro_f1 <= 1.0
    assert row.train_seconds >= 0.0
    assert row.model_size_mb > 0.0
    assert row.meta["n_labels"] == 2
