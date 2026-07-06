"""End-to-end test of the classical tier through the experiment runner.

Uses tiny synthetic data (no network, no torch) so it runs in CI and exercises
the whole spine: model interface -> metrics -> scorecard row.
"""

import numpy as np
import pytest

from emotion_classification.experiment import run_experiment
from emotion_classification.preprocessing import PreparedSplit

sklearn = pytest.importorskip("sklearn")

from emotion_classification.models.classical import (  # noqa: E402
    LogisticReg, NaiveBayes, RandomForest,
)


def _toy_data():
    # Two cleanly separable labels keyed off distinctive tokens.
    pos = ["happy joyful glad delighted", "joyful and glad today"]
    neg = ["angry furious mad rage", "furious and mad now"]
    texts = pos + neg
    label_names = ["joy", "anger"]
    Y = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)
    return PreparedSplit(texts, Y, label_names)


def test_logreg_fit_predict_shapes():
    data = _toy_data()
    model = LogisticReg().fit(data.texts, data.Y)
    proba = model.predict_proba(data.texts)
    assert proba.shape == (4, 2)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert model.size_mb() > 0


def test_constant_label_column_handled():
    # 'anger' is never positive -> _BinaryRelevance must not raise and predicts 0.
    texts = ["happy glad", "joyful glad", "glad happy"]
    Y = np.array([[1, 0], [1, 0], [1, 0]], dtype=np.float32)
    model = LogisticReg().fit(texts, Y)
    proba = model.predict_proba(texts)
    assert proba.shape == (3, 2)
    assert np.allclose(proba[:, 1], 0.0)  # constant-zero column


def test_naive_bayes_bow_backend_and_name():
    data = _toy_data()
    model = NaiveBayes(features="bow")
    assert model.name == "naive_bayes_bow"
    model.fit(data.texts, data.Y)
    proba = model.predict_proba(data.texts)
    assert proba.shape == (4, 2)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_random_forest_fit_predict():
    data = _toy_data()
    model = RandomForest(features="tfidf", n_estimators=10)
    assert model.name == "random_forest_tfidf"
    model.fit(data.texts, data.Y)
    proba = model.predict_proba(data.texts)
    assert proba.shape == (4, 2)


def test_unknown_features_rejected():
    with pytest.raises(ValueError):
        NaiveBayes(features="word2vec")


def test_run_experiment_emits_scorecard_row():
    data = _toy_data()
    row = run_experiment(LogisticReg(), data, data, dataset_name="toy",
                         extra_meta={"schema": "native", "features": "tfidf"})
    assert row.model == "logreg_tfidf"
    assert row.dataset == "toy"
    assert 0.0 <= row.macro_f1 <= 1.0
    assert row.train_seconds >= 0.0
    assert row.model_size_mb > 0.0
    assert row.meta["n_labels"] == 2
    assert row.meta["schema"] == "native"


def test_run_experiment_records_device_and_per_emotion():
    data = _toy_data()
    row = run_experiment(LogisticReg(), data, data, dataset_name="toy")
    # classical models run on CPU
    assert row.device == "cpu"
    # throughput (samples/sec) recorded for RQ1; roughly the inverse of latency
    assert row.throughput > 0.0
    # per-emotion breakdown covers every label, with support = positive counts
    assert set(row.per_label_f1) == {"joy", "anger"}
    assert set(row.per_label_support) == {"joy", "anger"}
    assert row.per_label_support["joy"] == 2
    assert all(0.0 <= f1 <= 1.0 for f1 in row.per_label_f1.values())


def test_run_experiment_bootstrap_populates_cis():
    data = _toy_data()
    row = run_experiment(LogisticReg(), data, data, dataset_name="toy", bootstrap=50)
    assert row.macro_f1_ci is not None and len(row.macro_f1_ci) == 2
    assert row.macro_f1_ci[0] <= row.macro_f1_ci[1]
    assert row.micro_f1_ci is not None
    assert set(row.per_label_f1_ci) == {"joy", "anger"}


def test_run_experiment_no_bootstrap_leaves_cis_empty():
    data = _toy_data()
    row = run_experiment(LogisticReg(), data, data, dataset_name="toy")
    assert row.macro_f1_ci is None
    assert row.per_label_f1_ci == {}
