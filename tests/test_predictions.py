"""Tests for the prediction cache (numpy + stdlib; no torch, no network)."""

import numpy as np
import pytest

from emotion_classification.predictions import (
    PredictionRecord, list_predictions, load_prediction, prediction_path,
    read_meta, save_prediction, slug,
)


def _record(dtype=np.float64):
    y_true = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 0], [1, 1, 0]], dtype=np.float32)
    y_prob = np.array([[0.9, 0.2, 0.1], [0.3, 0.8, 0.5000000001],
                       [0.1, 0.1, 0.4999999999], [0.7, 0.6, 0.2]], dtype=dtype)
    meta = {"model": "logreg_tfidf", "seed": 42, "eval_set": "go_emotions:test",
            "train_seconds": 1.5, "predict_seconds": 0.25}
    return PredictionRecord(y_true, y_prob, ["joy", "anger", "fear"], meta)


def test_record_reports_size():
    rec = _record()
    assert len(rec) == 4
    assert rec.n_labels == 3


def test_record_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        PredictionRecord(np.zeros((4, 3)), np.zeros((4, 2)), ["a", "b", "c"])


def test_record_rejects_wrong_label_count():
    with pytest.raises(ValueError):
        PredictionRecord(np.zeros((4, 3)), np.zeros((4, 3)), ["a", "b"])


def test_slug_makes_names_file_safe():
    assert slug("go_emotions:test") == "go_emotions-test"
    assert slug("go_emotions->sem_eval") == "go_emotions-sem_eval"
    assert slug("hf_bert-base-uncased") == "hf_bert-base-uncased"


def test_prediction_path_layout(tmp_path):
    path = prediction_path(tmp_path, "ch1 run", "logreg_tfidf", 42, "go_emotions:test")
    assert path == tmp_path / "ch1-run" / "logreg_tfidf__seed42__go_emotions-test.npz"


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_roundtrip_is_exact(tmp_path, dtype):
    rec = _record(dtype)
    path = save_prediction(rec, tmp_path / "nested" / "rec.npz")
    back = load_prediction(path)
    assert back.y_prob.dtype == dtype                 # dtype preserved, no rounding
    assert np.array_equal(back.y_prob, rec.y_prob)    # bit-for-bit
    assert np.array_equal(back.y_true, rec.y_true)
    assert back.label_names == ["joy", "anger", "fear"]
    assert back.meta == rec.meta


def test_save_leaves_no_temp_file(tmp_path):
    save_prediction(_record(), tmp_path / "rec.npz")
    assert [p.name for p in tmp_path.iterdir()] == ["rec.npz"]


def test_read_meta_returns_manifest_only(tmp_path):
    path = save_prediction(_record(), tmp_path / "rec.npz")
    assert read_meta(path)["eval_set"] == "go_emotions:test"


def test_list_predictions_walks_the_tree(tmp_path):
    save_prediction(_record(), prediction_path(tmp_path, "run", "m", 42, "a:test"))
    save_prediction(_record(), prediction_path(tmp_path, "run", "m", 43, "a:test"))
    metas = list_predictions(tmp_path)
    assert len(metas) == 2
    assert all(m["path"].endswith(".npz") for m in metas)
    assert all(m["model"] == "logreg_tfidf" for m in metas)
