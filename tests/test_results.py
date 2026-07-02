"""Tests for results persistence (stdlib json + csv; no pandas/torch)."""

import csv
import json

from emotion_classification.results import load_results, save_results
from emotion_classification.scorecard import Scorecard, ScorecardRow


def _card():
    return Scorecard([
        ScorecardRow(
            model="logreg_tfidf", dataset="go_emotions",
            macro_f1=0.45, micro_f1=0.55, subset_accuracy=0.30, ece=0.11,
            train_seconds=0.4, predict_latency_ms=0.02, model_size_mb=1.4,
            device="cpu",
            per_label_f1={"joy": 0.8, "grief": 0.1},
            per_label_support={"joy": 500, "grief": 6},
            meta={"schema": "native", "features": "tfidf", "n_train": 4000,
                  "n_test": 1500, "n_labels": 28, "seed": 42},
        ),
    ])


def test_save_results_writes_three_files(tmp_path):
    paths = save_results(_card(), tmp_path, "run1")
    assert paths["json"].exists()
    assert paths["scorecard"].exists()
    assert paths["per_emotion"].exists()


def test_json_roundtrips(tmp_path):
    paths = save_results(_card(), tmp_path, "run1")
    rows = load_results(paths["json"])
    assert rows[0]["model"] == "logreg_tfidf"
    assert rows[0]["device"] == "cpu"
    assert rows[0]["per_label_f1"]["grief"] == 0.1
    assert rows[0]["meta"]["schema"] == "native"


def test_scorecard_csv_flattens_meta(tmp_path):
    paths = save_results(_card(), tmp_path, "run1")
    with paths["scorecard"].open(encoding="utf-8") as fh:
        record = next(csv.DictReader(fh))
    assert record["model"] == "logreg_tfidf"
    assert record["device"] == "cpu"
    assert record["schema"] == "native"        # pulled from meta
    assert record["macro_f1"] == "0.45"
    assert record["n_labels"] == "28"


def test_per_emotion_long_format(tmp_path):
    paths = save_results(_card(), tmp_path, "run1")
    with paths["per_emotion"].open(encoding="utf-8") as fh:
        records = list(csv.DictReader(fh))
    by_emotion = {r["emotion"]: r for r in records}
    assert set(by_emotion) == {"joy", "grief"}
    assert by_emotion["grief"]["support"] == "6"
    assert by_emotion["joy"]["schema"] == "native"


def test_save_results_creates_missing_dir(tmp_path):
    target = tmp_path / "nested" / "results"
    paths = save_results(_card(), target, "run1")
    assert paths["json"].parent == target
    assert json.loads(paths["json"].read_text(encoding="utf-8"))  # non-empty
