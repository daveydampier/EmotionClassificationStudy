"""Tests for results persistence (stdlib json + csv; no pandas/torch)."""

import csv
import json

import pytest

from emotion_classification.results import (
    load_results, save_results, save_summary, summarize,
)
from emotion_classification.scorecard import Scorecard, ScorecardRow


def _card():
    return Scorecard([
        ScorecardRow(
            model="logreg_tfidf", dataset="go_emotions",
            macro_f1=0.45, micro_f1=0.55, subset_accuracy=0.30, ece=0.11,
            train_seconds=0.4, predict_latency_ms=0.02, throughput=50000.0,
            model_size_mb=1.4, device="cpu",
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
    assert record["throughput"] == "50000.0"   # efficiency axis (RQ1)
    assert record["n_labels"] == "28"


def test_per_emotion_long_format(tmp_path):
    paths = save_results(_card(), tmp_path, "run1")
    with paths["per_emotion"].open(encoding="utf-8") as fh:
        records = list(csv.DictReader(fh))
    by_emotion = {r["emotion"]: r for r in records}
    assert set(by_emotion) == {"joy", "grief"}
    assert by_emotion["grief"]["support"] == "6"
    assert by_emotion["joy"]["schema"] == "native"
    assert by_emotion["joy"]["seed"] == "42"   # seed column distinguishes multi-seed rows


def _multiseed_card():
    # same model config, two seeds; macro_f1 differs, size identical.
    return Scorecard([
        ScorecardRow(model="logreg_tfidf", dataset="go_emotions", macro_f1=0.40,
                     train_seconds=8.0, model_size_mb=12.0,
                     meta={"schema": "native", "features": "tfidf", "seed": 42}),
        ScorecardRow(model="logreg_tfidf", dataset="go_emotions", macro_f1=0.44,
                     train_seconds=9.0, model_size_mb=12.0,
                     meta={"schema": "native", "features": "tfidf", "seed": 43}),
    ])


def test_summarize_aggregates_seeds():
    summaries = summarize(_multiseed_card())
    assert len(summaries) == 1          # two seeds collapse to one model line
    row = summaries[0]
    assert row["n_seeds"] == 2
    assert row["macro_f1_mean"] == pytest.approx(0.42)   # mean of 0.40, 0.44
    assert row["macro_f1_std"] > 0                        # values differ
    assert row["model_size_mb_std"] == 0.0       # identical -> zero std


def test_summarize_accepts_row_dicts(tmp_path):
    # round-trip through JSON (dicts) still summarizes
    paths = save_results(_multiseed_card(), tmp_path, "ms")
    summaries = summarize(load_results(paths["json"]))
    assert summaries[0]["n_seeds"] == 2


def test_save_summary_writes_csv(tmp_path):
    path = save_summary(_multiseed_card(), tmp_path, "run1")
    with path.open(encoding="utf-8") as fh:
        rec = next(csv.DictReader(fh))
    assert rec["model"] == "logreg_tfidf"
    assert rec["n_seeds"] == "2"
    assert "macro_f1_mean" in rec and "macro_f1_std" in rec


def test_bootstrap_cis_persisted(tmp_path):
    card = Scorecard([ScorecardRow(
        model="m", dataset="d", macro_f1=0.45, micro_f1=0.55,
        macro_f1_ci=[0.43, 0.47], micro_f1_ci=[0.53, 0.57],
        per_label_f1={"joy": 0.8}, per_label_support={"joy": 100},
        per_label_f1_ci={"joy": [0.75, 0.85]},
        meta={"schema": "native", "seed": 42},
    )])
    paths = save_results(card, tmp_path, "ci")
    with paths["scorecard"].open(encoding="utf-8") as fh:
        rec = next(csv.DictReader(fh))
    assert rec["macro_f1_low"] == "0.43" and rec["macro_f1_high"] == "0.47"
    with paths["per_emotion"].open(encoding="utf-8") as fh:
        pe = next(csv.DictReader(fh))
    assert pe["f1_low"] == "0.75" and pe["f1_high"] == "0.85"


def test_save_results_creates_missing_dir(tmp_path):
    target = tmp_path / "nested" / "results"
    paths = save_results(_card(), target, "run1")
    assert paths["json"].parent == target
    assert json.loads(paths["json"].read_text(encoding="utf-8"))  # non-empty
