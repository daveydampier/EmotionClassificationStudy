"""Tests for the pipeline: eval-set fan-out, the prediction cache, limits, subsampling."""

import numpy as np
import pytest

from emotion_classification import corpora
from emotion_classification.corpora import CorpusSpec
from emotion_classification.errors import ConfigError
from emotion_classification.experiment import score
from emotion_classification.loaders import EmotionDataset
from emotion_classification.pipeline import (
    RunConfig, dataset_label, head, parse_eval_set, run_pipeline, subsample,
)
from emotion_classification.predictions import load_prediction
from tests.synthetic import synthetic_split

pytest.importorskip("sklearn")

SCORE_META_KEYS = ("schema", "features", "epochs", "subsample_train", "test_dataset", "eval_set")


def test_parse_eval_set():
    assert parse_eval_set("go_emotions:test") == ("go_emotions", "test")
    for bad in ("go_emotions", ":test", "go_emotions:", ""):
        with pytest.raises(ConfigError):
            parse_eval_set(bad)


def test_head_and_subsample_do_not_modify_their_input():
    split = synthetic_split(50, seed=1)
    texts, Y = list(split.texts), split.Y.copy()
    assert len(head(split, 10)) == 10
    assert len(subsample(split, 10, seed=3)) == 10
    assert split.texts == texts and np.array_equal(split.Y, Y)


def test_subsample_is_the_seeded_draw_the_matched_size_runs_used():
    split = synthetic_split(50, seed=1)
    sub = subsample(split, 12, seed=43)
    idx = np.sort(np.random.RandomState(43).choice(50, size=12, replace=False))
    assert sub.texts == [split.texts[i] for i in idx]
    assert np.array_equal(sub.Y, split.Y[idx])
    assert subsample(split, 50, seed=43) is split        # nothing to cut


def test_dataset_label_convention():
    cfg = RunConfig(models=["logreg"], dataset="toy")
    assert dataset_label(cfg, "toy:test") == "toy"
    assert dataset_label(cfg, "toy_b:test") == "toy->toy_b"
    assert dataset_label(cfg, "toy:validation") == "toy:validation"
    assert dataset_label(cfg, "toy_b:validation") == "toy->toy_b:validation"


def test_default_run_scores_the_test_split(toy_corpora):
    result = run_pipeline(RunConfig(models=["logreg"], dataset="toy"))
    assert [row.dataset for row in result.card.rows] == ["toy"]
    row = result.card.rows[0]
    assert row.model == "logreg_tfidf"
    assert row.meta["n_train"] == 160 and row.meta["n_test"] == 60
    assert row.meta["eval_set"] == "toy:test" and row.meta["test_dataset"] == "toy"
    assert result.prediction_paths == []                 # nothing cached unless asked


def test_one_fit_fans_out_to_every_eval_set(toy_corpora, tmp_path):
    cfg = RunConfig(models=["logreg"], dataset="toy", eval_sets=["toy:test", "toy_b:test"],
                    predictions_dir=tmp_path, run_name="fan")
    result = run_pipeline(cfg)
    assert [row.dataset for row in result.card.rows] == ["toy", "toy->toy_b"]
    assert sorted(path.name for path in result.prediction_paths) == [
        "logreg_tfidf__seed42__toy-test.npz",
        "logreg_tfidf__seed42__toy-validation.npz",      # cached for tuning, not scored
        "logreg_tfidf__seed42__toy_b-test.npz",
    ]
    assert all(path.parent == tmp_path / "fan" for path in result.prediction_paths)
    assert toy_corpora == {"toy": 1, "toy_b": 1}         # each corpus loaded once


def test_rescoring_the_cache_reproduces_the_scorecard_exactly(toy_corpora, tmp_path):
    cfg = RunConfig(models=["logreg", "naive_bayes"], dataset="toy",
                    eval_sets=["toy:test", "toy_b:test"], bootstrap=30,
                    predictions_dir=tmp_path, run_name="exact")
    result = run_pipeline(cfg)
    by_key = {(row.model, row.meta["eval_set"]): row for row in result.card.rows}
    rescored = 0
    for path in result.prediction_paths:
        record = load_prediction(path)
        key = (record.meta["model"], record.meta["eval_set"])
        if key not in by_key:            # the validation record has no scorecard row
            continue
        row = by_key[key]
        again = score(record, dataset_name=row.dataset, bootstrap=30,
                      extra_meta={k: row.meta[k] for k in SCORE_META_KEYS})
        assert again.as_dict() == row.as_dict()
        rescored += 1
    assert rescored == 4


def test_validation_caching_can_be_switched_off(toy_corpora, tmp_path):
    cfg = RunConfig(models=["logreg"], dataset="toy", val_split=None,
                    predictions_dir=tmp_path, run_name="noval")
    names = [path.name for path in run_pipeline(cfg).prediction_paths]
    assert names == ["logreg_tfidf__seed42__toy-test.npz"]


def test_validation_can_be_scored_when_named_explicitly(toy_corpora, tmp_path):
    cfg = RunConfig(models=["logreg"], dataset="toy",
                    eval_sets=["toy:validation", "toy:test"], predictions_dir=tmp_path)
    result = run_pipeline(cfg)
    assert [row.dataset for row in result.card.rows] == ["toy:validation", "toy"]
    assert len(result.prediction_paths) == 2             # no duplicate validation record


def test_each_seed_draws_its_own_subsample(toy_corpora, tmp_path):
    cfg = RunConfig(models=["logreg"], dataset="toy", seeds=[42, 43], subsample_train=50,
                    predictions_dir=tmp_path, run_name="sub")
    result = run_pipeline(cfg)
    assert [row.meta["n_train"] for row in result.card.rows] == [50, 50]
    assert [row.meta["seed"] for row in result.card.rows] == [42, 43]
    a, b = (load_prediction(p) for p in result.prediction_paths
            if p.name.endswith("toy-test.npz"))
    # LogReg is deterministic, so different probabilities mean different training rows.
    assert not np.array_equal(a.y_prob, b.y_prob)


def test_a_seed_gives_the_same_subsample_alone_or_in_a_sweep(toy_corpora, tmp_path):
    def test_record(result, seed):
        return next(load_prediction(p) for p in result.prediction_paths
                    if f"seed{seed}__toy-test" in p.name)

    sweep = run_pipeline(RunConfig(models=["logreg"], dataset="toy", seeds=[42, 43],
                                   subsample_train=50, predictions_dir=tmp_path / "sweep"))
    alone = run_pipeline(RunConfig(models=["logreg"], dataset="toy", seeds=[43],
                                   subsample_train=50, predictions_dir=tmp_path / "alone"))
    assert np.array_equal(test_record(sweep, 43).y_prob, test_record(alone, 43).y_prob)


def test_limits_cap_rows(toy_corpora):
    result = run_pipeline(RunConfig(models=["naive_bayes"], dataset="toy",
                                    limit_train=30, limit_eval=20))
    row = result.card.rows[0]
    assert row.meta["n_train"] == 30 and row.meta["n_test"] == 20


def test_multiple_seeds_and_models_give_one_row_each(toy_corpora):
    result = run_pipeline(RunConfig(models=["naive_bayes", "logreg"], dataset="toy",
                                    seeds=[1, 2, 3]))
    assert len(result.card) == 6
    assert [row.meta["seed"] for row in result.card.rows[:3]] == [1, 2, 3]


def test_a_different_label_space_is_rejected(toy_corpora, monkeypatch):
    other = CorpusSpec(name="toy_c", prefix="toy_c",
                       loader=lambda: EmotionDataset("toy_c", ["x", "y"], {"train": [], "test": []}),
                       native_labels=("x", "y"))
    monkeypatch.setitem(corpora.CORPORA, "toy_c", other)
    with pytest.raises(ConfigError, match="label space"):
        run_pipeline(RunConfig(models=["logreg"], dataset="toy", eval_sets=["toy_c:test"]))


def test_unknown_split_and_duplicate_eval_sets_are_rejected(toy_corpora):
    with pytest.raises(ConfigError, match="no split"):
        run_pipeline(RunConfig(models=["logreg"], dataset="toy", eval_sets=["toy:dev"]))
    with pytest.raises(ConfigError, match="duplicate"):
        run_pipeline(RunConfig(models=["logreg"], dataset="toy",
                               eval_sets=["toy:test", "toy:test"]))


def test_empty_models_or_seeds_are_rejected(toy_corpora):
    with pytest.raises(ConfigError):
        run_pipeline(RunConfig(models=[], dataset="toy"))
    with pytest.raises(ConfigError):
        run_pipeline(RunConfig(models=["logreg"], dataset="toy", seeds=[]))


def test_progress_lines_keep_the_macro_f1_marker(toy_corpora):
    lines = []
    run_pipeline(RunConfig(models=["logreg"], dataset="toy"), log=lines.append)
    assert any("=== logreg ===" in line for line in lines)
    assert any("macro_f1=" in line for line in lines)    # the GPU batch scripts grep for this
