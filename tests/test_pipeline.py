"""Tests for the pipeline: eval-set fan-out, the prediction cache, limits, subsampling."""

import json
from datetime import datetime

import numpy as np
import pytest

from emotion_classification import corpora
from emotion_classification.corpora import CorpusSpec
from emotion_classification.errors import ConfigError
from emotion_classification.experiment import fit_and_predict
from emotion_classification.loaders import EmotionDataset
from emotion_classification.models.registry import build_model
from emotion_classification.pipeline import (
    RunConfig, dataset_label, head, parse_eval_set, run_pipeline, score_record, subsample,
)
from emotion_classification.predictions import load_prediction
from emotion_classification.preprocessing import PreparedSplit, prepare_dataset
from tests.legacy_runner import legacy_run_experiment, without_timings
from tests.synthetic import LABELS, synthetic_split, toy_dataset

pytest.importorskip("sklearn")


def _register(monkeypatch, name, loader, native_labels=tuple(LABELS)):
    spec = CorpusSpec(name=name, prefix=name, loader=loader, native_labels=native_labels)
    monkeypatch.setitem(corpora.CORPORA, name, spec)


# -- small helpers ------------------------------------------------------------

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


# -- fan-out and the scorecard --------------------------------------------------

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


@pytest.mark.parametrize("model_name", ["logreg", "random_forest"])
def test_pipeline_matches_the_old_command_line_path(toy_corpora, model_name):
    """Old vs new for: --dataset toy --test-dataset toy_b --subsample-train 50 --seed 43."""
    train = prepare_dataset(corpora.get_corpus("toy").loader(), LABELS)["train"]
    test = prepare_dataset(corpora.get_corpus("toy_b").loader(), LABELS)["test"]
    # The old script drew the subsample with RandomState(seed), then ran with that seed.
    idx = np.sort(np.random.RandomState(43).choice(len(train), size=50, replace=False))
    sub = PreparedSplit([train.texts[i] for i in idx], train.Y[idx], list(LABELS))
    legacy = legacy_run_experiment(
        build_model(model_name), sub, test, dataset_name="toy->toy_b", seed=43, bootstrap=20,
        extra_meta={"schema": "native", "features": "tfidf", "epochs": None,
                    "subsample_train": 50, "test_dataset": "toy_b"})

    result = run_pipeline(RunConfig(models=[model_name], dataset="toy",
                                    eval_sets=["toy_b:test"], seeds=[43],
                                    subsample_train=50, bootstrap=20))
    new = without_timings(result.card.rows[0])
    assert new["meta"].pop("eval_set") == "toy_b:test"   # the one key the old row lacked
    assert new == without_timings(legacy)


def test_multiple_seeds_and_models_give_one_row_each(toy_corpora):
    result = run_pipeline(RunConfig(models=["naive_bayes", "logreg"], dataset="toy",
                                    seeds=[1, 2, 3]))
    assert len(result.card) == 6
    assert [row.meta["seed"] for row in result.card.rows[:3]] == [1, 2, 3]


def test_limits_cap_rows(toy_corpora):
    result = run_pipeline(RunConfig(models=["naive_bayes"], dataset="toy",
                                    limit_train=30, limit_eval=20))
    row = result.card.rows[0]
    assert row.meta["n_train"] == 30 and row.meta["n_test"] == 20


# -- the prediction cache --------------------------------------------------------

def test_cached_records_score_back_to_the_scorecard_on_their_own(toy_corpora, tmp_path):
    cfg = RunConfig(models=["logreg", "naive_bayes"], dataset="toy",
                    eval_sets=["toy:test", "toy_b:test"], bootstrap=30,
                    predictions_dir=tmp_path, run_name="exact")
    result = run_pipeline(cfg)
    by_key = {(row.model, row.meta["eval_set"]): row for row in result.card.rows}

    rebuilt = {}
    for path in result.prediction_paths:
        record = load_prediction(path)
        if record.meta["scored"]:
            # Nothing but the file and the bootstrap setting goes in.
            row = score_record(record, bootstrap=30)
            rebuilt[(row.model, row.meta["eval_set"])] = row
    assert set(rebuilt) == set(by_key) and len(rebuilt) == 4
    for key, row in by_key.items():
        assert rebuilt[key].as_dict() == row.as_dict()   # every field, timings included


def test_the_validation_record_is_flagged_as_unscored(toy_corpora, tmp_path):
    cfg = RunConfig(models=["logreg"], dataset="toy", predictions_dir=tmp_path)
    flags = {load_prediction(p).meta["eval_set"]: load_prediction(p).meta["scored"]
             for p in run_pipeline(cfg).prediction_paths}
    assert flags == {"toy:test": True, "toy:validation": False}


def test_the_manifest_describes_the_run(toy_corpora, tmp_path):
    cfg = RunConfig(models=["logreg"], dataset="toy", eval_sets=["toy_b:test"],
                    limit_train=100, limit_eval=30, predictions_dir=tmp_path, run_name="m")
    result = run_pipeline(cfg)
    meta = next(load_prediction(p).meta for p in result.prediction_paths
                if p.name.endswith("toy_b-test.npz"))
    assert meta["dataset"] == "toy->toy_b" and meta["test_dataset"] == "toy_b"
    assert meta["train_corpus"] == "toy" and meta["schema"] == "native"
    assert meta["model"] == "logreg_tfidf"               # the fitted model's own name
    assert meta["model_key"] == "logreg"                 # the registry name asked for
    assert meta["route"] == "S"
    assert meta["params"]["head.C"] == 1.0
    assert meta["limit_train"] == 100 and meta["limit_eval"] == 30
    assert meta["n_train"] == 100 and meta["n_eval"] == 30
    assert {"python", "numpy", "scikit-learn"} <= set(meta["versions"])
    datetime.fromisoformat(meta["created_utc"])          # parses as a timestamp
    assert json.loads(json.dumps(meta)) == meta


def test_score_record_needs_a_pipeline_manifest():
    train = synthetic_split(40, seed=1)
    record = fit_and_predict(build_model("naive_bayes"), train, {"toy:test": train})["toy:test"]
    with pytest.raises(ValueError, match="run_pipeline"):
        score_record(record)


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


def test_a_corpus_without_the_default_validation_split_still_runs(monkeypatch, tmp_path):
    _register(monkeypatch, "toy_nv", lambda: toy_dataset("toy_nv", 300, splits=("train", "test")))
    lines = []
    result = run_pipeline(RunConfig(models=["logreg"], dataset="toy_nv",
                                    predictions_dir=tmp_path), log=lines.append)
    assert [p.name for p in result.prediction_paths] == ["logreg_tfidf__seed42__toy_nv-test.npz"]
    assert any("no 'validation' split" in line for line in lines)   # said out loud


def test_a_validation_split_asked_for_by_name_must_exist(toy_corpora, tmp_path):
    with pytest.raises(ConfigError, match="no split 'valid'"):
        run_pipeline(RunConfig(models=["logreg"], dataset="toy", val_split="valid",
                               predictions_dir=tmp_path))


# -- subsampling -------------------------------------------------------------------

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


def test_the_header_reports_the_size_each_seed_trains_on(toy_corpora):
    lines = []
    run_pipeline(RunConfig(models=["naive_bayes"], dataset="toy", subsample_train=50),
                 log=lines.append)
    assert any("train=50" in line and "of 160" in line for line in lines)


# -- configuration mistakes are caught before anything expensive happens -------------

def test_an_unknown_model_fails_before_anything_is_loaded(toy_corpora):
    with pytest.raises(ConfigError, match="unknown model 'logregg'"):
        run_pipeline(RunConfig(models=["logreg", "logregg"], dataset="toy"))
    assert toy_corpora == {"toy": 0, "toy_b": 0}         # no corpus loaded, so nothing trained


def test_bad_eval_sets_fail_before_any_corpus_is_loaded(toy_corpora):
    for eval_sets, message in ((["toy:test", "nope:test"], "unknown corpus"),
                               (["toy"], "<corpus>:<split>"),
                               (["toy:test", "toy:test"], "duplicate")):
        with pytest.raises(ConfigError, match=message):
            run_pipeline(RunConfig(models=["logreg"], dataset="toy", eval_sets=eval_sets))
    assert toy_corpora == {"toy": 0, "toy_b": 0}


def test_a_different_label_space_is_rejected(toy_corpora, monkeypatch):
    _register(monkeypatch, "toy_c",
              lambda: EmotionDataset("toy_c", ["x", "y"], {"train": [], "test": []}),
              native_labels=("x", "y"))
    with pytest.raises(ConfigError, match="label space"):
        run_pipeline(RunConfig(models=["logreg"], dataset="toy", eval_sets=["toy_c:test"]))
    assert toy_corpora == {"toy": 0, "toy_b": 0}


def test_an_unknown_split_is_rejected(toy_corpora):
    with pytest.raises(ConfigError, match="no split"):
        run_pipeline(RunConfig(models=["logreg"], dataset="toy", eval_sets=["toy:dev"]))


def test_registry_and_loader_must_agree_on_labels(monkeypatch):
    # The registry promises LABELS; the loader hands back a different label set.
    _register(monkeypatch, "toy_bad",
              lambda: EmotionDataset("toy_bad", ["joy", "anger"], {"train": [], "test": []}))
    with pytest.raises(ConfigError, match="registry"):
        run_pipeline(RunConfig(models=["logreg"], dataset="toy_bad"))


def test_empty_models_or_seeds_are_rejected(toy_corpora):
    with pytest.raises(ConfigError):
        run_pipeline(RunConfig(models=[], dataset="toy"))
    with pytest.raises(ConfigError):
        run_pipeline(RunConfig(models=["logreg"], dataset="toy", seeds=[]))


def test_two_fits_that_would_share_a_cache_file_are_rejected(toy_corpora, tmp_path):
    for kwargs in ({"models": ["logreg", "logreg"]},                    # same model twice
                   {"models": ["hf_distilbert", "transformer"]},        # two names, one model
                   {"models": ["hf_org-a/tiny-bert", "hf_org-b/tiny-bert"]},  # same basename
                   {"models": ["logreg"], "seeds": [42, 42]}):          # same seed twice
        with pytest.raises(ConfigError, match="same prediction file"):
            run_pipeline(RunConfig(dataset="toy", predictions_dir=tmp_path, **kwargs))
    assert toy_corpora == {"toy": 0, "toy_b": 0}
    assert list(tmp_path.iterdir()) == []


def test_a_repeated_model_is_still_allowed_when_nothing_is_cached(toy_corpora):
    result = run_pipeline(RunConfig(models=["naive_bayes", "naive_bayes"], dataset="toy"))
    assert len(result.card) == 2                         # the old behavior, unchanged


def test_settings_the_manifest_cannot_store_fail_before_training(toy_corpora, tmp_path):
    cfg = RunConfig(models=["logreg"], dataset="toy", subsample_train=np.int64(50),
                    predictions_dir=tmp_path)
    with pytest.raises(ConfigError, match="manifest"):
        run_pipeline(cfg)
    assert toy_corpora == {"toy": 0, "toy_b": 0}
    assert list(tmp_path.iterdir()) == []


def test_progress_lines_keep_the_macro_f1_marker(toy_corpora):
    lines = []
    run_pipeline(RunConfig(models=["logreg"], dataset="toy"), log=lines.append)
    assert any("=== logreg ===" in line for line in lines)
    assert any("macro_f1=" in line for line in lines)    # the GPU batch scripts grep for this
