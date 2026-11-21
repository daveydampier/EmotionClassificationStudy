"""Tests for the split runner: fit once, predict on many, score from the record.

Tiny synthetic data and scikit-learn only (no network, no torch), so it runs in CI.
"""

import json

import numpy as np
import pytest

from emotion_classification.experiment import (
    fit_and_predict, model_params, run_experiment, score,
)
from emotion_classification.models.registry import build_model, model_names
from emotion_classification.predictions import load_prediction, save_prediction
from emotion_classification.preprocessing import PreparedSplit
from tests.legacy_runner import legacy_run_experiment, without_timings
from tests.synthetic import LABELS, synthetic_split

pytest.importorskip("sklearn")

from emotion_classification.models.classical import (  # noqa: E402
    LogisticReg, NaiveBayes, RandomForest,
)


@pytest.mark.parametrize("make_model", [
    lambda: NaiveBayes(),
    lambda: LogisticReg(),
    lambda: RandomForest(n_estimators=10),
], ids=["naive_bayes", "logreg", "random_forest"])
def test_run_experiment_matches_the_legacy_runner_exactly(make_model):
    train, test = synthetic_split(160, seed=1), synthetic_split(80, seed=2)
    kwargs = dict(dataset_name="toy", seed=7, bootstrap=40,
                  extra_meta={"schema": "native", "features": "tfidf"})
    legacy = legacy_run_experiment(make_model(), train, test, **kwargs)
    new = run_experiment(make_model(), train, test, **kwargs)
    assert without_timings(new) == without_timings(legacy)


def test_fit_and_predict_returns_one_record_per_eval_set():
    train = synthetic_split(120, seed=1)
    evals = {"toy:validation": synthetic_split(40, seed=2),
             "toy:test": synthetic_split(60, seed=3)}
    records = fit_and_predict(LogisticReg(), train, evals, seed=5,
                              meta={"run_name": "r", "schema": "native"})
    assert list(records) == ["toy:validation", "toy:test"]
    val, test = records["toy:validation"], records["toy:test"]
    assert len(val) == 40 and len(test) == 60
    assert test.label_names == LABELS
    assert np.array_equal(test.y_true, evals["toy:test"].Y)
    assert ((test.y_prob >= 0) & (test.y_prob <= 1)).all()
    for name, rec in records.items():
        m = rec.meta
        assert m["eval_set"] == name
        assert m["model"] == "logreg_tfidf" and m["seed"] == 5 and m["n_train"] == 120
        assert m["run_name"] == "r" and m["schema"] == "native"     # caller meta kept
        assert m["train_seconds"] >= 0 and m["predict_seconds"] >= 0
        assert m["model_size_mb"] > 0 and m["device"] == "cpu"
    assert val.meta["n_eval"] == 40 and test.meta["n_eval"] == 60
    assert val.meta["train_seconds"] == test.meta["train_seconds"]  # one fit, shared


def test_fit_and_predict_trains_exactly_once():
    class Counting(LogisticReg):
        fits = 0

        def fit(self, texts, Y):
            type(self).fits += 1
            return super().fit(texts, Y)

    train = synthetic_split(80, seed=1)
    evals = {f"toy:{i}": synthetic_split(20, seed=10 + i) for i in range(3)}
    fit_and_predict(Counting(), train, evals)
    assert Counting.fits == 1


def test_measured_values_win_over_caller_meta():
    train = synthetic_split(80, seed=1)
    records = fit_and_predict(LogisticReg(), train, {"toy:test": train}, seed=3,
                              meta={"seed": 999, "model": "spoofed"})
    assert records["toy:test"].meta["seed"] == 3
    assert records["toy:test"].meta["model"] == "logreg_tfidf"


def test_fit_and_predict_requires_an_eval_set():
    with pytest.raises(ValueError):
        fit_and_predict(LogisticReg(), synthetic_split(20, seed=1), {})


def test_fit_and_predict_rejects_a_different_label_space():
    train = synthetic_split(40, seed=1)
    other = PreparedSplit(["a b"], np.zeros((1, 2), dtype=np.float32), ["joy", "anger"])
    with pytest.raises(ValueError, match="labels"):
        fit_and_predict(LogisticReg(), train, {"other:test": other})


def test_fit_and_predict_records_the_model_hyperparameters():
    train = synthetic_split(60, seed=1)
    record = fit_and_predict(LogisticReg(C=0.5), train, {"toy:test": train})["toy:test"]
    params = record.meta["params"]
    assert params["head.C"] == 0.5
    assert params["features"] == "tfidf"
    assert params["vectorizer.max_features"] == 50_000


def test_model_params_are_flat_scalars_for_every_tier():
    logreg = model_params(LogisticReg(C=2.0, features="bow"))
    assert logreg["head.C"] == 2.0 and logreg["features"] == "bow"
    assert logreg["vectorizer.ngram_range"] == [1, 2]       # tuples become lists
    assert logreg["vectorizer.min_df"] == 2
    forest = model_params(RandomForest(n_estimators=7))
    assert forest["head.n_estimators"] == 7
    # Torch models: constructor arguments only; fitted and private state is left out.
    bilstm = model_params(build_model("bilstm", epochs=3))
    assert bilstm["epochs"] == 3 and bilstm["hidden_dim"] == 128
    assert not any(key.startswith("_") or key.endswith("_") for key in bilstm)
    bert = model_params(build_model("hf_bert-base-uncased"))
    assert bert["model_name"] == "bert-base-uncased" and bert["lr"] == 2e-5


def test_model_params_survive_json_for_every_registered_model():
    for name in model_names():
        params = model_params(build_model(name))
        assert params, name
        assert json.loads(json.dumps(params)) == params


def test_model_params_of_an_object_without_attributes_is_empty():
    assert model_params(object()) == {}


def test_score_from_cache_equals_score_in_process(tmp_path):
    train, test = synthetic_split(160, seed=1), synthetic_split(80, seed=2)
    record = fit_and_predict(LogisticReg(), train, {"toy:test": test}, seed=7)["toy:test"]
    in_process = score(record, dataset_name="toy", bootstrap=40)
    cached = score(load_prediction(save_prediction(record, tmp_path / "rec.npz")),
                   dataset_name="toy", bootstrap=40)
    assert cached.as_dict() == in_process.as_dict()   # every field, timings included


def test_score_defaults_dataset_to_the_eval_set_name():
    train = synthetic_split(60, seed=1)
    record = fit_and_predict(NaiveBayes(), train, {"toy:test": train})["toy:test"]
    assert score(record).dataset == "toy:test"


def test_score_derives_latency_and_throughput_from_the_record():
    train = synthetic_split(60, seed=1)
    record = fit_and_predict(NaiveBayes(), train, {"toy:test": train})["toy:test"]
    record.meta["predict_seconds"] = 0.5
    row = score(record)
    assert row.predict_latency_ms == pytest.approx(0.5 / 60 * 1000.0)
    assert row.throughput == pytest.approx(120.0)
