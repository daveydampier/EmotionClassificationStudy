"""Tests for the split runner: fit once, predict on many, score from the record.

Tiny synthetic data and scikit-learn only (no network, no torch), so it runs in CI.
"""

import time

import numpy as np
import pytest

from emotion_classification.experiment import (
    _detect_device, fit_and_predict, run_experiment, score, set_seed,
)
from emotion_classification.metrics import (
    bootstrap_f1_ci, bootstrap_per_label_f1_ci, evaluate, per_label_metrics,
)
from emotion_classification.predictions import load_prediction, save_prediction
from emotion_classification.preprocessing import PreparedSplit
from emotion_classification.scorecard import ScorecardRow
from tests.synthetic import LABELS, synthetic_split

pytest.importorskip("sklearn")

from emotion_classification.models.classical import (  # noqa: E402
    LogisticReg, NaiveBayes, RandomForest,
)

TIMING_FIELDS = {"train_seconds", "predict_latency_ms", "throughput"}


def _legacy_run_experiment(model, train, test, *, dataset_name, threshold=0.5, seed=42,
                           cost_usd=None, bootstrap=0, extra_meta=None):
    """Verbatim copy of ``run_experiment`` from before the runner was split.

    Kept as the reference the new code path must match exactly (timings aside).
    """
    set_seed(seed)

    t0 = time.perf_counter()
    model.fit(train.texts, train.Y)
    train_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_prob = model.predict_proba(test.texts)
    predict_seconds = time.perf_counter() - t1
    latency_ms = (predict_seconds / max(len(test), 1)) * 1000.0
    throughput = (len(test) / predict_seconds) if predict_seconds > 0 else 0.0

    scores = evaluate(test.Y, y_prob, threshold=threshold)
    per_label = per_label_metrics(test.Y, y_prob, test.label_names, threshold)
    per_label_f1 = {name: m["f1"] for name, m in per_label.items()}
    per_label_support = {name: m["support"] for name, m in per_label.items()}

    macro_f1_ci = micro_f1_ci = None
    per_label_f1_ci = {}
    if bootstrap:
        agg = bootstrap_f1_ci(test.Y, y_prob, threshold, n_boot=bootstrap, seed=seed)
        macro_f1_ci = list(agg["macro"])
        micro_f1_ci = list(agg["micro"])
        per_label_f1_ci = {
            name: list(bounds)
            for name, bounds in bootstrap_per_label_f1_ci(
                test.Y, y_prob, test.label_names, threshold, n_boot=bootstrap, seed=seed
            ).items()
        }

    meta = {
        "n_train": len(train),
        "n_test": len(test),
        "n_labels": test.n_labels,
        "labels": test.label_names,
        "threshold": threshold,
        "seed": seed,
    }
    if extra_meta:
        meta.update(extra_meta)

    return ScorecardRow(
        model=getattr(model, "name", model.__class__.__name__),
        dataset=dataset_name,
        macro_f1=scores["macro_f1"],
        micro_f1=scores["micro_f1"],
        subset_accuracy=scores["subset_accuracy"],
        ece=scores["ece"],
        train_seconds=train_seconds,
        predict_latency_ms=latency_ms,
        throughput=throughput,
        model_size_mb=model.size_mb(),
        cost_usd=cost_usd,
        device=_detect_device(model),
        per_label_f1=per_label_f1,
        per_label_support=per_label_support,
        macro_f1_ci=macro_f1_ci,
        micro_f1_ci=micro_f1_ci,
        per_label_f1_ci=per_label_f1_ci,
        meta=meta,
    )


def _without_timings(row):
    return {k: v for k, v in row.as_dict().items() if k not in TIMING_FIELDS}


@pytest.mark.parametrize("make_model", [
    lambda: NaiveBayes(),
    lambda: LogisticReg(),
    lambda: RandomForest(n_estimators=10),
], ids=["naive_bayes", "logreg", "random_forest"])
def test_run_experiment_matches_the_legacy_runner_exactly(make_model):
    train, test = synthetic_split(160, seed=1), synthetic_split(80, seed=2)
    kwargs = dict(dataset_name="toy", seed=7, bootstrap=40,
                  extra_meta={"schema": "native", "features": "tfidf"})
    legacy = _legacy_run_experiment(make_model(), train, test, **kwargs)
    new = run_experiment(make_model(), train, test, **kwargs)
    assert _without_timings(new) == _without_timings(legacy)


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
