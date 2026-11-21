"""The experiment runner as it was before the runner was split (reference only).

``legacy_run_experiment`` is a verbatim copy of ``run_experiment`` from before
``fit_and_predict`` and ``score`` existed. Nothing in the package uses it. Tests
compare the new code paths against it to prove the rewrite changed no result.
"""

import time

from emotion_classification.experiment import _detect_device, set_seed
from emotion_classification.metrics import (
    bootstrap_f1_ci, bootstrap_per_label_f1_ci, evaluate, per_label_metrics,
)
from emotion_classification.scorecard import ScorecardRow

TIMING_FIELDS = {"train_seconds", "predict_latency_ms", "throughput"}


def without_timings(row) -> dict:
    """A scorecard row as a dict, minus the fields that differ run to run."""
    return {k: v for k, v in row.as_dict().items() if k not in TIMING_FIELDS}


def legacy_run_experiment(model, train, test, *, dataset_name, threshold=0.5, seed=42,
                          cost_usd=None, bootstrap=0, extra_meta=None):
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
