"""Regression against the published table.

Needs the GoEmotions download and trains two classical models on the full corpus
(about a minute), so it is marked ``network`` and ``slow`` and does not run in CI:

    python -m pytest -m "network and slow" tests/test_regression_paper.py
"""

import csv

import pytest

from emotion_classification.config import PROJECT_ROOT
from emotion_classification.pipeline import RunConfig, run_pipeline, score_record
from emotion_classification.predictions import load_prediction

PAPER_TABLE = PROJECT_ROOT / "results" / "paper_final" / "data" / "go_native_final_scorecard.csv"

# The table was produced on the GPU server, and these rows do not reproduce
# bit-for-bit on another machine. The TF-IDF vocabulary is capped at 50,000 terms,
# and on GoEmotions that cut falls inside a block of ~24,000 terms tied at a count
# of 2. scikit-learn picks among ties with an unstable sort, so the rare-term tail
# of the vocabulary can differ between machines, and every classical model moves a
# little with it. Measured on a second machine: at most 5e-4 on the F1 axes and
# 2.4e-3 on subset accuracy (13 of 5,427 test rows change exact-match status).
# The tolerances sit above that noise and far below any difference the study
# interprets; a real regression (wrong split, labels, or features) moves these
# numbers by 1e-2 or more.
TOLERANCE = {"macro_f1": 2e-3, "micro_f1": 2e-3, "subset_accuracy": 5e-3}


@pytest.mark.network
@pytest.mark.slow
def test_classical_rows_match_the_published_table(tmp_path):
    pytest.importorskip("datasets")
    with PAPER_TABLE.open(newline="", encoding="utf-8") as fh:
        paper = {row["model"]: row for row in csv.DictReader(fh)}

    cfg = RunConfig(models=["naive_bayes", "logreg"], dataset="go_emotions",
                    schema="native", predictions_dir=tmp_path, run_name="regression")
    result = run_pipeline(cfg)

    assert [row.model for row in result.card.rows] == ["naive_bayes_tfidf", "logreg_tfidf"]
    for row in result.card.rows:
        published = paper[row.model]
        assert row.dataset == "go_emotions"
        assert row.meta["n_train"] == int(published["n_train"])
        assert row.meta["n_test"] == int(published["n_test"])
        for axis, tolerance in TOLERANCE.items():
            assert getattr(row, axis) == pytest.approx(float(published[axis]), abs=tolerance)

    # Validation and test were cached for both models ...
    assert sorted(path.name for path in result.prediction_paths) == [
        "logreg_tfidf__seed42__go_emotions-test.npz",
        "logreg_tfidf__seed42__go_emotions-validation.npz",
        "naive_bayes_tfidf__seed42__go_emotions-test.npz",
        "naive_bayes_tfidf__seed42__go_emotions-validation.npz",
    ]
    # ... and re-scoring the cache gives the scorecard row back exactly.
    by_model = {row.model: row for row in result.card.rows}
    for path in result.prediction_paths:
        record = load_prediction(path)
        if not record.meta["scored"]:
            assert record.meta["eval_set"] == "go_emotions:validation"
            assert len(record) == 5426
            continue
        row = by_model[record.meta["model"]]
        assert score_record(record).as_dict() == row.as_dict()
