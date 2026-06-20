"""Tests for scorecard assembly and normalization."""

from emotion_classification.scorecard import Scorecard, ScorecardRow


def _card():
    return Scorecard([
        # A: best F1 but slow + big.  B: worse F1 but fast + small.
        ScorecardRow(model="A", dataset="d", macro_f1=0.9, ece=0.2,
                     train_seconds=100.0, model_size_mb=400.0),
        ScorecardRow(model="B", dataset="d", macro_f1=0.7, ece=0.1,
                     train_seconds=10.0, model_size_mb=5.0),
    ])


def test_normalized_higher_is_better_for_f1():
    norm = {r["model"]: r for r in _card().normalized()}
    # A has the higher macro_f1 -> normalized 1.0; B -> 0.0
    assert norm["A"]["macro_f1"] == 1.0
    assert norm["B"]["macro_f1"] == 0.0


def test_normalized_lower_is_better_axes_inverted():
    norm = {r["model"]: r for r in _card().normalized()}
    # B is faster & smaller & better-calibrated -> should score 1.0 on those
    assert norm["B"]["train_seconds"] == 1.0
    assert norm["B"]["model_size_mb"] == 1.0
    assert norm["B"]["ece"] == 1.0
    assert norm["A"]["train_seconds"] == 0.0


def test_degenerate_axis_scores_one():
    card = Scorecard([
        ScorecardRow(model="A", dataset="d", macro_f1=0.5),
        ScorecardRow(model="B", dataset="d", macro_f1=0.5),
    ])
    norm = card.normalized()
    assert all(r["macro_f1"] == 1.0 for r in norm)


def test_cost_axis_absent_when_all_none():
    # cost_usd defaults to None -> not an active axis
    assert "cost_usd" not in _card().axes()


def test_markdown_has_header_and_rows():
    md = _card().to_markdown()
    assert "| model | dataset |" in md
    assert md.count("\n") >= 3  # header + separator + 2 rows
