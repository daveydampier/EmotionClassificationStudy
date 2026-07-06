"""Tests for the visualization module.

Skips cleanly when the plotting stack (matplotlib/seaborn) is not installed, so a
lean CI without the ``viz`` extra still passes. Uses the Agg backend so no window
is opened.
"""

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")  # headless; must precede pyplot import inside viz

pytest.importorskip("seaborn")

from emotion_classification import viz  # noqa: E402


def _scorecard_df():
    return pd.DataFrame([
        {"model": "naive_bayes_tfidf", "schema": "native", "macro_f1": 0.10,
         "train_seconds": 0.1, "model_size_mb": 6.0},
        {"model": "logreg_tfidf", "schema": "native", "macro_f1": 0.27,
         "train_seconds": 0.4, "model_size_mb": 1.4},
        {"model": "logreg_tfidf", "schema": "ekman6", "macro_f1": 0.45,
         "train_seconds": 0.3, "model_size_mb": 1.2},
    ])


def _per_emotion_df():
    return pd.DataFrame([
        {"model": "naive_bayes_tfidf", "emotion": "joy", "f1": 0.6, "support": 500},
        {"model": "naive_bayes_tfidf", "emotion": "grief", "f1": 0.0, "support": 6},
        {"model": "logreg_tfidf", "emotion": "joy", "f1": 0.8, "support": 500},
        {"model": "logreg_tfidf", "emotion": "grief", "f1": 0.1, "support": 6},
    ])


def test_pareto_scatter_returns_figure():
    fig = viz.pareto_scatter(_scorecard_df())
    assert fig.axes  # has at least one axis


def test_heatmap_returns_figure():
    fig = viz.per_emotion_heatmap(_per_emotion_df())
    assert fig.axes


def test_violin_returns_figure():
    fig = viz.per_emotion_violin(_per_emotion_df())
    assert fig.axes


def test_f1_vs_frequency_returns_figure():
    fig = viz.f1_vs_frequency(_per_emotion_df())
    ax = fig.axes[0]
    assert ax.get_xscale() == "log"


def test_granularity_bars_returns_figure():
    fig = viz.granularity_bars(_scorecard_df())
    assert fig.axes


def test_feature_bars_strips_suffix_and_returns_figure():
    df = pd.DataFrame([
        {"model": "logreg_tfidf", "features": "tfidf", "macro_f1": 0.44},
        {"model": "logreg_bow", "features": "bow", "macro_f1": 0.43},
        {"model": "naive_bayes_tfidf", "features": "tfidf", "macro_f1": 0.13},
        {"model": "naive_bayes_bow", "features": "bow", "macro_f1": 0.21},
    ])
    fig = viz.feature_bars(df)
    ax = fig.axes[0]
    # base-model tick labels have the feature suffix stripped
    labels = {t.get_text() for t in ax.get_xticklabels()}
    assert "logreg" in labels and "naive_bayes" in labels
    assert not any(l.endswith("_tfidf") or l.endswith("_bow") for l in labels)


def test_save_all_writes_pngs(tmp_path):
    paths = viz.save_all(_scorecard_df(), _per_emotion_df(), tmp_path, prefix="t_")
    # granularity included because >1 schema present
    assert "granularity" in paths
    for path in paths.values():
        assert path.exists() and path.stat().st_size > 0


def test_f1_vs_frequency_drops_zero_support():
    df = _per_emotion_df()
    df.loc[len(df)] = {"model": "logreg_tfidf", "emotion": "never",
                       "f1": 0.0, "support": 0}
    fig = viz.f1_vs_frequency(df)  # must not raise on log scale with a zero
    assert fig.axes
