"""The study's figures, generated from persisted scorecard results.

Every function takes a pandas DataFrame loaded from the CSVs written by
:mod:`emotion_classification.results` and returns a Matplotlib ``Figure`` (so the
caller decides whether to show or save). Figures never re-run models — they read
``results/`` — which keeps the analysis stage cheap and reproducible.

The five figures map onto the proposal's Analysis & Visualization plan:

* :func:`pareto_scatter`        — efficiency vs. macro-F1 (which models are dominated).
* :func:`per_emotion_heatmap`   — emotion x model F1 (exactly where tiers degrade).
* :func:`per_emotion_violin`    — F1 distribution per model (uniform vs. long tail).
* :func:`f1_vs_frequency`       — the core figure: per-emotion F1 vs. class frequency.
* :func:`granularity_bars`      — how performance shifts across label schemas.

This module imports matplotlib/seaborn/pandas at import time; it is intentionally
*not* imported by the package ``__init__`` so importing the library stays light.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Canonical order for the taxonomy sweep, coarsest last.
SCHEMA_ORDER = ["native", "ekman6", "sentiment3"]


def load_scorecard(csv_path) -> pd.DataFrame:
    """Load a ``*_scorecard.csv`` written by :func:`results.save_results`."""
    return pd.read_csv(csv_path)


def load_per_emotion(csv_path) -> pd.DataFrame:
    """Load a ``*_per_emotion.csv`` written by :func:`results.save_results`."""
    return pd.read_csv(csv_path)


def pareto_scatter(scorecard: pd.DataFrame, *, x: str = "train_seconds",
                   y: str = "macro_f1") -> plt.Figure:
    """Efficiency (``x``) vs. predictive quality (``y``), one point per model.

    ``x`` is typically ``train_seconds`` or ``model_size_mb`` — both span orders
    of magnitude, so a log x-axis is used. Points toward the upper-left are the
    efficient frontier; points dominated on both axes stand out.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(scorecard[x], scorecard[y], s=70, color="#2E5496", zorder=3)
    for _, row in scorecard.iterrows():
        ax.annotate(str(row["model"]), (row[x], row[y]),
                    xytext=(6, 4), textcoords="offset points", fontsize=8)
    if (scorecard[x] > 0).all():
        ax.set_xscale("log")
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(y.replace("_", " "))
    ax.set_title(f"Efficiency frontier: {y.replace('_', ' ')} vs. {x.replace('_', ' ')}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def per_emotion_heatmap(per_emotion: pd.DataFrame) -> plt.Figure:
    """Emotion x model heatmap of F1, emotions ordered by frequency (rarest last)."""
    order = (per_emotion.groupby("emotion")["support"].max()
             .sort_values(ascending=False).index)
    pivot = per_emotion.pivot_table(index="emotion", columns="model",
                                    values="f1").reindex(order)
    height = max(4.0, 0.3 * len(pivot))
    fig, ax = plt.subplots(figsize=(1.6 * max(1, pivot.shape[1]) + 2, height))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1,
                cbar_kws={"label": "F1"}, ax=ax)
    ax.set_title("Per-emotion F1 by model (emotions ordered by frequency)")
    ax.set_xlabel("model")
    ax.set_ylabel("emotion")
    fig.tight_layout()
    return fig


def per_emotion_violin(per_emotion: pd.DataFrame) -> plt.Figure:
    """Violin plot of the per-emotion F1 distribution for each model.

    A tight, high violin means the model performs uniformly across emotions; a
    wide violin with a long low tail is exactly the "non-uniform degradation"
    the study hypothesizes for lightweight models.
    """
    fig, ax = plt.subplots(figsize=(1.3 * max(1, per_emotion["model"].nunique()) + 2, 5))
    sns.violinplot(data=per_emotion, x="model", y="f1", cut=0,
                   inner="point", ax=ax)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Distribution of per-emotion F1 by model")
    ax.set_xlabel("model")
    ax.set_ylabel("per-emotion F1")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig


def f1_vs_frequency(per_emotion: pd.DataFrame) -> plt.Figure:
    """Core figure: per-emotion F1 against class frequency, one series per model.

    Tests whether the lightweight accuracy penalty concentrates in rare (low
    support) emotions. Log x-axis because class frequency is heavily skewed;
    labels with zero support in the eval set are dropped (undefined on a log
    scale). If the data carries bootstrap ``f1_low`` / ``f1_high`` columns, gray
    vertical CI whiskers are drawn — they balloon for rare emotions, showing those
    F1 estimates are the least certain.
    """
    data = per_emotion[per_emotion["support"] > 0]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    if {"f1_low", "f1_high"} <= set(data.columns):
        ci_rows = data.dropna(subset=["f1_low", "f1_high"])
        if len(ci_rows):
            ax.errorbar(
                ci_rows["support"], ci_rows["f1"],
                yerr=[(ci_rows["f1"] - ci_rows["f1_low"]).clip(lower=0),
                      (ci_rows["f1_high"] - ci_rows["f1"]).clip(lower=0)],
                fmt="none", ecolor="gray", alpha=0.4, capsize=2, zorder=1,
            )

    sns.scatterplot(data=data, x="support", y="f1", hue="model", s=55,
                    alpha=0.8, ax=ax, zorder=2)
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("class frequency (support, log scale)")
    ax.set_ylabel("per-emotion F1")
    ax.set_title("Per-emotion F1 vs. class frequency")
    ax.grid(True, alpha=0.3)
    ax.legend(title="model", fontsize=8)
    fig.tight_layout()
    return fig


def granularity_bars(scorecard: pd.DataFrame, *, metric: str = "macro_f1") -> plt.Figure:
    """Grouped bars of ``metric`` across label schemas (the taxonomy sweep)."""
    present = [s for s in SCHEMA_ORDER if s in set(scorecard["schema"])]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=scorecard, x="schema", y=metric, hue="model",
                order=present or None, ax=ax)
    ax.set_xlabel("label schema (fine → coarse)")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"{metric.replace('_', ' ')} across label granularity")
    ax.legend(title="model", fontsize=8)
    fig.tight_layout()
    return fig


def feature_bars(scorecard: pd.DataFrame, *, metric: str = "macro_f1") -> plt.Figure:
    """Grouped bars of ``metric`` per model, comparing feature backends.

    Expects a scorecard spanning both feature backends (``features`` in {tfidf,
    bow}); the trailing ``_<features>`` suffix is stripped from each model name so
    the two representations sit side by side for each base model.
    """
    df = scorecard.copy()
    df["base_model"] = [
        m[: -(len(str(f)) + 1)] if str(m).endswith(f"_{f}") else m
        for m, f in zip(df["model"], df["features"])
    ]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sns.barplot(data=df, x="base_model", y=metric, hue="features", ax=ax)
    ax.set_xlabel("model")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"{metric.replace('_', ' ')}: TF-IDF vs. Bag-of-Words")
    ax.legend(title="features")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    return fig


def save_all(scorecard: pd.DataFrame, per_emotion: pd.DataFrame, out_dir,
             *, prefix: str = "") -> dict:
    """Generate every figure and save as PNG under ``out_dir``. Returns paths."""
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Two Pareto views — one per cost axis (training time and model size) — since the
    # efficiency story spans both; `model_size_mb` falls back to the train-time view
    # only if the column is absent.
    figs = {
        "pareto_train": pareto_scatter(scorecard, x="train_seconds"),
        "heatmap": per_emotion_heatmap(per_emotion),
        "violin": per_emotion_violin(per_emotion),
        "f1_vs_frequency": f1_vs_frequency(per_emotion),
    }
    if "model_size_mb" in scorecard.columns:
        figs["pareto_size"] = pareto_scatter(scorecard, x="model_size_mb")
    if scorecard["schema"].nunique() > 1:
        figs["granularity"] = granularity_bars(scorecard)

    paths = {}
    for name, fig in figs.items():
        path = out / f"{prefix}{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths[name] = path
    return paths
