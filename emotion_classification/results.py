"""Persist scorecard results to disk for later analysis and plotting.

An experiment run is expensive (especially the transformer tier), so results are
written to ``results/`` once and the analysis/visualization stage reads them back
— figures never require re-running models. Three artifacts are written per run:

* ``<name>.json``            — the complete record: every :class:`ScorecardRow`
  as a dict, including ``meta``, ``device``, and the full per-emotion breakdown.
* ``<name>_scorecard.csv``   — one row per model, flat headline metrics (the
  Pareto / efficiency views).
* ``<name>_per_emotion.csv`` — long format (one row per model × emotion) with F1
  and class-frequency support — the shape the per-emotion figures consume.

Uses only the standard library (``json`` + ``csv``) so it imports and tests
without pandas.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from .scorecard import HIGHER_IS_BETTER, Scorecard

# Flat headline columns for the scorecard CSV; meta-derived ones are pulled from
# each row's ``meta`` dict.
_SCORECARD_FIELDS = [
    "model", "dataset", "schema", "features", "device",
    "macro_f1", "micro_f1", "subset_accuracy", "ece",
    "train_seconds", "predict_latency_ms", "model_size_mb", "cost_usd",
    "n_train", "n_test", "n_labels", "seed",
]
_META_FIELDS = {"schema", "features", "n_train", "n_test", "n_labels", "seed"}


def save_results(card: Scorecard, out_dir: str | Path, name: str = "run") -> dict[str, Path]:
    """Write ``card`` to ``out_dir`` as JSON + two CSVs. Returns the paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / f"{name}.json"
    scorecard_path = out / f"{name}_scorecard.csv"
    per_emotion_path = out / f"{name}_per_emotion.csv"

    rows = [r.as_dict() for r in card.rows]

    # 1. Full JSON record.
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # 2. Flat scorecard CSV.
    with scorecard_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_SCORECARD_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            meta = r.get("meta") or {}
            record = {k: r.get(k) for k in _SCORECARD_FIELDS}
            for k in _META_FIELDS:
                record[k] = meta.get(k)
            writer.writerow(record)

    # 3. Long per-emotion CSV (one row per model x seed x emotion).
    with per_emotion_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "dataset", "schema", "seed", "emotion", "f1", "support"])
        for r in rows:
            meta = r.get("meta") or {}
            schema, seed = meta.get("schema"), meta.get("seed")
            support = r.get("per_label_support") or {}
            for emotion, f1 in (r.get("per_label_f1") or {}).items():
                writer.writerow([r["model"], r["dataset"], schema, seed, emotion,
                                 f1, support.get(emotion)])

    return {"json": json_path, "scorecard": scorecard_path, "per_emotion": per_emotion_path}


def summarize(card) -> list[dict]:
    """Aggregate per-seed rows to mean/std per axis, grouped by model config.

    Accepts a :class:`Scorecard`, a list of :class:`ScorecardRow`, or a list of
    row dicts (e.g. from :func:`load_results`). Rows are grouped by
    ``(model, dataset, schema, features)`` so a multi-seed run collapses to one
    summary line per model with ``<axis>_mean`` / ``<axis>_std`` (sample std;
    0.0 for a single seed).
    """
    if isinstance(card, Scorecard):
        rows = [r.as_dict() for r in card.rows]
    else:
        rows = [r.as_dict() if hasattr(r, "as_dict") else r for r in card]

    groups: dict[tuple, list[dict]] = {}
    order: list[tuple] = []
    for r in rows:
        meta = r.get("meta") or {}
        key = (r["model"], r["dataset"], meta.get("schema"), meta.get("features"))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(r)

    summaries = []
    for key in order:
        model, dataset, schema, features = key
        grp = groups[key]
        summary = {"model": model, "dataset": dataset, "schema": schema,
                   "features": features, "n_seeds": len(grp)}
        for axis in HIGHER_IS_BETTER:
            vals = [r.get(axis) for r in grp if r.get(axis) is not None]
            if not vals:
                continue
            summary[f"{axis}_mean"] = statistics.fmean(vals)
            summary[f"{axis}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        summaries.append(summary)
    return summaries


def save_summary(card, out_dir: str | Path, name: str = "run") -> Path:
    """Write the multi-seed :func:`summarize` table to ``<name>_summary.csv``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summaries = summarize(card)

    axis_cols: list[str] = []
    for axis in HIGHER_IS_BETTER:
        if any(f"{axis}_mean" in s for s in summaries):
            axis_cols += [f"{axis}_mean", f"{axis}_std"]
    fields = ["model", "dataset", "schema", "features", "n_seeds", *axis_cols]

    path = out / f"{name}_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for s in summaries:
            writer.writerow(s)
    return path


def load_results(json_path: str | Path) -> list[dict]:
    """Read back the ``<name>.json`` record written by :func:`save_results`."""
    return json.loads(Path(json_path).read_text(encoding="utf-8"))
