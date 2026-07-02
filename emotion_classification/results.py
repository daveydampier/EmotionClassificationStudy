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
from pathlib import Path

from .scorecard import Scorecard

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

    # 3. Long per-emotion CSV.
    with per_emotion_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "dataset", "schema", "emotion", "f1", "support"])
        for r in rows:
            schema = (r.get("meta") or {}).get("schema")
            support = r.get("per_label_support") or {}
            for emotion, f1 in (r.get("per_label_f1") or {}).items():
                writer.writerow([r["model"], r["dataset"], schema, emotion,
                                 f1, support.get(emotion)])

    return {"json": json_path, "scorecard": scorecard_path, "per_emotion": per_emotion_path}


def load_results(json_path: str | Path) -> list[dict]:
    """Read back the ``<name>.json`` record written by :func:`save_results`."""
    return json.loads(Path(json_path).read_text(encoding="utf-8"))
