"""The deployment scorecard: one row per model, many axes.

The study's thesis is that accuracy alone is a poor model-selection signal. A
:class:`ScorecardRow` records, for a single (model, dataset) run, the predictive
quality *and* the real-world deployment costs:

==================  ===================================================  ========
Axis                Field                                                Better
==================  ===================================================  ========
Predictive          ``macro_f1``, ``micro_f1``, ``subset_accuracy``      higher
Calibration         ``ece``                                              lower
Efficiency (train)  ``train_seconds``                                    lower
Efficiency (infer)  ``predict_latency_ms`` (per sample)                  lower
Efficiency (infer)  ``throughput`` (samples/sec; inverse of latency)     higher
Footprint           ``model_size_mb``                                    lower
Cost                ``cost_usd`` (optional; for API/LLM tier)            lower
==================  ===================================================  ========

:meth:`Scorecard.normalized` min-max scales every axis to ``[0, 1]`` where 1 is
always "best" (lower-is-better axes are inverted), so heterogeneous units become
comparable — the input to radar / Pareto plots.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

# Axis name -> whether higher raw values are better.
HIGHER_IS_BETTER: dict[str, bool] = {
    "macro_f1": True,
    "micro_f1": True,
    "subset_accuracy": True,
    "ece": False,
    "train_seconds": False,
    "predict_latency_ms": False,
    "throughput": True,
    "model_size_mb": False,
    "cost_usd": False,
}


@dataclass
class ScorecardRow:
    """All measured axes for one (model, dataset) run."""

    model: str
    dataset: str

    # Predictive (from metrics.evaluate)
    macro_f1: float = 0.0
    micro_f1: float = 0.0
    subset_accuracy: float = 0.0

    # Calibration
    ece: float = 0.0

    # Efficiency / footprint / cost
    train_seconds: float = 0.0
    predict_latency_ms: float = 0.0
    throughput: float = 0.0  # samples/second at inference (RQ1); inverse of latency
    model_size_mb: float = 0.0
    cost_usd: Optional[float] = None

    # Hardware the timing/latency were measured on (fairness caveat: the
    # lightweight tier runs on CPU, the transformer on GPU, so ``train_seconds``
    # and ``predict_latency_ms`` are only comparable within the same device).
    device: Optional[str] = None

    # Per-emotion breakdown (backbone of RQ3). ``per_label_f1`` maps each label
    # to its F1; ``per_label_support`` maps each label to its positive count in
    # the evaluation set (its class frequency) — these two feed the core
    # per-emotion-F1-vs-frequency figure.
    per_label_f1: dict = field(default_factory=dict)
    per_label_support: dict = field(default_factory=dict)

    # Bootstrap confidence intervals (test-set sampling uncertainty), populated
    # only when a run requests them. ``*_ci`` are [low, high]; ``per_label_f1_ci``
    # maps label -> [low, high].
    macro_f1_ci: Optional[list] = None
    micro_f1_ci: Optional[list] = None
    per_label_f1_ci: dict = field(default_factory=dict)

    # Free-form extras (hyperparameters, label schema, n_samples, …)
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


class Scorecard:
    """A collection of :class:`ScorecardRow` with comparison helpers."""

    def __init__(self, rows: Optional[list[ScorecardRow]] = None) -> None:
        self.rows: list[ScorecardRow] = list(rows) if rows else []

    def add(self, row: ScorecardRow) -> None:
        self.rows.append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def axes(self) -> list[str]:
        """Axes actually present (non-None in at least one row)."""
        return [
            ax
            for ax in HIGHER_IS_BETTER
            if any(getattr(r, ax) is not None for r in self.rows)
        ]

    def normalized(self) -> list[dict[str, float]]:
        """Min-max scale each axis to [0, 1], 1 == best, across all rows.

        A degenerate axis (all rows equal, or a single row) scores 1.0 for every
        row on that axis — there is no spread to penalize.
        """
        if not self.rows:
            return []

        axes = self.axes()
        ranges: dict[str, tuple[float, float]] = {}
        for ax in axes:
            vals = [getattr(r, ax) for r in self.rows if getattr(r, ax) is not None]
            ranges[ax] = (min(vals), max(vals))

        out: list[dict[str, float]] = []
        for r in self.rows:
            scaled: dict[str, float] = {"model": r.model, "dataset": r.dataset}
            for ax in axes:
                raw = getattr(r, ax)
                if raw is None:
                    continue
                lo, hi = ranges[ax]
                if hi == lo:
                    norm = 1.0
                else:
                    norm = (raw - lo) / (hi - lo)
                    if not HIGHER_IS_BETTER[ax]:
                        norm = 1.0 - norm
                scaled[ax] = float(norm)
            out.append(scaled)
        return out

    def to_dataframe(self):
        """Raw rows as a pandas DataFrame (pandas imported lazily)."""
        import pandas as pd

        return pd.DataFrame([r.as_dict() for r in self.rows])

    def to_markdown(self, axes: Optional[list[str]] = None) -> str:
        """Compact Markdown table of raw values (no pandas dependency)."""
        axes = axes or self.axes()
        header = ["model", "dataset", *axes]
        lines = ["| " + " | ".join(header) + " |",
                 "| " + " | ".join("---" for _ in header) + " |"]
        for r in self.rows:
            cells = [r.model, r.dataset]
            for ax in axes:
                v = getattr(r, ax)
                cells.append("—" if v is None else f"{v:.4g}")
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)
