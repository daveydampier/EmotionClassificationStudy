"""The common model interface.

A model is anything that learns from ``(texts, Y)`` and produces per-label
probabilities for new texts. Keeping the surface this small is what lets every
tier — classical, deep, transformer, even a prompted LLM — flow through the same
experiment runner and land on the same scorecard.

``EmotionModel`` is a :class:`typing.Protocol`, so concrete classes don't need to
subclass it; they just need matching methods. :class:`BaseEmotionModel` is an
optional convenience base that supplies a default :meth:`predict`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EmotionModel(Protocol):
    """Structural interface every model tier satisfies."""

    name: str

    def fit(self, texts: list[str], Y: np.ndarray) -> "EmotionModel":
        """Train on texts and a multi-hot ``(n_samples, n_labels)`` target."""
        ...

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Per-label probabilities, shape ``(n_samples, n_labels)`` in [0, 1]."""
        ...

    def predict(self, texts: list[str], threshold: float = 0.5) -> np.ndarray:
        """Hard 0/1 predictions, shape ``(n_samples, n_labels)``."""
        ...

    def size_mb(self) -> float:
        """On-disk / in-memory footprint in megabytes (scorecard axis)."""
        ...


class BaseEmotionModel:
    """Convenience base: implements :meth:`predict` from :meth:`predict_proba`."""

    name: str = "base"

    def fit(self, texts: list[str], Y: np.ndarray):  # pragma: no cover - abstract
        raise NotImplementedError

    def predict_proba(self, texts: list[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def predict(self, texts: list[str], threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(texts) >= threshold).astype(np.int64)

    def size_mb(self) -> float:  # pragma: no cover - overridden per tier
        return 0.0
