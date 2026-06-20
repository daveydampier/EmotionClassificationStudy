"""Turn loaded datasets into model-ready arrays.

A loaded :class:`~emotion_classification.loaders.EmotionDataset` carries *native*
label names per example. For modeling we need:

* a fixed **target label vocabulary** (e.g. Ekman-6 (+ neutral)), and
* a **multi-hot** ``(n_samples, n_labels)`` indicator matrix.

:func:`prepare_split` projects native labels through a harmonization function
(see :mod:`emotion_classification.labels`) and builds that matrix. The result is
a :class:`PreparedSplit` of ``texts`` + ``Y`` that every model tier consumes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .loaders import EmotionDataset, Example

# A projector maps a list of native label names to target label names (possibly
# fewer, possibly empty if all were dropped). ``labels.goemotions_to_ekman`` and
# friends have exactly this signature.
Projector = Callable[[list[str]], list[str]]


@dataclass
class PreparedSplit:
    """Model-ready arrays for one split."""

    texts: list[str]
    Y: np.ndarray  # shape (n_samples, n_labels), dtype float32, values {0, 1}
    label_names: list[str]

    def __len__(self) -> int:
        return len(self.texts)

    @property
    def n_labels(self) -> int:
        return len(self.label_names)


def to_multihot(label_lists: Sequence[Sequence[str]], label_names: Sequence[str]) -> np.ndarray:
    """Build a multi-hot matrix from per-example label-name lists.

    Unknown labels (not in ``label_names``) are ignored — projection upstream is
    responsible for producing only valid target labels.
    """
    index = {name: i for i, name in enumerate(label_names)}
    Y = np.zeros((len(label_lists), len(label_names)), dtype=np.float32)
    for row, labels in enumerate(label_lists):
        for name in labels:
            col = index.get(name)
            if col is not None:
                Y[row, col] = 1.0
    return Y


def prepare_split(
    examples: Sequence[Example],
    label_names: Sequence[str],
    projector: Projector | None = None,
    *,
    drop_empty: bool = True,
) -> PreparedSplit:
    """Project one split's examples into ``(texts, Y)`` over ``label_names``.

    ``projector`` maps native labels into the target schema; pass ``None`` to use
    the native labels as-is (then ``label_names`` must be the native set). With
    ``drop_empty`` (default), examples whose labels all fall outside the target
    schema are removed so they don't become phantom all-zero rows.
    """
    texts: list[str] = []
    projected: list[list[str]] = []
    for ex in examples:
        labels = projector(list(ex.labels)) if projector is not None else list(ex.labels)
        if drop_empty and not labels:
            continue
        texts.append(ex.text)
        projected.append(labels)
    Y = to_multihot(projected, label_names)
    return PreparedSplit(texts=texts, Y=Y, label_names=list(label_names))


def prepare_dataset(
    ds: EmotionDataset,
    label_names: Sequence[str],
    projector: Projector | None = None,
    *,
    drop_empty: bool = True,
) -> dict[str, PreparedSplit]:
    """Apply :func:`prepare_split` to every split of a dataset."""
    return {
        split: prepare_split(examples, label_names, projector, drop_empty=drop_empty)
        for split, examples in ds.splits.items()
    }
