"""Dataset loaders that normalize public emotion datasets to one common form.

Every loader returns an :class:`EmotionDataset`: a mapping of split name ->
list of :class:`Example`, where ``labels`` is a list of the dataset's *native*
label names (multi-label friendly). Project those names into a shared schema
(e.g. Ekman-6) with :mod:`emotion_classification.labels`.

These loaders require the optional ``datasets`` (HuggingFace) dependency from
requirements.txt and hit the network on first use (results cached under the HF
cache dir). They are intentionally import-light: ``datasets`` is imported lazily
inside each function so this module imports without the dependency installed.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import labels as L


@dataclass
class Example:
    """One labeled text instance."""

    text: str
    labels: list[str]


@dataclass
class EmotionDataset:
    """A loaded dataset normalized across sources."""

    name: str
    label_names: list[str]
    splits: dict[str, list[Example]]
    multi_label: bool = True

    def __repr__(self) -> str:  # concise, avoids dumping every example
        sizes = ", ".join(f"{k}={len(v)}" for k, v in self.splits.items())
        return (f"EmotionDataset(name={self.name!r}, "
                f"labels={len(self.label_names)}, splits[{sizes}])")


def load_goemotions(config: str = "simplified") -> EmotionDataset:
    """GoEmotions (Demszky et al., 2020) — 27 emotions + neutral, Reddit.

    Uses the HF ``go_emotions`` dataset. The ``simplified`` config gives one row
    per example with ``labels`` as a list of integer indices into
    :data:`labels.GOEMOTIONS`. (The ``raw`` config carries per-rater columns and
    has a different schema — not handled here yet; it's what we'd use for the
    human-human kappa ceiling.)
    """
    if config != "simplified":
        raise NotImplementedError(
            "Only the 'simplified' GoEmotions config is supported so far. "
            "The 'raw' config (per-rater labels) needs separate handling."
        )
    from datasets import load_dataset

    ds = load_dataset("go_emotions", config)
    splits = {
        split: [
            Example(text=row["text"], labels=[L.GOEMOTIONS[i] for i in row["labels"]])
            for row in ds[split]
        ]
        for split in ds
    }
    return EmotionDataset("go_emotions", L.GOEMOTIONS, splits)


def load_semeval2018(language: str = "english") -> EmotionDataset:
    """SemEval-2018 Task 1 E-c (Mohammad et al.) — 11 labels, multi-label tweets.

    Uses HF ``sem_eval_2018_task_1`` config ``subtask5.<language>`` (english,
    spanish, arabic). The text column is ``Tweet`` and each of the 11 emotions
    is a boolean column.
    """
    from datasets import load_dataset

    ds = load_dataset("sem_eval_2018_task_1", f"subtask5.{language}")
    splits = {
        split: [
            Example(text=row["Tweet"], labels=[e for e in L.SEMEVAL2018 if row.get(e)])
            for row in ds[split]
        ]
        for split in ds
    }
    return EmotionDataset("sem_eval_2018_task_1", L.SEMEVAL2018, splits)


def load_brighter(language: str = "eng") -> EmotionDataset:
    """BRIGHTER / SemEval-2025 Task 11 — 6 Ekman emotions, 32 languages.

    Not implemented yet: confirm the exact HuggingFace dataset path/config for
    BRIGHTER before wiring this up (see the dataset table in README.md).
    """
    raise NotImplementedError(
        "BRIGHTER / SemEval-2025 Task 11 loader pending: confirm the HuggingFace "
        "dataset path and config name first."
    )


# Registry so callers can select a loader by short name.
LOADERS = {
    "go_emotions": load_goemotions,
    "sem_eval_2018_task_1": load_semeval2018,
    "brighter": load_brighter,
}
