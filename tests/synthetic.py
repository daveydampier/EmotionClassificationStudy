"""A small noisy multi-label corpus for tests (seeded; no network, no torch)."""

import numpy as np

from emotion_classification.loaders import EmotionDataset, Example
from emotion_classification.preprocessing import PreparedSplit

LABELS = ["joy", "anger", "sadness", "fear"]
VOCAB = {
    "joy": ["happy", "glad", "delighted", "smile"],
    "anger": ["furious", "mad", "rage", "annoyed"],
    "sadness": ["sad", "cry", "grief", "lonely"],
    "fear": ["scared", "afraid", "panic", "worried"],
}
FILLER = ["the", "today", "really", "and", "then", "it", "was", "so"]


def synthetic_split(n: int, seed: int) -> PreparedSplit:
    """``n`` texts mixing cue words for their labels with filler and some noise.

    Every text has at least one label, so no row is dropped as empty when the
    split is turned into a dataset and prepared again.
    """
    rng = np.random.RandomState(seed)
    texts, rows = [], []
    for _ in range(n):
        active = [j for j in range(len(LABELS)) if rng.rand() < 0.35]
        if not active:
            active = [int(rng.randint(len(LABELS)))]
        words = [VOCAB[LABELS[j]][rng.randint(4)] for j in active for _ in range(2)]
        words += [FILLER[rng.randint(len(FILLER))] for _ in range(4)]
        if rng.rand() < 0.2:   # noise: a cue word for a label that may not be active
            words.append(VOCAB[LABELS[rng.randint(len(LABELS))]][rng.randint(4)])
        rng.shuffle(words)
        texts.append(" ".join(words))
        row = np.zeros(len(LABELS), dtype=np.float32)
        row[active] = 1.0
        rows.append(row)
    return PreparedSplit(texts, np.vstack(rows), list(LABELS))


def toy_dataset(name: str, seed: int, splits=("train", "validation", "test")) -> EmotionDataset:
    """An in-memory corpus over ``LABELS`` with 160 / 40 / 60 rows, for registry tests."""
    sizes = {"train": (160, seed), "validation": (40, seed + 1), "test": (60, seed + 2)}

    def to_examples(split):
        return [Example(text, [LABELS[j] for j in np.flatnonzero(row)])
                for text, row in zip(split.texts, split.Y)]

    return EmotionDataset(name, list(LABELS), {
        split: to_examples(synthetic_split(*sizes[split])) for split in splits
    })
