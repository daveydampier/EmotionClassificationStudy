"""Label schemas and cross-dataset harmonization maps.

Pure-Python (no third-party deps) so it imports and tests anywhere, including
the current Python 3.14 interpreter without the venv installed.

Sources
-------
- GoEmotions taxonomy + Ekman/sentiment groupings: Demszky et al., ACL 2020,
  from the official repo files ``data/ekman_mapping.json`` and
  ``data/sentiment_mapping.json``.
- SemEval-2018 Task 1 E-c label set: Mohammad et al., SemEval-2018.
- BRIGHTER / SemEval-2025 Task 11 uses the 6 Ekman emotions (+ neutral in some
  languages), so it aligns with the Ekman target directly.

The shared target space for cross-dataset experiments is **Ekman-6 (+ neutral)**,
because GoEmotions has an official map to it and BRIGHTER already uses it. The
SemEval-2018 (Plutchik-based) projection to Ekman is lossy and documented below.
"""

from __future__ import annotations

# --- GoEmotions: 27 emotions + neutral, in official index order (0..27) ------
GOEMOTIONS: list[str] = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral",
]

NEUTRAL = "neutral"

# --- Ekman 6 (+ neutral): the shared cross-dataset target --------------------
EKMAN: list[str] = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

# --- Sentiment 3 (+ neutral): for the "granularity tax" analysis -------------
SENTIMENT: list[str] = ["positive", "negative", "ambiguous"]

# --- SemEval-2018 Task 1 E-c: 11 labels (Plutchik-based) ---------------------
SEMEVAL2018: list[str] = [
    "anger", "anticipation", "disgust", "fear", "joy", "love", "optimism",
    "pessimism", "sadness", "surprise", "trust",
]

# --- BRIGHTER / SemEval-2025 Task 11: the Ekman 6 ----------------------------
BRIGHTER: list[str] = list(EKMAN)

# ---------------------------------------------------------------------------
# GoEmotions 27 -> Ekman 6  (official ekman_mapping.json; neutral kept as-is)
# ---------------------------------------------------------------------------
EKMAN_GROUPS: dict[str, list[str]] = {
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": ["admiration", "amusement", "approval", "caring", "desire",
            "excitement", "gratitude", "joy", "love", "optimism", "pride",
            "relief"],
    "sadness": ["disappointment", "embarrassment", "grief", "remorse",
                "sadness"],
    "surprise": ["confusion", "curiosity", "realization", "surprise"],
}

# ---------------------------------------------------------------------------
# GoEmotions 27 -> sentiment  (official sentiment_mapping.json)
# ---------------------------------------------------------------------------
SENTIMENT_GROUPS: dict[str, list[str]] = {
    "positive": ["admiration", "amusement", "approval", "caring", "desire",
                 "excitement", "gratitude", "joy", "love", "optimism", "pride",
                 "relief"],
    "negative": ["anger", "annoyance", "disappointment", "disapproval",
                 "disgust", "embarrassment", "fear", "grief", "nervousness",
                 "remorse", "sadness"],
    "ambiguous": ["confusion", "curiosity", "realization", "surprise"],
}

# ---------------------------------------------------------------------------
# SemEval-2018 (Plutchik 11) -> Ekman 6.  Lossy:
#   * the 6 shared emotions map identically,
#   * love/optimism -> joy and pessimism -> sadness (nearest valence),
#   * anticipation/trust have NO Ekman equivalent -> None (dropped on project).
# ---------------------------------------------------------------------------
SEMEVAL2018_TO_EKMAN: dict[str, str | None] = {
    "anger": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "joy",
    "sadness": "sadness",
    "surprise": "surprise",
    "love": "joy",
    "optimism": "joy",
    "pessimism": "sadness",
    "anticipation": None,
    "trust": None,
}


def _invert(groups: dict[str, list[str]]) -> dict[str, str]:
    """Flatten {coarse: [fine, ...]} into {fine: coarse}."""
    out: dict[str, str] = {}
    for coarse, fine_labels in groups.items():
        for fine in fine_labels:
            out[fine] = coarse
    return out


# Per-label maps (include neutral -> neutral for completeness).
GOEMOTIONS_TO_EKMAN: dict[str, str] = {**_invert(EKMAN_GROUPS), NEUTRAL: NEUTRAL}
GOEMOTIONS_TO_SENTIMENT: dict[str, str] = {**_invert(SENTIMENT_GROUPS), NEUTRAL: NEUTRAL}


def map_labels(labels: "list[str] | set[str]", mapping: dict[str, str | None]) -> list[str]:
    """Project native labels into a coarser schema.

    Returns a sorted list of unique target labels. Labels mapping to ``None``
    (no equivalent in the target schema) are dropped. Unknown labels raise
    ``KeyError`` so silent typos can't slip through.
    """
    out: set[str] = set()
    for label in labels:
        target = mapping[label]
        if target is not None:
            out.add(target)
    return sorted(out)


def goemotions_to_ekman(labels: "list[str] | set[str]") -> list[str]:
    return map_labels(labels, GOEMOTIONS_TO_EKMAN)


def goemotions_to_sentiment(labels: "list[str] | set[str]") -> list[str]:
    return map_labels(labels, GOEMOTIONS_TO_SENTIMENT)


def semeval2018_to_ekman(labels: "list[str] | set[str]") -> list[str]:
    return map_labels(labels, SEMEVAL2018_TO_EKMAN)
