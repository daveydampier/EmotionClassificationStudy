"""Classical tier: Bag-of-Words / TF-IDF features + scikit-learn classifiers.

These are the study's **lightweight tier** — cheap to train, small on disk, and the
efficiency anchors the "cost of going lightweight" comparison is built around. Four
model families, each selectable over either feature backend (``features="tfidf"`` or
``features="bow"``):

* :class:`NaiveBayes`    — Multinomial NB (the lightweight extreme).
* :class:`LogisticReg`   — logistic regression, native ``predict_proba``.
* :class:`LinearSVM`     — linear SVM, probabilities via sigmoid calibration.
* :class:`RandomForest`  — random forest (the heaviest classical option).

All share a robust per-label binary-relevance head (:class:`_BinaryRelevance`) that
tolerates label columns with no positive (or no negative) training examples — common
in the fine-grained 27-class schema where rare emotions may be absent from a split.
Each model's ``name`` encodes the feature backend (e.g. ``naive_bayes_bow``) so the
scorecard distinguishes BoW from TF-IDF runs.
"""

from __future__ import annotations

import io
import pickle

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from .base import BaseEmotionModel


def _make_vectorizer(features: str, max_features: int, ngram_range, min_df: int):
    """Build the text vectorizer for the chosen feature backend."""
    if features == "tfidf":
        return TfidfVectorizer(
            max_features=max_features, ngram_range=ngram_range,
            min_df=min_df, sublinear_tf=True,
        )
    if features == "bow":
        return CountVectorizer(
            max_features=max_features, ngram_range=ngram_range, min_df=min_df,
        )
    raise ValueError(f"unknown features {features!r}; use 'tfidf' or 'bow'")


class _BinaryRelevance:
    """Fit one cloned binary estimator per label; handle constant columns.

    A column that is all-0 or all-1 in training has no decision to learn, so we
    record the constant and emit it directly at predict time instead of fitting
    (which would raise). Everything else gets a real fitted estimator exposing
    ``predict_proba``.
    """

    def __init__(self, estimator):
        self.estimator = estimator
        self.fitted_: list = []  # per label: ("const", value) or ("model", est)

    def fit(self, X, Y: np.ndarray):
        self.fitted_ = []
        for j in range(Y.shape[1]):
            col = Y[:, j].astype(int)
            classes = np.unique(col)
            if len(classes) < 2:
                self.fitted_.append(("const", float(classes[0])))
            else:
                est = clone(self.estimator).fit(X, col)
                self.fitted_.append(("model", est))
        return self

    def predict_proba(self, X) -> np.ndarray:
        n = X.shape[0]
        cols = []
        for kind, payload in self.fitted_:
            if kind == "const":
                cols.append(np.full(n, payload, dtype=np.float64))
            else:
                cols.append(payload.predict_proba(X)[:, 1])
        return np.column_stack(cols) if cols else np.zeros((n, 0))


class _VectorizedClassifier(BaseEmotionModel):
    """Shared vectorizer + binary-relevance plumbing for the classical variants."""

    def __init__(self, estimator, name: str, *, features: str = "tfidf",
                 max_features: int = 50_000, ngram_range: tuple[int, int] = (1, 2),
                 min_df: int = 2):
        self.name = name
        self.features = features
        self.vectorizer = _make_vectorizer(features, max_features, ngram_range, min_df)
        self.head = _BinaryRelevance(estimator)

    def fit(self, texts: list[str], Y: np.ndarray):
        X = self.vectorizer.fit_transform(texts)
        self.head.fit(X, np.asarray(Y))
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return self.head.predict_proba(X)

    def size_mb(self) -> float:
        buf = io.BytesIO()
        pickle.dump((self.vectorizer, self.head), buf)
        return buf.tell() / (1024 * 1024)


class NaiveBayes(_VectorizedClassifier):
    """Naive Bayes — the lightweight extreme (tiny, near-instant training).

    Uses :class:`~sklearn.naive_bayes.ComplementNB` rather than plain
    ``MultinomialNB``: Complement NB is designed for imbalanced text and, unlike
    Multinomial NB, does not collapse to all-negative predictions under the
    binary-relevance / 0.5-threshold setup on rare emotions.
    """

    def __init__(self, *, features: str = "tfidf", alpha: float = 1.0, **vec):
        super().__init__(
            ComplementNB(alpha=alpha),
            name=f"naive_bayes_{features}",
            features=features,
            **vec,
        )


class LogisticReg(_VectorizedClassifier):
    """Per-label logistic regression."""

    def __init__(self, *, features: str = "tfidf", C: float = 1.0,
                 max_iter: int = 1000, **vec):
        super().__init__(
            LogisticRegression(C=C, max_iter=max_iter, class_weight="balanced"),
            name=f"logreg_{features}",
            features=features,
            **vec,
        )


class LinearSVM(_VectorizedClassifier):
    """Per-label linear SVM, calibrated to emit probabilities.

    ``LinearSVC`` has no ``predict_proba``; :class:`~sklearn.calibration.
    CalibratedClassifierCV` wraps it with cross-validated Platt scaling so the
    calibration (ECE) axis is meaningful rather than a hard 0/1 stand-in.
    """

    def __init__(self, *, features: str = "tfidf", C: float = 1.0, cv: int = 3, **vec):
        base = LinearSVC(C=C, class_weight="balanced")
        super().__init__(
            CalibratedClassifierCV(base, cv=cv),
            name=f"linsvm_{features}",
            features=features,
            **vec,
        )


class RandomForest(_VectorizedClassifier):
    """Per-label random forest — the heaviest classical option (bigger, slower).

    Included partly to populate the middle of the size/training-time spectrum: it
    shows that "classical" is not automatically "lightweight."
    """

    def __init__(self, *, features: str = "tfidf", n_estimators: int = 200,
                 max_depth: "int | None" = None, n_jobs: int = -1,
                 random_state=None, **vec):
        # random_state=None (the default) makes the forest draw from the global
        # NumPy RNG, which the experiment runner seeds per run — so Random Forest
        # is reproducible for a given seed *and* varies across seeds (unlike the
        # deterministic NB/LogReg/SVM), giving it real multi-seed error bars.
        super().__init__(
            RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth, n_jobs=n_jobs,
                class_weight="balanced_subsample", random_state=random_state,
            ),
            name=f"random_forest_{features}",
            features=features,
            **vec,
        )
