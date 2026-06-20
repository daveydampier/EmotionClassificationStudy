"""Classical tier: TF-IDF features + linear classifiers (scikit-learn).

These are the study's **reference implementation** — cheap to train, strong
baselines, and the first end-to-end path through the scorecard. Two variants:

* :class:`TfidfLogReg`   — logistic regression, native ``predict_proba``.
* :class:`TfidfLinearSVM` — linear SVM, probabilities via sigmoid calibration.

Both wrap a shared, robust per-label binary-relevance head (see
:class:`_BinaryRelevance`) that tolerates label columns with no positive (or no
negative) training examples — handy when a harmonized schema leaves a rare
emotion empty in some split.
"""

from __future__ import annotations

import io
import pickle

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from .base import BaseEmotionModel


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


class _TfidfClassifier(BaseEmotionModel):
    """Shared TF-IDF + binary-relevance plumbing for the classical variants."""

    def __init__(self, estimator, name: str, *, max_features: int = 50_000,
                 ngram_range: tuple[int, int] = (1, 2), min_df: int = 2):
        self.name = name
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=ngram_range,
            min_df=min_df, sublinear_tf=True,
        )
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


class TfidfLogReg(_TfidfClassifier):
    """TF-IDF + per-label logistic regression."""

    def __init__(self, *, C: float = 1.0, max_iter: int = 1000, **tfidf):
        super().__init__(
            LogisticRegression(C=C, max_iter=max_iter, class_weight="balanced"),
            name="tfidf_logreg",
            **tfidf,
        )


class TfidfLinearSVM(_TfidfClassifier):
    """TF-IDF + per-label linear SVM, calibrated to emit probabilities.

    ``LinearSVC`` has no ``predict_proba``; :class:`~sklearn.calibration.
    CalibratedClassifierCV` wraps it with cross-validated Platt scaling so the
    calibration (ECE) axis is meaningful rather than a hard 0/1 stand-in.
    """

    def __init__(self, *, C: float = 1.0, cv: int = 3, **tfidf):
        base = LinearSVC(C=C, class_weight="balanced")
        super().__init__(
            CalibratedClassifierCV(base, cv=cv),
            name="tfidf_linearsvm",
            **tfidf,
        )
