"""Evaluation metrics for multi-label emotion classification.

This is the **stable contract** every model tier is scored against. Inputs are
always:

* ``y_true`` — ``(n_samples, n_labels)`` multi-hot ground truth (``{0, 1}``),
* ``y_prob`` — ``(n_samples, n_labels)`` predicted probabilities in ``[0, 1]``.

Thresholding ``y_prob`` at ``threshold`` yields binary predictions. Keeping
probabilities (not just hard labels) in the contract is deliberate: the
deployment scorecard's *calibration* axis (ECE) needs them.

Only numpy + scikit-learn are required (no torch / transformers), so metrics
import and test on any machine.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def _as_2d_float(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return a


def binarize(y_prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Hard 0/1 predictions from probabilities at a fixed threshold."""
    return (_as_2d_float(y_prob) >= threshold).astype(np.int64)


def classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Standard predictive axes: macro/micro F1, precision, recall, accuracy.

    ``accuracy`` is *subset (exact-match) accuracy* — the fraction of samples
    whose predicted label set exactly equals the true set. It is the strict,
    honest accuracy for multi-label problems; ``micro_f1`` is the more forgiving
    per-label view. Zero-division cases score 0 (the conservative choice).
    """
    y_true = _as_2d_float(y_true).astype(np.int64)
    y_pred = binarize(y_prob, threshold)

    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "subset_accuracy": float(np.mean(np.all(y_true == y_pred, axis=1))),
    }


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Expected Calibration Error over all label-probability scores.

    Flattens the multi-label probability matrix into individual ``(confidence,
    correct)`` pairs — where ``confidence = max(p, 1 - p)`` and a prediction is
    correct when it matches the true 0/1 — then bins by confidence and averages
    the |accuracy - confidence| gap, weighted by bin population. 0 is perfectly
    calibrated; higher is worse. This is the *calibration* axis of the scorecard.
    """
    y_true = _as_2d_float(y_true).ravel()
    p = _as_2d_float(y_prob).ravel()

    pred = (p >= 0.5).astype(np.int64)
    correct = (pred == y_true.astype(np.int64)).astype(np.float64)
    confidence = np.maximum(p, 1.0 - p)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    # Bin index for each score; clip so confidence == 1.0 lands in the last bin.
    idx = np.clip(np.digitize(confidence, bins) - 1, 0, n_bins - 1)

    n = len(p)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        bin_acc = float(correct[mask].mean())
        bin_conf = float(confidence[mask].mean())
        ece += (count / n) * abs(bin_acc - bin_conf)
    return float(ece)


def per_label_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: "list[str]",
    threshold: float = 0.5,
) -> "dict[str, dict[str, float]]":
    """Per-emotion precision, recall, F1, and support.

    Returns ``{label: {"f1", "precision", "recall", "support"}}``. ``support`` is
    the number of positive examples of that label in ``y_true`` (its class
    frequency in the evaluation set). This is the backbone of the study's RQ3 —
    whether per-emotion F1 degrades with rarity — so per-label F1 and support are
    reported together.
    """
    y_true_i = _as_2d_float(y_true).astype(np.int64)
    y_pred = binarize(y_prob, threshold)
    if y_true_i.shape[1] != len(label_names):
        raise ValueError(
            f"label_names has {len(label_names)} entries but y_true has "
            f"{y_true_i.shape[1]} columns"
        )

    f1s = f1_score(y_true_i, y_pred, average=None, zero_division=0)
    ps = precision_score(y_true_i, y_pred, average=None, zero_division=0)
    rs = recall_score(y_true_i, y_pred, average=None, zero_division=0)
    support = y_true_i.sum(axis=0)

    return {
        name: {
            "f1": float(f1s[i]),
            "precision": float(ps[i]),
            "recall": float(rs[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(label_names)
    }


def per_label_f1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: "list[str]",
    threshold: float = 0.5,
) -> "dict[str, float]":
    """Per-emotion F1 only, as ``{label: f1}`` (convenience over :func:`per_label_metrics`)."""
    return {k: v["f1"] for k, v in per_label_metrics(y_true, y_prob, label_names, threshold).items()}


def evaluate(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5, n_bins: int = 10
) -> dict[str, float]:
    """All probability-based metrics in one call (predictive + calibration)."""
    out = classification_metrics(y_true, y_prob, threshold)
    out["ece"] = expected_calibration_error(y_true, y_prob, n_bins)
    return out


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals (test-set sampling uncertainty)
# ---------------------------------------------------------------------------
# These quantify how precisely a score was estimated on a *finite* test set,
# independent of model stochasticity (which multi-seed runs measure). They are
# especially informative for rare-emotion per-label F1, whose CIs balloon when
# support is small — the honest error bars for the RQ3 core figure.


def _bootstrap_percentiles(samples: np.ndarray, ci: float) -> "tuple":
    a = (1.0 - ci) / 2.0
    lo = np.quantile(samples, a, axis=0)
    hi = np.quantile(samples, 1.0 - a, axis=0)
    return lo, hi


def bootstrap_f1_ci(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5,
    n_boot: int = 1000, ci: float = 0.95, seed: int = 0,
) -> "dict[str, tuple]":
    """Bootstrap CI for macro- and micro-F1.

    Resamples the evaluation rows with replacement ``n_boot`` times (predictions
    are *not* recomputed — no retraining), returning
    ``{"macro": (low, high), "micro": (low, high)}`` at confidence ``ci``.
    """
    y_true = _as_2d_float(y_true).astype(np.int64)
    y_pred = binarize(y_prob, threshold)
    n = y_true.shape[0]
    rng = np.random.default_rng(seed)

    macro, micro = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        macro.append(f1_score(yt, yp, average="macro", zero_division=0))
        micro.append(f1_score(yt, yp, average="micro", zero_division=0))

    mlo, mhi = _bootstrap_percentiles(np.asarray(macro), ci)
    ulo, uhi = _bootstrap_percentiles(np.asarray(micro), ci)
    return {"macro": (float(mlo), float(mhi)), "micro": (float(ulo), float(uhi))}


def bootstrap_per_label_f1_ci(
    y_true: np.ndarray, y_prob: np.ndarray, label_names: "list[str]",
    threshold: float = 0.5, n_boot: int = 1000, ci: float = 0.95, seed: int = 0,
) -> "dict[str, tuple]":
    """Bootstrap CI for each label's F1 → ``{label: (low, high)}``.

    This is the backbone of honest error bars on the per-emotion figure: a rare
    emotion (small support) gets a wide interval, showing its F1 is estimated
    from few examples.
    """
    y_true = _as_2d_float(y_true).astype(np.int64)
    y_pred = binarize(y_prob, threshold)
    if y_true.shape[1] != len(label_names):
        raise ValueError(
            f"label_names has {len(label_names)} entries but y_true has "
            f"{y_true.shape[1]} columns"
        )
    n = y_true.shape[0]
    rng = np.random.default_rng(seed)

    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        samples.append(f1_score(y_true[idx], y_pred[idx], average=None, zero_division=0))
    lo, hi = _bootstrap_percentiles(np.asarray(samples), ci)  # each shape (n_labels,)
    return {name: (float(lo[i]), float(hi[i])) for i, name in enumerate(label_names)}
