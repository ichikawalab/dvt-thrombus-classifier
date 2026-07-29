"""Classification metrics with deterministic threshold construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "OperatingPoint",
    "brier_score",
    "confusion_at",
    "operating_point",
    "roc_auc",
    "roc_curve",
    "roc_thresholds",
    "threshold_at_sensitivity",
    "youden_threshold",
]


def roc_thresholds(scores: np.ndarray) -> np.ndarray:
    """Return thresholds between consecutive unique scores."""
    unique = np.unique(np.asarray(scores, dtype=float))
    if unique.size == 1:
        return np.array([-np.inf, unique[0], np.inf])
    lower = np.concatenate(([-np.inf], unique))
    upper = np.concatenate((unique, [np.inf]))
    with np.errstate(invalid="ignore"):
        thresholds = (lower + upper) / 2.0
    thresholds[0] = -np.inf
    thresholds[-1] = np.inf
    return thresholds


def roc_curve(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, ...]:
    """Sensitivity and specificity over the full threshold grid.

    Returns ``(thresholds, sensitivity, specificity)``. Positives satisfy
    ``score >= threshold``.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if y_true.shape != scores.shape:
        raise ValueError("y_true and scores must have the same shape")
    if not np.isin(y_true, (0, 1)).all():
        raise ValueError("y_true must contain only 0 and 1")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")

    thresholds = roc_thresholds(scores)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("both classes must be present to compute a ROC curve")

    # predicted[i, j] would be scores[j] >= thresholds[i]; done via searchsorted
    # to keep memory flat for large inputs.
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_y = y_true[order]
    cum_pos = np.concatenate(([0], np.cumsum(sorted_y == 1)))
    cum_neg = np.concatenate(([0], np.cumsum(sorted_y == 0)))

    # number of samples with score < threshold
    idx = np.searchsorted(sorted_scores, thresholds, side="left")
    fn = cum_pos[idx]  # positives predicted negative
    tn = cum_neg[idx]  # negatives predicted negative
    sensitivity = (n_pos - fn) / n_pos
    specificity = tn / n_neg
    return thresholds, sensitivity, specificity


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Calculate trapezoidal area under the ROC curve."""
    _, sensitivity, specificity = roc_curve(y_true, scores)
    fpr = 1.0 - specificity
    area = float(-np.trapezoid(sensitivity, fpr))
    if area <= 0.0:
        return 0.0
    if area >= 1.0:
        return 1.0
    return area


def youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximising Youden's J, with sensitivity-first tie-breaking."""
    thresholds, sensitivity, specificity = roc_curve(y_true, scores)
    youden = sensitivity + specificity - 1.0
    best = youden == youden.max()
    best &= sensitivity == sensitivity[best].max()
    threshold = float(thresholds[best].max())
    if threshold == -np.inf:
        return float(np.min(scores))
    if threshold == np.inf:
        return float(np.nextafter(np.max(scores), np.inf))
    return threshold


def threshold_at_sensitivity(y_true: np.ndarray, scores: np.ndarray, target: float) -> float:
    """Highest threshold whose sensitivity is at least ``target``.

    Taking the highest such threshold gives the best attainable specificity
    subject to the sensitivity constraint.  Returns ``nan`` if the constraint
    cannot be met.
    """
    if not 0.0 < target <= 1.0:
        raise ValueError("target must lie in (0, 1]")
    thresholds, sensitivity, _ = roc_curve(y_true, scores)
    eligible = thresholds[sensitivity >= target]
    if eligible.size == 0:
        return float("nan")
    return float(eligible.max())


def confusion_at(
    y_true: np.ndarray, scores: np.ndarray, threshold: float | np.ndarray
) -> tuple[int, ...]:
    """Return ``(tp, tn, fp, fn)`` for ``score >= threshold``.

    ``threshold`` may be an array of per-observation thresholds, which is what
    cross-validation needs when each fold contributes its own.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if y_true.shape != scores.shape:
        raise ValueError("y_true and scores must have the same shape")
    if not np.isin(y_true, (0, 1)).all():
        raise ValueError("y_true must contain only 0 and 1")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    try:
        predicted = scores >= threshold
    except ValueError as error:
        raise ValueError("threshold must be scalar or match the score shape") from error
    if predicted.shape != scores.shape:
        raise ValueError("threshold must be scalar or match the score shape")
    tp = int(np.sum(predicted & (y_true == 1)))
    tn = int(np.sum(~predicted & (y_true == 0)))
    fp = int(np.sum(predicted & (y_true == 0)))
    fn = int(np.sum(~predicted & (y_true == 1)))
    return tp, tn, fp, fn


def brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Mean squared difference between predicted probability and outcome."""
    y_true = np.asarray(y_true, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if y_true.shape != probabilities.shape:
        raise ValueError("y_true and probabilities must have the same shape")
    if not np.isin(y_true, (0, 1)).all():
        raise ValueError("y_true must contain only 0 and 1")
    if (
        not np.isfinite(probabilities).all()
        or not np.logical_and(probabilities >= 0, probabilities <= 1).all()
    ):
        raise ValueError("probabilities must be finite and lie in [0, 1]")
    return float(np.mean((probabilities - y_true) ** 2))


@dataclass(frozen=True)
class OperatingPoint:
    """Metrics at a single decision threshold."""

    threshold: float
    tp: int
    tn: int
    fp: int
    fn: int
    accuracy: float
    precision: float
    recall: float
    specificity: float
    npv: float
    f1_score: float


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float("nan") if denominator == 0 else numerator / denominator


def operating_point(
    y_true: np.ndarray, scores: np.ndarray, threshold: float | np.ndarray
) -> OperatingPoint:
    """Full metric set at ``threshold``.

    An array of per-observation thresholds is accepted; the reported
    ``threshold`` is then their mean.
    """
    tp, tn, fp, fn = confusion_at(y_true, scores, threshold)
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        f1 = float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return OperatingPoint(
        threshold=float(np.mean(threshold)),
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        accuracy=_safe_ratio(tp + tn, tp + tn + fp + fn),
        precision=precision,
        recall=recall,
        specificity=_safe_ratio(tn, tn + fp),
        npv=_safe_ratio(tn, tn + fn),
        f1_score=f1,
    )
