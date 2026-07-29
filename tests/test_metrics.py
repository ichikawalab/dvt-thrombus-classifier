"""Unit tests for classification metrics."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from dvt_thrombus.metrics import (
    brier_score,
    confusion_at,
    operating_point,
    roc_auc,
    roc_curve,
    roc_thresholds,
    threshold_at_sensitivity,
    youden_threshold,
)


def test_threshold_grid_brackets_the_scores():
    thresholds = roc_thresholds(np.array([0.1, 0.4, 0.4, 0.9]))
    assert thresholds[0] == -np.inf
    assert thresholds[-1] == np.inf
    # One threshold per unique value, plus the two infinities.
    assert len(thresholds) == 4
    assert thresholds[1] == pytest.approx(0.25)


def test_constant_scores_have_a_finite_candidate_threshold():
    thresholds = roc_thresholds(np.array([0.4, 0.4]))
    np.testing.assert_allclose(thresholds, [-np.inf, 0.4, np.inf])


def test_roc_endpoints_are_degenerate_classifiers():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    _, sensitivity, specificity = roc_curve(y, scores)
    assert (sensitivity[0], specificity[0]) == (1.0, 0.0)
    assert (sensitivity[-1], specificity[-1]) == (0.0, 1.0)


def test_perfect_and_inverted_separation():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    inverted_auc = roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1]))
    assert inverted_auc == 0.0
    assert not np.signbit(inverted_auc)


def test_ties_receive_half_credit():
    # Every score identical: the classifier is uninformative.
    y = np.array([0, 1, 0, 1])
    assert roc_auc(y, np.full(4, 0.5)) == pytest.approx(0.5)


@pytest.mark.parametrize("seed", range(5))
def test_auc_agrees_with_sklearn(seed: int):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=200)
    scores = rng.random(200) + 0.3 * y
    assert roc_auc(y, scores) == pytest.approx(roc_auc_score(y, scores), abs=1e-12)


def test_auc_survives_heavy_ties():
    """Coarse scores create long vertical and horizontal ROC segments."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=300)
    scores = np.round(rng.random(300) + 0.4 * y, 1)
    assert roc_auc(y, scores) == pytest.approx(roc_auc_score(y, scores), abs=1e-12)


def test_youden_threshold_separates_cleanly():
    y = np.array([0, 0, 1, 1])
    threshold = youden_threshold(y, np.array([0.1, 0.2, 0.8, 0.9]))
    tp, tn, fp, fn = confusion_at(y, np.array([0.1, 0.2, 0.8, 0.9]), threshold)
    assert (tp, tn, fp, fn) == (2, 2, 0, 0)


def test_youden_tie_prefers_sensitivity_and_returns_finite_threshold():
    y = np.array([0, 0, 1, 1])
    inverted = np.array([0.9, 0.8, 0.2, 0.1])
    threshold = youden_threshold(y, inverted)
    assert threshold == inverted.min()


def test_threshold_at_sensitivity_meets_the_constraint():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=400)
    scores = rng.random(400) + 0.5 * y
    threshold = threshold_at_sensitivity(y, scores, 0.95)
    assert operating_point(y, scores, threshold).recall >= 0.95


def test_threshold_at_sensitivity_is_the_tightest_available():
    """A higher threshold would violate the constraint."""
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, size=200)
    scores = rng.random(200) + 0.5 * y
    threshold = threshold_at_sensitivity(y, scores, 0.9)
    grid = roc_thresholds(scores)
    higher = grid[grid > threshold]
    assert all(operating_point(y, scores, t).recall < 0.9 for t in higher if np.isfinite(t))


def test_brier_score_rewards_confident_correctness():
    y = np.array([0, 1])
    assert brier_score(y, np.array([0.0, 1.0])) == 0.0
    assert brier_score(y, np.array([1.0, 0.0])) == 1.0


def test_rejects_single_class_input():
    with pytest.raises(ValueError, match="both classes"):
        roc_auc(np.zeros(5, dtype=int), np.linspace(0, 1, 5))


def test_rejects_non_binary_labels():
    with pytest.raises(ValueError, match="only 0 and 1"):
        roc_auc(np.array([0, 1, 2]), np.array([0.1, 0.2, 0.3]))
