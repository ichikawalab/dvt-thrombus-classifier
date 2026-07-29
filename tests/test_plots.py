"""Numerical checks for publication figures."""

from __future__ import annotations

import numpy as np
import pytest

from dvt_thrombus.metrics import roc_auc
from dvt_thrombus.plots import _GRID, _interpolate_roc, _mean_roc


def test_interpolated_roc_preserves_auc_with_vertical_segments():
    labels = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    scores = np.array([0.1, 0.9, 0.8, 0.4, 0.7, 0.6, 0.2, 0.5])
    interpolated = _interpolate_roc(labels, scores)

    assert np.trapezoid(interpolated, _GRID) == pytest.approx(roc_auc(labels, scores), abs=1e-4)


def test_mean_roc_matches_mean_repetition_auc():
    labels = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    probabilities = np.column_stack(
        (
            np.array([0.1, 0.9, 0.8, 0.4, 0.7, 0.6, 0.2, 0.5]),
            np.array([0.2, 0.8, 0.9, 0.3, 0.6, 0.5, 0.1, 0.7]),
        )
    )
    expected = np.mean([roc_auc(labels, probabilities[:, column]) for column in range(2)])

    assert np.trapezoid(_mean_roc(labels, probabilities), _GRID) == pytest.approx(
        expected, abs=1e-4
    )
