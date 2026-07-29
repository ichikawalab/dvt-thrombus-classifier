"""Probability-based ensembles of individual model predictions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["average_ensemble"]


def _stack(probabilities: Sequence[np.ndarray]) -> np.ndarray:
    if len(probabilities) == 0:
        raise ValueError("at least one model is required")
    matrix = np.column_stack([np.asarray(p, dtype=float) for p in probabilities])
    if np.isnan(matrix).any():
        raise ValueError("predicted probabilities contain NaN")
    return matrix


def average_ensemble(probabilities: Sequence[np.ndarray]) -> np.ndarray:
    """Unweighted mean of the thrombus-positive probabilities across models."""
    return _stack(probabilities).mean(axis=1)
