"""Validation-derived thresholds must be fixed before test evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dvt_thrombus.analysis import evaluate_at_fold_thresholds
from dvt_thrombus.predictions import PredictionSet


@pytest.fixture
def prediction_set() -> PredictionSet:
    labels = np.array([0, 1, 0, 1, 0, 1])
    frame = pd.DataFrame(
        {
            "image_id": [f"img_{index}" for index in range(6)],
            "patient_id": [f"patient_{index}" for index in range(6)],
            "true_label": labels,
            "fold_1": [1, 1, 2, 2, 3, 3],
            "trial_1": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7],
        }
    )
    return PredictionSet("M", frame)


def _thresholds(value: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "repetition": [1, 1, 1],
            "fold": [1, 2, 3],
            "validation_threshold": value,
        }
    )


def test_counts_are_consistent(prediction_set):
    table = evaluate_at_fold_thresholds(
        prediction_set,
        prediction_set.probability_matrix(),
        _thresholds(0.5),
        "validation_threshold",
        "M",
    )
    assert table[["TP", "TN", "FP", "FN"]].sum(axis=1).eq(6).all()


def test_per_fold_thresholds_are_applied_separately(prediction_set):
    thresholds = _thresholds(0.5)
    thresholds.loc[thresholds["fold"] == 1, "validation_threshold"] = 0.0
    table = evaluate_at_fold_thresholds(
        prediction_set,
        prediction_set.probability_matrix(),
        thresholds,
        "validation_threshold",
        "M",
    )
    assert table.loc[0, "FP"] == 1


def test_missing_column_is_reported(prediction_set):
    with pytest.raises(ValueError, match="missing column"):
        evaluate_at_fold_thresholds(
            prediction_set,
            prediction_set.probability_matrix(),
            pd.DataFrame({"repetition": [1]}),
            "validation_threshold",
            "M",
        )


def test_missing_fold_is_reported(prediction_set):
    with pytest.raises(KeyError, match="no threshold for fold"):
        evaluate_at_fold_thresholds(
            prediction_set,
            prediction_set.probability_matrix(),
            _thresholds(0.5).iloc[:-1],
            "validation_threshold",
            "M",
        )
