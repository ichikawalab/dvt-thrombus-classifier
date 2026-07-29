"""Tests for the global image-characteristic analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dvt_thrombus.image_features import (
    baseline_predictions,
    paired_within_patient,
    summarise_image_dimensions,
    univariate_separability,
)
from dvt_thrombus.metrics import roc_auc
from dvt_thrombus.splitting import iter_splits


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(0)
    n_patients, per_patient = 30, 4
    patient_id = np.repeat([f"pt_{i:02d}" for i in range(n_patients)], per_patient)
    label = np.tile([0, 0, 1, 1], n_patients)
    return pd.DataFrame(
        {
            "image_id": [f"image_{i:03d}" for i in range(len(label))],
            "patient_id": patient_id,
            "label": label,
            # Intensity carries the signal; the rest is noise.
            "mean_intensity": 50 + 20 * label + rng.normal(0, 3, len(label)),
            "sd_intensity": 40 + rng.normal(0, 3, len(label)),
            "width_px": np.tile([640, 800], len(label) // 2),
            "height_px": np.tile([480, 600], len(label) // 2),
        }
    )


def test_univariate_reports_positive_class_auc(synthetic):
    table = univariate_separability(synthetic)
    assert table.iloc[0]["feature"] == "mean_intensity"
    assert table.loc[table["feature"] == "mean_intensity", "auc"].item() > 0.95


def test_baseline_recovers_a_planted_signal(synthetic):
    predictions = baseline_predictions(synthetic, ("mean_intensity",), n_folds=5, n_repetitions=3)
    aucs = [
        roc_auc(predictions["label"], predictions[f"trial_{repetition}"])
        for repetition in range(1, 4)
    ]
    assert min(aucs) > 0.95


def test_image_dimensions_are_descriptive_only(synthetic):
    summary = summarise_image_dimensions(synthetic).set_index("dimension")
    assert summary.loc["width_px", "median"] == 720
    assert summary.loc["height_px", "min"] == 480
    assert set(summary.index).isdisjoint({"mean_intensity", "sd_intensity"})


def test_baseline_uses_configured_outer_folds(synthetic):
    predictions = baseline_predictions(synthetic, n_folds=5, n_repetitions=2)
    expected = np.full((len(synthetic), 2), -1)
    for split in iter_splits(
        synthetic["label"].to_numpy(),
        synthetic["patient_id"].to_numpy(),
        n_repetitions=2,
        n_folds=5,
        base_seed=42,
    ):
        expected[split.test, split.repetition] = split.fold + 1
    assert np.array_equal(predictions[["fold_1", "fold_2"]].to_numpy(), expected)


def test_paired_within_patient_detects_the_planted_offset(synthetic):
    result = paired_within_patient(synthetic)
    assert result["n_patients"] == 30
    assert result["mean_difference"] == pytest.approx(20, abs=2)
    assert result["ci_lower"] < result["mean_difference"] < result["ci_upper"]
    assert result["n_positive_higher"] == 30


def test_paired_within_patient_needs_mixed_patients():
    frame = pd.DataFrame(
        {"patient_id": ["a", "a", "b", "b"], "label": [0, 0, 1, 1], "mean_intensity": [1, 2, 3, 4]}
    )
    with pytest.raises(ValueError, match="both classes"):
        paired_within_patient(frame)
