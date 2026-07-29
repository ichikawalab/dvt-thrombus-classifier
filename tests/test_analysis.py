"""Tests for fold-specific selection, resampling, and calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dvt_thrombus.analysis import (
    bootstrap_auc_ci,
    bootstrap_auc_difference,
    bootstrap_operating_point_ci,
    build_selected_predictions,
    build_selected_validation_predictions,
    calibration_table,
    derive_validation_thresholds,
    select_models,
    summarise,
)
from dvt_thrombus.ensemble import average_ensemble
from dvt_thrombus.predictions import PredictionSet


def _prediction_set(
    name: str,
    probabilities: np.ndarray,
    labels: np.ndarray,
    patients: np.ndarray,
    folds: np.ndarray | None = None,
) -> PredictionSet:
    frame = pd.DataFrame(
        {
            "image_id": [f"img_{index:03d}" for index in range(len(labels))],
            "patient_id": patients,
            "true_label": labels,
        }
    )
    if folds is None:
        folds = np.tile([1, 2], (len(labels), probabilities.shape[1] // 2 + 1))[
            :, : probabilities.shape[1]
        ]
    for column in range(probabilities.shape[1]):
        frame[f"fold_{column + 1}"] = folds[:, column]
        frame[f"trial_{column + 1}"] = probabilities[:, column]
    return PredictionSet(model=name, frame=frame)


@pytest.fixture
def cohort():
    rng = np.random.default_rng(0)
    n_patients, per_patient = 40, 4
    patients = np.repeat([f"pt_{index:02d}" for index in range(n_patients)], per_patient)
    labels = rng.integers(0, 2, size=n_patients * per_patient)
    signal = labels[:, None]
    strong = np.clip(rng.normal(0.5 + 0.35 * signal, 0.12, (len(labels), 5)), 0, 1)
    weak = np.clip(rng.normal(0.5 + 0.15 * signal, 0.20, (len(labels), 5)), 0, 1)
    return labels, patients, strong, weak


def test_average_ensemble_rejects_nan():
    with pytest.raises(ValueError, match="NaN"):
        average_ensemble([np.array([0.5, np.nan])])


def test_summarise_reports_only_mean_and_sd():
    values = np.linspace(0.90, 0.95, 10)
    frame = pd.DataFrame(
        {
            "Model": "M",
            "Trial": [f"trial_{index}" for index in range(1, 11)],
            "AUC": values,
            "Brier": np.linspace(0.10, 0.12, 10),
        }
    )
    row = summarise(frame).iloc[0]
    assert set(row.index) == {
        "Model",
        "n_repetitions",
        "AUC_mean",
        "AUC_sd",
        "Brier_mean",
        "Brier_sd",
    }
    assert row["AUC_sd"] == pytest.approx(values.std(ddof=1))


def test_patient_bootstrap_is_non_degenerate(cohort):
    labels, patients, strong, _ = cohort
    result = bootstrap_auc_ci(labels, strong, patients, n_resamples=300, seed=1)
    assert result["ci_lower"] < result["auc_mean"] < result["ci_upper"]


def test_paired_difference_is_zero_for_identical_models(cohort):
    labels, patients, strong, _ = cohort
    result = bootstrap_auc_difference(labels, strong, strong, patients, 200, seed=1)
    assert result["difference"] == pytest.approx(0.0)
    assert result["ci_lower"] == pytest.approx(0.0)
    assert result["ci_upper"] == pytest.approx(0.0)


def test_paired_difference_detects_gap(cohort):
    labels, patients, strong, weak = cohort
    result = bootstrap_auc_difference(labels, strong, weak, patients, 400, seed=1)
    assert result["difference"] > 0
    assert result["ci_lower"] > 0


def test_operating_point_bootstrap_reports_patient_intervals(cohort):
    labels, patients, strong, _ = cohort
    thresholds = np.full_like(strong, 0.5)
    result = bootstrap_operating_point_ci(
        labels, strong, thresholds, patients, n_resamples=300, seed=1
    )
    for metric in ("Sensitivity", "Specificity", "PPV", "NPV"):
        assert 0 <= result[f"{metric}_ci_lower"] <= result[f"{metric}_mean"]
        assert result[f"{metric}_mean"] <= result[f"{metric}_ci_upper"] <= 1


def test_operating_point_bootstrap_requires_aligned_matrices(cohort):
    labels, patients, strong, _ = cohort
    with pytest.raises(ValueError, match="identical shapes"):
        bootstrap_operating_point_ci(
            labels,
            strong,
            np.full((len(labels), 1), 0.5),
            patients,
            n_resamples=10,
        )


def test_selection_is_per_fold_and_breaks_ties_by_model_name():
    rows = []
    scores = {
        (1, 1): {"A": 0.9, "B": 0.8, "C": 0.8},
        (1, 2): {"A": 0.6, "B": 0.7, "C": 0.95},
    }
    for (repetition, fold), model_scores in scores.items():
        rows.extend(
            {
                "repetition": repetition,
                "fold": fold,
                "Model": model,
                "val_auc": auc,
            }
            for model, auc in model_scores.items()
        )
    selection = select_models(pd.DataFrame(rows), k=2)
    assert selection[(1, 1)] == ("A", "B")
    assert selection[(1, 2)] == ("C", "B")


def test_selected_predictions_apply_membership_to_each_fold():
    labels = np.array([0, 1, 0, 1])
    patients = np.array(["p1", "p2", "p3", "p4"])
    folds = np.array([[1], [1], [2], [2]])
    a = np.array([[0.1], [0.2], [0.3], [0.4]])
    b = np.array([[0.9], [0.8], [0.7], [0.6]])
    sets = {
        "A": _prediction_set("A", a, labels, patients, folds),
        "B": _prediction_set("B", b, labels, patients, folds),
    }
    matrix = build_selected_predictions(sets, {(1, 1): ("A",), (1, 2): ("B",)})
    np.testing.assert_allclose(matrix[:, 0], [0.1, 0.2, 0.7, 0.6])


def test_selected_predictions_reject_missing_fold():
    labels = np.array([0, 1, 0, 1])
    patients = np.array(["p1", "p2", "p3", "p4"])
    probabilities = np.array([[0.1], [0.8], [0.2], [0.9]])
    folds = np.array([[1], [1], [2], [2]])
    sets = {"A": _prediction_set("A", probabilities, labels, patients, folds)}
    with pytest.raises(ValueError, match="selection keys"):
        build_selected_predictions(sets, {(1, 1): ("A",)})


def test_validation_predictions_are_aligned_before_averaging():
    base = pd.DataFrame(
        {
            "image_id": ["i1", "i2"],
            "repetition": [1, 1],
            "fold": [1, 1],
            "true_label": [0, 1],
            "probability": [0.2, 0.8],
        }
    )
    reversed_rows = base.iloc[::-1].copy()
    reversed_rows["probability"] = [0.6, 0.4]
    result = build_selected_validation_predictions(
        {"A": base, "B": reversed_rows}, {(1, 1): ("A", "B")}
    )
    np.testing.assert_allclose(result["probability"], [0.3, 0.7])


def test_validation_thresholds_include_high_sensitivity_and_youden():
    table = pd.DataFrame(
        {
            "repetition": 1,
            "fold": 1,
            "true_label": [0, 0, 1, 1],
            "probability": [0.1, 0.4, 0.6, 0.9],
        }
    )
    result = derive_validation_thresholds(table, target_sensitivity=1.0)
    assert result.loc[0, "high_sensitivity_threshold"] <= 0.6
    assert result.loc[0, "high_sensitivity_validation_sensitivity"] == 1.0
    assert np.isfinite(result.loc[0, "youden_threshold"])


def test_calibration_bins_are_within_range(cohort):
    labels, _, strong, _ = cohort
    table = calibration_table(labels, strong, "strong", n_bins=10)
    assert table["observed_frequency"].between(0, 1).all()
    assert table["n"].sum() == strong.size


def test_calibration_keeps_each_probability_with_its_image_label():
    labels = np.array([0, 1])
    probabilities = np.array([[0.1, 0.2], [0.8, 0.9]])
    table = calibration_table(labels, probabilities, "model", n_bins=2)
    assert table.loc[table["bin_upper"] == 0.5, "observed_frequency"].item() == 0
    assert table.loc[table["bin_lower"] == 0.5, "observed_frequency"].item() == 1
