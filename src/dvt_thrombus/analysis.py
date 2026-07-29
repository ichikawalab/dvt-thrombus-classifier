"""Leakage-controlled evaluation for repeated patient-level cross-validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias

import numpy as np
import pandas as pd

from .ensemble import average_ensemble
from .metrics import (
    brier_score,
    operating_point,
    roc_auc,
    threshold_at_sensitivity,
    youden_threshold,
)
from .predictions import PredictionSet

SelectionKey: TypeAlias = tuple[int, int]  # (1-indexed repetition, outer fold)
ModelSelection: TypeAlias = dict[SelectionKey, tuple[str, ...]]

__all__ = [
    "ModelSelection",
    "bootstrap_auc_ci",
    "bootstrap_auc_difference",
    "bootstrap_operating_point_ci",
    "build_threshold_matrix",
    "build_selected_predictions",
    "build_selected_validation_predictions",
    "calibration_table",
    "derive_validation_thresholds",
    "evaluate_at_fold_thresholds",
    "evaluate_repetitions",
    "fixed_model_selection",
    "select_models",
    "summarise",
]


def _selection_keys(reference: PredictionSet) -> set[SelectionKey]:
    keys: set[SelectionKey] = set()
    for repetition in range(1, reference.n_repetitions + 1):
        folds = reference.frame[f"fold_{repetition}"].to_numpy(dtype=int)
        keys.update((repetition, int(fold)) for fold in np.unique(folds))
    return keys


def select_models(validation_auc: pd.DataFrame, k: int) -> ModelSelection:
    """Select the top ``k`` models within each repetition and outer fold."""
    required = {"Model", "repetition", "fold", "val_auc"}
    missing = required - set(validation_auc.columns)
    if missing:
        raise ValueError(f"validation AUC is missing column(s) {sorted(missing)}")
    if k < 1:
        raise ValueError("k must be positive")
    if validation_auc.duplicated(["Model", "repetition", "fold"]).any():
        raise ValueError("validation AUC contains duplicate model/repetition/fold rows")
    if not np.isfinite(validation_auc["val_auc"].to_numpy(dtype=float)).all():
        raise ValueError("validation AUC must be finite")

    expected_models = set(validation_auc["Model"])
    selection: ModelSelection = {}
    grouped = validation_auc.groupby(["repetition", "fold"], sort=True)
    for (repetition, fold), group in grouped:
        models = set(group["Model"])
        if models != expected_models:
            missing_models = sorted(expected_models - models)
            raise ValueError(
                f"repetition {repetition} fold {fold}: missing model(s) {missing_models}"
            )
        if len(group) < k:
            raise ValueError(
                f"repetition {repetition} fold {fold}: only {len(group)} models available for k={k}"
            )
        ranked = group.sort_values(["val_auc", "Model"], ascending=[False, True], kind="mergesort")
        selection[(int(repetition), int(fold))] = tuple(ranked["Model"].iloc[:k])
    return selection


def fixed_model_selection(
    keys: Sequence[SelectionKey] | set[SelectionKey], members: Sequence[str]
) -> ModelSelection:
    """Use the same configured model members for every outer fold."""
    chosen = tuple(members)
    if not chosen:
        raise ValueError("at least one model is required")
    return {key: chosen for key in sorted(keys)}


def _validate_selection(sets: Mapping[str, PredictionSet], selection: ModelSelection) -> None:
    if not sets:
        raise ValueError("at least one prediction set is required")
    reference = next(iter(sets.values()))
    expected = _selection_keys(reference)
    if set(selection) != expected:
        missing = sorted(expected - set(selection))
        extra = sorted(set(selection) - expected)
        raise ValueError(
            f"selection keys do not match test folds; missing={missing}, extra={extra}"
        )
    unknown = sorted({model for members in selection.values() for model in members} - set(sets))
    if unknown:
        raise KeyError(f"selection refers to unknown model(s) {unknown}")
    if any(not members for members in selection.values()):
        raise ValueError("every selection must contain at least one model")


def build_selected_predictions(
    sets: Mapping[str, PredictionSet], selection: ModelSelection
) -> np.ndarray:
    """Build held-out predictions using fold-specific model membership."""
    _validate_selection(sets, selection)
    reference = next(iter(sets.values()))
    result = np.full((len(reference.frame), reference.n_repetitions), np.nan)

    for repetition in range(1, reference.n_repetitions + 1):
        folds = reference.frame[f"fold_{repetition}"].to_numpy(dtype=int)
        for fold in np.unique(folds):
            mask = folds == fold
            members = selection[(repetition, int(fold))]
            result[mask, repetition - 1] = average_ensemble(
                [sets[model].probabilities(repetition)[mask] for model in members]
            )

    if np.isnan(result).any():
        raise RuntimeError("some held-out observations did not receive an ensemble prediction")
    return result


def _normalise_validation_table(table: pd.DataFrame, model: str) -> pd.DataFrame:
    columns = ["image_id", "repetition", "fold", "true_label", "probability"]
    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError(f"{model}: validation predictions missing column(s) {sorted(missing)}")
    if table.duplicated(["image_id", "repetition", "fold"]).any():
        raise ValueError(f"{model}: duplicate validation prediction rows")
    result = table[columns].copy()
    result["image_id"] = result["image_id"].astype(str)
    if not result["true_label"].isin((0, 1)).all():
        raise ValueError(f"{model}: true_label must contain only 0 and 1")
    probabilities = result["probability"].to_numpy(dtype=float)
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError(f"{model}: validation probabilities must lie in [0, 1]")
    return result


def build_selected_validation_predictions(
    validation_predictions: Mapping[str, pd.DataFrame],
    selection: ModelSelection,
) -> pd.DataFrame:
    """Average validation probabilities for the members selected in each fold."""
    normalised = {
        model: _normalise_validation_table(table, model)
        for model, table in validation_predictions.items()
    }
    selected_models = {model for members in selection.values() for model in members}
    unknown = sorted(selected_models - set(normalised))
    if unknown:
        raise KeyError(f"validation predictions missing model(s) {unknown}")

    rows: list[pd.DataFrame] = []
    for (repetition, fold), members in sorted(selection.items()):
        aligned: pd.DataFrame | None = None
        probabilities: list[np.ndarray] = []
        for model in members:
            table = normalised[model]
            current = table.loc[
                (table["repetition"] == repetition) & (table["fold"] == fold),
                ["image_id", "true_label", "probability"],
            ].sort_values("image_id")
            if current.empty:
                raise ValueError(
                    f"{model}: no validation predictions for repetition {repetition} fold {fold}"
                )
            identity = current[["image_id", "true_label"]].reset_index(drop=True)
            if aligned is None:
                aligned = identity
            elif not aligned.equals(identity):
                raise ValueError(
                    f"validation rows differ across models for repetition {repetition} fold {fold}"
                )
            probabilities.append(current["probability"].to_numpy(dtype=float))

        assert aligned is not None
        aligned["repetition"] = repetition
        aligned["fold"] = fold
        aligned["probability"] = average_ensemble(probabilities)
        rows.append(aligned)
    return pd.concat(rows, ignore_index=True)


def derive_validation_thresholds(
    validation_predictions: pd.DataFrame, target_sensitivity: float = 0.95
) -> pd.DataFrame:
    """Derive high-sensitivity and Youden thresholds from validation data only."""
    required = {"repetition", "fold", "true_label", "probability"}
    missing = required - set(validation_predictions.columns)
    if missing:
        raise ValueError(f"validation predictions missing column(s) {sorted(missing)}")

    rows = []
    grouped = validation_predictions.groupby(["repetition", "fold"], sort=True)
    for (repetition, fold), group in grouped:
        labels = group["true_label"].to_numpy(dtype=int)
        probabilities = group["probability"].to_numpy(dtype=float)
        high_threshold = threshold_at_sensitivity(labels, probabilities, target_sensitivity)
        youden = youden_threshold(labels, probabilities)
        high_point = operating_point(labels, probabilities, high_threshold)
        youden_point = operating_point(labels, probabilities, youden)
        rows.append(
            {
                "repetition": int(repetition),
                "fold": int(fold),
                "n_validation": len(group),
                "validation_auc": roc_auc(labels, probabilities),
                "high_sensitivity_threshold": high_threshold,
                "high_sensitivity_validation_sensitivity": high_point.recall,
                "high_sensitivity_validation_specificity": high_point.specificity,
                "youden_threshold": youden,
                "youden_validation_sensitivity": youden_point.recall,
                "youden_validation_specificity": youden_point.specificity,
            }
        )
    return pd.DataFrame(rows)


def evaluate_repetitions(
    y_true: np.ndarray, probability_matrix: np.ndarray, model: str
) -> pd.DataFrame:
    """Calculate threshold-free metrics after pooling the five test folds."""
    rows = []
    for index in range(probability_matrix.shape[1]):
        scores = probability_matrix[:, index]
        rows.append(
            {
                "Model": model,
                "Trial": f"trial_{index + 1}",
                "AUC": roc_auc(y_true, scores),
                "Brier": brier_score(y_true, scores),
            }
        )
    return pd.DataFrame(rows)


def evaluate_at_fold_thresholds(
    reference: PredictionSet,
    probability_matrix: np.ndarray,
    thresholds: pd.DataFrame,
    threshold_column: str,
    model: str,
) -> pd.DataFrame:
    """Apply one validation-derived threshold to each corresponding test fold."""
    threshold_matrix = build_threshold_matrix(reference, thresholds, threshold_column)
    rows = []
    for repetition in range(1, reference.n_repetitions + 1):
        applied = threshold_matrix[:, repetition - 1]
        point = operating_point(reference.y_true, probability_matrix[:, repetition - 1], applied)
        rows.append(
            {
                "Model": model,
                "Trial": f"trial_{repetition}",
                "mean_threshold": point.threshold,
                "TP": point.tp,
                "TN": point.tn,
                "FP": point.fp,
                "FN": point.fn,
                "Sensitivity": point.recall,
                "Specificity": point.specificity,
                "PPV": point.precision,
                "NPV": point.npv,
            }
        )
    return pd.DataFrame(rows)


def build_threshold_matrix(
    reference: PredictionSet,
    thresholds: pd.DataFrame,
    threshold_column: str,
) -> np.ndarray:
    """Map each validation-derived fold threshold to its held-out observations."""
    required = {"repetition", "fold", threshold_column}
    missing = required - set(thresholds.columns)
    if missing:
        raise ValueError(f"thresholds missing column(s) {sorted(missing)}")
    if thresholds.duplicated(["repetition", "fold"]).any():
        raise ValueError("thresholds contain duplicate repetition/fold rows")

    lookup = thresholds.set_index(["repetition", "fold"])[threshold_column].to_dict()
    matrix = np.empty((len(reference.frame), reference.n_repetitions), dtype=float)
    for repetition in range(1, reference.n_repetitions + 1):
        folds = reference.frame[f"fold_{repetition}"].to_numpy(dtype=int)
        missing_folds = sorted(
            int(fold) for fold in np.unique(folds) if (repetition, int(fold)) not in lookup
        )
        if missing_folds:
            raise KeyError(f"repetition {repetition}: no threshold for fold(s) {missing_folds}")
        matrix[:, repetition - 1] = [lookup[(repetition, int(fold))] for fold in folds]
    return matrix


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and repetition-to-repetition SD across repeated cross-validation."""
    rows = []
    for model, group in results.groupby("Model", sort=False):
        rows.append(
            {
                "Model": model,
                "n_repetitions": len(group),
                "AUC_mean": float(group["AUC"].mean()),
                "AUC_sd": float(group["AUC"].std(ddof=1)),
                "Brier_mean": float(group["Brier"].mean()),
                "Brier_sd": float(group["Brier"].std(ddof=1)),
            }
        )
    return pd.DataFrame(rows).sort_values("AUC_mean", ascending=False).reset_index(drop=True)


def _patient_resample_indices(
    patient_id: np.ndarray, n_resamples: int, rng: np.random.Generator
) -> list[np.ndarray]:
    unique_patients = np.unique(patient_id)
    index_by_patient = {
        patient: np.flatnonzero(patient_id == patient) for patient in unique_patients
    }
    return [
        np.concatenate(
            [
                index_by_patient[patient]
                for patient in rng.choice(unique_patients, size=unique_patients.size, replace=True)
            ]
        )
        for _ in range(n_resamples)
    ]


def _validate_bootstrap_design(
    y_true: np.ndarray,
    patient_id: np.ndarray,
    n_resamples: int,
    confidence: float,
    *matrices: np.ndarray,
) -> None:
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    if len(y_true) != len(patient_id):
        raise ValueError("labels and patients must align")
    if not matrices:
        raise ValueError("at least one probability matrix is required")
    reference_shape = matrices[0].shape
    if len(reference_shape) != 2 or reference_shape[0] != len(y_true):
        raise ValueError("probability matrices must be two-dimensional and align with labels")
    if any(matrix.shape != reference_shape for matrix in matrices[1:]):
        raise ValueError("probability matrices must have identical shapes")
    if any(not np.isfinite(matrix).all() for matrix in matrices):
        raise ValueError("probability matrices must be finite")


def _mean_auc(
    y_true: np.ndarray, probability_matrix: np.ndarray, index: np.ndarray | None = None
) -> float:
    if index is None:
        labels, matrix = y_true, probability_matrix
    else:
        labels, matrix = y_true[index], probability_matrix[index]
    return float(np.mean([roc_auc(labels, matrix[:, column]) for column in range(matrix.shape[1])]))


def _bootstrap_interval(values: Sequence[float], confidence: float) -> tuple[float, float]:
    alpha = (1.0 - confidence) / 2.0
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("no valid bootstrap resamples contained both classes")
    return float(np.quantile(array, alpha)), float(np.quantile(array, 1.0 - alpha))


def bootstrap_auc_ci(
    y_true: np.ndarray,
    probability_matrix: np.ndarray,
    patient_id: np.ndarray,
    n_resamples: int = 5000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Patient-cluster bootstrap interval for mean AUC across repetitions."""
    _validate_bootstrap_design(y_true, patient_id, n_resamples, confidence, probability_matrix)
    draws = _patient_resample_indices(patient_id, n_resamples, np.random.default_rng(seed))
    values = [
        _mean_auc(y_true, probability_matrix, index)
        for index in draws
        if np.unique(y_true[index]).size == 2
    ]
    lower, upper = _bootstrap_interval(values, confidence)
    return {
        "auc_mean": _mean_auc(y_true, probability_matrix),
        "ci_lower": lower,
        "ci_upper": upper,
        "n_resamples": len(values),
    }


def bootstrap_auc_difference(
    y_true: np.ndarray,
    probability_matrix_a: np.ndarray,
    probability_matrix_b: np.ndarray,
    patient_id: np.ndarray,
    n_resamples: int = 5000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Paired patient-cluster bootstrap interval for AUC(a) minus AUC(b)."""
    _validate_bootstrap_design(
        y_true,
        patient_id,
        n_resamples,
        confidence,
        probability_matrix_a,
        probability_matrix_b,
    )
    draws = _patient_resample_indices(patient_id, n_resamples, np.random.default_rng(seed))
    values = [
        _mean_auc(y_true, probability_matrix_a, index)
        - _mean_auc(y_true, probability_matrix_b, index)
        for index in draws
        if np.unique(y_true[index]).size == 2
    ]
    lower, upper = _bootstrap_interval(values, confidence)
    return {
        "difference": _mean_auc(y_true, probability_matrix_a)
        - _mean_auc(y_true, probability_matrix_b),
        "ci_lower": lower,
        "ci_upper": upper,
        "n_resamples": len(values),
    }


def _mean_operating_metrics(
    y_true: np.ndarray,
    probability_matrix: np.ndarray,
    threshold_matrix: np.ndarray,
    index: np.ndarray | None = None,
) -> dict[str, float]:
    labels = y_true if index is None else y_true[index]
    probabilities = probability_matrix if index is None else probability_matrix[index]
    thresholds = threshold_matrix if index is None else threshold_matrix[index]
    rows = [
        operating_point(labels, probabilities[:, column], thresholds[:, column])
        for column in range(probabilities.shape[1])
    ]
    return {
        "Sensitivity": float(np.mean([row.recall for row in rows])),
        "Specificity": float(np.mean([row.specificity for row in rows])),
        "PPV": float(np.mean([row.precision for row in rows])),
        "NPV": float(np.mean([row.npv for row in rows])),
    }


def bootstrap_operating_point_ci(
    y_true: np.ndarray,
    probability_matrix: np.ndarray,
    threshold_matrix: np.ndarray,
    patient_id: np.ndarray,
    n_resamples: int = 5000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Patient-cluster bootstrap intervals for mean repeated-CV operating metrics."""
    if probability_matrix.shape != threshold_matrix.shape:
        raise ValueError("probability and threshold matrices must have identical shapes")
    _validate_bootstrap_design(
        y_true, patient_id, n_resamples, confidence, probability_matrix, threshold_matrix
    )

    draws = _patient_resample_indices(patient_id, n_resamples, np.random.default_rng(seed))
    point = _mean_operating_metrics(y_true, probability_matrix, threshold_matrix)
    sampled = [
        _mean_operating_metrics(y_true, probability_matrix, threshold_matrix, index)
        for index in draws
        if np.unique(y_true[index]).size == 2
    ]
    result: dict[str, float] = {"n_resamples": len(sampled)}
    for metric, value in point.items():
        values = [row[metric] for row in sampled if np.isfinite(row[metric])]
        lower, upper = _bootstrap_interval(values, confidence)
        result[f"{metric}_mean"] = value
        result[f"{metric}_ci_lower"] = lower
        result[f"{metric}_ci_upper"] = upper
    return result


def calibration_table(
    y_true: np.ndarray, probability_matrix: np.ndarray, model: str, n_bins: int = 10
) -> pd.DataFrame:
    """Reliability-diagram data pooled descriptively over repetitions."""
    scores = probability_matrix.ravel()
    outcomes = np.repeat(np.asarray(y_true, dtype=float), probability_matrix.shape[1])
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    assignment = np.clip(np.digitize(scores, edges[1:-1]), 0, n_bins - 1)

    rows = []
    for bin_index in range(n_bins):
        mask = assignment == bin_index
        if mask.any():
            rows.append(
                {
                    "Model": model,
                    "bin_lower": edges[bin_index],
                    "bin_upper": edges[bin_index + 1],
                    "n": int(mask.sum()),
                    "mean_predicted": float(scores[mask].mean()),
                    "observed_frequency": float(outcomes[mask].mean()),
                }
            )
    return pd.DataFrame(rows)
