"""Assemble the study analysis from held-out predictions."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import (
    ModelSelection,
    bootstrap_auc_ci,
    bootstrap_auc_difference,
    bootstrap_operating_point_ci,
    build_selected_predictions,
    build_selected_validation_predictions,
    build_threshold_matrix,
    calibration_table,
    derive_validation_thresholds,
    evaluate_at_fold_thresholds,
    evaluate_repetitions,
    fixed_model_selection,
    select_models,
    summarise,
)
from .config import ARCHITECTURES
from .image_features import (
    FEATURE_COLUMNS,
    baseline_predictions,
    paired_within_patient,
    summarise_image_dimensions,
    univariate_separability,
)
from .metrics import roc_auc
from .plots import (
    plot_auc_distribution,
    plot_calibration,
    plot_confusion_matrices,
    plot_mean_roc,
)
from .predictions import PredictionSet, load_prediction_set

__all__ = ["run_report"]

TOP1 = "Validation-selected Top-1 Individual"
TOP3 = "Validation-selected Top-3 Average Ensemble"
ALL6 = "All-6 Average Ensemble"


def _cohort_composition(patient_id: np.ndarray, y_true: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"patient_id": patient_id, "label": y_true})
    per_patient = frame.groupby("patient_id")["label"].agg(["size", "sum"])
    per_patient["kind"] = np.where(
        (per_patient["sum"] > 0) & (per_patient["sum"] < per_patient["size"]),
        "mixed",
        np.where(per_patient["sum"] > 0, "positive only", "negative only"),
    )
    counts = per_patient["kind"].value_counts()
    return pd.DataFrame(
        {
            "quantity": [
                "patients",
                "images",
                "positive images",
                "negative images",
                "patients with both classes",
                "patients positive only",
                "patients negative only",
                "images per patient (mean)",
                "images per patient (min)",
                "images per patient (max)",
            ],
            "value": [
                len(per_patient),
                len(frame),
                int(y_true.sum()),
                int((y_true == 0).sum()),
                int(counts.get("mixed", 0)),
                int(counts.get("positive only", 0)),
                int(counts.get("negative only", 0)),
                round(float(per_patient["size"].mean()), 2),
                int(per_patient["size"].min()),
                int(per_patient["size"].max()),
            ],
        }
    )


def _load_run_artifacts(
    runs_dir: Path,
) -> tuple[dict[str, PredictionSet], pd.DataFrame, dict[str, pd.DataFrame]]:
    sets: dict[str, PredictionSet] = {}
    auc_rows: list[dict[str, object]] = []
    predictions: dict[str, pd.DataFrame] = {}
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        test_path = run_dir / "predictions.csv"
        predictions_path = run_dir / "validation_predictions.csv"
        existing = [path.exists() for path in (test_path, predictions_path)]
        if not any(existing):
            continue
        if not all(existing):
            raise FileNotFoundError(
                f"{run_dir}: predictions.csv and validation_predictions.csv are required"
            )
        sets[run_dir.name] = load_prediction_set(test_path, model=run_dir.name)
        validation = pd.read_csv(predictions_path, dtype={"image_id": str, "patient_id": str})
        predictions[run_dir.name] = validation
        for (repetition, fold), group in validation.groupby(["repetition", "fold"], sort=True):
            auc_rows.append(
                {
                    "Model": run_dir.name,
                    "repetition": int(repetition),
                    "fold": int(fold),
                    "val_auc": roc_auc(
                        group["true_label"].to_numpy(dtype=int),
                        group["probability"].to_numpy(dtype=float),
                    ),
                }
            )

    if not auc_rows:
        raise FileNotFoundError(f"no complete per-model run artifacts found in {runs_dir}")

    reference_name, reference = next(iter(sets.items()))
    for name, candidate in sets.items():
        if name == reference_name:
            continue
        identity_columns = ["image_id", "patient_id", "true_label", *reference.fold_columns]
        if candidate.fold_columns != reference.fold_columns or not candidate.frame[
            identity_columns
        ].equals(reference.frame[identity_columns]):
            raise ValueError(
                f"{name}: cases, labels, or folds differ from {reference_name}; "
                "paired analysis is invalid"
            )
    return sets, pd.DataFrame(auc_rows), predictions


def _validate_models(
    sets: dict[str, PredictionSet],
    validation_predictions: dict[str, pd.DataFrame],
) -> list[str]:
    expected = set(ARCHITECTURES)
    predicted = set(sets)
    validated = set(validation_predictions)
    if predicted != expected:
        raise ValueError(
            f"prediction models must be the six configured architectures; "
            f"missing={sorted(expected - predicted)}, extra={sorted(predicted - expected)}"
        )
    if validated != expected:
        raise ValueError(
            f"validation models must match prediction models; "
            f"missing={sorted(expected - validated)}, extra={sorted(validated - expected)}"
        )
    reference = sets[next(iter(ARCHITECTURES))]
    if reference.n_repetitions != 10:
        raise ValueError(
            f"the analysis protocol requires 10 repetitions, found {reference.n_repetitions}"
        )
    expected_folds = set(range(1, 6))
    for column in reference.fold_columns:
        observed = set(reference.frame[column].to_numpy(dtype=int))
        if observed != expected_folds:
            raise ValueError(
                f"{column}: the analysis protocol requires folds 1-5, found {sorted(observed)}"
            )
    return list(ARCHITECTURES)


def _selection_rows(selections: dict[str, ModelSelection]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "strategy": strategy,
                "repetition": repetition,
                "fold": fold,
                "members": ", ".join(members),
            }
            for strategy in (TOP1, TOP3)
            for selection in (selections[strategy],)
            for (repetition, fold), members in sorted(selection.items())
        ]
    )


def _summarise_operating_points(table: pd.DataFrame) -> pd.DataFrame:
    metrics = ["Sensitivity", "Specificity", "PPV", "NPV", "TP", "TN", "FP", "FN"]
    rows = []
    for model, group in table.groupby("Model", sort=False):
        row: dict[str, object] = {"Model": model, "n_repetitions": len(group)}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def _selection_frequencies(selections: dict[str, ModelSelection]) -> pd.DataFrame:
    rows = []
    for strategy in (TOP1, TOP3):
        counts = Counter(model for members in selections[strategy].values() for model in members)
        denominator = len(selections[strategy])
        for model in ARCHITECTURES:
            rows.append(
                {
                    "strategy": strategy,
                    "Model": model,
                    "selected_folds": counts[model],
                    "total_folds": denominator,
                    "selection_frequency": counts[model] / denominator,
                }
            )
    return pd.DataFrame(rows)


def _image_feature_report(
    features_path: Path,
    output_dir: Path,
    reference: PredictionSet,
    top3_probabilities: np.ndarray,
    n_resamples: int,
) -> dict[str, object]:
    features = pd.read_csv(features_path, dtype={"image_id": str, "patient_id": str})
    features["image_id"] = features["image_id"].astype(str)
    features["patient_id"] = features["patient_id"].astype(str)
    if features["image_id"].duplicated().any():
        raise ValueError("image features contain duplicate image_id values")
    indexed = features.set_index("image_id", drop=False)
    expected_ids = reference.frame["image_id"].astype(str)
    missing = sorted(set(expected_ids) - set(indexed.index))
    extra = sorted(set(indexed.index) - set(expected_ids))
    if missing or extra:
        raise ValueError(
            f"image features do not match test predictions; missing={missing}, extra={extra}"
        )
    features = indexed.loc[expected_ids].reset_index(drop=True)
    if not np.array_equal(
        features["patient_id"].astype(str).to_numpy(), reference.patient_id
    ) or not np.array_equal(features["label"].to_numpy(dtype=int), reference.y_true):
        raise ValueError("image-feature patient IDs or labels do not match test predictions")

    summarise_image_dimensions(features).to_csv(
        output_dir / "source_image_dimensions.csv", index=False
    )
    univariate_separability(features).to_csv(
        output_dir / "image_feature_univariate_auc.csv", index=False
    )
    predictions = baseline_predictions(features, FEATURE_COLUMNS)
    expected_folds = reference.frame[reference.fold_columns].to_numpy(dtype=int)
    predicted_folds = predictions[reference.fold_columns].to_numpy(dtype=int)
    if not np.array_equal(predicted_folds, expected_folds):
        raise RuntimeError("image-feature baseline folds differ from deep-model folds")
    predictions.to_csv(output_dir / "image_feature_baseline_predictions.csv", index=False)
    probability_columns = [column for column in predictions.columns if column.startswith("trial_")]
    interval = bootstrap_auc_ci(
        predictions["label"].to_numpy(dtype=int),
        predictions[probability_columns].to_numpy(dtype=float),
        predictions["patient_id"].to_numpy(dtype=str),
        n_resamples=n_resamples,
    )
    pd.DataFrame([{**interval, "features": ", ".join(FEATURE_COLUMNS)}]).to_csv(
        output_dir / "image_feature_baseline_auc.csv", index=False
    )
    baseline_matrix = predictions[probability_columns].to_numpy(dtype=float)
    difference = bootstrap_auc_difference(
        reference.y_true,
        top3_probabilities,
        baseline_matrix,
        reference.patient_id,
        n_resamples=n_resamples,
    )
    pd.DataFrame(
        [
            {
                "comparison": f"{TOP3} minus image-feature baseline",
                **difference,
            }
        ]
    ).to_csv(output_dir / "auc_difference_top3_vs_image_features.csv", index=False)
    paired = paired_within_patient(features, n_resamples=n_resamples)
    pd.DataFrame([paired]).to_csv(
        output_dir / "within_patient_intensity_difference.csv", index=False
    )
    return {
        "image_feature_baseline_auc": round(interval["auc_mean"], 4),
        "image_feature_baseline_ci": [
            round(interval["ci_lower"], 4),
            round(interval["ci_upper"], 4),
        ],
        "top3_minus_image_feature_baseline": {
            key: round(float(value), 4) for key, value in difference.items()
        },
        "within_patient_intensity": {
            "n_patients": paired["n_patients"],
            "mean_difference": round(paired["mean_difference"], 2),
            "ci_lower": round(paired["ci_lower"], 2),
            "ci_upper": round(paired["ci_upper"], 2),
            "n_positive_higher": paired["n_positive_higher"],
        },
    }


def run_report(
    runs_dir: Path,
    output_dir: Path,
    image_features: Path | None = None,
    n_resamples: int = 5000,
    target_sensitivity: float = 0.95,
) -> dict[str, object]:
    """Write analysis tables and figures from untouched test predictions."""
    if not 0.0 < target_sensitivity <= 1.0:
        raise ValueError("target_sensitivity must lie in (0, 1]")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"{output_dir} is not empty; use a clean directory for one result set"
        )

    sets, validation_auc, validation_predictions = _load_run_artifacts(runs_dir)
    models = _validate_models(sets, validation_predictions)
    reference = sets[models[0]]
    keys = {
        (repetition, int(fold))
        for repetition in range(1, reference.n_repetitions + 1)
        for fold in np.unique(reference.frame[f"fold_{repetition}"])
    }

    selections: dict[str, ModelSelection] = {
        **{model: fixed_model_selection(keys, (model,)) for model in models},
        TOP1: select_models(validation_auc, k=1),
        TOP3: select_models(validation_auc, k=3),
        ALL6: fixed_model_selection(keys, models),
    }
    matrices = {
        strategy: build_selected_predictions(sets, selection)
        for strategy, selection in selections.items()
    }

    directories = {
        name: output_dir / name
        for name in (
            "performance",
            "operating_points",
            "calibration",
            "ensemble",
            "cohort",
        )
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    performance_dir = directories["performance"]
    operating_dir = directories["operating_points"]
    calibration_dir = directories["calibration"]
    ensemble_dir = directories["ensemble"]
    cohort_dir = directories["cohort"]

    _selection_rows(selections).to_csv(ensemble_dir / "ensemble_membership.csv", index=False)
    _selection_frequencies(selections).to_csv(
        ensemble_dir / "selection_frequencies.csv", index=False
    )
    _cohort_composition(reference.patient_id, reference.y_true).to_csv(
        cohort_dir / "cohort_composition.csv", index=False
    )

    thresholds: dict[str, pd.DataFrame] = {}
    for strategy, selection in selections.items():
        selected_validation = build_selected_validation_predictions(
            validation_predictions, selection
        )
        thresholds[strategy] = derive_validation_thresholds(selected_validation, target_sensitivity)
    pd.concat(
        [table.assign(Model=strategy) for strategy, table in thresholds.items()],
        ignore_index=True,
    ).to_csv(operating_dir / "validation_thresholds.csv", index=False)

    per_repetition = pd.concat(
        [
            evaluate_repetitions(reference.y_true, matrix, strategy)
            for strategy, matrix in matrices.items()
        ],
        ignore_index=True,
    )
    per_repetition.to_csv(performance_dir / "per_repetition_metrics.csv", index=False)

    summary_metrics = summarise(per_repetition)
    bootstrap = (
        pd.DataFrame(
            [
                {
                    "Model": strategy,
                    **bootstrap_auc_ci(
                        reference.y_true,
                        matrices[strategy],
                        reference.patient_id,
                        n_resamples,
                    ),
                }
                for strategy in matrices
            ]
        )
        .drop(columns="auc_mean")
        .rename(
            columns={
                "ci_lower": "AUC_patient_ci_lower",
                "ci_upper": "AUC_patient_ci_upper",
            }
        )
    )
    summary_metrics = summary_metrics.merge(bootstrap, on="Model")
    summary_metrics.to_csv(performance_dir / "summary_metrics.csv", index=False)

    high_sensitivity = {
        strategy: evaluate_at_fold_thresholds(
            reference,
            matrices[strategy],
            thresholds[strategy],
            "high_sensitivity_threshold",
            strategy,
        )
        for strategy in matrices
    }
    high_sensitivity_table = pd.concat(high_sensitivity.values(), ignore_index=True)
    high_sensitivity_table.to_csv(
        operating_dir / "operating_points_high_sensitivity.csv", index=False
    )
    high_sensitivity_summary = _summarise_operating_points(high_sensitivity_table)
    high_sensitivity_intervals = pd.DataFrame(
        [
            {
                "Model": strategy,
                **bootstrap_operating_point_ci(
                    reference.y_true,
                    matrices[strategy],
                    build_threshold_matrix(
                        reference,
                        thresholds[strategy],
                        "high_sensitivity_threshold",
                    ),
                    reference.patient_id,
                    n_resamples=n_resamples,
                ),
            }
            for strategy in matrices
        ]
    )
    high_sensitivity_summary.drop(
        columns=[
            "Sensitivity_mean",
            "Specificity_mean",
            "PPV_mean",
            "NPV_mean",
        ]
    ).merge(high_sensitivity_intervals, on="Model").to_csv(
        operating_dir / "operating_point_summary_high_sensitivity.csv", index=False
    )
    youden = {
        strategy: evaluate_at_fold_thresholds(
            reference,
            matrices[strategy],
            thresholds[strategy],
            "youden_threshold",
            strategy,
        )
        for strategy in matrices
    }
    youden_table = pd.concat(youden.values(), ignore_index=True)
    youden_table.to_csv(operating_dir / "operating_points_youden.csv", index=False)
    youden_summary = _summarise_operating_points(youden_table)
    youden_intervals = pd.DataFrame(
        [
            {
                "Model": strategy,
                **bootstrap_operating_point_ci(
                    reference.y_true,
                    matrices[strategy],
                    build_threshold_matrix(reference, thresholds[strategy], "youden_threshold"),
                    reference.patient_id,
                    n_resamples=n_resamples,
                ),
            }
            for strategy in matrices
        ]
    )
    youden_summary.drop(
        columns=[
            "Sensitivity_mean",
            "Specificity_mean",
            "PPV_mean",
            "NPV_mean",
        ]
    ).merge(youden_intervals, on="Model").to_csv(
        operating_dir / "operating_point_summary_youden.csv", index=False
    )

    primary_difference = bootstrap_auc_difference(
        reference.y_true,
        matrices[TOP3],
        matrices[TOP1],
        reference.patient_id,
        n_resamples,
    )
    pd.DataFrame([{"comparison": f"{TOP3} minus {TOP1}", **primary_difference}]).to_csv(
        performance_dir / "auc_difference_top3_vs_top1.csv", index=False
    )
    vit_difference = bootstrap_auc_difference(
        reference.y_true,
        matrices[TOP3],
        matrices["ViT"],
        reference.patient_id,
        n_resamples,
    )
    pd.DataFrame([{"comparison": f"{TOP3} minus ViT", **vit_difference}]).to_csv(
        performance_dir / "auc_difference_top3_vs_vit.csv", index=False
    )

    calibration = {
        strategy: calibration_table(reference.y_true, matrix, strategy)
        for strategy, matrix in matrices.items()
    }
    pd.concat(calibration.values(), ignore_index=True).to_csv(
        calibration_dir / "calibration.csv", index=False
    )

    plot_mean_roc(reference.y_true, matrices, performance_dir / "roc_curves.png")
    plot_auc_distribution(per_repetition, performance_dir / "auc_distribution.png")
    plot_calibration(
        {strategy: calibration[strategy] for strategy in ("ViT", TOP3, ALL6)},
        calibration_dir / "calibration.png",
    )
    plot_confusion_matrices(
        {strategy: high_sensitivity[strategy] for strategy in ("ViT", TOP3)},
        operating_dir / "confusion_matrices.png",
    )

    result: dict[str, object] = {
        "n_images": len(reference.y_true),
        "n_patients": int(np.unique(reference.patient_id).size),
        "models": list(matrices),
        "primary_comparison": {
            "contrast": f"{TOP3} minus {TOP1}",
            **{key: round(float(value), 4) for key, value in primary_difference.items()},
        },
        "secondary_comparison": {
            "contrast": f"{TOP3} minus ViT",
            **{key: round(float(value), 4) for key, value in vit_difference.items()},
        },
    }
    if image_features is not None:
        image_dir = output_dir / "image_characteristics"
        image_dir.mkdir(parents=True, exist_ok=True)
        result["image_characteristics"] = _image_feature_report(
            image_features, image_dir, reference, matrices[TOP3], n_resamples
        )
    return result
