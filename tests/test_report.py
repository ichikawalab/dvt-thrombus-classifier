"""End-to-end test for the statistical report."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dvt_thrombus.config import ARCHITECTURES
from dvt_thrombus.report import ALL6, TOP1, TOP3, run_report
from dvt_thrombus.splitting import iter_splits


def test_report_uses_complete_fold_specific_artifacts(tmp_path):
    runs = tmp_path / "runs"
    labels = np.tile([0, 1], 10)
    patient_ids = np.array([f"patient_{index // 2}" for index in range(len(labels))])
    fold_matrix = np.full((len(labels), 10), -1)
    for split in iter_splits(labels, patient_ids, 10, 5, 42):
        fold_matrix[split.test, split.repetition] = split.fold + 1

    for model_index, model in enumerate(ARCHITECTURES):
        run = runs / model
        run.mkdir(parents=True)
        prediction = pd.DataFrame(
            {
                "image_id": [f"image_{index}" for index in range(len(labels))],
                "patient_id": patient_ids,
                "true_label": labels,
            }
        )
        for repetition in range(1, 11):
            prediction[f"fold_{repetition}"] = fold_matrix[:, repetition - 1]
            signal = 0.2 + 0.6 * labels
            prediction[f"trial_{repetition}"] = np.clip(
                signal + 0.005 * model_index + 0.002 * repetition, 0, 1
            )
        prediction.to_csv(run / "predictions.csv", index=False)

        validation_rows = []
        for repetition in range(1, 11):
            for fold in range(1, 6):
                for index, label in enumerate((0, 1, 0, 1)):
                    validation_rows.append(
                        {
                            "image_id": f"validation_{repetition}_{fold}_{index}",
                            "patient_id": f"validation_patient_{index}",
                            "repetition": repetition,
                            "fold": fold,
                            "true_label": label,
                            "probability": 0.2 + 0.6 * label + 0.005 * model_index,
                        }
                    )
        pd.DataFrame(validation_rows).to_csv(run / "validation_predictions.csv", index=False)

    image_features = pd.DataFrame(
        {
            "image_id": [f"image_{index}" for index in range(len(labels))],
            "patient_id": patient_ids,
            "label": labels,
            "mean_intensity": 50 + labels * (8 + np.arange(len(labels)) / 10),
            "sd_intensity": 30 + labels,
            "width_px": np.tile([640, 800], len(labels) // 2),
            "height_px": np.tile([480, 600], len(labels) // 2),
        }
    )
    image_features_path = tmp_path / "image_features.csv"
    image_features.to_csv(image_features_path, index=False)

    output = tmp_path / "report"
    summary = run_report(runs, output, image_features=image_features_path, n_resamples=30)

    assert {TOP1, TOP3, ALL6}.issubset(summary["models"])
    expected = {
        "performance/summary_metrics.csv",
        "performance/auc_difference_top3_vs_top1.csv",
        "performance/auc_difference_top3_vs_vit.csv",
        "image_characteristics/auc_difference_top3_vs_image_features.csv",
        "operating_points/operating_points_high_sensitivity.csv",
        "operating_points/operating_points_youden.csv",
        "operating_points/validation_thresholds.csv",
        "ensemble/ensemble_membership.csv",
        "ensemble/selection_frequencies.csv",
        "calibration/calibration.png",
        "operating_points/confusion_matrices.png",
        "image_characteristics/image_feature_baseline_predictions.csv",
        "image_characteristics/source_image_dimensions.csv",
        "image_characteristics/within_patient_intensity_difference.csv",
    }
    written = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    assert expected.issubset(written)
    assert not any(path.is_file() for path in output.iterdir())
    assert summary["secondary_comparison"]["contrast"] == f"{TOP3} minus ViT"
    operating = pd.read_csv(
        output / "operating_points" / "operating_point_summary_high_sensitivity.csv"
    )
    assert {
        "Sensitivity_ci_lower",
        "Sensitivity_ci_upper",
        "Specificity_ci_lower",
        "Specificity_ci_upper",
    }.issubset(operating.columns)
