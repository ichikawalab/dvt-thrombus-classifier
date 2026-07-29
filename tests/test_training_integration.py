"""End-to-end training smoke test on synthetic images.

Skipped unless the ``train`` extra is installed.  Uses a tiny backbone, two
repetitions and two epochs: this checks that the pipeline runs and writes the
expected artefacts, not that it learns anything.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("pytorch_lightning")

from PIL import Image  # noqa: E402

from dvt_thrombus.config import TrainingConfig  # noqa: E402
from dvt_thrombus.data import load_input_table  # noqa: E402
from dvt_thrombus.training import cross_validate  # noqa: E402

N_PATIENTS = 12
IMAGES_PER_PATIENT = 2


@pytest.fixture(scope="module")
def synthetic_cohort(tmp_path_factory) -> tuple[Path, Path]:
    """Bright squares for positives, dark for negatives."""
    root = tmp_path_factory.mktemp("cohort")
    rng = np.random.default_rng(0)
    rows = []
    for patient in range(N_PATIENTS):
        label = patient % 2
        for index in range(IMAGES_PER_PATIENT):
            name = f"pt{patient:02d}_{index}.png"
            intensity = 170 if label else 80
            pixels = np.clip(rng.normal(intensity, 15, (64, 64)), 0, 255).astype(np.uint8)
            Image.fromarray(pixels, mode="L").convert("RGB").save(root / name)
            rows.append({"image_path": name, "label": label, "patient_id": f"pt_{patient:02d}"})

    csv_path = root / "input.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path, root


def test_cross_validation_writes_expected_outputs(synthetic_cohort, tmp_path):
    csv_path, data_root = synthetic_cohort
    frame = load_input_table(csv_path, data_root)

    config = TrainingConfig(
        arch="resnet18.a1_in1k",
        pretrained=False,
        n_repetitions=2,
        n_folds=3,
        max_epochs=2,
        min_epochs=1,
        warmup_epochs=1,
        early_stopping_patience=1,
        batch_size=4,
        num_workers=0,
        precision="32-true",
    )
    output = tmp_path / "run"
    predictions = cross_validate(frame=frame, arch=config.arch, config=config, output_dir=output)

    for name in (
        "predictions.csv",
        "validation_predictions.csv",
        "run_metadata.json",
    ):
        assert (output / name).exists(), f"{name} was not written"

    # Every image receives exactly one held-out prediction per repetition.
    assert len(predictions) == N_PATIENTS * IMAGES_PER_PATIENT
    for repetition in (1, 2):
        probabilities = predictions[f"trial_{repetition}"]
        assert probabilities.notna().all()
        assert probabilities.between(0, 1).all()
        assert set(predictions[f"fold_{repetition}"]) == {1, 2, 3}

    # The threshold must come from validation data, so no test image may appear
    # in the validation predictions of the fold that held it out.
    validation = pd.read_csv(output / "validation_predictions.csv")
    assert validation["patient_id"].notna().all()
    for repetition in (1, 2):
        for fold in (1, 2, 3):
            held_out = set(predictions.loc[predictions[f"fold_{repetition}"] == fold, "image_id"])
            validated = set(
                validation.loc[
                    (validation["repetition"] == repetition) & (validation["fold"] == fold),
                    "image_id",
                ]
            )
            assert not (held_out & validated), "threshold was fitted on test images"

    # Exactly one checkpoint is written for each fold.
    assert len(list((output / "checkpoints").glob("*.ckpt"))) == 6


def test_prediction_format_matches_training_output(synthetic_cohort, tmp_path):
    """Training output must load through the prediction-table reader."""
    from dvt_thrombus.predictions import load_prediction_set

    csv_path, data_root = synthetic_cohort
    frame = load_input_table(csv_path, data_root)
    config = TrainingConfig(
        arch="resnet18.a1_in1k",
        pretrained=False,
        n_repetitions=2,
        n_folds=3,
        max_epochs=1,
        min_epochs=1,
        warmup_epochs=1,
        batch_size=4,
        # Worker start-up dominates on these tiny datasets, and fp32 keeps the
        # test independent of GPU capability.
        num_workers=0,
        precision="32-true",
    )
    output = tmp_path / "format"
    cross_validate(frame, config.arch, config, output)

    prediction_set = load_prediction_set(output / "predictions.csv", model="synthetic")
    assert prediction_set.n_repetitions == 2
    assert prediction_set.probability_matrix().shape == (len(frame), 2)
