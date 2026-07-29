"""Manifest validation and the properties augmentation must satisfy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from dvt_thrombus.data import UltrasoundDataset, build_transforms, load_input_table  # noqa: E402

INPUT_SIZE = (64, 64)
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


@pytest.fixture
def cohort(tmp_path: Path) -> tuple[Path, Path]:
    """Two images whose top half is bright and bottom half dark."""
    gradient = np.zeros((64, 64), dtype=np.uint8)
    gradient[:32, :] = 220
    gradient[32:, :] = 40

    rows = []
    for index in range(2):
        name = f"img_{index}.png"
        Image.fromarray(gradient, mode="L").convert("RGB").save(tmp_path / name)
        rows.append({"image_path": name, "label": index, "patient_id": f"pt_{index}"})

    csv_path = tmp_path / "input.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path, tmp_path


def _dataset(cohort, augment: bool) -> UltrasoundDataset:
    csv_path, root = cohort
    frame = load_input_table(csv_path, root)
    return UltrasoundDataset(frame, build_transforms(INPUT_SIZE, MEAN, STD, augment=augment))


def test_image_id_defaults_to_the_path(cohort):
    csv_path, root = cohort
    frame = load_input_table(csv_path, root)
    assert (frame["image_id"] == frame["image_path"]).all()


def test_explicit_image_id_is_used(cohort, tmp_path):
    csv_path, root = cohort
    frame = pd.read_csv(csv_path)
    frame["image_id"] = ["a", "b"]
    frame.to_csv(csv_path, index=False)

    loaded = load_input_table(csv_path, root)
    dataset = UltrasoundDataset(
        loaded, build_transforms(INPUT_SIZE, MEAN, STD, augment=False), return_id=True
    )
    assert dataset[0][2] == "a"


def test_duplicate_image_id_is_rejected(cohort):
    csv_path, root = cohort
    frame = pd.read_csv(csv_path)
    frame["image_id"] = ["same", "same"]
    frame.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="image_id must be unique"):
        load_input_table(csv_path, root)


def test_missing_image_fails_before_training_starts(cohort):
    csv_path, root = cohort
    frame = pd.read_csv(csv_path)
    frame.loc[0, "image_path"] = "absent.png"
    frame.to_csv(csv_path, index=False)
    with pytest.raises(FileNotFoundError):
        load_input_table(csv_path, root)


def test_evaluation_transform_is_deterministic(cohort):
    dataset = _dataset(cohort, augment=False)
    torch.testing.assert_close(dataset[0][0], dataset[0][0])


def test_augmentation_is_stochastic(cohort):
    dataset = _dataset(cohort, augment=True)
    torch.manual_seed(0)
    draws = [dataset[0][0] for _ in range(5)]
    assert any(not torch.allclose(draws[0], other) for other in draws[1:])


def test_augmentation_preserves_shape_and_range(cohort):
    dataset = _dataset(cohort, augment=True)
    torch.manual_seed(0)
    image = dataset[0][0]
    assert image.shape == (3, *INPUT_SIZE)
    assert torch.isfinite(image).all()


def test_augmentation_never_flips_vertically(cohort):
    """Depth is physically meaningful in B-mode: the near field stays on top.

    The source image is bright on top and dark below, so a vertical flip would
    invert that ordering.  Rotation and translation can erode the margin, so the
    check allows for that but not for a reversal.
    """
    dataset = _dataset(cohort, augment=True)
    torch.manual_seed(0)
    for _ in range(30):
        image = dataset[0][0]
        top = image[:, : INPUT_SIZE[0] // 2, :].mean()
        bottom = image[:, INPUT_SIZE[0] // 2 :, :].mean()
        assert top > bottom, "the bright half moved to the bottom: a vertical flip occurred"
