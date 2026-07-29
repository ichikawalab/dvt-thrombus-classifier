"""Dataset definition, input validation and image transforms."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

__all__ = [
    "REQUIRED_COLUMNS",
    "UltrasoundDataset",
    "build_transforms",
    "load_input_table",
]

REQUIRED_COLUMNS = ("image_path", "label", "patient_id")


def load_input_table(csv_path: str | Path, data_root: str | Path) -> pd.DataFrame:
    """Read and validate the training manifest.

    Adds an absolute ``resolved_path`` column and fails immediately if any image
    is missing, rather than part-way through a multi-hour run.

    An optional ``image_id`` column labels rows in every output file. Use opaque
    identifiers when paths contain sensitive information. It defaults to
    ``image_path``.
    """
    csv_path, data_root = Path(csv_path), Path(data_root)
    frame = pd.read_csv(
        csv_path,
        dtype={"image_path": str, "patient_id": str, "image_id": str},
    )

    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing required column(s) {missing}")
    if frame[list(REQUIRED_COLUMNS)].isna().to_numpy().any():
        raise ValueError(f"{csv_path}: required columns must not contain missing values")
    if not frame["label"].isin((0, 1)).all():
        raise ValueError(f"{csv_path}: label must be 0 (negative) or 1 (positive)")

    frame = frame.copy()
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["resolved_path"] = [str(data_root / p) for p in frame["image_path"]]

    if "image_id" not in frame.columns:
        frame["image_id"] = frame["image_path"]
    frame["image_id"] = frame["image_id"].astype(str)
    if frame["image_id"].duplicated().any():
        raise ValueError(f"{csv_path}: image_id must be unique")

    absent = [p for p in frame["resolved_path"] if not Path(p).exists()]
    if absent:
        raise FileNotFoundError(
            f"{len(absent)} image(s) listed in {csv_path} do not exist, e.g. {absent[0]}"
        )
    return frame.reset_index(drop=True)


def build_transforms(
    input_size: tuple[int, int],
    mean: tuple[float, ...],
    std: tuple[float, ...],
    augment: bool,
) -> Callable:
    """Build evaluation transforms or mild anatomically plausible augmentation.

    Training uses horizontal flips, affine variation, and brightness/contrast
    jitter. Vertical flips and elastic deformation are excluded because they can
    violate ultrasound depth orientation or vessel geometry.
    """
    if not augment:
        return transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


class UltrasoundDataset(Dataset):
    """B-mode images with binary thrombus labels.

    Images are converted to RGB because the backbones are ImageNet-pretrained
    and expect three channels; the three channels carry identical grayscale
    values.
    """

    def __init__(
        self, frame: pd.DataFrame, transform: Callable, *, return_id: bool = False
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.return_id = return_id

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(row["resolved_path"]) as handle:
            image = self.transform(handle.convert("RGB"))
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        if self.return_id:
            return image, label, str(row["image_id"])
        return image, label
