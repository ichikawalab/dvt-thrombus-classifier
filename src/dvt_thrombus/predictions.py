"""Loading and validation of held-out prediction tables.

A prediction table holds, for one architecture, the held-out probability
assigned to every image in every repetition of the cross-validation:

===========  ==========  ==========  ========  =======  ========
image_id     patient_id  true_label  fold_1    trial_1  ...
===========  ==========  ==========  ========  =======  ========
example_001  example_01  0           4         0.0252   ...
===========  ==========  ==========  ========  =======  ========

``fold_i`` records which of the five outer folds held the image out during
repetition *i*, and ``trial_i`` is the probability the model assigned to the
thrombus-positive class on that held-out prediction.  Because the folds
partition the cohort, each repetition contributes exactly one prediction per
image, and the metrics for a repetition are computed on all images pooled
across the five folds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["PredictionSet", "load_prediction_set"]

_TRIAL_RE = re.compile(r"^trial_(\d+)$")
_FOLD_RE = re.compile(r"^fold_(\d+)$")

REQUIRED_COLUMNS = ("image_id", "patient_id", "true_label")


@dataclass(frozen=True)
class PredictionSet:
    """Held-out probabilities for one architecture."""

    model: str
    frame: pd.DataFrame

    @property
    def trial_columns(self) -> list[str]:
        columns = [c for c in self.frame.columns if _TRIAL_RE.match(c)]
        return sorted(columns, key=lambda c: int(_TRIAL_RE.match(c).group(1)))

    @property
    def fold_columns(self) -> list[str]:
        columns = [c for c in self.frame.columns if _FOLD_RE.match(c)]
        return sorted(columns, key=lambda c: int(_FOLD_RE.match(c).group(1)))

    @property
    def n_repetitions(self) -> int:
        return len(self.trial_columns)

    @property
    def y_true(self) -> np.ndarray:
        return self.frame["true_label"].to_numpy(dtype=int)

    @property
    def patient_id(self) -> np.ndarray:
        return self.frame["patient_id"].to_numpy(dtype=str)

    @property
    def image_id(self) -> np.ndarray:
        return self.frame["image_id"].to_numpy(dtype=str)

    def probabilities(self, repetition: int) -> np.ndarray:
        """Probabilities for a 1-indexed repetition."""
        return self.frame[f"trial_{repetition}"].to_numpy(dtype=float)

    def probability_matrix(self) -> np.ndarray:
        """``(n_images, n_repetitions)`` matrix of held-out probabilities."""
        return self.frame[self.trial_columns].to_numpy(dtype=float)


def _validate(frame: pd.DataFrame, source: Path) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{source}: missing required column(s) {missing}")
    if not frame["true_label"].isin((0, 1)).all():
        raise ValueError(f"{source}: true_label must contain only 0 and 1")
    if frame["image_id"].duplicated().any():
        raise ValueError(f"{source}: image_id must be unique")
    if frame[list(REQUIRED_COLUMNS)].isna().to_numpy().any():
        raise ValueError(f"{source}: required columns must not contain missing values")

    trial_columns = [c for c in frame.columns if _TRIAL_RE.match(c)]
    fold_columns = [c for c in frame.columns if _FOLD_RE.match(c)]
    if not trial_columns:
        raise ValueError(f"{source}: no trial_* columns found")
    trial_numbers = {int(_TRIAL_RE.match(column).group(1)) for column in trial_columns}
    fold_numbers = {int(_FOLD_RE.match(column).group(1)) for column in fold_columns}
    if trial_numbers != fold_numbers:
        raise ValueError(f"{source}: trial_* and fold_* repetitions must match")
    expected_repetitions = set(range(1, len(trial_numbers) + 1))
    if trial_numbers != expected_repetitions:
        raise ValueError(f"{source}: repetition numbers must be consecutive from 1")
    probabilities = frame[trial_columns].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{source}: predicted probabilities must be finite")
    if probabilities.min() < 0.0 or probabilities.max() > 1.0:
        raise ValueError(f"{source}: predicted probabilities must lie in [0, 1]")
    folds = frame[fold_columns].to_numpy(dtype=float)
    if not np.isfinite(folds).all():
        raise ValueError(f"{source}: fold assignments must be positive integers")
    integer_folds = np.equal(folds, folds.astype(int)).all()
    if (folds < 1).any() or not integer_folds:
        raise ValueError(f"{source}: fold assignments must be positive integers")


def load_prediction_set(path: str | Path, model: str | None = None) -> PredictionSet:
    """Read and validate a single prediction table."""
    path = Path(path)
    frame = pd.read_csv(path, dtype={"image_id": str, "patient_id": str})
    _validate(frame, path)
    return PredictionSet(model=model or path.stem, frame=frame)
