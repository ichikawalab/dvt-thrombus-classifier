"""Patient-separated analysis of global image characteristics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import roc_auc
from .splitting import iter_splits

__all__ = [
    "FEATURE_COLUMNS",
    "baseline_predictions",
    "paired_within_patient",
    "summarise_image_dimensions",
    "univariate_separability",
]

FEATURE_COLUMNS = ("mean_intensity", "sd_intensity")
DIMENSION_COLUMNS = ("width_px", "height_px")


def _require(frame: pd.DataFrame, columns) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"missing column(s) {missing}")


def univariate_separability(features: pd.DataFrame) -> pd.DataFrame:
    """Positive-class AUC of each feature on its own."""
    _require(features, ("label", *FEATURE_COLUMNS))
    labels = features["label"].to_numpy(dtype=int)
    rows = []
    for column in FEATURE_COLUMNS:
        values = features[column].to_numpy(dtype=float)
        auc = roc_auc(labels, values)
        rows.append({"feature": column, "auc": auc})
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


def summarise_image_dimensions(features: pd.DataFrame) -> pd.DataFrame:
    """Describe source-image dimensions without using them as predictors."""
    _require(features, DIMENSION_COLUMNS)
    rows = []
    for column in DIMENSION_COLUMNS:
        values = features[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"{column} must contain positive finite values")
        rows.append(
            {
                "dimension": column,
                "n_images": len(values),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "median": float(np.median(values)),
                "q1": float(np.quantile(values, 0.25)),
                "q3": float(np.quantile(values, 0.75)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def baseline_predictions(
    features: pd.DataFrame,
    columns=FEATURE_COLUMNS,
    n_folds: int = 5,
    n_repetitions: int = 10,
    base_seed: int = 42,
) -> pd.DataFrame:
    """Return repeated out-of-fold probabilities from a logistic regression.

    The outer test folds are identical to those used for the deep models.
    Logistic regression has no model-selection step, so each fit uses all
    non-test patients.
    """
    _require(features, ("image_id", "label", "patient_id", *columns))
    if features["image_id"].duplicated().any():
        raise ValueError("image_id must be unique")
    labels = features["label"].to_numpy(dtype=int)
    groups = features["patient_id"].to_numpy(dtype=str)
    design = features[list(columns)].to_numpy(dtype=float)
    if not np.isfinite(design).all():
        raise ValueError("image features must be finite")

    result = features[["image_id", "patient_id", "label"]].copy()
    probabilities = np.full((len(features), n_repetitions), np.nan)
    folds = np.full((len(features), n_repetitions), -1, dtype=int)
    for split in iter_splits(labels, groups, n_repetitions, n_folds, base_seed):
        development = np.concatenate((split.train, split.validation))
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(design[development], labels[development])
        probabilities[split.test, split.repetition] = model.predict_proba(design[split.test])[:, 1]
        folds[split.test, split.repetition] = split.fold + 1

    if np.isnan(probabilities).any() or (folds < 1).any():
        raise RuntimeError("out-of-fold predictions are incomplete")
    for repetition in range(n_repetitions):
        result[f"fold_{repetition + 1}"] = folds[:, repetition]
        result[f"trial_{repetition + 1}"] = probabilities[:, repetition]
    return result


def paired_within_patient(
    features: pd.DataFrame,
    column: str = "mean_intensity",
    n_resamples: int = 5000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Paired comparison of ``column`` between classes, within patients.

    Only patients contributing both a positive and a negative image are used.
    The confidence interval resamples patients.
    """
    _require(features, ("patient_id", "label", column))
    if n_resamples < 1 or not 0 < confidence < 1:
        raise ValueError("n_resamples must be positive and confidence lie in (0, 1)")
    if not np.isfinite(features[column].to_numpy(dtype=float)).all():
        raise ValueError(f"{column} must be finite")
    positive, negative = [], []
    for _, subset in features.groupby("patient_id"):
        if subset["label"].nunique() < 2:
            continue
        positive.append(subset.loc[subset["label"] == 1, column].mean())
        negative.append(subset.loc[subset["label"] == 0, column].mean())

    if len(positive) < 2:
        raise ValueError("fewer than two patients contribute both classes")

    positive_arr, negative_arr = np.asarray(positive), np.asarray(negative)
    difference = positive_arr - negative_arr
    rng = np.random.default_rng(seed)
    means = np.mean(
        rng.choice(difference, size=(n_resamples, len(difference)), replace=True),
        axis=1,
    )
    alpha = (1.0 - confidence) / 2.0
    return {
        "column": column,
        "n_patients": len(difference),
        "mean_difference": float(difference.mean()),
        "ci_lower": float(np.quantile(means, alpha)),
        "ci_upper": float(np.quantile(means, 1.0 - alpha)),
        "n_positive_higher": int((difference > 0).sum()),
        "n_resamples": n_resamples,
    }
