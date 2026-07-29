"""Repeated patient-level, class-stratified cross-validation splits."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

__all__ = ["Split", "assert_no_patient_leakage", "iter_splits"]


@dataclass(frozen=True)
class Split:
    """Index arrays for one outer fold of one repetition."""

    repetition: int  # 0-indexed
    fold: int  # 0-indexed
    seed: int
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def assert_no_patient_leakage(groups: np.ndarray, split: Split) -> None:
    """Raise if any patient appears in more than one of the three subsets."""
    subsets = {
        "train": set(groups[split.train]),
        "validation": set(groups[split.validation]),
        "test": set(groups[split.test]),
    }
    names = list(subsets)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            shared = subsets[first] & subsets[second]
            if shared:
                raise RuntimeError(
                    f"repetition {split.repetition + 1} fold {split.fold + 1}: "
                    f"patient(s) {sorted(shared)} appear in both {first} and {second}"
                )


def iter_splits(
    labels: np.ndarray,
    groups: np.ndarray,
    n_repetitions: int,
    n_folds: int,
    base_seed: int,
) -> Iterator[Split]:
    """Yield every (repetition, fold) split.

    Each repetition partitions patients into ``n_folds`` outer folds; one is held
    out for testing.  The remaining folds are split once more, again by patient,
    to carve out a validation set used for early stopping, checkpoint selection,
    decision-threshold selection and ensemble-member selection.  With five folds
    this gives roughly 3:1:1 train:validation:test.

    Seeds are ``base_seed + repetition`` and depend on nothing else, so every
    architecture receives identical splits and comparisons remain paired.
    """
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    if labels.ndim != 1 or groups.ndim != 1 or len(labels) != len(groups):
        raise ValueError("labels and groups must be aligned one-dimensional arrays")
    if n_repetitions < 1 or n_folds < 3:
        raise ValueError("n_repetitions must be positive and n_folds at least 3")
    if np.unique(groups).size < n_folds:
        raise ValueError("the number of patients must be at least n_folds")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("labels must contain both binary classes")
    n_samples = len(labels)

    for repetition in range(n_repetitions):
        seed = base_seed + repetition
        outer = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        for fold, (development, test) in enumerate(
            outer.split(np.zeros(n_samples), labels, groups)
        ):
            inner = StratifiedGroupKFold(n_splits=n_folds - 1, shuffle=True, random_state=seed)
            train_local, validation_local = next(
                inner.split(np.zeros(len(development)), labels[development], groups[development])
            )
            split = Split(
                repetition=repetition,
                fold=fold,
                seed=seed,
                train=development[train_local],
                validation=development[validation_local],
                test=test,
            )
            assert_no_patient_leakage(groups, split)
            for name, indices in (
                ("train", split.train),
                ("validation", split.validation),
                ("test", split.test),
            ):
                if np.unique(labels[indices]).size != 2:
                    raise ValueError(
                        f"repetition {repetition + 1} fold {fold + 1}: "
                        f"{name} subset does not contain both classes"
                    )
            yield split
