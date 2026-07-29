"""Tests for the cross-validation design.

These guard the two properties the study depends on: no patient may span a
split, and the splits must be identical for every architecture.
"""

from __future__ import annotations

import numpy as np
import pytest

from dvt_thrombus.splitting import Split, assert_no_patient_leakage, iter_splits


@pytest.fixture
def cohort():
    rng = np.random.default_rng(0)
    patients = np.repeat([f"pt_{i:02d}" for i in range(40)], 4)
    # Label is a patient-level property for half the patients, mixed for the rest.
    labels = rng.integers(0, 2, size=len(patients))
    return labels, patients


def test_no_patient_spans_two_subsets(cohort):
    labels, patients = cohort
    for split in iter_splits(labels, patients, n_repetitions=3, n_folds=5, base_seed=42):
        assert_no_patient_leakage(patients, split)


def test_every_image_is_tested_exactly_once_per_repetition(cohort):
    labels, patients = cohort
    per_repetition: dict[int, list[np.ndarray]] = {}
    for split in iter_splits(labels, patients, n_repetitions=3, n_folds=5, base_seed=42):
        per_repetition.setdefault(split.repetition, []).append(split.test)

    for repetition, folds in per_repetition.items():
        combined = np.concatenate(folds)
        assert len(combined) == len(labels), f"repetition {repetition} does not cover the cohort"
        assert len(np.unique(combined)) == len(labels), "an image was tested twice"


def test_subsets_are_disjoint_and_exhaustive(cohort):
    labels, patients = cohort
    for split in iter_splits(labels, patients, n_repetitions=2, n_folds=5, base_seed=42):
        combined = np.concatenate([split.train, split.validation, split.test])
        assert len(np.unique(combined)) == len(labels)


def test_splits_depend_only_on_the_seed(cohort):
    """Re-running for another architecture must give byte-identical splits."""
    labels, patients = cohort
    first = list(iter_splits(labels, patients, 2, 5, 42))
    second = list(iter_splits(labels, patients, 2, 5, 42))
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a.train, b.train)
        np.testing.assert_array_equal(a.validation, b.validation)
        np.testing.assert_array_equal(a.test, b.test)


def test_a_different_base_seed_gives_different_splits(cohort):
    labels, patients = cohort
    a = next(iter_splits(labels, patients, 1, 5, 42))
    b = next(iter_splits(labels, patients, 1, 5, 99))
    assert not np.array_equal(a.test, b.test)


def test_leakage_is_detected():
    groups = np.array(["p1", "p1", "p2", "p2"])
    leaking = Split(
        repetition=0,
        fold=0,
        seed=42,
        train=np.array([0]),
        validation=np.array([1]),  # same patient as train
        test=np.array([2, 3]),
    )
    with pytest.raises(RuntimeError, match="appear in both"):
        assert_no_patient_leakage(groups, leaking)


def test_too_few_patients_for_the_requested_folds_is_rejected():
    labels = np.array([0, 1, 0, 1])
    patients = np.array(["a", "b", "c", "d"])
    with pytest.raises(ValueError, match="number of patients"):
        list(iter_splits(labels, patients, n_repetitions=1, n_folds=5, base_seed=42))
