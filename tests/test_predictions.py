"""Validation of prediction tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dvt_thrombus.predictions import load_prediction_set


def _table(n: int = 6, n_reps: int = 2) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "image_id": [f"img_{i:03d}" for i in range(n)],
            "patient_id": [f"pt_{i // 2:03d}" for i in range(n)],
            "true_label": [0, 1] * (n // 2),
        }
    )
    for r in range(1, n_reps + 1):
        frame[f"fold_{r}"] = [1, 2] * (n // 2)
        frame[f"trial_{r}"] = [0.1 * (i + 1) for i in range(n)]
    return frame


def test_trial_columns_sort_numerically(tmp_path: Path):
    frame = _table(n_reps=11)
    path = tmp_path / "M.csv"
    frame.to_csv(path, index=False)
    columns = load_prediction_set(path).trial_columns
    assert columns[-1] == "trial_11", "trial_10 must not sort before trial_2"
    assert load_prediction_set(path).n_repetitions == 11


def test_rejects_probability_out_of_range(tmp_path: Path):
    frame = _table()
    frame.loc[0, "trial_1"] = 1.5
    frame.to_csv(tmp_path / "M.csv", index=False)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        load_prediction_set(tmp_path / "M.csv")


def test_rejects_missing_probability(tmp_path: Path):
    frame = _table()
    frame.loc[0, "trial_1"] = None
    frame.to_csv(tmp_path / "M.csv", index=False)
    with pytest.raises(ValueError, match="finite"):
        load_prediction_set(tmp_path / "M.csv")


def test_rejects_missing_fold_column(tmp_path: Path):
    frame = _table()
    frame = frame.drop(columns="fold_2")
    frame.to_csv(tmp_path / "M.csv", index=False)
    with pytest.raises(ValueError, match=r"trial_\* and fold_\*"):
        load_prediction_set(tmp_path / "M.csv")


def test_rejects_nonconsecutive_repetition_numbers(tmp_path: Path):
    frame = _table().rename(columns={"trial_1": "trial_3", "fold_1": "fold_3"})
    frame.to_csv(tmp_path / "M.csv", index=False)
    with pytest.raises(ValueError, match="consecutive"):
        load_prediction_set(tmp_path / "M.csv")


def test_rejects_duplicate_image_id(tmp_path: Path):
    frame = _table()
    frame.loc[1, "image_id"] = frame.loc[0, "image_id"]
    frame.to_csv(tmp_path / "M.csv", index=False)
    with pytest.raises(ValueError, match="unique"):
        load_prediction_set(tmp_path / "M.csv")
