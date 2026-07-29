"""Tests for strict configuration validation."""

from __future__ import annotations

import pytest

from dvt_thrombus.config import TrainingConfig


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_repetitions", 0),
        ("n_folds", 2),
        ("batch_size", 0),
        ("num_workers", -1),
        ("learning_rate", 0),
        ("weight_decay", -1),
        ("dropout_rate", 1),
        ("warmup_epochs", 101),
        ("min_epochs", 0),
        ("min_epochs", 101),
        ("early_stopping_patience", -1),
    ],
)
def test_invalid_training_options_are_rejected(field, value):
    with pytest.raises(ValueError):
        TrainingConfig(**{field: value})
