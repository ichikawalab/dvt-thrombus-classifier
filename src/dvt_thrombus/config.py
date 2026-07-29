"""Configuration for the study training protocol."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

__all__ = ["ARCHITECTURES", "TrainingConfig", "load_config"]

# timm identifiers keyed by the architecture display name.
ARCHITECTURES: dict[str, str] = {
    "ResNet50": "resnet50.a1_in1k",
    "DenseNet121": "densenet121.ra_in1k",
    "InceptionV3": "inception_v3.tv_in1k",
    "ConvNeXtV2": "convnextv2_base.fcmae_ft_in22k_in1k_384",
    "ViT": "vit_base_patch16_384.augreg_in21k_ft_in1k",
    "SwinT": "swin_base_patch4_window12_384.ms_in22k_ft_in1k",
}


@dataclass
class TrainingConfig:
    """Hyperparameters and the cross-validation design."""

    arch: str = ARCHITECTURES["ViT"]
    pretrained: bool = True

    # Cross-validation.  Seeds are base_seed + repetition index and are shared
    # across architectures so that every model sees identical splits.
    n_repetitions: int = 10
    n_folds: int = 5
    base_seed: int = 42

    # Optimisation.
    learning_rate: float = 1e-5
    weight_decay: float = 1e-3
    batch_size: int = 16
    max_epochs: int = 100
    min_epochs: int = 10
    warmup_epochs: int = 5
    # min_epochs guarantees the schedule has left warm-up before this can fire.
    early_stopping_patience: int = 5

    # Classification head appended to the pretrained backbone.
    hidden_dim: int = 512
    dropout_rate: float = 0.5

    # Runtime.
    num_workers: int = 4
    # bf16 has the dynamic range of fp32 without loss scaling, so it is more
    # robust than fp16 on Ampere and later GPUs.  Lightning falls back to
    # 16-mixed where bf16 is unsupported, and to 32-true on CPU.  Use 32-true
    # when bit-level reproducibility matters; mixed precision is not
    # reproducible across GPU models.
    precision: str = "bf16-mixed"

    def __post_init__(self) -> None:
        if self.n_repetitions < 1:
            raise ValueError("n_repetitions must be positive")
        if self.n_folds < 3:
            raise ValueError("n_folds must be at least 3")
        if self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers non-negative")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if not 0 <= self.dropout_rate < 1:
            raise ValueError("dropout_rate must lie in [0, 1)")
        if not 0 <= self.warmup_epochs <= self.max_epochs:
            raise ValueError("warmup_epochs must lie in [0, max_epochs]")
        if not 1 <= self.min_epochs <= self.max_epochs:
            raise ValueError("min_epochs must lie in [1, max_epochs]")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative")

    def seed_for(self, repetition: int) -> int:
        """Seed for a 0-indexed repetition."""
        return self.base_seed + repetition


@dataclass
class Config:
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _filter_known(section: dict[str, Any], cls: type) -> dict[str, Any]:
    known = {f.name for f in fields(cls)}
    unknown = set(section) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} option(s): {sorted(unknown)}")
    return section


def load_config(path: str | Path) -> Config:
    """Read a YAML configuration file."""
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    training = TrainingConfig(**_filter_known(raw.get("training", {}), TrainingConfig))
    unknown_sections = set(raw) - {"training"}
    if unknown_sections:
        raise ValueError(f"unknown configuration section(s): {sorted(unknown_sections)}")
    return Config(training=training)
