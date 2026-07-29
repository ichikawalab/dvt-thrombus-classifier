"""Backbone, classification head and the Lightning training module."""

from __future__ import annotations

import math
from dataclasses import asdict

import pytorch_lightning as pl
import timm
import torch
from timm.data import resolve_data_config
from torch import nn

from .config import TrainingConfig

__all__ = ["ThrombusClassifier", "LitThrombusClassifier", "resolve_input_config"]


def resolve_input_config(arch: str) -> tuple[tuple[int, int], tuple[float, ...], tuple[float, ...]]:
    """Return ``(input_size, mean, std)`` expected by a timm architecture."""
    model = timm.create_model(arch, pretrained=False)
    config = resolve_data_config({}, model=model)
    del model
    return tuple(config["input_size"][1:]), tuple(config["mean"]), tuple(config["std"])


class ThrombusClassifier(nn.Module):
    """ImageNet-pretrained backbone with a two-class head.

    The whole network is fine-tuned; the backbone is not frozen.
    """

    def __init__(
        self,
        arch: str,
        hidden_dim: int = 512,
        dropout_rate: float = 0.5,
        num_classes: int = 2,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.backbone.num_features),
            nn.Linear(self.backbone.num_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(images))


class LitThrombusClassifier(pl.LightningModule):
    """Training loop: class-weighted cross-entropy, AdamW, warm-up then cosine decay."""

    def __init__(
        self,
        arch: str,
        config: TrainingConfig,
        class_weights: list[float] | None = None,
        pretrained: bool | None = None,
    ) -> None:
        super().__init__()
        # Only primitives are recorded, so a checkpoint unpickles under
        # torch.load's default weights_only=True and loading someone else's
        # checkpoint cannot execute arbitrary code.
        self.save_hyperparameters({"arch": arch, **asdict(config)})
        self.config = config
        self.model = ThrombusClassifier(
            arch=arch,
            hidden_dim=config.hidden_dim,
            dropout_rate=config.dropout_rate,
            pretrained=config.pretrained if pretrained is None else pretrained,
        )
        # Registered non-persistently so the class weights stay out of the
        # checkpoint: they are a property of one training fold, not of the
        # model, and keeping them out lets a checkpoint be reloaded with
        # strict=True without knowing which fold produced it.
        self.register_buffer(
            "class_weights",
            torch.tensor(class_weights, dtype=torch.float) if class_weights else None,
            persistent=False,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def _step(self, batch, stage: str) -> torch.Tensor:
        images, labels = batch[0], batch[1]
        loss = nn.functional.cross_entropy(self(images), labels, weight=self.class_weights)
        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=(stage == "val"),
            batch_size=images.size(0),
        )
        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        warmup = self.config.warmup_epochs
        total = self.config.max_epochs

        def factor(epoch: int) -> float:
            if epoch < warmup:
                return 0.1 + 0.9 * epoch / warmup
            progress = (epoch - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
