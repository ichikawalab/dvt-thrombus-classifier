"""Inference on new images using the trained fold models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .data import UltrasoundDataset, build_transforms
from .model import LitThrombusClassifier, resolve_input_config

__all__ = ["load_fold_model", "predict_frame"]


def load_fold_model(
    checkpoint: str | Path, arch: str, config: TrainingConfig
) -> LitThrombusClassifier:
    """Restore one fold model from a weights-only checkpoint."""
    state = torch.load(Path(checkpoint), map_location="cpu")["state_dict"]
    model = LitThrombusClassifier(arch=arch, config=config, class_weights=None, pretrained=False)
    model.load_state_dict(state, strict=True)
    return model.eval()


@torch.no_grad()
def predict_frame(
    frame: pd.DataFrame,
    checkpoints: list[Path],
    arch: str,
    config: TrainingConfig,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Average thrombus-positive probabilities across fold models.

    Averaging over the fold models is an ensemble over models that saw different
    training subsets.  It is not a held-out estimate for images that were part of
    training, so this is meaningful only for genuinely new cases.

    ``threshold`` must be derived from validation data and never tuned on
    the images being predicted.
    """
    if not checkpoints:
        raise ValueError("at least one checkpoint is required")
    if threshold is not None and not 0 <= threshold <= 1:
        raise ValueError("threshold must lie in [0, 1]")

    input_size, mean, std = resolve_input_config(arch)
    transform = build_transforms(input_size, mean, std, augment=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        UltrasoundDataset(frame, transform, return_id=True),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    total = np.zeros(len(frame), dtype=float)
    image_ids: list[str] = []
    for position, checkpoint in enumerate(checkpoints):
        model = load_fold_model(checkpoint, arch, config).to(device)
        offset, batch_ids = 0, []
        for images, _labels, ids in loader:
            probability = torch.softmax(model(images.to(device)), dim=1)[:, 1].cpu().numpy()
            total[offset : offset + len(probability)] += probability
            offset += len(probability)
            batch_ids.extend(ids)
        if offset != len(frame):
            raise RuntimeError(f"{checkpoint}: prediction count does not match input rows")
        if position == 0:
            image_ids = batch_ids
        elif batch_ids != image_ids:
            raise RuntimeError(f"{checkpoint}: prediction row order changed across checkpoints")
        del model
        torch.cuda.empty_cache()

    result = pd.DataFrame(
        {
            "image_id": image_ids,
            "probability": total / len(checkpoints),
            "n_models": len(checkpoints),
        }
    )
    if threshold is not None:
        result["threshold"] = threshold
        result["predicted_label"] = (result["probability"] >= threshold).astype(int)
    return result
