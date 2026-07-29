"""Case-level Grad-CAM visualisation.

Maps target the thrombus-positive class by default. Transformer activations are
reshaped to a spatial grid; these maps are exploratory and are neither attention
maps nor causal explanations.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .model import ThrombusClassifier

__all__ = ["POSITIVE_CLASS", "resolve_target_layer", "generate_maps"]

POSITIVE_CLASS = 1


def resolve_target_layer(
    model: ThrombusClassifier, arch: str, input_size: tuple[int, int]
) -> tuple[torch.nn.Module, Callable | None]:
    """Return the layer to hook and, for transformers, a reshape function.

    The reshape function maps a sequence of tokens back to ``(N, C, H, W)`` so
    that Grad-CAM can treat it like a convolutional feature map.
    """
    name = arch.lower()
    backbone = model.backbone

    if "resnet" in name:
        return backbone.layer4[-1], None
    if "densenet" in name:
        return backbone.features[-1], None
    if "inception" in name:
        return backbone.Mixed_7c, None
    if "convnext" in name:
        return backbone.stages[-1].blocks[-1].conv_dw, None

    if "vit" in name:
        height, width = input_size[0] // 16, input_size[1] // 16

        def reshape_vit(tensor: torch.Tensor) -> torch.Tensor:
            # Drop the class token, then restore the patch grid.
            patches = tensor[:, 1:, :]
            grid = patches.reshape(tensor.size(0), height, width, tensor.size(2))
            return grid.permute(0, 3, 1, 2)

        return backbone.blocks[-1].norm1, reshape_vit

    if "swin" in name:
        height, width = input_size[0] // 32, input_size[1] // 32

        def reshape_swin(tensor: torch.Tensor) -> torch.Tensor:
            grid = tensor.reshape(tensor.size(0), height, width, tensor.size(-1))
            return grid.permute(0, 3, 1, 2)

        return backbone.layers[-1].blocks[-1].norm2, reshape_swin

    raise ValueError(f"no Grad-CAM target layer is defined for architecture {arch!r}")


def generate_maps(
    model: ThrombusClassifier,
    images: torch.Tensor,
    arch: str,
    input_size: tuple[int, int],
    target_class: int | None = POSITIVE_CLASS,
) -> np.ndarray:
    """Grad-CAM maps for a batch, as ``(N, H, W)`` arrays in ``[0, 1]``.

    ``target_class=None`` reverts to the predicted class, which is useful for
    inspecting individual failures but should not be used for a figure that
    averages or compares maps across cases.
    """
    layer, reshape = resolve_target_layer(model, arch, input_size)
    cam = GradCAM(model=model, target_layers=[layer], reshape_transform=reshape)

    if target_class is None:
        with torch.no_grad():
            predicted = model(images).argmax(dim=1).tolist()
        targets = [ClassifierOutputTarget(c) for c in predicted]
    else:
        targets = [ClassifierOutputTarget(target_class)] * images.size(0)

    return cam(input_tensor=images, targets=targets)


def save_map(
    grayscale: np.ndarray, image: np.ndarray, destination: Path, alpha: float = 0.45
) -> None:
    """Write a heatmap overlaid on the source image.

    ``image`` must be ``(H, W, 3)`` in ``[0, 1]``; ``grayscale`` is ``(H, W)``.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale), cv2.COLORMAP_MAGMA)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    overlay = (1 - alpha) * image + alpha * heatmap
    overlay = np.uint8(255 * overlay / max(overlay.max(), 1e-8))
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
