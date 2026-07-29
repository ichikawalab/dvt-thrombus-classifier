"""Repeated patient-level cross-validation training."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from .config import ARCHITECTURES, TrainingConfig
from .data import UltrasoundDataset, build_transforms
from .metrics import roc_auc
from .model import LitThrombusClassifier, resolve_input_config
from .splitting import Split, iter_splits

__all__ = ["cross_validate"]


def _architecture_name(identifier: str) -> str:
    return next((name for name, value in ARCHITECTURES.items() if value == identifier), identifier)


class EpochProgress(pl.Callback):
    """Print one concise status line after each training epoch."""

    def __init__(self, context: str) -> None:
        self.context = context
        self.best_epoch = 0
        self.best_val_loss = float("inf")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = trainer.callback_metrics
        train_loss = float(metrics["train_loss"])
        val_loss = float(metrics["val_loss"])
        epoch = trainer.current_epoch + 1
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch

        print(
            f"{self.context} epoch {epoch}/{trainer.max_epochs}: "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}",
            flush=True,
        )


def _seed_everything(seed: int) -> None:
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _loader(
    frame: pd.DataFrame, transform, config: TrainingConfig, *, shuffle: bool, return_id: bool
) -> DataLoader:
    return DataLoader(
        UltrasoundDataset(frame, transform, return_id=return_id),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )


@torch.no_grad()
def _predict(model: LitThrombusClassifier, loader: DataLoader, device) -> tuple[np.ndarray, ...]:
    """Return ``(image_ids, probabilities)`` for the thrombus-positive class."""
    model.eval()
    ids: list[str] = []
    probabilities: list[np.ndarray] = []
    for images, _labels, image_ids in loader:
        logits = model(images.to(device))
        probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        ids.extend(image_ids)
    return np.asarray(ids), np.concatenate(probabilities)


def _fit_one_fold(
    frame: pd.DataFrame,
    split: Split,
    arch: str,
    config: TrainingConfig,
    transforms_: tuple,
    output_dir: Path,
) -> tuple[LitThrombusClassifier, int]:
    train_transform, eval_transform = transforms_
    train_frame = frame.iloc[split.train]

    weights = compute_class_weight(
        "balanced",
        classes=np.unique(train_frame["label"].to_numpy()),
        y=train_frame["label"].to_numpy(),
    )
    module = LitThrombusClassifier(arch=arch, config=config, class_weights=weights.tolist())

    checkpoint = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename=f"rep{split.repetition + 1:02d}_fold{split.fold + 1}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_weights_only=True,
        enable_version_counter=False,
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=config.early_stopping_patience,
        check_on_train_epoch_end=False,
    )
    architecture = _architecture_name(arch)
    context = (
        f"{architecture} repetition {split.repetition + 1}/{config.n_repetitions} "
        f"fold {split.fold + 1}/{config.n_folds}"
    )
    progress = EpochProgress(context)
    print(f"Starting {context}", flush=True)
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        min_epochs=config.min_epochs,
        accelerator="auto",
        precision=config.precision,
        callbacks=[checkpoint, early_stopping, progress],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(
        module,
        _loader(train_frame, train_transform, config, shuffle=True, return_id=False),
        _loader(
            frame.iloc[split.validation], eval_transform, config, shuffle=False, return_id=True
        ),
    )

    best_path = Path(checkpoint.best_model_path)
    state = torch.load(best_path, map_location="cpu")["state_dict"]
    best = LitThrombusClassifier(arch=arch, config=config, class_weights=None, pretrained=False)
    best.load_state_dict(state, strict=True)
    return best, progress.best_epoch


def cross_validate(
    frame: pd.DataFrame, arch: str, config: TrainingConfig, output_dir: str | Path
) -> pd.DataFrame:
    """Run the full repeated cross-validation and write results to ``output_dir``.

    Writes ``predictions.csv`` (held-out probability per image per repetition),
    ``validation_predictions.csv`` and ``run_metadata.json``.
    """
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} is not empty; use a clean directory for one run")
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    input_size, mean, std = resolve_input_config(arch)
    transforms_ = (
        build_transforms(input_size, mean, std, augment=True),
        build_transforms(input_size, mean, std, augment=False),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels = frame["label"].to_numpy(dtype=int)
    groups = frame["patient_id"].to_numpy(dtype=str)
    id_to_row = {image_id: i for i, image_id in enumerate(frame["image_id"])}

    n_images, n_reps = len(frame), config.n_repetitions
    test_probability = np.full((n_images, n_reps), np.nan)
    fold_of = np.full((n_images, n_reps), -1, dtype=int)
    validation_rows: list[pd.DataFrame] = []

    for split in iter_splits(labels, groups, n_reps, config.n_folds, config.base_seed):
        _seed_everything(split.seed)
        model, best_epoch = _fit_one_fold(frame, split, arch, config, transforms_, output_dir)
        model = model.to(device)

        # Validation predictions support threshold and ensemble selection.
        validation_frame = frame.iloc[split.validation]
        validation_ids, validation_probability = _predict(
            model,
            _loader(validation_frame, transforms_[1], config, shuffle=False, return_id=True),
            device,
        )
        label_by_image = validation_frame.set_index("image_id")["label"]
        validation_labels = label_by_image.loc[validation_ids].to_numpy(dtype=int)
        patient_by_image = validation_frame.set_index("image_id")["patient_id"]
        validation_rows.append(
            pd.DataFrame(
                {
                    "image_id": validation_ids,
                    "patient_id": patient_by_image.loc[validation_ids].to_numpy(),
                    "repetition": split.repetition + 1,
                    "fold": split.fold + 1,
                    "true_label": validation_labels,
                    "probability": validation_probability,
                }
            )
        )

        test_ids, test_probs = _predict(
            model,
            _loader(frame.iloc[split.test], transforms_[1], config, shuffle=False, return_id=True),
            device,
        )
        rows = [id_to_row[i] for i in test_ids]
        test_probability[rows, split.repetition] = test_probs
        fold_of[rows, split.repetition] = split.fold + 1

        validation_auc = roc_auc(validation_labels, validation_probability)
        print(
            f"Completed {_architecture_name(arch)}: repetition {split.repetition + 1}/{n_reps}, "
            f"fold {split.fold + 1}/{config.n_folds}, best_epoch={best_epoch}, "
            f"val_AUC={validation_auc:.4f}",
            flush=True,
        )
        del model
        torch.cuda.empty_cache()

    if np.isnan(test_probability).any():
        raise RuntimeError("some images received no held-out prediction")

    predictions = pd.DataFrame(
        {
            "image_id": frame["image_id"].to_numpy(),
            "patient_id": groups,
            "true_label": labels,
        }
    )
    for repetition in range(n_reps):
        predictions[f"fold_{repetition + 1}"] = fold_of[:, repetition]
        predictions[f"trial_{repetition + 1}"] = test_probability[:, repetition]

    predictions.to_csv(output_dir / "predictions.csv", index=False)
    # Ensemble thresholds require the member-level validation probabilities.
    pd.concat(validation_rows, ignore_index=True).to_csv(
        output_dir / "validation_predictions.csv", index=False
    )
    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "arch": arch,
                "augment": True,
                "config": asdict(config),
                "n_images": n_images,
                "n_patients": int(len(np.unique(groups))),
                "torch": torch.__version__,
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return predictions
