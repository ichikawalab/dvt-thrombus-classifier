"""Render what training augmentation does to real images.

    uv run --extra train python scripts\\preview_augmentation.py ^
        --input-csv data\\input.csv --data-root data ^
        --output outputs\\augmentation_preview.png

Each row is one image: the evaluation view on the left, then independent draws
from the training pipeline.  The transforms come from
``dvt_thrombus.data.build_transforms``, the same call training uses, so the
figure cannot drift away from what the models actually see.

The rendered image is the normalised tensor mapped back to display range, which
is what reaches the network rather than the file on disk.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from dvt_thrombus.config import ARCHITECTURES  # noqa: E402
from dvt_thrombus.data import UltrasoundDataset, build_transforms, load_input_table  # noqa: E402
from dvt_thrombus.model import resolve_input_config  # noqa: E402
from dvt_thrombus.plots import PUBLICATION_STYLE  # noqa: E402


def to_display(tensor: torch.Tensor, mean: tuple[float, ...], std: tuple[float, ...]) -> np.ndarray:
    array = tensor.numpy().transpose(1, 2, 0) * np.asarray(std) + np.asarray(mean)
    return np.clip(array, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arch", default="ViT")
    parser.add_argument("--n-images", type=int, default=4)
    parser.add_argument("--n-draws", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--image-ids", nargs="*", help="specific image_id values; default is a balanced sample"
    )
    args = parser.parse_args()

    arch = ARCHITECTURES.get(args.arch, args.arch)
    input_size, mean, std = resolve_input_config(arch)
    frame = load_input_table(args.input_csv, args.data_root)
    plt.rcParams.update(PUBLICATION_STYLE)

    if args.image_ids:
        selected = frame[frame["image_id"].isin(args.image_ids)]
        if selected.empty:
            raise SystemExit("none of the requested image_id values are in the manifest")
    else:
        # Balance the sample across classes so the figure shows both.
        rng = np.random.default_rng(args.seed)
        per_class = max(1, args.n_images // 2)
        positions = []
        for label in (1, 0):
            candidates = np.flatnonzero(frame["label"].to_numpy() == label)
            positions.extend(rng.permutation(candidates)[:per_class])
        selected = frame.iloc[positions]

    eval_transform = build_transforms(input_size, mean, std, augment=False)
    train_transform = build_transforms(input_size, mean, std, augment=True)

    n_rows, n_cols = len(selected), args.n_draws + 1
    figure, axes = plt.subplots(n_rows, n_cols, figsize=(1.9 * n_cols, 2.0 * n_rows), squeeze=False)

    torch.manual_seed(args.seed)
    for row in range(len(selected)):
        single = selected.iloc[[row]]
        record = selected.iloc[row]
        evaluation_image = UltrasoundDataset(single, eval_transform)[0][0]
        axes[row][0].imshow(to_display(evaluation_image, mean, std))
        axes[row][0].set_ylabel(
            f"{record['image_id']}\n{'positive' if record['label'] else 'negative'}", fontsize=7
        )
        if row == 0:
            axes[row][0].set_title("evaluation", fontsize=9)

        augmented_dataset = UltrasoundDataset(single, train_transform)
        for column in range(1, n_cols):
            axes[row][column].imshow(to_display(augmented_dataset[0][0], mean, std))
            if row == 0:
                axes[row][column].set_title(f"draw {column}", fontsize=9)

    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(args.output, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(figure)
    print(f"wrote {args.output} ({n_rows} images x {args.n_draws} draws)")


if __name__ == "__main__":
    main()
