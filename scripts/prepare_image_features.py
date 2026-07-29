"""Derive global image features used by the statistical report.

Requires the images, so it runs only at the originating site::

    uv run --extra train python scripts\\prepare_image_features.py ^
        --input-csv data\\input.csv --data-root data ^
        --output outputs\\image_features.csv

The output contains scalar features, source-image dimensions, and opaque
identifiers. It contains no pixel data or source paths.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from dvt_thrombus.data import load_input_table


def features_for(path: Path) -> dict[str, float | int]:
    with Image.open(path) as handle:
        width_px, height_px = handle.size
        pixels = np.asarray(handle.convert("L"), dtype=float)
    return {
        "width_px": width_px,
        "height_px": height_px,
        "mean_intensity": float(pixels.mean()),
        "sd_intensity": float(pixels.std()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/image_features.csv"))
    args = parser.parse_args()

    frame = load_input_table(args.input_csv, args.data_root)
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            {
                "image_id": str(row["image_id"]),
                "patient_id": str(row["patient_id"]),
                "label": int(row["label"]),
                **features_for(Path(row["resolved_path"])),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(rows)} images)")


if __name__ == "__main__":
    main()
