"""Command-line entry points.

Every command that needs PyTorch imports it inside its own function.  That is
deliberate: the analysis stack is installed by ``uv sync`` alone, and importing
this module must not fail when the optional ``train`` extra is absent.  Only
``dvt-stats`` is expected to work in that configuration, and it does.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ARCHITECTURES, Config, TrainingConfig, load_config

# Present when running from a checkout; absent when installed as a wheel, in
# which case the dataclass defaults (which are identical) are used.
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def _resolve_arch(name: str) -> str:
    """Accept either a configured display name or a timm identifier."""
    return ARCHITECTURES.get(name, name)


def _load(path: Path | None) -> Config:
    if path is None:
        return Config()
    if not path.exists():
        if path == DEFAULT_CONFIG:
            return Config()
        raise FileNotFoundError(f"configuration file does not exist: {path}")
    return load_config(path)


def _config_from_args(args: argparse.Namespace) -> TrainingConfig:
    config = _load(args.config).training
    for field in ("batch_size", "num_workers", "n_repetitions", "base_seed"):
        value = getattr(args, field, None)
        if value is not None:
            setattr(config, field, value)
    config.__post_init__()
    return config


# --------------------------------------------------------------------------- train


def train_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Repeated patient-level cross-validation.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--arch", default="ViT", help="short name or timm identifier")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--n-repetitions", type=int)
    parser.add_argument("--base-seed", type=int)
    args = parser.parse_args(argv)

    from .data import load_input_table
    from .training import cross_validate

    config = _config_from_args(args)
    config.arch = _resolve_arch(args.arch)
    frame = load_input_table(args.input_csv, args.data_root)
    run_dir = args.output_root / args.run_name

    cross_validate(
        frame=frame,
        arch=config.arch,
        config=config,
        output_dir=run_dir,
    )
    print(f"wrote {args.output_root / args.run_name}")


# ------------------------------------------------------------------------- predict


def predict_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Average predictions over fold models.")
    parser.add_argument("--checkpoints", type=Path, required=True, help="directory of .ckpt files")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--arch", default="ViT")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--threshold",
        type=float,
        help="decision threshold derived from validation data",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    args = parser.parse_args(argv)
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        parser.error("--threshold must lie in [0, 1]")

    from .data import load_input_table
    from .inference import predict_frame

    config = _config_from_args(args)
    checkpoints = sorted(args.checkpoints.glob("*.ckpt"))
    if not checkpoints:
        raise SystemExit(f"no checkpoints found in {args.checkpoints}")

    frame = load_input_table(args.input_csv, args.data_root)
    result = predict_frame(frame, checkpoints, _resolve_arch(args.arch), config, args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(result)} rows, {len(checkpoints)} fold models)")


# ------------------------------------------------------------------------- gradcam


def gradcam_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Grad-CAM maps for individual cases.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--arch", default="ViT")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--target-class",
        choices=("1", "0", "pred"),
        default="1",
        help=(
            "class to explain; '1' (default) asks where the model finds evidence of "
            "thrombus, 'pred' explains whatever the model predicted and is intended "
            "for inspecting individual failures"
        ),
    )
    parser.add_argument("--limit", type=int, help="only process the first N images")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    import numpy as np
    import torch

    from .data import UltrasoundDataset, build_transforms, load_input_table
    from .gradcam import generate_maps, save_map
    from .inference import load_fold_model
    from .model import resolve_input_config

    config = _config_from_args(args)
    arch = _resolve_arch(args.arch)
    input_size, mean, std = resolve_input_config(arch)

    frame = load_input_table(args.input_csv, args.data_root)
    if args.limit:
        frame = frame.head(args.limit)

    model = load_fold_model(args.checkpoint, arch, config).model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataset = UltrasoundDataset(
        frame, build_transforms(input_size, mean, std, augment=False), return_id=True
    )
    target = None if args.target_class == "pred" else int(args.target_class)
    mean_arr, std_arr = np.asarray(mean), np.asarray(std)

    for index in range(len(dataset)):
        image, _label, image_id = dataset[index]
        batch = image.unsqueeze(0).to(device)
        grayscale = generate_maps(model, batch, arch, input_size, target)[0]
        rgb = np.clip(image.numpy().transpose(1, 2, 0) * std_arr + mean_arr, 0, 1)
        save_map(grayscale, rgb, args.output / f"{Path(image_id).stem}.png")

    print(f"wrote {len(dataset)} maps to {args.output} (target class: {args.target_class})")


# --------------------------------------------------------------------------- stats


def stats_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the statistical analysis protocol.")
    parser.add_argument(
        "--runs",
        type=Path,
        required=True,
        help="directory containing one complete training-output directory per architecture",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/stats"))
    parser.add_argument(
        "--image-features",
        type=Path,
        help="optional local CSV of global image characteristics",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--target-sensitivity", type=float, default=0.95)
    args = parser.parse_args(argv)
    if args.bootstrap_resamples < 1:
        parser.error("--bootstrap-resamples must be positive")

    from .report import run_report

    args.output.mkdir(parents=True, exist_ok=True)
    summary = run_report(
        runs_dir=args.runs,
        output_dir=args.output,
        image_features=args.image_features,
        n_resamples=args.bootstrap_resamples,
        target_sensitivity=args.target_sensitivity,
    )
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {args.output}")


# ----------------------------------------------------------------------- benchmark


def benchmark_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Measure model parameters, profiler FLOPs, and batch-one latency."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--timed-runs", type=int, default=50)
    args = parser.parse_args(argv)

    import pandas as pd

    from .complexity import benchmark_architecture

    rows = []
    for name, architecture in ARCHITECTURES.items():
        result = benchmark_architecture(
            architecture,
            device_name=args.device,
            warmup_runs=args.warmup_runs,
            timed_runs=args.timed_runs,
        )
        rows.append({"Model": name, **result.as_dict()})
        print(f"{name}: {result.inference_ms_mean:.2f} ms", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"wrote {args.output}")
