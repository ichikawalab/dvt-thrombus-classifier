"""Architecture size, profiler FLOPs, and batch-one inference latency."""

from __future__ import annotations

import platform
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

from .model import ThrombusClassifier, resolve_input_config

__all__ = ["ArchitectureBenchmark", "benchmark_architecture"]


@dataclass(frozen=True)
class ArchitectureBenchmark:
    architecture: str
    input_height: int
    input_width: int
    trainable_parameters: int
    profiler_flops: int
    device: str
    hardware: str
    warmup_runs: int
    timed_runs: int
    inference_ms_mean: float
    inference_ms_sd: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return platform.processor() or "CPU"


def _flops(model: torch.nn.Module, example: torch.Tensor) -> int:
    with (
        torch.inference_mode(),
        profile(activities=[ProfilerActivity.CPU], with_flops=True) as profiler,
    ):
        model(example)
    return int(sum(event.flops for event in profiler.key_averages()))


def benchmark_architecture(
    architecture: str,
    *,
    device_name: str = "auto",
    warmup_runs: int = 10,
    timed_runs: int = 50,
) -> ArchitectureBenchmark:
    """Benchmark one randomly initialised architecture at batch size one."""
    if warmup_runs < 0 or timed_runs < 2:
        raise ValueError("warmup_runs must be non-negative and timed_runs must be at least 2")
    if device_name not in {"auto", "cpu", "cuda"}:
        raise ValueError("device_name must be auto, cpu, or cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    use_cuda = device_name == "cuda" or (device_name == "auto" and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    input_size, _, _ = resolve_input_config(architecture)
    cpu_model = ThrombusClassifier(architecture, pretrained=False).eval()
    cpu_example = torch.zeros((1, 3, *input_size))
    parameter_count = sum(parameter.numel() for parameter in cpu_model.parameters())
    flops = _flops(cpu_model, cpu_example)

    model = cpu_model.to(device)
    example = cpu_example.to(device)
    with torch.inference_mode():
        for _ in range(warmup_runs):
            model(example)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        elapsed = []
        for _ in range(timed_runs):
            start = time.perf_counter()
            model(example)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed.append((time.perf_counter() - start) * 1000.0)

    return ArchitectureBenchmark(
        architecture=architecture,
        input_height=input_size[0],
        input_width=input_size[1],
        trainable_parameters=parameter_count,
        profiler_flops=flops,
        device=str(device),
        hardware=_device_name(device),
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
        inference_ms_mean=float(np.mean(elapsed)),
        inference_ms_sd=float(np.std(elapsed, ddof=1)),
    )
