from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GpuProfile:
    device: str
    cuda_available: bool
    vram_gb: float
    preferred_precision: str


def detect_gpu_profile(torch_module: Any | None = None) -> GpuProfile:
    """Return safe local-inference defaults without requiring PyTorch at import time."""
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[import-not-found]
        except ImportError:
            return GpuProfile("cpu", False, 0.0, "fp32")

    cuda = torch_module.cuda
    if not cuda.is_available():
        return GpuProfile("cpu", False, 0.0, "fp32")

    bytes_available = cuda.get_device_properties(0).total_memory
    vram_gb = round(bytes_available / (1024 ** 3), 2)
    precision = "bf16" if vram_gb >= 16 else "fp16"
    return GpuProfile("cuda:0", True, vram_gb, precision)
