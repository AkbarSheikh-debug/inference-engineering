# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
"""A first-order roofline model of one decode step.

One decode step reads every weight once, reads each sequence's KV cache, and
does about 2 FLOPs per parameter per token. The step takes as long as the
slower of the memory side and the compute side. This is a ceiling, not a
forecast: it ignores attention FLOPs, kernel launch overhead, communication,
and the fact that real kernels do not reach datasheet peaks.
"""

from dataclasses import dataclass

from .hardware import Gpu
from .kv import kv_bytes_per_token, weight_bytes
from .models import ModelConfig


@dataclass(frozen=True)
class Step:
    memory_s: float
    compute_s: float

    @property
    def seconds(self) -> float:
        return max(self.memory_s, self.compute_s)

    @property
    def bound(self) -> str:
        return "memory" if self.memory_s >= self.compute_s else "compute"


def decode_step(
    gpu: Gpu,
    model: ModelConfig,
    batch: int = 1,
    context: int = 0,
    bytes_per_param: float = 2.0,
    kv_dtype_bytes: int = 2,
) -> Step:
    """Time for one decode step that emits one token for each of `batch` sequences."""
    bytes_moved = weight_bytes(model, bytes_per_param) + batch * context * kv_bytes_per_token(
        model, kv_dtype_bytes
    )
    flops = 2.0 * model.params * batch
    return Step(memory_s=bytes_moved / gpu.bandwidth, compute_s=flops / gpu.peak_bf16)


def weight_intensity(batch: int, bytes_per_param: float = 2.0) -> float:
    """FLOP/byte of a weight matmul when weights dominate the bytes moved.

    Each weight is used for 2 FLOPs per token in the batch and costs
    `bytes_per_param` bytes to fetch, so intensity is 2 * batch / bytes_per_param.
    """
    return 2.0 * batch / bytes_per_param


def ridge_batch(gpu: Gpu, bytes_per_param: float = 2.0) -> float:
    """Batch size at which weight matmuls reach the ridge point."""
    return gpu.ridge() * bytes_per_param / 2.0
