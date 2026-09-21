# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
"""KV-cache sizing."""

from .hardware import Gpu
from .models import ModelConfig


def kv_bytes_per_token(model: ModelConfig, dtype_bytes: int = 2) -> int:
    """Bytes of K and V stored per token, across all layers.

    2 (K and V) x layers x KV heads x head_dim x bytes per element.
    """
    return 2 * model.n_layers * model.n_kv_heads * model.head_dim * dtype_bytes


def kv_bytes(model: ModelConfig, tokens: int, dtype_bytes: int = 2) -> int:
    return tokens * kv_bytes_per_token(model, dtype_bytes)


def weight_bytes(model: ModelConfig, bytes_per_param: float = 2.0) -> float:
    return model.params * bytes_per_param


def kv_pool_bytes(
    gpu: Gpu,
    model: ModelConfig,
    bytes_per_param: float = 2.0,
    mem_util: float = 0.9,
) -> float:
    """HBM left for the KV cache after weights, under a memory-utilisation cap.

    Ignores activation workspace and CUDA context, so it is an upper bound.
    """
    return max(0.0, gpu.hbm_bytes * mem_util - weight_bytes(model, bytes_per_param))


def max_sequences(
    gpu: Gpu,
    model: ModelConfig,
    context: int,
    bytes_per_param: float = 2.0,
    kv_dtype_bytes: int = 2,
    mem_util: float = 0.9,
) -> int:
    """How many full-context sequences fit in the KV pool."""
    per_seq = kv_bytes(model, context, kv_dtype_bytes)
    return int(kv_pool_bytes(gpu, model, bytes_per_param, mem_util) // per_seq)
