# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
"""Published architecture configurations of a few reference models.

Values come from the model papers and public config files:
Llama 2 (arXiv:2307.09288) and Llama 3 (arXiv:2407.21783). Parameter counts
are the published approximate totals.
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ModelConfig:
    name: str
    n_layers: int
    hidden: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    params: float  # total parameters, approximate
    vocab: int

    def with_kv_heads(self, n_kv_heads: int, name: str | None = None) -> "ModelConfig":
        """Counterfactual: same model, different KV-head count (MHA, GQA, MQA)."""
        return replace(self, n_kv_heads=n_kv_heads, name=name or f"{self.name} ({n_kv_heads} KV heads)")


LLAMA2_7B = ModelConfig(
    name="Llama-2-7B",
    n_layers=32,
    hidden=4096,
    n_heads=32,
    n_kv_heads=32,  # full multi-head attention
    head_dim=128,
    params=6.74e9,
    vocab=32_000,
)

LLAMA3_8B = ModelConfig(
    name="Llama-3-8B",
    n_layers=32,
    hidden=4096,
    n_heads=32,
    n_kv_heads=8,
    head_dim=128,
    params=8.03e9,
    vocab=128_256,
)

LLAMA3_70B = ModelConfig(
    name="Llama-3-70B",
    n_layers=80,
    hidden=8192,
    n_heads=64,
    n_kv_heads=8,
    head_dim=128,
    params=70.6e9,
    vocab=128_256,
)

ALL_MODELS = (LLAMA2_7B, LLAMA3_8B, LLAMA3_70B)
