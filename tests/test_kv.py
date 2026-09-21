# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
import pytest

from ie.hardware import A100_80GB, H100_80GB
from ie.kv import kv_bytes, kv_bytes_per_token, kv_pool_bytes, max_sequences, weight_bytes
from ie.models import LLAMA2_7B, LLAMA3_8B, LLAMA3_70B
from ie.units import GiB, KiB


def test_kv_bytes_per_token_reference_models():
    assert kv_bytes_per_token(LLAMA2_7B) == 512 * KiB  # 2 x 32 x 32 x 128 x 2
    assert kv_bytes_per_token(LLAMA3_8B) == 128 * KiB  # 2 x 32 x 8 x 128 x 2
    assert kv_bytes_per_token(LLAMA3_70B) == 320 * KiB  # 2 x 80 x 8 x 128 x 2


def test_llama3_8b_at_8k_is_exactly_one_gib():
    assert kv_bytes(LLAMA3_8B, 8192) == GiB


def test_fp8_cache_halves_bytes():
    assert kv_bytes_per_token(LLAMA3_8B, dtype_bytes=1) * 2 == kv_bytes_per_token(LLAMA3_8B)


def test_gqa_counterfactuals():
    mha = LLAMA3_8B.with_kv_heads(32)
    mqa = LLAMA3_8B.with_kv_heads(1)
    assert kv_bytes_per_token(mha) == 4 * kv_bytes_per_token(LLAMA3_8B)
    assert kv_bytes_per_token(LLAMA3_8B) == 8 * kv_bytes_per_token(mqa)


def test_sequences_that_fit_on_one_gpu():
    assert max_sequences(H100_80GB, LLAMA3_8B, 8192) == 57
    assert max_sequences(H100_80GB, LLAMA3_8B.with_kv_heads(32), 8192) == 14
    assert max_sequences(A100_80GB, LLAMA3_8B, 8192) == 57  # capacity, not bandwidth, decides this


def test_70b_bf16_does_not_fit_one_gpu():
    assert weight_bytes(LLAMA3_70B) > H100_80GB.hbm_bytes
    assert kv_pool_bytes(H100_80GB, LLAMA3_70B) == 0.0
    assert max_sequences(H100_80GB, LLAMA3_70B, 8192) == 0


def test_kv_pool_shrinks_with_weight_precision():
    bf16 = kv_pool_bytes(H100_80GB, LLAMA3_8B, bytes_per_param=2.0)
    int8 = kv_pool_bytes(H100_80GB, LLAMA3_8B, bytes_per_param=1.0)
    assert int8 - bf16 == pytest.approx(LLAMA3_8B.params)
