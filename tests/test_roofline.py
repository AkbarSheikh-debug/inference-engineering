# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
import pytest

from ie.hardware import A100_80GB, H100_80GB
from ie.kv import weight_bytes
from ie.models import LLAMA3_8B
from ie.quant import bytes_per_param
from ie.roofline import decode_step, ridge_batch, weight_intensity


def test_batch_one_decode_is_memory_bound_and_reads_weights_once():
    step = decode_step(H100_80GB, LLAMA3_8B, batch=1)
    assert step.bound == "memory"
    assert step.seconds == pytest.approx(weight_bytes(LLAMA3_8B) / H100_80GB.bandwidth)
    assert 1 / step.seconds == pytest.approx(209, abs=1)


def test_a100_batch_one_ceiling():
    assert 1 / decode_step(A100_80GB, LLAMA3_8B).seconds == pytest.approx(127, abs=1)


def test_large_batch_becomes_compute_bound():
    assert decode_step(H100_80GB, LLAMA3_8B, batch=1024).bound == "compute"


def test_memory_and_compute_meet_at_the_ridge_batch():
    b = ridge_batch(H100_80GB)
    step = decode_step(H100_80GB, LLAMA3_8B, batch=1)
    assert step.memory_s / (step.compute_s * b) == pytest.approx(1.0, rel=1e-6)


def test_intensity_of_weight_matmul_is_batch_over_half_bytes():
    assert weight_intensity(1, 2.0) == 1.0
    assert weight_intensity(64, 2.0) == 64.0
    assert weight_intensity(1, 0.5) == 4.0  # smaller weights raise intensity at the same batch


def test_ridge_batch_shrinks_with_quantisation():
    assert ridge_batch(H100_80GB, 1.0) == pytest.approx(ridge_batch(H100_80GB, 2.0) / 2)


def test_kv_reads_add_to_step_time_and_do_not_amortise_over_batch():
    base = decode_step(H100_80GB, LLAMA3_8B, batch=32, context=0)
    long = decode_step(H100_80GB, LLAMA3_8B, batch=32, context=8192)
    assert long.memory_s > 2 * base.memory_s  # 32 GiB of KV against 15 GiB of weights


def test_int4_group_scale_overhead():
    assert bytes_per_param(4, 128) == pytest.approx(0.515625)
    step = decode_step(H100_80GB, LLAMA3_8B, bytes_per_param=bytes_per_param(4, 128))
    assert step.seconds * 1e3 == pytest.approx(1.24, abs=0.01)
