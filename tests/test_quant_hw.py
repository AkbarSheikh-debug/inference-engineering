# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
import pytest

from ie.hardware import A100_80GB, ALL_GPUS, H100_80GB
from ie.quant import bits_per_weight, bytes_per_param
from ie.units import GiB


def test_plain_bit_widths():
    assert bytes_per_param(16) == 2.0
    assert bytes_per_param(8) == 1.0
    assert bytes_per_param(4) == 0.5


def test_group_scales_add_overhead():
    assert bits_per_weight(4, 128) == 4.125
    assert bits_per_weight(4, 32) == 4.5
    assert bits_per_weight(8, 128, scale_bits=32) == 8.25


def test_ridge_points():
    assert A100_80GB.ridge() == pytest.approx(153.0, abs=0.1)
    assert H100_80GB.ridge() == pytest.approx(295.2, abs=0.1)


def test_roofline_shape():
    g = H100_80GB
    assert g.attainable(1.0) == g.bandwidth  # 1 FLOP/byte: bandwidth-limited
    assert g.attainable(10_000) == g.peak_bf16  # far right: compute-limited
    assert g.attainable(g.ridge()) == pytest.approx(g.peak_bf16)


def test_capacity_is_binary_and_bandwidth_is_decimal():
    for g in ALL_GPUS:
        assert g.hbm_bytes == 80 * GiB
        assert g.bandwidth % 1e9 == 0
        assert g.source
