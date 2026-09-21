# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
import pytest

from ie.gpumem import (
    gemm_flops,
    gemm_intensity,
    gemm_read_bytes,
    gemm_read_bytes_counted,
    warp_access,
)
from ie.units import GiB


def test_consecutive_words_use_four_sectors_and_waste_nothing():
    a = warp_access(4)
    assert (a.sectors, a.useful_bytes, a.moved_bytes, a.efficiency) == (4, 128, 128, 1.0)


def test_scattered_four_byte_loads_move_32_sectors_at_12_5_percent():
    a = warp_access(128)
    assert (a.sectors, a.moved_bytes) == (32, 1024)
    assert a.efficiency == pytest.approx(0.125)


def test_the_penalty_plateaus_once_every_thread_owns_a_sector():
    assert {warp_access(s).efficiency for s in (32, 64, 128, 256, 4096)} == {0.125}


def test_efficiency_falls_as_stride_grows_until_the_plateau():
    effs = [warp_access(s).efficiency for s in (4, 8, 16, 32)]
    assert effs == sorted(effs, reverse=True) == [1.0, 0.5, 0.25, 0.125]


def test_one_byte_scattered_loads_hit_3_125_percent():
    assert warp_access(128, elem_bytes=1).efficiency == pytest.approx(0.03125)


def test_a_misaligned_start_costs_one_extra_sector():
    assert warp_access(4, base=4).sectors == 5


@pytest.mark.parametrize("shape", [(64, 64, 64, 16, 16), (100, 90, 50, 32, 24), (17, 5, 3, 4, 4), (4096, 4096, 128, 128, 128)])
def test_closed_form_traffic_matches_walking_the_tile_grid(shape):
    assert gemm_read_bytes(*shape) == gemm_read_bytes_counted(*shape)


def test_square_tile_traffic_is_two_mnk_b_over_t():
    m = n = k = 4096
    assert gemm_read_bytes(m, n, k, 128, 128, 2) == 2 * m * n * k * 2 // 128 == 2 * GiB


def test_no_reuse_intensity_is_half_a_flop_per_byte_in_fp16():
    assert gemm_intensity(4096, 4096, 4096, 1, 1, 2) == pytest.approx(0.5, abs=1e-3)


def test_intensity_grows_with_tile_size():
    vals = [gemm_intensity(4096, 4096, 4096, t, t, 2) for t in (1, 16, 32, 64, 128, 256)]
    assert vals == sorted(vals)
    assert vals[4] == pytest.approx(63.0, abs=0.1)
    assert vals[5] == pytest.approx(124.1, abs=0.1)


def test_flops():
    assert gemm_flops(4096, 4096, 4096) == 2 * 4096**3
