# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
import numpy as np
import pytest

from ie.attention import (
    flash_attention,
    hbm_elements_flash,
    hbm_elements_standard,
    naive_attention,
    online_softmax_state,
    score_matrix_bytes,
)
from ie.units import GiB, MiB


@pytest.fixture(scope="module")
def qkv():
    rng = np.random.default_rng(0)
    return tuple(rng.normal(size=(97, 16)) for _ in range(3))


@pytest.mark.parametrize("blocks", [(64, 64), (37, 29), (97, 97), (7, 3), (1, 1)])
def test_tiled_attention_equals_direct_attention(qkv, blocks):
    out, _ = flash_attention(*qkv, *blocks)
    np.testing.assert_allclose(out, naive_attention(*qkv), rtol=0, atol=1e-12)


def test_block_order_does_not_change_the_answer(qkv):
    fwd, _ = flash_attention(*qkv, 37, 29)
    rev, _ = flash_attention(*qkv, 37, 29, reverse_kv=True)
    np.testing.assert_allclose(fwd, rev, atol=1e-12)


def test_only_one_block_of_scores_is_ever_held(qkv):
    n = qkv[0].shape[0]
    _, largest = flash_attention(*qkv, 16, 8)
    assert largest == 16 * 8 < n * n


def test_large_scores_do_not_overflow():
    rng = np.random.default_rng(1)
    q, k, v = (rng.normal(size=(40, 8)) * 60 for _ in range(3))
    out, _ = flash_attention(q, k, v, 9, 7)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, naive_attention(q, k, v), atol=1e-10)


def test_the_eight_by_eight_trace():
    (m1, l1, _), (m2, l2, alpha) = online_softmax_state([[0.515, 0.385, 0.405, 0.435], [0.560, 0.300, 0.370, 0.320]])
    assert (round(m1, 3), round(l1, 3)) == (0.515, 3.697)
    assert (round(m2, 3), round(l2, 3), round(alpha, 3)) == (0.560, 6.919, 0.956)
    ref = sum(np.exp(np.array([0.515, 0.385, 0.405, 0.435, 0.560, 0.300, 0.370, 0.320]) - 0.560))
    assert l2 == pytest.approx(ref)


def test_traffic_ratio_is_two_b_over_d_and_does_not_grow_with_n():
    for n in (1024, 4096, 16384, 65536):
        ratio = hbm_elements_standard(n, 128) / hbm_elements_flash(n, 128, 128)
        assert ratio == pytest.approx(2.0, abs=0.01), n


def test_traffic_is_still_quadratic_in_n():
    # 2nd (1 + n/B): quadrupling n multiplies traffic by 4 x (1 + 128) / (1 + 32), close to 16
    t1, t2 = hbm_elements_flash(4096, 128, 128), hbm_elements_flash(16384, 128, 128)
    assert t2 / t1 == pytest.approx(4 * 129 / 33)
    assert 15 < t2 / t1 < 16


def test_a_bigger_query_block_cuts_traffic_by_about_the_block_ratio():
    # quadrupling B turns (1 + n/64) into (1 + n/256): 129 / 33 at n = 8192, just under 4
    small, big = hbm_elements_flash(8192, 128, 64), hbm_elements_flash(8192, 128, 256)
    assert small / big == pytest.approx(129 / 33)


def test_absolute_traffic_at_n_4096():
    assert hbm_elements_standard(4096, 128) * 2 / MiB == pytest.approx(132.0, abs=0.5)
    assert hbm_elements_flash(4096, 128, 128) * 2 / MiB == pytest.approx(66.0, abs=0.5)


def test_score_matrix_storage_is_quadratic():
    assert score_matrix_bytes(16384) == 16 * score_matrix_bytes(4096)
    assert score_matrix_bytes(4096) == 32 * MiB
    assert score_matrix_bytes(32768, heads=32) == 64 * GiB
