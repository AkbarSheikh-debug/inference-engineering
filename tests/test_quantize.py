# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
import numpy as np
import pytest

from ie.quantize import (
    asymmetric_params,
    dequantize,
    fake_quantize,
    qmax,
    quantize,
    relative_rms_error,
    symmetric_scale,
    synthetic_weights,
)

ROW = np.array([-0.10, 0.05, 0.22, 0.41, 0.63, 0.86, 0.34, 0.12])


def test_qmax():
    assert qmax(8) == 127 and qmax(4) == 7


def test_symmetric_and_asymmetric_rules_on_a_skewed_row():
    s_sym = symmetric_scale(ROW)
    s_asym, z = asymmetric_params(ROW)
    assert s_sym == pytest.approx(0.86 / 127)
    assert s_asym == pytest.approx(0.96 / 255)
    assert z == 27
    q_sym = quantize(ROW, s_sym)
    q_asym = quantize(ROW, s_asym, z, 0, 255)
    assert q_sym.tolist() == [-15, 7, 32, 61, 93, 127, 50, 18]
    assert q_asym.tolist() == [0, 40, 85, 136, 194, 255, 117, 59]
    worst_sym = np.max(np.abs(ROW - dequantize(q_sym, s_sym)))
    worst_asym = np.max(np.abs(ROW - dequantize(q_asym, s_asym, z)))
    assert round(worst_sym, 4) == 0.0033 and round(worst_asym, 4) == 0.0016
    assert s_sym / s_asym == pytest.approx(1.80, abs=0.005)


def test_error_never_exceeds_half_a_step():
    s, z = asymmetric_params(ROW)
    err = np.abs(ROW - dequantize(quantize(ROW, s, z, 0, 255), s, z))
    assert err.max() <= s / 2 + 1e-12


def test_finer_granularity_lowers_error_at_both_bit_widths():
    w = synthetic_weights()
    for bits in (8, 4):
        t = relative_rms_error(w, fake_quantize(w, bits, "tensor"))
        c = relative_rms_error(w, fake_quantize(w, bits, "channel"))
        g = relative_rms_error(w, fake_quantize(w, bits, "group", 128))
        assert t > c > g, bits


def test_fewer_bits_means_more_error():
    w = synthetic_weights()
    for gran in ("tensor", "channel", "group"):
        assert relative_rms_error(w, fake_quantize(w, 4, gran)) > 4 * relative_rms_error(w, fake_quantize(w, 8, gran))


def test_per_channel_error_is_bounded_by_half_a_row_step():
    w = synthetic_weights()
    step = np.max(np.abs(w), axis=1, keepdims=True) / qmax(8)
    assert np.all(np.abs(w - fake_quantize(w, 8, "channel")) <= step / 2 + 1e-12)


def test_bad_arguments_are_rejected():
    w = np.ones((4, 100))
    with pytest.raises(ValueError):
        fake_quantize(w, 8, "group", 128)
    with pytest.raises(ValueError):
        fake_quantize(w, 8, "banana")


def test_all_zero_matrix_survives():
    assert np.all(fake_quantize(np.zeros((2, 128)), 8, "group") == 0)
