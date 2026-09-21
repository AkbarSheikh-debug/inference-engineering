# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
"""Lab 04: integer quantization, from one row of numbers to a whole matrix.

    python labs/cpu/04_quantization.py

Part 1 quantizes one skewed row of eight values with the symmetric rule and the
asymmetric (zero-point) rule and prints both. Part 2 quantizes a synthetic
weight matrix at three granularities and reports the error, along with the
storage each scheme really costs once its scales are counted.
"""

import numpy as np

from ie.quant import bits_per_weight
from ie.quantize import (
    asymmetric_params,
    dequantize,
    fake_quantize,
    quantize,
    relative_rms_error,
    symmetric_scale,
    synthetic_weights,
)

ROW = np.array([-0.10, 0.05, 0.22, 0.41, 0.63, 0.86, 0.34, 0.12])


def one_row() -> None:
    s_sym = symmetric_scale(ROW)
    s_asym, z = asymmetric_params(ROW)
    q_sym = quantize(ROW, s_sym)
    q_asym = quantize(ROW, s_asym, z, 0, 255)
    d_sym, d_asym = dequantize(q_sym, s_sym), dequantize(q_asym, s_asym, z)
    print(f"symmetric:  step {s_sym:.6f}, zero-point 0")
    print(f"asymmetric: step {s_asym:.6f}, zero-point {z}\n")
    print(f"{'x':>7}{'sym int':>9}{'dequant':>9}{'error':>8}{'asym int':>10}{'dequant':>9}{'error':>8}")
    for i, x in enumerate(ROW):
        print(f"{x:>7.2f}{q_sym[i]:>9}{d_sym[i]:>9.4f}{abs(x - d_sym[i]):>8.4f}"
              f"{q_asym[i]:>10}{d_asym[i]:>9.4f}{abs(x - d_asym[i]):>8.4f}")
    worst_sym, worst_asym = np.max(np.abs(ROW - d_sym)), np.max(np.abs(ROW - d_asym))
    print(f"{'worst':>7}{'':>9}{'':>9}{worst_sym:>8.4f}{'':>10}{'':>9}{worst_asym:>8.4f}")
    print(f"\nGuaranteed-bound improvement = step ratio {s_sym / s_asym:.2f}\n")


def granularity() -> None:
    w = synthetic_weights()
    print(f"synthetic weights {w.shape}: log-normal row scales, 0.2% outliers at 20x")
    print(f"{'bits':>5}{'granularity':>16}{'rel. RMS error':>16}{'bits/weight':>13}")
    for bits in (8, 4):
        for name, gran, size in (("per-tensor", "tensor", None), ("per-channel", "channel", None),
                                 ("group of 128", "group", 128)):
            err = relative_rms_error(w, fake_quantize(w, bits, gran, size or 128))
            print(f"{bits:>5}{name:>16}{err:>16.5f}{bits_per_weight(bits, size):>13.3f}")


if __name__ == "__main__":
    one_row()
    granularity()
