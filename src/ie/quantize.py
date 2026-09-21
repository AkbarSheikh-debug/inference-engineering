# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
"""Integer quantization: affine mapping, and scale granularity.

Integer quantization maps real values to integers with a scale s and an
optional zero-point z:

    q = round(x / s) + z          x_hat = s * (q - z)

Symmetric quantization fixes z = 0. Asymmetric quantization picks z so the
integer range covers an interval that is not centred on zero.
"""

import numpy as np


def qmax(bits: int) -> int:
    """Largest magnitude of a symmetric signed grid: 127 for INT8, 7 for INT4."""
    return 2 ** (bits - 1) - 1


def symmetric_scale(x: np.ndarray, bits: int = 8) -> float:
    return float(np.max(np.abs(x))) / qmax(bits)


def asymmetric_params(x: np.ndarray, bits: int = 8) -> tuple[float, int]:
    """Scale and zero-point that stretch the unsigned range [0, 2^bits - 1] over [min, max]."""
    lo, hi = float(np.min(x)), float(np.max(x))
    scale = (hi - lo) / (2**bits - 1)
    return scale, int(round(-lo / scale))


def quantize(x: np.ndarray, scale: float, zero: int = 0, lo: int = -127, hi: int = 127) -> np.ndarray:
    return np.clip(np.rint(x / scale) + zero, lo, hi).astype(np.int32)


def dequantize(q: np.ndarray, scale: float, zero: int = 0) -> np.ndarray:
    return scale * (q - zero)


def fake_quantize(w: np.ndarray, bits: int, granularity: str, group_size: int = 128) -> np.ndarray:
    """Symmetric quantize-then-dequantize of a 2-D weight matrix.

    granularity: "tensor" (one scale), "channel" (one per row), or "group"
    (one per `group_size` consecutive weights within a row).
    """
    m = qmax(bits)
    rows, cols = w.shape
    if granularity == "tensor":
        blocks = w.reshape(1, -1)
    elif granularity == "channel":
        blocks = w
    elif granularity == "group":
        if cols % group_size:
            raise ValueError("columns must be a multiple of group_size")
        blocks = w.reshape(-1, group_size)
    else:
        raise ValueError(f"unknown granularity {granularity!r}")
    scale = np.max(np.abs(blocks), axis=1, keepdims=True) / m
    scale = np.where(scale == 0, 1.0, scale)
    q = np.clip(np.rint(blocks / scale), -m, m)
    return (q * scale).reshape(rows, cols)


def relative_rms_error(w: np.ndarray, w_hat: np.ndarray) -> float:
    return float(np.linalg.norm(w - w_hat) / np.linalg.norm(w))


def synthetic_weights(rows: int = 256, cols: int = 1024, seed: int = 0) -> np.ndarray:
    """A weight matrix with the two features that make granularity matter.

    Rows have different magnitudes (log-normal row scales), and 0.2 percent of
    entries are outliers 20 times larger. Synthetic: it shows the mechanism, and
    is not a measurement of any real model.
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(rows, cols)) * rng.lognormal(mean=0.0, sigma=0.5, size=(rows, 1))
    outliers = rng.random(size=w.shape) < 0.002
    return np.where(outliers, w * 20.0, w)
