# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
"""Lab 07: tiled attention with online softmax is exact, and what it does to traffic.

    python labs/cpu/07_flash_attention.py

Part 1 traces one row of scores through two blocks and shows the rescaling factor.
Part 2 runs the tiled algorithm against direct attention. Part 3 compares HBM
traffic and score-matrix storage as the sequence grows.
"""

import numpy as np

from ie.attention import (
    flash_attention,
    hbm_elements_flash,
    hbm_elements_standard,
    naive_attention,
    online_softmax_state,
    score_matrix_bytes,
)
from ie.units import GiB, MiB

ROW = [[0.515, 0.385, 0.405, 0.435], [0.560, 0.300, 0.370, 0.320]]


def trace() -> None:
    print("one row of scores in two blocks of four")
    for i, (m, l, alpha) in enumerate(online_softmax_state(ROW), 1):
        note = "first block, nothing to rescale" if i == 1 else f"rescale factor alpha = {alpha:.3f}"
        print(f"  after block {i}: m = {m:.3f}, l = {l:.3f}, {note}")
    print()


def exactness() -> None:
    rng = np.random.default_rng(0)
    n, d = 200, 32
    q, k, v = (rng.normal(size=(n, d)) for _ in range(3))
    ref = naive_attention(q, k, v)
    print(f"n = {n}, d = {d}: largest difference from direct attention")
    for bq, bkv in ((64, 64), (37, 29), (200, 200), (7, 3)):
        out, block = flash_attention(q, k, v, bq, bkv)
        print(f"  blocks {bq:>3} x {bkv:<3}: {np.max(np.abs(out - ref)):.1e}   largest score block {block:>6} of {n * n}")
    out, _ = flash_attention(q, k, v, 37, 29, reverse_kv=True)
    print(f"  reversed block order: {np.max(np.abs(out - ref)):.1e}\n")


def traffic() -> None:
    d, bq = 128, 128
    print(f"head dimension {d}, query block {bq}, FP16")
    print(f"{'n':>8}{'standard MiB':>14}{'tiled MiB':>11}{'ratio':>7}{'S per head GiB':>16}")
    for n in (1024, 4096, 16384, 65536):
        std = hbm_elements_standard(n, d) * 2
        fl = hbm_elements_flash(n, d, bq) * 2
        print(f"{n:>8}{std / MiB:>14,.0f}{fl / MiB:>11,.0f}{std / fl:>7.2f}{score_matrix_bytes(n) / GiB:>16.2f}")


if __name__ == "__main__":
    trace()
    exactness()
    traffic()
