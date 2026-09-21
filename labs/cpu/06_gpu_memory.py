# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
"""Lab 06: coalescing and tiling, the two levers of on-chip memory.

    python labs/cpu/06_gpu_memory.py

Part 1: how many 32-byte sectors a warp's 32 loads touch, for a range of strides.
Part 2: HBM-side traffic and arithmetic intensity of a tiled matrix multiply as the
tile grows, with the closed form checked against an explicit walk of the tile grid.
"""

from ie.gpumem import (
    LINE,
    SECTOR,
    gemm_flops,
    gemm_intensity,
    gemm_read_bytes,
    gemm_read_bytes_counted,
    warp_access,
)
from ie.units import GiB


def coalescing() -> None:
    print(f"warp of 32 threads, 4-byte loads, {SECTOR}-byte sectors, {LINE}-byte lines\n")
    print(f"{'stride':>8}{'sectors':>9}{'useful B':>10}{'moved B':>9}{'efficiency':>12}")
    for stride in (4, 8, 16, 32, 64, 128, 256):
        a = warp_access(stride)
        print(f"{stride:>7}B{a.sectors:>9}{a.useful_bytes:>10}{a.moved_bytes:>9}{a.efficiency:>12.1%}")
    one = warp_access(128, elem_bytes=1)
    print(f"\n1-byte loads at stride 128: {one.sectors} sectors, efficiency {one.efficiency:.3%}\n")


def tiling() -> None:
    m = n = k = 4096
    b = 2
    print(f"C = A x B with m = n = k = {m}, FP16 ({b} bytes per element), {gemm_flops(m, n, k) / 1e9:.1f} GFLOP\n")
    print(f"{'tile':>8}{'reads GiB':>11}{'FLOP/byte':>11}")
    for t in (1, 16, 32, 64, 128, 256):
        reads = gemm_read_bytes(m, n, k, t, t, b)
        print(f"{t:>4}x{t:<3}{reads / GiB:>11.3f}{gemm_intensity(m, n, k, t, t, b):>11.1f}")
    small = (100, 90, 50, 32, 24)
    assert gemm_read_bytes(*small) == gemm_read_bytes_counted(*small)
    print("\nclosed form matches an explicit walk of the tile grid, including tiles that do not divide the matrix")


if __name__ == "__main__":
    coalescing()
    tiling()
