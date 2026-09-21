# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
"""GPU memory arithmetic: coalescing and tiling.

Two models, both first-order:

* A warp's 32 loads are served in 32-byte sectors. Bytes moved is the number of
  distinct sectors touched times 32, whatever the threads actually needed.
* A tiled matrix multiply loads each tile of A and B once per output tile it
  contributes to, so its traffic depends on the tile shape, not just the matrix.
"""

from dataclasses import dataclass
from math import ceil

SECTOR = 32  # bytes: the smallest unit the memory system fetches
LINE = 128  # bytes: four sectors
WARP = 32  # threads


@dataclass(frozen=True)
class WarpAccess:
    stride_bytes: int
    sectors: int
    useful_bytes: int
    moved_bytes: int

    @property
    def efficiency(self) -> float:
        return self.useful_bytes / self.moved_bytes


def warp_access(stride_bytes: int, elem_bytes: int = 4, threads: int = WARP, base: int = 0) -> WarpAccess:
    """One warp load: thread i reads `elem_bytes` at `base + i * stride_bytes`."""
    sectors: set[int] = set()
    for t in range(threads):
        start = base + t * stride_bytes
        sectors.update(range(start // SECTOR, (start + elem_bytes - 1) // SECTOR + 1))
    return WarpAccess(
        stride_bytes=stride_bytes,
        sectors=len(sectors),
        useful_bytes=threads * elem_bytes,
        moved_bytes=len(sectors) * SECTOR,
    )


def gemm_flops(m: int, n: int, k: int) -> int:
    return 2 * m * n * k


def gemm_read_bytes(m: int, n: int, k: int, tile_m: int, tile_n: int, elem_bytes: int = 2) -> int:
    """Bytes of A and B loaded when C is built from tile_m x tile_n output tiles.

    Every output tile loads its `tile_m` rows of A and `tile_n` columns of B in
    full (length k). Summing over the tile grid gives, exactly and for any
    tile size that need not divide the matrix,
    k * elem_bytes * (m * ceil(n / tile_n) + n * ceil(m / tile_m)).
    """
    return k * elem_bytes * (m * ceil(n / tile_n) + n * ceil(m / tile_m))


def gemm_read_bytes_counted(m: int, n: int, k: int, tile_m: int, tile_n: int, elem_bytes: int = 2) -> int:
    """The same quantity, by walking the tile grid. Slow; used to check the closed form."""
    total = 0
    for i in range(0, m, tile_m):
        tm = min(tile_m, m - i)
        for j in range(0, n, tile_n):
            tn = min(tile_n, n - j)
            total += (tm * k + k * tn) * elem_bytes
    return total


def gemm_intensity(m: int, n: int, k: int, tile_m: int, tile_n: int, elem_bytes: int = 2) -> float:
    """FLOP per byte between this tile's staging level and the memory above it (C write included)."""
    moved = gemm_read_bytes(m, n, k, tile_m, tile_n, elem_bytes) + m * n * elem_bytes
    return gemm_flops(m, n, k) / moved
