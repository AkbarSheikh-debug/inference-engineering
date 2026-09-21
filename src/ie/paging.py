# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
"""A toy paged KV-cache allocator.

Illustrates the idea behind PagedAttention (Kwon et al., arXiv:2309.06180):
hand out fixed-size blocks on demand and keep a per-sequence block table,
instead of reserving one contiguous max-length slab per sequence. This is a
bookkeeping model, not a GPU implementation.
"""

import math
from dataclasses import dataclass, field

import numpy as np


class OutOfBlocks(Exception):
    pass


@dataclass
class BlockAllocator:
    n_blocks: int
    block_size: int
    _free: list[int] = field(init=False)
    tables: dict[int, list[int]] = field(default_factory=dict)
    lengths: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._free = list(range(self.n_blocks - 1, -1, -1))

    @property
    def free_blocks(self) -> int:
        return len(self._free)

    def add_sequence(self, seq_id: int) -> None:
        if seq_id in self.tables:
            raise ValueError(f"sequence {seq_id} already exists")
        self.tables[seq_id] = []
        self.lengths[seq_id] = 0

    def append_tokens(self, seq_id: int, n: int = 1) -> None:
        """Grow a sequence by n tokens, allocating blocks only when the last one fills."""
        for _ in range(n):
            if self.lengths[seq_id] == len(self.tables[seq_id]) * self.block_size:
                if not self._free:
                    raise OutOfBlocks(f"no free block for sequence {seq_id}")
                self.tables[seq_id].append(self._free.pop())
            self.lengths[seq_id] += 1

    def free_sequence(self, seq_id: int) -> None:
        self._free.extend(reversed(self.tables.pop(seq_id)))
        del self.lengths[seq_id]

    def wasted_tokens(self) -> int:
        """Slots allocated but not yet holding a token (internal fragmentation)."""
        return sum(len(t) * self.block_size - self.lengths[s] for s, t in self.tables.items())


def compare_policies(
    max_len: int = 4096,
    block_size: int = 16,
    pool_seqs: int = 64,
    requests: int = 2000,
    seed: int = 0,
) -> dict:
    """Contiguous max-length reservation versus paged blocks, on synthetic requests.

    Request lengths are seeded lognormal (median 400 tokens, sigma 0.8), clipped to
    [16, max_len]. Lengths are known at admission, a simplification real engines lack.
    """
    rng = np.random.default_rng(seed)
    lengths = np.clip(rng.lognormal(mean=math.log(400), sigma=0.8, size=requests), 16, max_len)
    lengths = lengths.astype(int).tolist()

    pool_tokens = pool_seqs * max_len
    alloc = BlockAllocator(n_blocks=pool_tokens // block_size, block_size=block_size)
    admitted = 0
    for i, n in enumerate(lengths):
        if math.ceil(n / block_size) > alloc.free_blocks:
            break
        alloc.add_sequence(i)
        alloc.append_tokens(i, n)
        admitted += 1

    contiguous = pool_tokens // max_len  # each request reserves max_len tokens
    return {
        "mean_length": float(np.mean(lengths)),
        "pool_tokens": pool_tokens,
        "contiguous_admitted": contiguous,
        "contiguous_used": sum(lengths[:contiguous]),
        "contiguous_waste": contiguous * max_len - sum(lengths[:contiguous]),
        "paged_admitted": admitted,
        "paged_used": sum(lengths[:admitted]),
        "paged_waste": alloc.wasted_tokens(),
    }
