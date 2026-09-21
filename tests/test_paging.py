# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
import pytest

from ie.paging import BlockAllocator, OutOfBlocks


def test_blocks_are_allocated_only_when_the_last_one_fills():
    a = BlockAllocator(n_blocks=8, block_size=4)
    a.add_sequence(0)
    a.append_tokens(0, 1)
    assert len(a.tables[0]) == 1
    a.append_tokens(0, 3)  # fills the first block exactly
    assert len(a.tables[0]) == 1
    a.append_tokens(0, 1)
    assert len(a.tables[0]) == 2
    assert a.lengths[0] == 5


def test_waste_is_bounded_by_one_block_per_sequence():
    a = BlockAllocator(n_blocks=64, block_size=16)
    for i, n in enumerate([1, 15, 16, 17, 100]):
        a.add_sequence(i)
        a.append_tokens(i, n)
    assert a.wasted_tokens() < 5 * 16
    assert a.wasted_tokens() == 15 + 1 + 0 + 15 + 12


def test_freed_blocks_are_reused():
    a = BlockAllocator(n_blocks=2, block_size=4)
    a.add_sequence(0)
    a.append_tokens(0, 8)
    assert a.free_blocks == 0
    a.free_sequence(0)
    assert a.free_blocks == 2
    a.add_sequence(1)
    a.append_tokens(1, 8)


def test_running_out_of_blocks_raises():
    a = BlockAllocator(n_blocks=1, block_size=4)
    a.add_sequence(0)
    with pytest.raises(OutOfBlocks):
        a.append_tokens(0, 5)


def test_block_tables_do_not_overlap():
    a = BlockAllocator(n_blocks=16, block_size=4)
    for i in range(3):
        a.add_sequence(i)
        a.append_tokens(i, 10)
    blocks = [b for t in a.tables.values() for b in t]
    assert len(blocks) == len(set(blocks))


def test_duplicate_sequence_is_rejected():
    a = BlockAllocator(n_blocks=4, block_size=4)
    a.add_sequence(0)
    with pytest.raises(ValueError):
        a.add_sequence(0)
