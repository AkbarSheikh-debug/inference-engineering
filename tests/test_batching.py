# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
import pytest

from ie.batching import (
    EXAMPLE_LENGTHS,
    EXAMPLE_SLOTS,
    continuous_schedule,
    continuous_window_tokens,
    makespan,
    static_schedule,
    static_window_tokens,
    token_budget_split,
    utilisation,
)


@pytest.mark.parametrize("fn", [static_schedule, continuous_schedule])
def test_schedules_run_every_request_once_without_overlap(fn):
    rows = fn(EXAMPLE_LENGTHS, EXAMPLE_SLOTS)
    seen = sorted(name for row in rows for name, _, _ in row)
    assert seen == sorted(EXAMPLE_LENGTHS)
    for row in rows:
        for (_, _, end), (_, start, _) in zip(row, row[1:]):
            assert start >= end
        for name, start, end in row:
            assert end - start == EXAMPLE_LENGTHS[name]


def test_continuous_beats_static_on_identical_work():
    s = static_schedule(EXAMPLE_LENGTHS, EXAMPLE_SLOTS)
    c = continuous_schedule(EXAMPLE_LENGTHS, EXAMPLE_SLOTS)
    assert (makespan(s), makespan(c)) == (15, 10)
    assert round(utilisation(s), 2) == 0.60 and round(utilisation(c), 2) == 0.90


def test_ten_second_window_example():
    wants = [32, 80, 200, 400]
    st = static_window_tokens(wants, rate=40, window=10)
    ct = continuous_window_tokens(len(wants), rate=40, window=10)
    assert (st, ct) == (712, 1600)
    assert round(ct / st, 1) == 2.2


def test_static_never_credits_more_than_a_request_wants():
    assert static_window_tokens([10], rate=40, window=10) == 10


def test_token_budget_example():
    r = token_budget_split(budget=4096, decode_seqs=24, chunk=1024)
    assert r == {"decode": 24, "prefill_chunks": 3, "prefill": 3072, "spare": 1000}
    assert round(24 / 4096 * 100, 1) == 0.6
