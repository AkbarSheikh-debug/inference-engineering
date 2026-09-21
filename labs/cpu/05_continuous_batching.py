# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
"""Lab 05: static versus continuous batching, and what a token budget buys.

    python labs/cpu/05_continuous_batching.py

Toy schedules with equal-cost decode steps: they show the mechanism and its
size in a stated example, not the throughput of any engine.
"""

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


def schedules() -> None:
    print(f"requests (decode steps): {EXAMPLE_LENGTHS}, {EXAMPLE_SLOTS} slots\n")
    for name, fn in (("static", static_schedule), ("continuous", continuous_schedule)):
        rows = fn(EXAMPLE_LENGTHS, EXAMPLE_SLOTS)
        print(f"{name}: {makespan(rows)} steps, slots busy {utilisation(rows):.0%}")
        for i, row in enumerate(rows, 1):
            print(f"  slot {i}: " + "  ".join(f"{n}[{a}-{b}]" for n, a, b in row))
    print()


def window() -> None:
    wants, rate, secs = [32, 80, 200, 400], 40, 10
    st = static_window_tokens(wants, rate, secs)
    ct = continuous_window_tokens(len(wants), rate, secs)
    print(f"ten-second window, 4 slots at {rate} tokens/s, requests wanting {wants} tokens")
    print(f"  static:     {st:,.0f} tokens\n  continuous: {ct:,.0f} tokens  ({ct / st:.1f}x)\n")


def budget() -> None:
    r = token_budget_split(budget=4096, decode_seqs=24, chunk=1024)
    print("token budget 4096, 24 decoding sequences, prefill chunks of 1024:")
    print(f"  decode {r['decode']} tokens ({r['decode'] / 4096:.1%} of the budget), "
          f"{r['prefill_chunks']} prefill chunks = {r['prefill']:,} tokens, spare {r['spare']:,}")


if __name__ == "__main__":
    schedules()
    window()
    budget()
