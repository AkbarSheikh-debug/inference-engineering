# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
"""Static versus continuous batching, as scheduling arithmetic.

Toy models with equal-cost steps. They show the mechanism and its size in a
stated example; they are not the throughput of any engine.
"""

EXAMPLE_LENGTHS = {"A": 3, "B": 10, "C": 5, "D": 6, "E": 4, "F": 5, "G": 3}
EXAMPLE_SLOTS = 4

Schedule = list[list[tuple[str, int, int]]]  # per slot: (request, start step, end step)


def static_schedule(lengths: dict[str, int], slots: int) -> Schedule:
    """Batches of `slots` requests, each lasting as long as its longest member."""
    rows: Schedule = [[] for _ in range(slots)]
    names = list(lengths)
    t = 0
    for i in range(0, len(names), slots):
        batch = names[i:i + slots]
        for slot, name in enumerate(batch):
            rows[slot].append((name, t, t + lengths[name]))
        t += max(lengths[n] for n in batch)
    return rows


def continuous_schedule(lengths: dict[str, int], slots: int) -> Schedule:
    """Each queued request takes the slot that frees earliest."""
    rows: Schedule = [[] for _ in range(slots)]
    free_at = [0] * slots
    for name, n in lengths.items():
        slot = min(range(slots), key=lambda s: (free_at[s], s))
        rows[slot].append((name, free_at[slot], free_at[slot] + n))
        free_at[slot] += n
    return rows


def makespan(rows: Schedule) -> int:
    return max(end for row in rows for _, _, end in row)


def utilisation(rows: Schedule) -> float:
    busy = sum(end - start for row in rows for _, start, end in row)
    return busy / (len(rows) * makespan(rows))


def static_window_tokens(wants: list[int], rate: float, window: float) -> float:
    """Tokens emitted in a window when every request keeps its slot until the batch ends.

    A finished request emits nothing more but is not replaced, so each request
    contributes min(tokens wanted, rate x window).
    """
    return sum(min(w, rate * window) for w in wants)


def continuous_window_tokens(slots: int, rate: float, window: float) -> float:
    """Tokens emitted in a window when a freed slot is refilled immediately.

    Assumes the queue never runs dry, so every slot is busy for the whole window.
    """
    return slots * rate * window


def token_budget_split(budget: int, decode_seqs: int, chunk: int) -> dict[str, int]:
    """Spend one forward pass's token budget: decodes first (1 token each), then whole prefill chunks."""
    prefill_chunks = (budget - decode_seqs) // chunk
    prefill = prefill_chunks * chunk
    return {
        "decode": decode_seqs,
        "prefill_chunks": prefill_chunks,
        "prefill": prefill,
        "spare": budget - decode_seqs - prefill,
    }
