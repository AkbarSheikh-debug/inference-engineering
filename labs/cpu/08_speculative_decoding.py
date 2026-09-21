# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
"""Lab 08: speculative decoding is exact, and how much it can speed things up.

    python labs/cpu/08_speculative_decoding.py

Part 1 runs the accept/reject rule on the four-token example. Part 2 samples a
million tokens through the rule with a deliberately bad draft and compares the
result with the target distribution. Part 3 prints expected tokens per target
pass and the speedup once the draft's own cost is counted.
"""

import numpy as np

from ie.speculative import (
    accept_probability,
    expected_tokens,
    residual_distribution,
    simulate_tokens_per_pass,
    speculative_step,
    speedup,
)

TOKENS = ["sunny", "warm", "cloudy", "beautiful", "nice"]
P = np.array([0.45, 0.22, 0.15, 0.08, 0.10])  # target
Q = np.array([0.10, 0.08, 0.12, 0.30, 0.40])  # draft: confident in the wrong words


def worked_example() -> None:
    print("draft proposed 'beautiful': q = 0.30, target p = 0.08")
    print(f"  accept probability = min(1, p/q) = {accept_probability(0.08, 0.30):.3f}")
    res, z = residual_distribution(P, Q)
    print(f"  on rejection resample from the residual, normaliser Z = {z:.2f}")
    for name, r in zip(TOKENS, res):
        if r > 0:
            print(f"    {name:<9} {r:.3f}")
    print()


def exactness(samples: int = 1_000_000) -> None:
    rng = np.random.default_rng(0)
    counts = np.zeros(len(P))
    accepted = 0
    for _ in range(samples):
        x, ok = speculative_step(P, Q, rng)
        counts[x] += 1
        accepted += ok
    freq = counts / samples
    print(f"{samples:,} tokens through the rule with a poor draft (accepted {accepted / samples:.1%} of drafts)")
    print(f"{'token':<11}{'target p':>10}{'emitted':>10}")
    for name, p, f in zip(TOKENS, P, freq):
        print(f"{name:<11}{p:>10.3f}{f:>10.3f}")
    print(f"largest gap: {np.max(np.abs(freq - P)):.4f}   (sampling noise is about 0.001)\n")


def speedups() -> None:
    rng = np.random.default_rng(1)
    print("expected tokens per target pass, E = (1 - a^(k+1)) / (1 - a); speedup counts a draft step at 5% of a target step")
    print(f"{'alpha':>6}{'k':>3}{'E[tokens]':>11}{'simulated':>11}{'speedup':>9}")
    for alpha in (0.5, 0.7, 0.9):
        for k in (1, 4, 8):
            sim = simulate_tokens_per_pass(alpha, k, 100_000, rng)
            print(f"{alpha:>6}{k:>3}{expected_tokens(alpha, k):>11.3f}{sim:>11.3f}{speedup(alpha, k, 0.05):>9.2f}")


if __name__ == "__main__":
    worked_example()
    exactness()
    speedups()
