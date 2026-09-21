# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
"""Speculative decoding: the accept/reject rule, its exactness, and expected speedup.

Follows Leviathan, Kalman and Matias (arXiv:2211.17192) and Chen et al.
(arXiv:2302.01318). p is the target model's distribution over the next token,
q the draft model's. Nothing here needs a real model: the rule acts on
distributions, which is exactly why it is provably exact for any draft.
"""

from __future__ import annotations

import numpy as np


def accept_probability(p_x: float, q_x: float) -> float:
    return min(1.0, p_x / q_x)


def residual_distribution(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalised max(0, p - q) and its normaliser Z, which equals the rejection probability."""
    r = np.maximum(0.0, p - q)
    z = float(r.sum())
    return (r / z if z > 0 else p.copy()), z


def speculative_step(p: np.ndarray, q: np.ndarray, rng: np.random.Generator) -> tuple[int, bool]:
    """Draft one token from q, then accept it or resample from the residual."""
    x = int(rng.choice(len(q), p=q))
    if rng.random() < accept_probability(p[x], q[x]):
        return x, True
    res, _ = residual_distribution(p, q)
    return int(rng.choice(len(p), p=res)), False


def verify_block(
    p_rows: np.ndarray,
    q_rows: np.ndarray,
    draft_tokens: list[int],
    rng: np.random.Generator,
) -> list[int]:
    """Verify k drafted tokens against one target pass.

    p_rows has k + 1 rows (the last is the target's distribution after all k
    drafts); q_rows has k rows. Tokens are accepted in order. At the first
    rejection a token is drawn from the residual and verification stops. If all
    k are accepted, a bonus token is drawn from the target's last row.
    """
    out: list[int] = []
    for i, x in enumerate(draft_tokens):
        if rng.random() < accept_probability(p_rows[i][x], q_rows[i][x]):
            out.append(x)
        else:
            res, _ = residual_distribution(p_rows[i], q_rows[i])
            out.append(int(rng.choice(len(res), p=res)))
            return out
    out.append(int(rng.choice(p_rows.shape[1], p=p_rows[len(draft_tokens)])))
    return out


def expected_tokens(alpha: float, k: int) -> float:
    """Expected tokens per target pass if each drafted token is accepted independently with prob alpha."""
    if alpha >= 1.0:
        return float(k + 1)
    return (1.0 - alpha ** (k + 1)) / (1.0 - alpha)


def speedup(alpha: float, k: int, draft_cost: float = 0.0) -> float:
    """Wall-clock speedup over plain decoding.

    draft_cost is the cost of one draft step as a fraction of one target step. One
    iteration costs k draft steps plus one target pass.
    """
    return expected_tokens(alpha, k) / (k * draft_cost + 1.0)


def simulate_tokens_per_pass(alpha: float, k: int, trials: int, rng: np.random.Generator) -> float:
    """Monte Carlo of the same quantity, by drawing the acceptance of each drafted token."""
    total = 0
    for _ in range(trials):
        accepted = 0
        while accepted < k and rng.random() < alpha:
            accepted += 1
        total += accepted + 1  # the resampled or bonus token
    return total / trials
