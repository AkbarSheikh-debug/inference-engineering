# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
"""Exact tiled attention with online softmax, and a first-order HBM traffic model.

The tiled routine is the FlashAttention algorithm (Dao et al., 2022) in NumPy:
it never forms the n x n score matrix, only one block of scores at a time, and
keeps a running maximum m, running sum l and running output o per query row.
It is float64 so equality with the direct computation can be checked to 1e-12.
Causal masking is omitted; it changes which blocks are needed, not the argument.
"""

from __future__ import annotations

from math import ceil, sqrt

import numpy as np


def naive_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """softmax(q k^T / sqrt(d)) v, materialising the full score matrix."""
    s = q @ k.T / sqrt(q.shape[1])
    s = s - s.max(axis=1, keepdims=True)
    p = np.exp(s)
    return (p / p.sum(axis=1, keepdims=True)) @ v


def flash_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    block_q: int = 64,
    block_kv: int = 64,
    reverse_kv: bool = False,
) -> tuple[np.ndarray, int]:
    """Tiled attention. Returns the output and the largest score block ever held (elements)."""
    n, d = q.shape
    out = np.empty_like(q)
    largest_block = 0
    kv_starts = list(range(0, k.shape[0], block_kv))
    if reverse_kv:
        kv_starts.reverse()
    for i in range(0, n, block_q):
        qi = q[i:i + block_q]
        m = np.full(qi.shape[0], -np.inf)
        l = np.zeros(qi.shape[0])
        o = np.zeros((qi.shape[0], v.shape[1]))
        for j in kv_starts:
            s = qi @ k[j:j + block_kv].T / sqrt(d)  # one tile of scores, "in SRAM"
            largest_block = max(largest_block, s.size)
            m_new = np.maximum(m, s.max(axis=1))
            alpha = np.exp(m - m_new)  # rescales everything accumulated so far
            p = np.exp(s - m_new[:, None])
            l = l * alpha + p.sum(axis=1)
            o = o * alpha[:, None] + p @ v[j:j + block_kv]
            m = m_new
        out[i:i + block_q] = o / l[:, None]
    return out, largest_block


def online_softmax_state(score_blocks: list[list[float]]) -> list[tuple[float, float, float]]:
    """Running (m, l, alpha) of one row after each block of scores."""
    m, l = -np.inf, 0.0
    trace = []
    for block in score_blocks:
        m_new = max(m, max(block))
        alpha = float(np.exp(m - m_new)) if np.isfinite(m) else 0.0
        l = l * alpha + sum(np.exp(s - m_new) for s in block)
        m = m_new
        trace.append((float(m), float(l), alpha))
    return trace


def hbm_elements_standard(n: int, d: int) -> int:
    """Elements moved: read Q and K, write S, read S, write P, read P, read V, write O."""
    return 4 * n * n + 4 * n * d


def hbm_elements_flash(n: int, d: int, block_q: int) -> int:
    """Elements moved: Q once, O once, and K and V re-read once per query block."""
    return 2 * n * d + 2 * n * d * ceil(n / block_q)


def score_matrix_bytes(n: int, heads: int = 1, batch: int = 1, elem_bytes: int = 2) -> int:
    """Storage for the full score matrix that standard attention materialises."""
    return batch * heads * n * n * elem_bytes
