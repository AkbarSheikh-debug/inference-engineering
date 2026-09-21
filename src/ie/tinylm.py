# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
"""A tiny decoder-only transformer in NumPy, with and without a KV cache.

Random weights, no training: the point is to count what generation does, not
to produce good text. It has the real structure of a Llama-style block
(RMSNorm, grouped-query attention, gated MLP, residuals, LM head) with one
deliberate omission: no positional encoding. Rotary embeddings rotate q and k
by absolute position, and a cache stores keys that were already rotated, so
adding them changes none of the arguments made here.

Everything runs in float64 so the cached and uncached paths can be compared
to machine precision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TinyConfig:
    vocab: int = 64
    d_model: int = 32
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int = 2
    head_dim: int = 8
    d_ff: int = 64
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be a multiple of n_kv_heads")


@dataclass
class Stats:
    flops: int = 0
    forward_calls: int = 0
    positions_processed: int = 0  # token positions pushed through the network


class KVCache:
    """Per-layer key and value tensors, shape (tokens, kv_heads, head_dim)."""

    def __init__(self, n_layers: int) -> None:
        self.k: list[np.ndarray | None] = [None] * n_layers
        self.v: list[np.ndarray | None] = [None] * n_layers

    def append(self, layer: int, k: np.ndarray, v: np.ndarray) -> None:
        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k, v
        else:
            self.k[layer] = np.concatenate([self.k[layer], k], axis=0)
            self.v[layer] = np.concatenate([self.v[layer], v], axis=0)

    def length(self) -> int:
        first = self.k[0]
        return 0 if first is None else first.shape[0]

    def nbytes(self, dtype_bytes: int) -> int:
        """Storage if held at `dtype_bytes` per element (the arrays are float64)."""
        elems = sum(a.size for a in self.k if a is not None) + sum(a.size for a in self.v if a is not None)
        return elems * dtype_bytes


@dataclass
class GenResult:
    tokens: list[int]
    logits: np.ndarray  # logits used to pick each generated token, (n_new, vocab)
    stats: Stats
    cache: KVCache | None


def _rmsnorm(x: np.ndarray) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


class TinyLM:
    def __init__(self, cfg: TinyConfig = TinyConfig()) -> None:
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)

        def w(rows: int, cols: int) -> np.ndarray:
            return rng.normal(0.0, 1.0 / np.sqrt(rows), size=(rows, cols))

        nh, nkv, hd, d, f = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.d_model, cfg.d_ff
        self.emb = rng.normal(0.0, 0.5, size=(cfg.vocab, d))
        self.layers = [
            {
                "wq": w(d, nh * hd),
                "wk": w(d, nkv * hd),
                "wv": w(d, nkv * hd),
                "wo": w(nh * hd, d),
                "wg": w(d, f),
                "wu": w(d, f),
                "wd": w(f, d),
            }
            for _ in range(cfg.n_layers)
        ]
        self.lm_head = w(d, cfg.vocab)

    def params_read_per_step(self) -> int:
        """Weights streamed from memory on every forward pass.

        The embedding table is excluded: a lookup touches one row, not the table.
        """
        per_layer = sum(m.size for m in self.layers[0].values())
        return per_layer * self.cfg.n_layers + self.lm_head.size

    def forward(self, ids: np.ndarray, cache: KVCache | None = None, stats: Stats | None = None) -> np.ndarray:
        cfg = self.cfg
        nh, nkv, hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        group = nh // nkv
        st = stats if stats is not None else Stats()
        t = len(ids)
        past = cache.length() if cache is not None else 0

        def mm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            st.flops += 2 * a.shape[0] * a.shape[1] * b.shape[1]
            return a @ b

        st.forward_calls += 1
        st.positions_processed += t
        x = self.emb[ids]
        for li, layer in enumerate(self.layers):
            h = _rmsnorm(x)
            q = mm(h, layer["wq"]).reshape(t, nh, hd)
            k = mm(h, layer["wk"]).reshape(t, nkv, hd)
            v = mm(h, layer["wv"]).reshape(t, nkv, hd)
            if cache is not None:
                cache.append(li, k, v)
                keys, values = cache.k[li], cache.v[li]
            else:
                keys, values = k, v
            s = keys.shape[0]
            keys_h = np.repeat(keys, group, axis=1)  # each KV head serves `group` query heads
            values_h = np.repeat(values, group, axis=1)

            scores = np.einsum("thd,shd->hts", q, keys_h) / np.sqrt(hd)
            st.flops += 2 * nh * t * s * hd  # q . k
            visible = np.arange(s)[None, :] <= (past + np.arange(t))[:, None]
            scores = np.where(visible[None, :, :], scores, -np.inf)
            attn = _softmax(scores)
            out = np.einsum("hts,shd->thd", attn, values_h).reshape(t, nh * hd)
            st.flops += 2 * nh * t * s * hd  # attn . v

            x = x + mm(out, layer["wo"])
            h = _rmsnorm(x)
            x = x + mm(_silu(mm(h, layer["wg"])) * mm(h, layer["wu"]), layer["wd"])
        return mm(_rmsnorm(x), self.lm_head)

    def generate(self, prompt: list[int], n_new: int, use_cache: bool = True) -> GenResult:
        """Greedy generation. With a cache: one prefill pass, then one token per step."""
        stats = Stats()
        ids = list(prompt)
        cache = KVCache(self.cfg.n_layers) if use_cache else None
        picked: list[np.ndarray] = []

        if use_cache:
            logits = self.forward(np.array(ids), cache, stats)  # prefill: all prompt positions at once
            for step in range(n_new):
                picked.append(logits[-1])
                ids.append(int(np.argmax(logits[-1])))
                if step + 1 < n_new:
                    logits = self.forward(np.array([ids[-1]]), cache, stats)  # decode: one position
        else:
            for _ in range(n_new):
                logits = self.forward(np.array(ids), None, stats)  # recompute the whole sequence
                picked.append(logits[-1])
                ids.append(int(np.argmax(logits[-1])))
        return GenResult(ids[len(prompt):], np.stack(picked), stats, cache)
