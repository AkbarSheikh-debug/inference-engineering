# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
import numpy as np
import pytest

from ie.tinylm import TinyConfig, TinyLM


@pytest.fixture(scope="module")
def model():
    return TinyLM(TinyConfig())


def _prompt(n, vocab=64, seed=1):
    return [int(t) for t in np.random.default_rng(seed).integers(0, vocab, size=n)]


def test_cache_does_not_change_the_output(model):
    prompt = _prompt(32)
    slow = model.generate(prompt, 32, use_cache=False)
    fast = model.generate(prompt, 32, use_cache=True)
    assert slow.tokens == fast.tokens
    np.testing.assert_allclose(slow.logits, fast.logits, atol=1e-9)


def test_position_counts_match_the_closed_forms(model):
    p, n = 32, 32
    slow = model.generate(_prompt(p), n, use_cache=False).stats
    fast = model.generate(_prompt(p), n, use_cache=True).stats
    assert slow.positions_processed == n * p + n * (n - 1) // 2  # 1520
    assert fast.positions_processed == p + n - 1  # 63
    assert slow.flops > 20 * fast.flops


def test_cache_holds_one_entry_per_token_seen(model):
    res = model.generate(_prompt(10), 5, use_cache=True)
    assert res.cache.length() == 10 + 5 - 1  # the last token is never fed back


def test_attention_is_causal(model):
    ids = np.array(_prompt(12))
    full = model.forward(ids)
    prefix = model.forward(ids[:7])
    np.testing.assert_allclose(full[:7], prefix, atol=1e-9)


def test_chunked_prefill_matches_one_shot(model):
    ids = np.array(_prompt(12))
    from ie.tinylm import KVCache

    cache = KVCache(model.cfg.n_layers)
    a = model.forward(ids[:5], cache)
    b = model.forward(ids[5:], cache)
    np.testing.assert_allclose(np.concatenate([a, b]), model.forward(ids), atol=1e-9)


def test_gqa_cache_is_smaller_than_mha_cache():
    gqa = TinyLM(TinyConfig(n_kv_heads=2)).generate(_prompt(16), 4).cache
    mha = TinyLM(TinyConfig(n_kv_heads=4)).generate(_prompt(16), 4).cache
    assert mha.nbytes(2) == 2 * gqa.nbytes(2)


def test_invalid_head_grouping_is_rejected():
    with pytest.raises(ValueError):
        TinyConfig(n_heads=4, n_kv_heads=3)
