# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
"""Fail if an article quotes a number the code does not produce."""

from pathlib import Path

import numpy as np

from ie.hardware import A100_80GB as A100, H100_80GB as H100
from ie.kv import kv_bytes, kv_bytes_per_token, kv_pool_bytes, max_sequences, weight_bytes
from ie.models import LLAMA2_7B, LLAMA3_8B, LLAMA3_70B
from ie.paging import compare_policies
from ie.quant import bytes_per_param
from ie.roofline import decode_step, ridge_batch, weight_intensity
from ie.tinylm import TinyConfig, TinyLM
from ie.units import GB, GiB, KiB, to_gib

ROOT = Path(__file__).resolve().parents[1]


def article(name: str) -> str:
    return (ROOT / "curriculum" / name).read_text(encoding="utf-8")


def test_article_02_ceiling_table_and_ridges():
    t = article("02-why-decode-is-slow.md")
    for label, bpp in (("BF16", bytes_per_param(16)), ("INT8", bytes_per_param(8)),
                       ("INT4, group size 128", bytes_per_param(4, 128))):
        h = decode_step(H100, LLAMA3_8B, bytes_per_param=bpp).seconds
        a = decode_step(A100, LLAMA3_8B, bytes_per_param=bpp).seconds
        row = (f"| {label} | {weight_bytes(LLAMA3_8B, bpp) / GB:.2f} GB | {h * 1e3:.2f} | {1 / h:.0f} "
               f"| {a * 1e3:.2f} | {1 / a:.0f} |")
        assert row in t, row
    for gpu in (A100, H100):
        assert f"{gpu.ridge():.0f} FLOP/byte" in t
    assert f"{ridge_batch(H100, 1.0):.0f} instead of {ridge_batch(H100):.0f}" in t


def test_article_02_utilisation_and_prefill():
    t = article("02-why-decode-is-slow.md")
    for batch in (1, 16, 64):
        share = H100.attainable(weight_intensity(batch)) / H100.peak_bf16
        assert f"{share * 100:.1f} percent" in t, batch
    assert f"**{decode_step(H100, LLAMA3_8B, batch=2048).seconds * 1e3:.1f} ms**" in t
    step = decode_step(H100, LLAMA3_8B)
    assert f"{step.memory_s * 1e3:.2f} ms" in t
    assert f"{weight_bytes(LLAMA3_8B) / GB:.2f} GB" in t


def test_article_03_kv_tables():
    t = article("03-kv-cache.md")
    rows = (("Llama-2-7B (MHA)", LLAMA2_7B), ("Llama-3-8B (GQA)", LLAMA3_8B), ("Llama-3-70B (GQA)", LLAMA3_70B))
    for label, m in rows:
        row = f"| {label} | {m.n_kv_heads} | {kv_bytes_per_token(m) // KiB} | {to_gib(kv_bytes(m, 8192)):.2f} |"
        assert row in t, row
    for label, heads in (("MHA", 32), ("GQA (real Llama-3-8B)", 8), ("MQA", 1)):
        m = LLAMA3_8B.with_kv_heads(heads)
        row = f"| {label} | {heads} | {kv_bytes_per_token(m) // KiB} | {max_sequences(H100, m, 8192)} |"
        assert row in t, row


def test_article_03_capacity_and_crossover():
    t = article("03-kv-cache.md")
    assert f"{to_gib(weight_bytes(LLAMA3_8B)):.2f} GiB" in t
    assert f"**{to_gib(kv_pool_bytes(H100, LLAMA3_8B)):.2f} GiB pool**" in t
    assert f"**{max_sequences(H100, LLAMA3_8B, 8192)} concurrent sequences**" in t
    crossover = weight_bytes(LLAMA3_8B) / kv_bytes_per_token(LLAMA3_8B)
    assert f"{crossover:,.0f}" in t
    for batch in (16, 64):
        assert f"{crossover / batch:,.0f}" in t


def test_article_03_tiny_decoder_and_paging():
    t = article("03-kv-cache.md")
    model = TinyLM(TinyConfig())
    prompt = [int(x) for x in np.random.default_rng(1).integers(0, 64, size=32)]
    slow = model.generate(prompt, 32, use_cache=False).stats
    fast = model.generate(prompt, 32, use_cache=True).stats
    assert f"| Positions processed | {slow.positions_processed:,} | {fast.positions_processed} |" in t
    assert f"| Matmul FLOPs | {slow.flops:,} | {fast.flops:,} |" in t
    assert f"{slow.flops / fast.flops:.1f} times" in t

    r = compare_policies()
    pool = r["pool_tokens"]
    assert f"| Requests admitted | {r['contiguous_admitted']} | {r['paged_admitted']} |" in t
    assert (f"| Pool slots holding tokens | {r['contiguous_used'] / pool:.1%} | "
            f"{r['paged_used'] / pool:.1%} |") in t
    assert f"mean about {r['mean_length']:.0f} tokens" in t


def test_article_01_position_counts():
    t = article("01-request-lifecycle.md")
    assert "1,520" in t and "63" in t
    assert f"{LLAMA3_8B.vocab:,}" in t
    assert 2 * LLAMA3_8B.params / 1e9 > 16  # "about 16 GFLOP per token"
    assert "16 GB" in t and round(weight_bytes(LLAMA3_8B) / GB) == 16


def test_article_04_one_row_table():
    from ie.quantize import asymmetric_params, dequantize, quantize, symmetric_scale

    t = article("04-quantization.md")
    row = np.array([-0.10, 0.05, 0.22, 0.41, 0.63, 0.86, 0.34, 0.12])
    s_sym = symmetric_scale(row)
    s_asym, z = asymmetric_params(row)
    q_sym, q_asym = quantize(row, s_sym), quantize(row, s_asym, z, 0, 255)
    d_sym, d_asym = dequantize(q_sym, s_sym), dequantize(q_asym, s_asym, z)
    for i, x in enumerate(row):
        line = (f"| {x:.2f} | {q_sym[i]} | {d_sym[i]:.4f} | {abs(x - d_sym[i]):.4f} | {q_asym[i]} | "
                f"{d_asym[i]:.4f} | {abs(x - d_asym[i]):.4f} |")
        assert line in t, line
    assert f"| worst | | | {np.max(np.abs(row - d_sym)):.4f} | | | {np.max(np.abs(row - d_asym)):.4f} |" in t
    assert f"{s_sym:.6f}" in t and f"{s_asym:.6f}" in t and f"zero-point {z}" in t
    assert f"**{s_sym / s_asym:.2f}**" in t


def test_article_04_granularity_table_and_bytes():
    from ie.quant import bits_per_weight
    from ie.quantize import fake_quantize, relative_rms_error, synthetic_weights

    t = article("04-quantization.md")
    w = synthetic_weights()
    for bits in (8, 4):
        for label, gran, size in (("per-tensor", "tensor", None), ("per-channel", "channel", None),
                                  ("group of 128", "group", 128)):
            err = relative_rms_error(w, fake_quantize(w, bits, gran, size or 128))
            line = f"| {bits} | {label} | {err:.5f} | {bits_per_weight(bits, size):.3f} |"
            assert line in t, line
    assert f"{bytes_per_param(4, 128):.3f} bytes" in t
    assert f"{2 / bytes_per_param(4, 128):.1f}x" in t


def test_article_05_schedule_window_and_budget():
    from ie.batching import (EXAMPLE_LENGTHS, EXAMPLE_SLOTS, continuous_schedule, continuous_window_tokens,
                             makespan, static_schedule, static_window_tokens, token_budget_split, utilisation)

    t = article("05-continuous-batching.md")
    s = static_schedule(EXAMPLE_LENGTHS, EXAMPLE_SLOTS)
    c = continuous_schedule(EXAMPLE_LENGTHS, EXAMPLE_SLOTS)
    assert f"| Steps to finish all seven | {makespan(s)} | {makespan(c)} |" in t
    assert f"| Slots busy | {utilisation(s):.0%} | {utilisation(c):.0%} |" in t
    assert sum(EXAMPLE_LENGTHS.values()) == 36 and "36 slot-steps" in t
    st = static_window_tokens([32, 80, 200, 400], 40, 10)
    ct = continuous_window_tokens(4, 40, 10)
    assert f"**{st:,.0f}**" in t and f"**{ct:,.0f}**" in t and f"**{ct / st:.1f}**" in t
    r = token_budget_split(4096, 24, 1024)
    assert f"**{r['decode'] / 4096 * 100:.1f} percent**" in t
    assert f"{r['prefill']:,}" in t and f"**{r['spare']:,} tokens**" in t
    assert f"{max_sequences(H100, LLAMA3_8B, 8192)} sequences" in t
