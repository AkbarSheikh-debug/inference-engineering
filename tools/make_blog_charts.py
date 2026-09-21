# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
"""Draw the figures for the long-form blog post into blog/images/.

    pip install -e ".[blog]"
    python tools/make_blog_charts.py

Every number plotted comes from the library in src/ie, the same code the
articles and tests use, so a figure can never drift from the text. They are
datasheet arithmetic and small simulations, not benchmarks, and each figure
says so in its footer.
"""

import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ie.attention import hbm_elements_flash, hbm_elements_standard, score_matrix_bytes  # noqa: E402
from ie.batching import (EXAMPLE_LENGTHS, EXAMPLE_SLOTS, continuous_schedule, continuous_window_tokens,  # noqa: E402
                         static_schedule, static_window_tokens, utilisation)
from ie.gpumem import gemm_intensity, warp_access  # noqa: E402
from ie.hardware import A100_80GB, H100_80GB  # noqa: E402
from ie.kv import kv_bytes_per_token, kv_pool_bytes, max_sequences, weight_bytes  # noqa: E402
from ie.models import LLAMA2_7B, LLAMA3_8B, LLAMA3_70B  # noqa: E402
from ie.paging import compare_policies  # noqa: E402
from ie.quant import bits_per_weight, bytes_per_param  # noqa: E402
from ie.quantize import (asymmetric_params, dequantize, fake_quantize, quantize, relative_rms_error,  # noqa: E402
                         symmetric_scale, synthetic_weights)
from ie.roofline import decode_step, ridge_batch  # noqa: E402
from ie.speculative import expected_tokens, speculative_step, speedup  # noqa: E402
from ie.units import GB, GiB, KiB  # noqa: E402

OUT = ROOT / "blog" / "images"
DPI = 140
INK, GRAY, GRID = "#111827", "#6b7280", "#e5e7eb"
BLUE, PURPLE, TEAL, ORANGE, RED, GREEN = "#0369a1", "#7c3aed", "#0f766e", "#ea580c", "#dc2626", "#059669"
FOOTER = ("Computed by github.com/AkbarSheikh-debug/inference-engineering  |  "
          "datasheet arithmetic and small simulations, not benchmarks")

plt.rcParams.update({
    "font.size": 12, "axes.edgecolor": GRAY, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
})


def canvas(ncols: int = 1):
    fig, axes = plt.subplots(1, ncols, figsize=(10, 5.63), dpi=DPI)
    return fig, axes


def finish(fig, name: str, title: str, subtitle: str = "", left: float = 0.09) -> None:
    size = 17 if len(title) <= 58 else 14.5 if len(title) <= 70 else 12.5
    fig.text(0.05, 0.965, title, fontsize=size, fontweight="bold", va="top", color=INK)
    if subtitle:
        fig.text(0.05, 0.895, textwrap.fill(subtitle, 112), fontsize=11, color=GRAY, va="top", linespacing=1.35)
    fig.text(0.05, 0.02, FOOTER, fontsize=8.5, color=GRAY)
    fig.subplots_adjust(top=0.75, bottom=0.17, left=left, right=0.96, wspace=0.28)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, facecolor="white")
    plt.close(fig)
    print("wrote", (OUT / name).relative_to(ROOT))


def label_bars(ax, bars, fmt="{:.0f}", dy=0.0, fontsize=11):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h), ha="center", va="bottom", fontsize=fontsize, color=INK)


def c01_decode_ceilings():
    fig, ax = canvas()
    labels = ["BF16", "INT8", "INT4\n(group 128)"]
    bpps = [bytes_per_param(16), bytes_per_param(8), bytes_per_param(4, 128)]
    x = np.arange(3)
    for i, (gpu, color) in enumerate(((A100_80GB, PURPLE), (H100_80GB, BLUE))):
        vals = [1 / decode_step(gpu, LLAMA3_8B, bytes_per_param=b).seconds for b in bpps]
        bars = ax.bar(x + (i - 0.5) * 0.36, vals, 0.34, color=color, label=gpu.name)
        label_bars(ax, bars, "{:.0f}", dy=8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("tokens per second, one sequence")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 950)
    finish(fig, "c01-decode-ceilings.png", "The batch-1 decode ceiling is set by bytes, not FLOPs",
           "Llama-3-8B, one sequence, weights only. Halve the bytes per weight and the ceiling doubles.")


def c02_batch_scaling():
    fig, ax = canvas()
    batches = np.array([2**i for i in range(0, 12)])
    steps = [decode_step(H100_80GB, LLAMA3_8B, batch=int(b)) for b in batches]
    total = [b / s.seconds for b, s in zip(batches, steps)]
    per_seq = [1 / s.seconds for s in steps]
    ax.plot(batches, total, color=BLUE, lw=3, marker="o", label="aggregate tokens/s (all sequences)")
    ax.plot(batches, per_seq, color=ORANGE, lw=3, marker="s", label="tokens/s for one sequence")
    ridge = ridge_batch(H100_80GB)
    ax.axvline(ridge, color=GRAY, ls="--")
    ax.text(ridge * 1.08, 300, f"ridge point\nbatch about {ridge:.0f}", color=GRAY, fontsize=11)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("batch size (sequences decoded together)")
    ax.set_ylabel("tokens per second")
    ax.legend(loc="upper left")
    finish(fig, "c02-batch-scaling.png", "Batching is nearly free, until the roofline bends",
           "H100, Llama-3-8B BF16, KV cache ignored. Left of the ridge, throughput scales with batch; right of it, it flattens.")


def c03_kv_bytes():
    fig, ax = canvas()
    rows = [("Llama-3-8B if MQA (1 KV head)", LLAMA3_8B.with_kv_heads(1), GREEN),
            ("Llama-3-8B (GQA, 8 KV heads)", LLAMA3_8B, BLUE),
            ("Llama-3-70B (GQA, 8 KV heads)", LLAMA3_70B, PURPLE),
            ("Llama-3-8B if MHA (32 KV heads)", LLAMA3_8B.with_kv_heads(32), ORANGE),
            ("Llama-2-7B (MHA)", LLAMA2_7B, RED)]
    vals = [kv_bytes_per_token(m) / KiB for _, m, _ in rows]
    bars = ax.barh([r[0] for r in rows], vals, color=[r[2] for r in rows])
    for b, v in zip(bars, vals):
        ax.text(v + 6, b.get_y() + b.get_height() / 2, f"{v:.0f} KiB", va="center", fontsize=12)
    ax.set_xlim(0, 620)
    ax.set_xlabel("KV cache per token (KiB, BF16)")
    ax.grid(axis="y", visible=False)
    finish(fig, "c03-kv-bytes-per-token.png", "How many KV heads you keep decides the cache size",
           "2 x layers x KV heads x head_dim x 2 bytes. Eight KV heads is 4x smaller than 32.", left=0.33)


def c04_users_vs_context():
    fig, ax = canvas()
    ctx = np.array([2048 * 2**i for i in range(7)])
    series = [("GQA, 8 KV heads (real Llama-3-8B)", LLAMA3_8B, BLUE),
              ("MHA, 32 KV heads (counterfactual)", LLAMA3_8B.with_kv_heads(32), ORANGE),
              ("MQA, 1 KV head (counterfactual)", LLAMA3_8B.with_kv_heads(1), GREEN)]
    for name, m, color in series:
        vals = np.array([max_sequences(H100_80GB, m, int(c)) for c in ctx], dtype=float)
        vals[vals == 0] = np.nan
        ax.plot(ctx, vals, color=color, lw=3, marker="o", label=name)
        if np.isnan(vals[-1]):
            ax.annotate("does not fit", (ctx[-1], 1), xytext=(ctx[-1] * 0.45, 1.4), color=color, fontsize=11)
    ax.axhline(1, color=RED, ls="--", lw=1.2)
    ax.text(2300, 1.12, "one user", color=RED, fontsize=11)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("context length per user (tokens)")
    ax.set_ylabel("concurrent users on one 80 GiB GPU")
    ax.legend(loc="upper right")
    finish(fig, "c04-users-vs-context.png", "Doubling the context halves your users, whatever the attention scheme",
           "H100 80 GiB, BF16 weights and cache, 90% memory cap. GQA shifts the line up by 4x; it does not change its slope.")


def c05_recompute():
    fig, ax = canvas()
    p = 128
    n = np.arange(1, 513)
    no_cache = n * p + n * (n - 1) // 2
    cache = p + n - 1
    ax.plot(n, no_cache, color=RED, lw=3, label="no cache: recompute the whole sequence every step")
    ax.plot(n, cache, color=BLUE, lw=3, label="KV cache: prefill once, then one position per step")
    ax.set_yscale("log")
    ax.set_xlabel("tokens generated")
    ax.set_ylabel("token positions pushed through the network")
    ratio = no_cache[-1] / cache[-1]
    ax.annotate(f"{ratio:.0f}x more work\nat 512 tokens", (512, no_cache[-1]), xytext=(330, no_cache[-1] * 0.18),
                arrowprops=dict(arrowstyle="->", color=GRAY), color=INK, fontsize=12)
    ax.legend(loc="upper left")
    finish(fig, "c05-recompute-vs-cache.png", "Without a KV cache, every step redoes the whole past",
           "128-token prompt. Positions processed: N x P + N(N-1)/2 without a cache, P + N - 1 with one. The cache is exact.")


def c06_weight_bytes():
    fig, ax = canvas()
    names = ["BF16\n16 bits", "INT8\n8 bits", "INT4, group 128\n4.125 bits"]
    bpps = [bytes_per_param(16), bytes_per_param(8), bytes_per_param(4, 128)]
    vals = [weight_bytes(LLAMA3_8B, b) / GB for b in bpps]
    bars = ax.bar(names, vals, color=[ORANGE, BLUE, TEAL], width=0.55)
    label_bars(ax, bars, "{:.2f} GB", dy=0.25, fontsize=13)
    ax.set_ylabel("weight bytes read per decode step (GB)")
    ax.set_ylim(0, 19)
    ax.text(2, 9.5, f"{bits_per_weight(4, 128):.3f} bits, not 4:\nthe group scales cost\n16 bits per 128 weights", ha="center",
            color=GRAY, fontsize=11)
    finish(fig, "c06-weight-bytes.png", "Quantization shrinks the thing being moved",
           "Llama-3-8B, 8.03 billion parameters. Group-wise scales make '4-bit' really 4.125 bits per weight.")


def c07_granularity():
    fig, ax = canvas()
    w = synthetic_weights()
    x = np.arange(2)
    grans = [("per-tensor", "tensor", PURPLE), ("per-channel", "channel", BLUE), ("group of 128", "group", TEAL)]
    for i, (name, g, color) in enumerate(grans):
        vals = [relative_rms_error(w, fake_quantize(w, bits, g, 128)) for bits in (8, 4)]
        bars = ax.bar(x + (i - 1) * 0.26, vals, 0.24, color=color, label=name)
        label_bars(ax, bars, "{:.3f}", dy=0.008, fontsize=10)
    ax.set_xticks(x, ["INT8", "INT4"])
    ax.set_ylabel("relative RMS error")
    ax.set_ylim(0, 0.9)
    ax.legend(loc="upper left")
    finish(fig, "c07-quantization-granularity.png", "Finer scales tame outliers; 4 bits is far less forgiving",
           "Synthetic 256 x 1024 weights with row-scale variation and 0.2% outliers. A mechanism demo, not a model measurement.")


def c08_sym_vs_asym():
    fig, ax = canvas()
    row = np.array([-0.10, 0.05, 0.22, 0.41, 0.63, 0.86, 0.34, 0.12])
    s_sym = symmetric_scale(row)
    s_asym, z = asymmetric_params(row)
    e_sym = np.abs(row - dequantize(quantize(row, s_sym), s_sym))
    e_asym = np.abs(row - dequantize(quantize(row, s_asym, z, 0, 255), s_asym, z))
    x = np.arange(len(row))
    ax.bar(x - 0.2, e_sym, 0.38, color=ORANGE, label=f"symmetric (step {s_sym:.4f})")
    ax.bar(x + 0.2, e_asym, 0.38, color=BLUE, label=f"asymmetric with zero-point {z} (step {s_asym:.4f})")
    ax.set_xticks(x, [f"{v:.2f}" for v in row])
    ax.set_xlabel("value being quantized (a skewed row)")
    ax.set_ylabel("absolute error after INT8 round trip")
    ax.legend(loc="upper right")
    finish(fig, "c08-symmetric-vs-asymmetric.png", "A zero-point buys accuracy on skewed data, at a runtime cost",
           f"Worst-case bound improves by the step ratio, {s_sym / s_asym:.2f}x. Individual values can still favour either rule.")


def c09_coalescing():
    fig, ax = canvas()
    strides = [4, 8, 12, 16, 24, 32, 48, 64, 128, 256]
    eff = [warp_access(s).efficiency * 100 for s in strides]
    ax.plot(range(len(strides)), eff, color=BLUE, lw=3, marker="o")
    ax.axhline(12.5, color=RED, ls="--", lw=1.3)
    ax.text(0.1, 15, "plateau: 12.5% (8x penalty)", color=RED, fontsize=12)
    ax.set_xticks(range(len(strides)), [f"{s} B" for s in strides])
    ax.set_xlabel("stride between consecutive threads' addresses (4-byte loads)")
    ax.set_ylabel("useful share of bytes moved (%)")
    ax.set_ylim(0, 105)
    finish(fig, "c09-coalescing.png", "Scatter a warp's loads and most of each trip is wasted",
           "32 threads, 4 bytes each, fetched in 32-byte sectors. Efficiency stops falling once every thread owns a sector.")


def c10_tiling():
    fig, ax = canvas()
    tiles = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    inten = [gemm_intensity(4096, 4096, 4096, t, t, 2) for t in tiles]
    ax.plot(tiles, inten, color=BLUE, lw=3, marker="o", label="tiled matrix multiply, FP16")
    for gpu, color in ((A100_80GB, PURPLE), (H100_80GB, TEAL)):
        ax.axhline(gpu.ridge(), color=color, ls="--", lw=1.3)
        ax.text(1.1, gpu.ridge() * 1.06, f"{gpu.name.split()[0]} ridge {gpu.ridge():.0f}", color=color, fontsize=11)
    ax.axhline(1, color=ORANGE, ls="--", lw=1.3)
    ax.text(1.1, 1.1, "batch-1 decode: about 1 FLOP/byte, whatever the tile", color=ORANGE, fontsize=11)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("output tile size (T x T)")
    ax.set_ylabel("FLOP per byte read")
    ax.legend(loc="lower right")
    finish(fig, "c10-tiling-intensity.png", "Tiling is how a kernel moves right on the roofline",
           "4096^3 FP16 matrix multiply. Intensity is about tile / bytes. Real kernels get the rest from L2 reuse and larger tiles.")


def c11_flash():
    fig, (a, b) = canvas(2)
    n = np.array([2**i for i in range(10, 18)])
    s1 = np.array([score_matrix_bytes(int(v)) / GiB for v in n])
    s32 = np.array([score_matrix_bytes(int(v), heads=32) / GiB for v in n])
    a.plot(n, s1, color=BLUE, lw=3, marker="o", label="1 head")
    a.plot(n, s32, color=ORANGE, lw=3, marker="s", label="32 heads")
    a.axhline(80, color=RED, ls="--")
    a.text(1100, 95, "80 GiB of HBM", color=RED, fontsize=11)
    a.set(xscale="log", yscale="log", xlabel="sequence length n", ylabel="GiB")
    a.set_title("Score matrix storage (FP16)", fontsize=12.5, loc="left")
    a.legend(loc="lower right")
    std = np.array([hbm_elements_standard(int(v), 128) * 2 / GiB for v in n])
    fl = np.array([hbm_elements_flash(int(v), 128, 128) * 2 / GiB for v in n])
    b.plot(n, std, color=ORANGE, lw=3, marker="o", label="standard attention")
    b.plot(n, fl, color=TEAL, lw=3, marker="s", label="tiled (FlashAttention)")
    b.set(xscale="log", yscale="log", xlabel="sequence length n", ylabel="GiB moved through HBM")
    b.set_title("HBM traffic, d = 128, block 128", fontsize=12.5, loc="left")
    b.legend(loc="upper left")
    b.text(0.55, 0.12, f"gap: {std[3] / fl[3]:.1f}x at every n\nboth still grow as n^2", transform=b.transAxes, color=GRAY, fontsize=11)
    finish(fig, "c11-flashattention.png", "FlashAttention removes the n x n matrix, not the quadratic",
           "Left: the matrix that no longer exists. Right: traffic stays parallel to the baseline, lower by about 2B/d.")


def c12_speculative():
    fig, (a, b) = canvas(2)
    ks = np.arange(0, 13)
    for alpha, color in ((0.5, ORANGE), (0.7, BLUE), (0.9, TEAL)):
        a.plot(ks, [expected_tokens(alpha, int(k)) for k in ks], color=color, lw=3, marker="o", ms=4, label=f"acceptance {alpha}")
        b.plot(ks, [speedup(alpha, int(k), 0.05) for k in ks], color=color, lw=3, marker="o", ms=4, label=f"acceptance {alpha}")
    a.set(xlabel="tokens drafted per step (k)", ylabel="tokens per target pass")
    a.set_title("Expected tokens per pass", fontsize=12.5, loc="left")
    b.axhline(1, color=RED, ls="--")
    b.text(6.2, 1.06, "no speedup", color=RED, fontsize=11)
    b.set(xlabel="tokens drafted per step (k)", ylabel="speedup vs plain decoding")
    b.set_title("Speedup if a draft step costs 5%", fontsize=12.5, loc="left")
    a.legend(loc="upper left")
    finish(fig, "c12-speculative-decoding.png", "Speculative decoding: acceptance rate beats draft length",
           "1 + a + a^2 + ... + a^k tokens per target pass. Extra drafts stop paying, then start costing.")


def c13_exactness(samples: int = 200_000):
    fig, ax = canvas()
    p = np.array([0.45, 0.22, 0.15, 0.08, 0.10])
    q = np.array([0.10, 0.08, 0.12, 0.30, 0.40])
    rng = np.random.default_rng(0)
    counts = np.zeros(5)
    accepted = 0
    for _ in range(samples):
        x, ok = speculative_step(p, q, rng)
        counts[x] += 1
        accepted += ok
    x = np.arange(5)
    ax.bar(x - 0.2, p, 0.38, color=BLUE, label="target distribution p")
    ax.bar(x + 0.2, counts / samples, 0.38, color=ORANGE, label=f"emitted by the rule ({samples:,} samples)")
    ax.scatter(x, q, marker="D", s=80, color=GRAY, zorder=5, label="draft distribution q (a poor draft)")
    ax.set_xticks(x, ["sunny", "warm", "cloudy", "beautiful", "nice"])
    ax.set_ylabel("probability")
    ax.legend(loc="upper right")
    finish(fig, "c13-speculative-exactness.png", "Even with a poor draft, the output distribution is exactly the target's",
           f"Only {accepted / samples:.0%} of drafts were accepted, yet the emitted frequencies match p. A bad draft costs speed, never correctness.")


def c14_batching():
    fig, (a, b) = canvas(2)
    static_tok = static_window_tokens([32, 80, 200, 400], 40, 10)
    cont_tok = continuous_window_tokens(4, 40, 10)
    bars = a.bar(["static", "continuous"], [static_tok, cont_tok], color=[ORANGE, TEAL], width=0.55)
    label_bars(a, bars, "{:,.0f}", dy=20, fontsize=13)
    a.set_ylabel("tokens emitted in 10 seconds")
    a.set_title(f"Same 4 slots, {cont_tok / static_tok:.1f}x the tokens", fontsize=12.5, loc="left")
    a.set_ylim(0, 1900)
    st = static_schedule(EXAMPLE_LENGTHS, EXAMPLE_SLOTS)
    ct = continuous_schedule(EXAMPLE_LENGTHS, EXAMPLE_SLOTS)
    bars = b.bar(["static", "continuous"], [utilisation(st) * 100, utilisation(ct) * 100], color=[ORANGE, TEAL], width=0.55)
    label_bars(b, bars, "{:.0f}%", dy=1.5, fontsize=13)
    b.set_ylabel("share of slot-steps doing work")
    b.set_title("7 requests on 4 slots: busy time", fontsize=12.5, loc="left")
    b.set_ylim(0, 110)
    finish(fig, "c14-continuous-batching.png", "Continuous batching refills a seat the moment it frees",
           "Per-request latency is unchanged. Throughput rises because finished requests no longer hold a slot idle.")


def c15_paging():
    fig, (a, b) = canvas(2)
    r = compare_policies()
    pool = r["pool_tokens"]
    bars = a.bar(["contiguous\nslabs", "paged\nblocks"], [r["contiguous_admitted"], r["paged_admitted"]], color=[ORANGE, TEAL], width=0.55)
    label_bars(a, bars, "{:.0f}", dy=8, fontsize=13)
    a.set_ylabel("requests that fit in the same pool")
    a.set_ylim(0, 560)
    bars = b.bar(["contiguous\nslabs", "paged\nblocks"], [r["contiguous_used"] / pool * 100, r["paged_used"] / pool * 100],
                 color=[ORANGE, TEAL], width=0.55)
    label_bars(b, bars, "{:.1f}%", dy=1.5, fontsize=13)
    b.set_ylabel("pool slots holding real tokens (%)")
    b.set_ylim(0, 115)
    finish(fig, "c15-paged-vs-contiguous.png", "Paging turns wasted tail space into users",
           "Synthetic lognormal request lengths and a 4,096-token maximum: the size of the gap depends on those choices; the mechanism does not.")


def c16_hbm_budget():
    fig, ax = canvas()
    total = H100_80GB.hbm_bytes / GiB
    weights = weight_bytes(LLAMA3_8B) / GiB
    pool = kv_pool_bytes(H100_80GB, LLAMA3_8B) / GiB
    headroom = total - weights - pool
    left = 0
    for label, v, color in (("weights", weights, BLUE), ("KV cache pool", pool, TEAL), ("headroom", headroom, GRAY)):
        ax.barh([0], [v], left=left, color=color, height=0.5)
        if v > 20:
            ax.text(left + v / 2, 0, f"{label}\n{v:.1f} GiB", ha="center", va="center", color="white", fontsize=13, fontweight="bold")
        else:  # a narrow segment: label it above the bar instead of overflowing it
            ax.text(left + v / 2, 0.32, f"{label}\n{v:.1f} GiB", ha="center", va="bottom", color=GRAY, fontsize=11.5, fontweight="bold")
        left += v
    ax.set_xlim(0, total)
    ax.set_ylim(-1.2, 1.2)
    ax.set_yticks([])
    ax.grid(False)
    ax.set_xlabel("GiB of HBM on an 80 GiB GPU")
    ax.text(total / 2, -0.95, f"At 8K context, 1 GiB per conversation: about {int(pool)} conversations fit (an upper bound).",
            ha="center", fontsize=12, color=INK)
    finish(fig, "c16-hbm-budget.png", "After the weights, whatever is left is your user capacity",
           "Llama-3-8B BF16 on an H100 80 GiB with a 90% memory cap. Activation workspace ignored.")


def c17_prefill_vs_decode():
    fig, (a, b) = canvas(2)
    dec = decode_step(H100_80GB, LLAMA3_8B, batch=1)
    pre = decode_step(H100_80GB, LLAMA3_8B, batch=2048)
    names = ["decode step\n(1 token)", "prefill\n(2,048 tokens)"]
    bars = a.bar(names, [dec.seconds * 1e3, pre.seconds * 1e3], color=[ORANGE, TEAL], width=0.55)
    label_bars(a, bars, "{:.1f} ms", dy=0.6, fontsize=13)
    a.set_ylabel("time for one forward pass (ms)")
    a.set_title("The pass takes longer...", fontsize=12.5, loc="left")
    per = [dec.seconds * 1e3, pre.seconds * 1e3 / 2048]
    bars = b.bar(names, per, color=[ORANGE, TEAL], width=0.55)
    label_bars(b, bars, "{:.3f} ms", dy=0.1, fontsize=13)
    b.set_ylabel("ms per token")
    b.set_ylim(0, 5.9)
    b.set_title(f"...but each token costs {per[0] / per[1]:.0f}x less", fontsize=12.5, loc="left")
    finish(fig, "c17-prefill-vs-decode.png", "Same GPU, same weights, opposite bottleneck",
           "Prefill is compute-bound; a decode step is memory-bound. Llama-3-8B BF16 on an H100, ceilings from datasheet peaks.")


def c18_kv_crossover():
    fig, ax = canvas()
    b = np.array([1, 2, 4, 8, 16, 32, 64, 128])
    cross = weight_bytes(LLAMA3_8B) / kv_bytes_per_token(LLAMA3_8B) / b
    ax.plot(b, cross, color=BLUE, lw=3, marker="o")
    ax.fill_between(b, cross, cross.max() * 4, color=ORANGE, alpha=0.15)
    ax.text(2, cross[0] * 2.2, "KV-cache reads dominate each step", color=ORANGE, fontsize=13, fontweight="bold")
    ax.text(1.15, 900, "weight reads dominate", color=BLUE, fontsize=13, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylim(500, cross.max() * 4)
    ax.set_xlabel("batch size")
    ax.set_ylabel("context length where KV reads equal weight reads")
    finish(fig, "c18-kv-crossover.png", "At long context and large batch, the cache is the bill",
           f"Llama-3-8B BF16. Crossover at batch x context = {weight_bytes(LLAMA3_8B) / kv_bytes_per_token(LLAMA3_8B):,.0f} tokens.")


def main() -> int:
    for fn in (c01_decode_ceilings, c02_batch_scaling, c03_kv_bytes, c04_users_vs_context, c05_recompute,
               c06_weight_bytes, c07_granularity, c08_sym_vs_asym, c09_coalescing, c10_tiling, c11_flash,
               c12_speculative, c13_exactness, c14_batching, c15_paging, c16_hbm_budget, c17_prefill_vs_decode,
               c18_kv_crossover):
        fn()
    return 0


if __name__ == "__main__":
    sys.exit(main())
