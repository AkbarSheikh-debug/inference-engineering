# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
"""The long-form post must stay consistent with the code: images exist, numbers match, HTML is in sync."""

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np

from ie.attention import score_matrix_bytes
from ie.batching import (EXAMPLE_LENGTHS, EXAMPLE_SLOTS, continuous_schedule, continuous_window_tokens, makespan,
                         static_schedule, static_window_tokens, token_budget_split, utilisation)
from ie.hardware import A100_80GB as A100, H100_80GB as H100
from ie.kv import kv_bytes_per_token, max_sequences, weight_bytes
from ie.models import LLAMA3_8B
from ie.quant import bytes_per_param
from ie.quantize import asymmetric_params, symmetric_scale
from ie.roofline import decode_step, ridge_batch
from ie.speculative import expected_tokens, speedup
from ie.units import GB, GiB, KiB

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "blog" / "inference-engineering-substack.md"
HTML = ROOT / "blog" / "inference-engineering-substack.html"
RAW = "https://raw.githubusercontent.com/AkbarSheikh-debug/inference-engineering/main/"


def _load():
    spec = importlib.util.spec_from_file_location("blog_html", ROOT / "tools" / "blog_html.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["blog_html"] = mod
    spec.loader.exec_module(mod)
    return mod


text = MD.read_text(encoding="utf-8")
IMAGES = re.findall(r"!\[[^\]]*\]\((" + re.escape(RAW) + r"([^)]+))\)", text)


def test_the_post_has_many_distinct_images_and_every_one_exists_in_the_repo():
    assert len(IMAGES) >= 30
    assert len({p for _, p in IMAGES}) == len(IMAGES), "an image is used twice"
    for _, path in IMAGES:
        assert (ROOT / path).exists(), path


def test_every_image_is_followed_by_a_caption():
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("!["):
            j = i + 1
            while not lines[j].strip():
                j += 1
            assert lines[j].startswith("*") and lines[j].rstrip().endswith("*"), line


def test_no_paragraph_is_repeated():
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40 and not p.startswith("```")]
    assert len(paragraphs) == len(set(paragraphs)), "a paragraph appears twice"


def test_no_em_dashes_and_no_tooling_mentions():
    assert "—" not in text
    banned = ("cl" + "aude", "anthro" + "pic")  # built from pieces so this file never contains the words itself
    assert not any(word in text.lower() for word in banned)


def test_headline_numbers_match_the_code():
    dec = decode_step(H100, LLAMA3_8B)
    expected = [
        f"{H100.attainable(1.0) / H100.peak_bf16 * 100:.1f} percent",
        f"{dec.memory_s * 1e3:.2f} ms",
        f"{2 * LLAMA3_8B.params / H100.peak_bf16 * 1e3:.3f} ms",
        f"{decode_step(H100, LLAMA3_8B, batch=2048).seconds * 1e3:.1f} ms",
        f"{ridge_batch(H100):.0f}", f"{A100.ridge():.0f}",
        f"{weight_bytes(LLAMA3_8B) / GB:.2f} GB",
    ]
    for bpp in (bytes_per_param(16), bytes_per_param(8), bytes_per_param(4, 128)):
        for gpu in (H100, A100):
            expected.append(f"{1 / decode_step(gpu, LLAMA3_8B, bytes_per_param=bpp).seconds:.0f}")
    for s in expected:
        assert s in text, s


def test_kv_and_capacity_numbers():
    per_tok = kv_bytes_per_token(LLAMA3_8B)
    crossover = weight_bytes(LLAMA3_8B) / per_tok
    for s in (f"{per_tok // KiB} KiB per token", f"**{per_tok * 8192 // GiB} GiB**",
              f"{max_sequences(H100, LLAMA3_8B, 8192)} concurrent conversations",
              f"{max_sequences(H100, LLAMA3_8B.with_kv_heads(32), 8192)} with full multi-head",
              f"{max_sequences(H100, LLAMA3_8B.with_kv_heads(1), 8192)} with multi-query",
              f"{crossover:,.0f}", f"{crossover / 16:,.0f}", f"{crossover / 64:,.0f}"):
        assert s in text, s


def test_quantization_flash_and_speculative_numbers():
    row = np.array([-0.10, 0.05, 0.22, 0.41, 0.63, 0.86, 0.34, 0.12])
    ratio = symmetric_scale(row) / asymmetric_params(row)[0]
    assert f"{ratio:.2f}" in text
    assert f"{score_matrix_bytes(65536) / GiB:.0f} GiB" in text
    assert f"{score_matrix_bytes(32768, heads=32) / GiB:.0f} GiB" in text
    assert f"{expected_tokens(0.7, 4):.2f} tokens per pass" in text
    assert f"{speedup(0.7, 4, 0.05):.2f}x speedup" in text
    assert f"about {speedup(0.3, 8, 0.3):.2f}" in text


def test_batching_numbers():
    s = static_schedule(EXAMPLE_LENGTHS, EXAMPLE_SLOTS)
    c = continuous_schedule(EXAMPLE_LENGTHS, EXAMPLE_SLOTS)
    st, ct = static_window_tokens([32, 80, 200, 400], 40, 10), continuous_window_tokens(4, 40, 10)
    r = token_budget_split(4096, 24, 1024)
    for s_ in (f"{makespan(s)} steps", f"{makespan(c)} steps", f"{utilisation(s):.0%}".replace("%", " percent"),
               f"{utilisation(c):.0%}".replace("%", " percent"), f"**{st:,.0f}**", f"**{ct:,.0f}**", f"{ct / st:.1f}",
               f"{r['decode'] / 4096 * 100:.1f} percent", f"{r['prefill']:,}", f"{r['spare']:,}"):
        assert s_ in text, s_


def test_all_external_links_are_https_and_the_repo_is_linked():
    urls = re.findall(r"\]\((https?://[^)]+)\)", text)
    assert urls and all(u.startswith("https://") for u in urls)
    assert "https://github.com/AkbarSheikh-debug/inference-engineering" in urls


def test_html_is_in_sync_with_the_markdown_and_well_formed():
    mod = _load()
    rendered = mod.render(text)
    assert HTML.exists() and HTML.read_text(encoding="utf-8") == rendered, "run python tools/blog_html.py"
    assert rendered.count("<figure>") == len(IMAGES) == rendered.count("<figcaption>")
    body = rendered.split('<article id="post">')[1].split("</article>")[0]
    assert "**" not in body and "](" not in body and "```" not in body
    assert body.count("<pre>") == body.count("</pre>") and body.count("<ul>") == body.count("</ul>")
    assert body.count("<ol>") == body.count("</ol>")
