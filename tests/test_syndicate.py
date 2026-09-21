# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO = "example-owner/inference-engineering"


def _load():
    spec = importlib.util.spec_from_file_location("syndicate", ROOT / "tools" / "syndicate.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["syndicate"] = mod
    spec.loader.exec_module(mod)
    return mod


syn = _load()
ARTICLES = sorted(p for p in (ROOT / "curriculum").glob("*.md") if not p.name.startswith("00-"))


@pytest.mark.parametrize("platform", syn.PLATFORMS)
@pytest.mark.parametrize("article", ARTICLES, ids=lambda p: p.name)
def test_output_has_no_repository_relative_references(article, platform):
    out = syn.transform(article.read_text(encoding="utf-8"), platform, REPO, article.name)
    assert "```mermaid" not in out and "diagram:start" not in out
    assert not re.search(r"\]\(\.\./", out), "relative parent link left behind"
    assert not re.search(r"\]\(\d\d-[\w-]+\.md\)", out), "relative article link left behind"
    assert not re.search(r"\]\((?!https?://)[^)]*\.(svg|png)\)", out), "non-absolute image left behind"
    assert f"Originally published at [https://github.com/{REPO}/blob/main/curriculum/{article.name}]" in out
    assert "Copyright (c) 2026 Akbar Arif" in out  # the footer survives


def test_devto_front_matter():
    article = next(p for p in ARTICLES if p.name.startswith("03-"))
    out = syn.transform(article.read_text(encoding="utf-8"), "devto", REPO, article.name)
    head = out.split("---")[1]
    assert "published: false" in head
    assert f"canonical_url: https://github.com/{REPO}/blob/main/curriculum/{article.name}" in head
    tags = re.search(r"tags: (.+)", head).group(1).split(", ")
    assert 1 <= len(tags) <= 4
    assert not out.split("---", 2)[2].lstrip().startswith("# 03.")  # the title lives in the front matter


@pytest.mark.parametrize("article", ARTICLES, ids=lambda p: p.name)
def test_devto_front_matter_is_valid_yaml_even_when_titles_contain_colons(article):
    out = syn.transform(article.read_text(encoding="utf-8"), "devto", REPO, article.name)
    head = out.split("---")[1]
    title_line = next(l for l in head.splitlines() if l.startswith("title: "))
    desc_line = next(l for l in head.splitlines() if l.startswith("description: "))
    for line in (title_line, desc_line):
        value = line.split(": ", 1)[1]
        assert value.startswith('"') and value.endswith('"') and len(value) >= 2, line
    assert desc_line.split(": ", 1)[1][1].isupper()


def test_other_platforms_keep_a_clean_title_heading():
    article = next(p for p in ARTICLES if p.name.startswith("03-"))
    for platform in ("hashnode", "huggingface"):
        out = syn.transform(article.read_text(encoding="utf-8"), platform, REPO, article.name)
        assert out.startswith("# Why the KV cache exists, and why it fills your GPU\n")


def test_diagram_markers_become_images_on_the_raw_host():
    article = next(p for p in ARTICLES if p.name.startswith("01-"))
    out = syn.transform(article.read_text(encoding="utf-8"), "hashnode", REPO, article.name)
    assert f"![Request lifecycle from prompt to streamed tokens](https://raw.githubusercontent.com/{REPO}/main/diagrams/export/d01-request-lifecycle.png)" in out


def test_svg_figures_are_replaced_by_their_png():
    article = next(p for p in ARTICLES if p.name.startswith("02-"))
    out = syn.transform(article.read_text(encoding="utf-8"), "huggingface", REPO, article.name)
    assert f"https://raw.githubusercontent.com/{REPO}/main/diagrams/export/d04-roofline.png" in out


def test_description_is_the_question():
    article = next(p for p in ARTICLES if p.name.startswith("08-"))
    _, description, _ = syn.split_article(article.read_text(encoding="utf-8"))
    assert description.startswith("decoding is sequential")
