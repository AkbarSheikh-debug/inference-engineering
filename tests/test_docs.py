# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_no_em_dashes_and_no_broken_links():
    assert _load("check_docs").problems() == []


def test_embedded_diagrams_match_their_sources():
    sync = _load("sync_diagrams")
    for f in sync.targets():
        text = f.read_text(encoding="utf-8")
        assert sync.MARKED.sub(lambda m: sync.embed(m.group(1)), text) == text, f.name
    assert (ROOT / "diagrams" / "README.md").read_text(encoding="utf-8") == sync.gallery()


def test_every_marker_points_at_an_existing_source():
    sync = _load("sync_diagrams")
    for f in sync.targets():
        for name in sync.MARKED.findall(f.read_text(encoding="utf-8")):
            assert (sync.SRC / f"{name}.mmd").exists(), f"{f.name}: {name}"


def test_batching_schedule_claims():
    b = _load("batching_svg")
    static = b.static_schedule(b.LENGTHS, b.SLOTS)
    cont = b.continuous_schedule(b.LENGTHS, b.SLOTS)
    assert (b.makespan(static), b.makespan(cont)) == (15, 10)
    assert round(b.utilisation(static), 2) == 0.60
    assert round(b.utilisation(cont), 2) == 0.90
    assert sum(b.LENGTHS.values()) == 36  # identical work under both policies


FOOTER = (
    "\n---\n\nCopyright (c) 2026 Akbar Arif. Released under the [MIT License](../LICENSE). "
    "You may reuse and adapt this article as long as the copyright and license notice stays with it.\n"
)


def test_every_article_carries_the_copyright_footer():
    articles = sorted((ROOT / "curriculum").glob("*.md"))
    assert len(articles) >= 6
    for f in articles:
        assert f.read_text(encoding="utf-8").endswith(FOOTER), f.name
