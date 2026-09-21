# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
"""Turn each article into Markdown ready to paste into a blogging platform.

    python tools/export_diagrams.py     # once, so the diagram images exist
    python tools/syndicate.py           # writes dist/syndication/<platform>/*.md

The repository stays the source of truth. Each output swaps Mermaid blocks for
images, turns repository-relative links into absolute GitHub links, adds the
front matter the platform expects, and ends with a pointer to the canonical
copy. Check each platform's current Markdown support (tables, images, front
matter) before publishing, and set the canonical URL in its settings if it has
one. For Medium, import the story from a published URL instead of pasting.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "diagrams" / "src"
PLATFORMS = ("devto", "hashnode", "huggingface")
MARKED = re.compile(r"<!-- diagram:start (\S+) -->.*?<!-- diagram:end -->", re.S)
SVG_IMAGE = re.compile(r"!\[([^\]]*)\]\(\.\./diagrams/export/([\w-]+)\.svg\)")
ARTICLE_LINK = re.compile(r"\]\((\d\d-[\w-]+\.md)\)")
TAGS = "llm, gpu, ai, performance"  # DEV allows at most four


def default_repo() -> str:
    try:
        url = subprocess.run(["git", "config", "--get", "remote.origin.url"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "OWNER/inference-engineering"
    return re.sub(r"^.*github\.com[:/]", "", url).removesuffix(".git")


def diagram_title(name: str) -> str:
    first = (SRC / f"{name}.mmd").read_text(encoding="utf-8").splitlines()[0]
    m = re.match(r"%% title: (.+)", first)
    return m.group(1) if m else name


def _yaml(value: str) -> str:
    """Double-quoted YAML scalar; titles such as 'Speculative decoding: more tokens' contain a colon."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def split_article(text: str) -> tuple[str, str, str]:
    """(title without its number, one-line description, body after the H1)."""
    lines = text.splitlines()
    title = re.sub(r"^#\s*\d+\.\s*", "", lines[0])
    m = re.search(r"\*\*Question this answers:\*\*\s*(.+)", text)
    description = m.group(1).strip() if m else title
    return title, description, "\n".join(lines[1:]).lstrip("\n")


def transform(text: str, platform: str, repo: str, filename: str) -> str:
    raw = f"https://raw.githubusercontent.com/{repo}/main"
    blob = f"https://github.com/{repo}/blob/main"
    canonical = f"{blob}/curriculum/{filename}"
    title, description, body = split_article(text)

    body = MARKED.sub(lambda m: f"![{diagram_title(m.group(1))}]({raw}/diagrams/export/{m.group(1)}.png)", body)
    body = SVG_IMAGE.sub(lambda m: f"![{m.group(1)}]({raw}/diagrams/export/{m.group(2)}.png)", body)
    body = ARTICLE_LINK.sub(lambda m: f"]({blob}/curriculum/{m.group(1)})", body)
    body = body.replace("](../LICENSE)", f"]({blob}/LICENSE)")
    body = body.rstrip("\n") + (
        f"\n\nOriginally published at [{canonical}]({canonical}). "
        f"Source, runnable labs and tests: [github.com/{repo}](https://github.com/{repo}).\n"
    )

    if platform == "devto":
        short = description[:1].upper() + description[1:]
        short = short if len(short) <= 155 else short[:152].rstrip() + "..."
        front = (f"---\ntitle: {_yaml(title)}\npublished: false\ndescription: {_yaml(short)}\n"
                 f"tags: {TAGS}\ncanonical_url: {canonical}\n---\n\n")
        return front + body
    return f"# {title}\n\n{body}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=default_repo(), help="GitHub owner/name")
    ap.add_argument("--out", type=Path, default=ROOT / "dist" / "syndication")
    args = ap.parse_args()

    articles = sorted(p for p in (ROOT / "curriculum").glob("*.md") if not p.name.startswith("00-"))
    missing = sorted({m.group(1) for a in articles for m in MARKED.finditer(a.read_text(encoding="utf-8"))
                      if not (ROOT / "diagrams" / "export" / f"{m.group(1)}.png").exists()})
    if missing:
        print("warning: no PNG yet for " + ", ".join(missing) + " (run tools/export_diagrams.py first)")

    for platform in PLATFORMS:
        folder = args.out / platform
        folder.mkdir(parents=True, exist_ok=True)
        for article in articles:
            out = transform(article.read_text(encoding="utf-8"), platform, args.repo, article.name)
            (folder / article.name).write_text(out, encoding="utf-8", newline="\n")
        print(f"wrote {len(articles)} articles to {folder.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
