# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
"""Convert the long-form post to HTML that pastes cleanly into Substack.

    python tools/blog_html.py

Substack's editor has no tables and does not convert pasted Markdown, but it
keeps headings, lists, links, code blocks and images when you paste rich text.
Open blog/inference-engineering-substack.html in a browser, click "Copy post
body", and paste into a new Substack post. The title and subtitle go in
Substack's own fields. Below the post the page lists every image with the
section it belongs in, in case you prefer to upload images by hand.

Handles only the Markdown this post uses: headings, paragraphs, bullet and
numbered lists, fenced code, images with an italic caption line, links, bold,
italic and inline code.
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "blog" / "inference-engineering-substack.md"
OUT = ROOT / "blog" / "inference-engineering-substack.html"

IMAGE_LINE = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$")
CAPTION_LINE = re.compile(r"^\*(?!\*)(.+)\*$")
BULLET = re.compile(r"^-\s+(.*)")
NUMBERED = re.compile(r"^\d+\.\s+(.*)")
META = re.compile(r"^<!--\s*(TITLE|SUBTITLE):\s*(.*?)\s*-->$")


def inline(text: str) -> str:
    codes: list[str] = []

    def stash(m: re.Match) -> str:
        codes.append(m.group(1))
        return f"\x00{len(codes) - 1}\x00"

    text = re.sub(r"`([^`]+)`", stash, text)
    text = html.escape(text, quote=False)
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)",
                  lambda m: f'<img src="{m.group(2)}" alt="{html.escape(m.group(1), quote=True)}">', text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", text)
    return re.sub(r"\x00(\d+)\x00", lambda m: f"<code>{html.escape(codes[int(m.group(1))])}</code>", text)


def convert(md: str) -> tuple[str, str, str, list[tuple[str, str, str]]]:
    """Returns (title, subtitle, body html, image manifest of (file, alt, section))."""
    lines = md.splitlines()
    out: list[str] = []
    meta = {"TITLE": "", "SUBTITLE": ""}
    images: list[tuple[str, str, str]] = []
    section = "Top of the post"
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        m = META.match(line.strip())
        if m:
            meta[m.group(1)] = m.group(2)
            i += 1
            continue
        if line.startswith("```"):
            i += 1
            block = []
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            out.append("<pre><code>" + html.escape("\n".join(block)) + "</code></pre>")
            continue
        h = re.match(r"^(#{1,3})\s+(.*)", line)
        if h:
            level = len(h.group(1))
            if level == 2:
                section = h.group(2)
            out.append(f"<h{level}>{inline(h.group(2))}</h{level}>")
            i += 1
            continue
        im = IMAGE_LINE.match(line.strip())
        if im:
            alt, src = im.groups()
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            caption = ""
            cm = CAPTION_LINE.match(lines[j].strip()) if j < len(lines) else None
            if cm:
                caption = f"<figcaption>{inline(cm.group(1))}</figcaption>"
                i = j
            images.append((src.rsplit("/", 1)[-1], alt, section))
            out.append(f'<figure><img src="{src}" alt="{html.escape(alt, quote=True)}">{caption}</figure>')
            i += 1
            continue
        if BULLET.match(line) or NUMBERED.match(line):
            ordered = bool(NUMBERED.match(line))
            pattern = NUMBERED if ordered else BULLET
            items = []
            while i < len(lines) and pattern.match(lines[i]):
                items.append(f"<li>{inline(pattern.match(lines[i]).group(1))}</li>")
                i += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(items) + f"</{tag}>")
            continue
        para = [line.strip()]
        i += 1
        while i < len(lines) and lines[i].strip() and not (
            lines[i].startswith(("```", "#")) or BULLET.match(lines[i]) or NUMBERED.match(lines[i])
            or IMAGE_LINE.match(lines[i].strip())
        ):
            para.append(lines[i].strip())
            i += 1
        out.append(f"<p>{inline(' '.join(para))}</p>")
    return meta["TITLE"], meta["SUBTITLE"], "\n".join(out), images


PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body {{ font: 18px/1.65 Georgia, 'Times New Roman', serif; color: #1f2937; max-width: 720px; margin: 0 auto; padding: 24px; }}
  h1, h2, h3 {{ font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; line-height: 1.25; color: #111827; }}
  h2 {{ margin-top: 2.2em; }} pre {{ background: #f3f4f6; padding: 14px; overflow-x: auto; font-size: 14px; line-height: 1.45; }}
  code {{ font-family: Consolas, 'Courier New', monospace; font-size: 0.9em; background: #f3f4f6; padding: 1px 4px; }}
  pre code {{ background: none; padding: 0; }} figure {{ margin: 1.6em 0; }} figure img {{ max-width: 100%; height: auto; }}
  figcaption {{ font-size: 0.85em; color: #6b7280; text-align: center; margin-top: 6px; }}
  .tools {{ font-family: -apple-system, 'Segoe UI', sans-serif; background: #eff6ff; border: 1px solid #bfdbfe; padding: 14px 18px; margin-bottom: 32px; font-size: 15px; line-height: 1.5; }}
  .tools button {{ font-size: 15px; padding: 8px 16px; background: #0369a1; color: #fff; border: 0; cursor: pointer; }}
  .tools code {{ user-select: all; }} .manifest {{ font-family: -apple-system, 'Segoe UI', sans-serif; font-size: 14px; margin-top: 56px; border-top: 2px solid #e5e7eb; padding-top: 16px; }}
  .manifest li {{ margin-bottom: 6px; }}
</style></head><body>
<div class="tools">
  <p><strong>Substack title:</strong> <code>{title}</code><br><strong>Substack subtitle:</strong> <code>{subtitle}</code></p>
  <p><button onclick="copyPost()" id="copybtn">Copy post body</button> then paste into the Substack editor body.</p>
  <p>If images do not import on paste, upload them by hand from <code>blog/images/</code> and <code>diagrams/export/</code>; the list at the bottom says where each one goes.</p>
</div>
<article id="post">
{body}
</article>
<div class="manifest"><h3>Image checklist ({count} images)</h3><ol>
{manifest}
</ol></div>
<script>
function copyPost() {{
  const el = document.getElementById('post'); const r = document.createRange(); r.selectNodeContents(el);
  const s = window.getSelection(); s.removeAllRanges(); s.addRange(r); document.execCommand('copy'); s.removeAllRanges();
  document.getElementById('copybtn').textContent = 'Copied. Now paste into Substack';
}}
</script></body></html>
"""


def render(md: str) -> str:
    title, subtitle, body, images = convert(md)
    manifest = "\n".join(f"<li><code>{html.escape(f)}</code> in section <em>{html.escape(s)}</em>: {html.escape(a)}</li>"
                         for f, a, s in images)
    return PAGE.format(title=html.escape(title), subtitle=html.escape(subtitle), body=body, count=len(images), manifest=manifest)


def main() -> int:
    OUT.write_text(render(SRC.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
