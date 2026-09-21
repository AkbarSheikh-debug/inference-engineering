# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
"""Render every diagram to a cropped PNG with a headless browser.

    python tools/export_diagrams.py

Some blogging platforms cannot render Mermaid, so tools/syndicate.py links to
these images instead. Needs Microsoft Edge, Chrome or Chromium on the machine,
Pillow (`pip install pillow`), and network access on first use, because Mermaid
is loaded from a CDN. Mermaid sources are rendered by Mermaid; the generated
SVG figures are rendered as they are.
"""

import html
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "diagrams" / "src"
EXPORT = ROOT / "diagrams" / "export"

BROWSERS = [
    "msedge", "microsoft-edge", "google-chrome", "chrome", "chromium", "chromium-browser",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]

MERMAID_PAGE = """<html><body style="margin:0;padding:24px;background:#fff;width:1300px">
<script type="module">
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
mermaid.initialize({{startOnLoad: true}});
</script>
<pre class="mermaid">{code}</pre></body></html>"""

IMAGE_PAGE = '<html><body style="margin:0;padding:24px;background:#fff"><img src="{src}"></body></html>'


def find_browser() -> str:
    for candidate in BROWSERS:
        found = shutil.which(candidate) or (candidate if Path(candidate).exists() else None)
        if found:
            return found
    sys.exit("No Edge, Chrome or Chromium found. Install one, or render the diagrams by hand.")


def screenshot(browser: str, page: Path, out: Path) -> None:
    subprocess.run(
        [browser, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--window-size=1400,2400",
         "--force-device-scale-factor=2", "--virtual-time-budget=20000", f"--screenshot={out}", page.as_uri()],
        check=True, capture_output=True, timeout=180,
    )


def crop_to_content(png: Path, margin: int = 40) -> None:
    from PIL import Image, ImageChops

    im = Image.open(png).convert("RGB")
    box = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))).getbbox()
    if box is None:
        raise RuntimeError(f"{png.name} rendered blank")
    left, top, right, bottom = box
    im.crop((max(0, left - margin), max(0, top - margin), min(im.width, right + margin), min(im.height, bottom + margin))).save(png)


def main() -> int:
    browser = find_browser()
    EXPORT.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, str]] = []  # (name, page html)
    for mmd in sorted(SRC.glob("*.mmd")):
        jobs.append((mmd.stem, MERMAID_PAGE.format(code=html.escape(mmd.read_text(encoding="utf-8")))))
    for svg in sorted(EXPORT.glob("*.svg")):
        jobs.append((svg.stem, IMAGE_PAGE.format(src=svg.as_uri())))

    with tempfile.TemporaryDirectory() as tmp:
        for name, page_html in jobs:
            page = Path(tmp) / f"{name}.html"
            page.write_text(page_html, encoding="utf-8")
            out = EXPORT / f"{name}.png"
            screenshot(browser, page, out)
            crop_to_content(out)
            print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
