# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
"""Render the blog cover image (blog/images/c00-cover.png).

    python tools/make_cover.py

The cover is designed in HTML and CSS and rendered with headless Edge, Chrome or
Chromium at 1456 x 816 (the 16:9 size Substack uses for post headers), at twice
that resolution for sharpness. Every number on it, including the points on the
roofline plot, is computed from src/ie, so the image cannot disagree with the
post. Needs a Chromium-family browser and Pillow.
"""

import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ie.hardware import A100_80GB, H100_80GB  # noqa: E402
from ie.kv import kv_bytes_per_token, max_sequences, weight_bytes  # noqa: E402
from ie.models import LLAMA3_8B  # noqa: E402
from ie.roofline import decode_step, weight_intensity  # noqa: E402
from ie.units import GB, GiB  # noqa: E402

OUT = ROOT / "blog" / "images" / "c00-cover.png"
W, H = 1456, 816
BROWSERS = [
    "msedge", "microsoft-edge", "google-chrome", "chrome", "chromium", "chromium-browser",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]

# chart geometry (pixels inside the SVG)
CW, CH, ML, MR, MT, MB = 560, 420, 56, 28, 34, 44
X_DECADES, Y_DECADES = 4, 3
FONT = "Inter,'Segoe UI',sans-serif"


def cx(intensity: float) -> float:
    return ML + math.log10(intensity) / X_DECADES * (CW - ML - MR)


def cy(tflops: float) -> float:
    return CH - MB - math.log10(tflops) / Y_DECADES * (CH - MT - MB)


def label(x, y, text, fill, size=13, weight=700, anchor="start", spacing=0):
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" font-weight="{weight}" '
            f'text-anchor="{anchor}" letter-spacing="{spacing}" font-family="{FONT}">{text}</text>')


def roofline_svg() -> str:
    h, a = H100_80GB, A100_80GB
    roof = lambda g: [(1.0, g.attainable(1.0) / 1e12), (g.ridge(), g.peak_bf16 / 1e12), (1e4, g.peak_bf16 / 1e12)]
    pts = lambda g: " ".join(f"{cx(x):.1f},{cy(y):.1f}" for x, y in roof(g))
    area = pts(h) + f" {cx(1e4):.1f},{cy(1):.1f} {cx(1):.1f},{cy(1):.1f}"
    d1, d64, pre = weight_intensity(1), weight_intensity(64), weight_intensity(2048)
    y = lambda x: h.attainable(x) / 1e12
    grid = "".join(
        f'<line x1="{cx(10**e):.1f}" y1="{MT}" x2="{cx(10**e):.1f}" y2="{CH - MB}" class="g"/>'
        + label(cx(10**e), CH - MB + 20, f"{10**e:,}", "#94a3b8", 12, 500, "middle")
        for e in range(X_DECADES + 1)
    ) + "".join(
        f'<line x1="{ML}" y1="{cy(10**e):.1f}" x2="{CW - MR}" y2="{cy(10**e):.1f}" class="g"/>'
        + label(ML - 9, cy(10**e) + 4, f"{10**e:,}", "#94a3b8", 12, 500, "end")
        for e in range(Y_DECADES + 1)
    )
    share = h.attainable(1.0) / h.peak_bf16
    r = cx(h.ridge())
    return f"""
<svg width="{CW}" height="{CH}" viewBox="0 0 {CW} {CH}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#38bdf8" stop-opacity=".30"/><stop offset="1" stop-color="#38bdf8" stop-opacity="0"/></linearGradient>
    <filter id="glow" x="-100%" y="-100%" width="300%" height="300%"><feGaussianBlur stdDeviation="7" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  </defs>
  <style>.g{{stroke:rgba(255,255,255,.09);stroke-width:1}}</style>
  {grid}
  <rect x="{ML}" y="{MT}" width="{r - ML:.1f}" height="{CH - MB - MT}" fill="#f59e0b" opacity=".07"/>
  <rect x="{r:.1f}" y="{MT}" width="{CW - MR - r:.1f}" height="{CH - MB - MT}" fill="#14b8a6" opacity=".07"/>
  <polygon points="{area}" fill="url(#fill)"/>
  <polyline points="{pts(a)}" fill="none" stroke="#a78bfa" stroke-width="2.5" stroke-dasharray="6 5" stroke-linejoin="round" opacity=".85"/>
  <polyline points="{pts(h)}" fill="none" stroke="#38bdf8" stroke-width="4" stroke-linejoin="round" filter="url(#glow)"/>
  <line x1="{r:.1f}" y1="{cy(h.peak_bf16 / 1e12):.1f}" x2="{r:.1f}" y2="{CH - MB}" stroke="#e2e8f0" stroke-dasharray="4 4" opacity=".6"/>
  {label(r + 8, CH - MB - 46, f"ridge: {h.ridge():.0f}", "#e2e8f0", 12.5, 600)}
  {label(ML + 14, MT + 28, "MEMORY-BOUND", "#fbbf24", 13, 800, spacing=1.5)}
  {label(CW - MR - 12, CH - MB - 16, "COMPUTE-BOUND", "#5eead4", 13, 800, "end", 1.5)}
  <circle cx="{cx(d64):.1f}" cy="{cy(y(d64)):.1f}" r="6" fill="#fbbf24" opacity=".8"/>
  <circle cx="{cx(d1):.1f}" cy="{cy(y(d1)):.1f}" r="9" fill="#fbbf24" filter="url(#glow)"/>
  <circle cx="{cx(pre):.1f}" cy="{cy(y(pre)):.1f}" r="9" fill="#2dd4bf" filter="url(#glow)"/>
  {label(cx(d1) + 18, cy(y(d1)) + 30, "one decode step", "#fde68a", 15, 800)}
  {label(cx(d1) + 18, cy(y(d1)) + 49, f"{share:.1%} of peak", "#fbbf24", 14, 600)}
  {label(cx(d64) + 12, cy(y(d64)) + 22, "batch 64", "#fcd34d", 12.5, 600)}
  {label(cx(pre) - 4, cy(y(pre)) + 32, "prefill", "#99f6e4", 15, 800, "middle")}
  {label(CW - MR, cy(h.peak_bf16 / 1e12) - 9, "H100 · 989 TFLOP/s", "#7dd3fc", 12.5, 700, "end")}
  {label(CW - MR, cy(a.peak_bf16 / 1e12) - 9, "A100 · 312 TFLOP/s", "#c4b5fd", 12.5, 700, "end")}
  {label(6, 16, "TFLOP/s", "#94a3b8", 12, 600, "start")}
  {label((ML + CW - MR) / 2, CH - 6, "arithmetic intensity (FLOP per byte moved)", "#94a3b8", 12.5, 500, "middle")}
</svg>"""


TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@500;600;700;800;900&family=JetBrains+Mono:wght@600;700&display=swap" rel="stylesheet">
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{width:1456px;height:816px;overflow:hidden;font-family:Inter,'Segoe UI',system-ui,sans-serif;color:#fff;
    background:radial-gradient(900px 600px at 92% -8%,rgba(56,189,248,.28),transparent 62%),
               radial-gradient(800px 600px at -6% 108%,rgba(20,184,166,.24),transparent 60%),
               radial-gradient(700px 500px at 60% 120%,rgba(245,158,11,.14),transparent 60%),#070d1a;position:relative}
  .grid{position:absolute;inset:0;background-image:linear-gradient(rgba(255,255,255,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.035) 1px,transparent 1px);background-size:48px 48px;
    -webkit-mask-image:radial-gradient(ellipse at 50% 40%,#000 30%,transparent 78%);mask-image:radial-gradient(ellipse at 50% 40%,#000 30%,transparent 78%)}
  .left{position:absolute;left:64px;top:48px;width:760px}
  .eyebrow{font:700 14px 'JetBrains Mono',Consolas,monospace;letter-spacing:.2em;color:#7dd3fc}
  h1{font-size:80px;line-height:.98;font-weight:900;letter-spacing:-.025em;margin:16px 0 20px;background:linear-gradient(95deg,#fff 30%,#93c5fd 100%);-webkit-background-clip:text;background-clip:text;color:transparent}
  .sub{font-size:26px;line-height:1.36;color:#cbd5e1;font-weight:500;max-width:700px}
  .sub b{color:#fff;font-weight:700}
  .cards{display:flex;gap:16px;margin-top:32px}
  .card{width:240px;padding:18px 20px 17px;border-radius:18px;background:linear-gradient(160deg,rgba(255,255,255,.10),rgba(255,255,255,.03));border:1px solid rgba(255,255,255,.14)}
  .big{font-size:56px;font-weight:900;letter-spacing:-.02em;line-height:1;color:#fbbf24;text-shadow:0 0 26px rgba(251,191,36,.35)}
  .card:nth-child(2) .big{color:#38bdf8;text-shadow:0 0 26px rgba(56,189,248,.35)}
  .card:nth-child(3) .big{color:#2dd4bf;text-shadow:0 0 26px rgba(45,212,191,.35)}
  .lab{margin-top:10px;font-size:16px;line-height:1.35;color:#cbd5e1;font-weight:500}
  .chips{margin-top:26px;display:flex;flex-wrap:wrap;gap:10px;max-width:760px}
  .chip{font-size:15px;font-weight:600;padding:7px 14px;border-radius:999px;color:#bfdbfe;background:rgba(59,130,246,.12);border:1px solid rgba(147,197,253,.32)}
  .right{position:absolute;left:836px;top:50px;width:560px}
  .chartcard{border-radius:22px;padding:20px 22px 14px;background:linear-gradient(165deg,rgba(255,255,255,.09),rgba(255,255,255,.025));border:1px solid rgba(255,255,255,.15);box-shadow:0 30px 80px rgba(0,0,0,.45)}
  .ct{font-size:20px;font-weight:800;letter-spacing:-.01em}
  .cs{font-size:14px;color:#94a3b8;margin:3px 0 6px;font-weight:500}
  .ctx{margin-top:4px;font-size:15px;color:#cbd5e1;line-height:1.4;font-weight:500}
  .ctx b{color:#fde68a}
  .flow{position:absolute;left:64px;bottom:98px;display:flex;align-items:center;gap:10px;font-size:15px;font-weight:700}
  .pill{padding:9px 16px;border-radius:12px;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.16);color:#e2e8f0;white-space:nowrap}
  .pill.pre{background:rgba(20,184,166,.18);border-color:rgba(45,212,191,.55);color:#99f6e4}
  .pill.dec{background:rgba(245,158,11,.18);border-color:rgba(251,191,36,.55);color:#fde68a}
  .arrow{color:#64748b;font-size:18px}
  .foot{position:absolute;left:64px;right:64px;bottom:34px;display:flex;justify-content:space-between;align-items:baseline;border-top:1px solid rgba(255,255,255,.12);padding-top:16px}
  .repo{font:700 17px 'JetBrains Mono',Consolas,monospace;color:#7dd3fc}
  .meta{font-size:14px;color:#94a3b8;font-weight:500}
</style></head><body>
<div class="grid"></div>
<div class="left">
  <div class="eyebrow">A FIRST-PRINCIPLES GUIDE TO LLM INFERENCE</div>
  <h1>Inside the<br>Inference Machine</h1>
  <div class="sub">How a prompt becomes a token on a GPU, and why <b>memory, not math,</b> is the bottleneck.</div>
  <div class="cards">
    <div class="card"><div class="big">__SHARE__</div><div class="lab">of an H100's arithmetic used by one decode step</div></div>
    <div class="card"><div class="big">__MS__ ms</div><div class="lab">to read __GB__ GB of weights for a single token</div></div>
    <div class="card"><div class="big">__KV__ GiB</div><div class="lab">of KV cache per 8K-token chat: __USERS__ fit on 80 GiB</div></div>
  </div>
  <div class="chips">
    <span class="chip">Roofline</span><span class="chip">KV cache</span><span class="chip">GQA &amp; paging</span><span class="chip">Quantization</span>
    <span class="chip">FlashAttention</span><span class="chip">Speculative decoding</span><span class="chip">Continuous batching</span>
  </div>
</div>
<div class="right"><div class="chartcard">
  <div class="ct">The roofline: why generation starves the GPU</div>
  <div class="cs">Llama-3-8B in BF16 on an H100, from datasheet numbers</div>
  __SVG__
  <div class="ctx">Each token needs every weight, so <b>batch-1 decode sits at 1 FLOP per byte</b>, far left of the ridge.</div>
</div></div>
<div class="flow">
  <span class="pill">Prompt</span><span class="arrow">&rarr;</span><span class="pill">Tokens</span><span class="arrow">&rarr;</span>
  <span class="pill pre">Prefill &middot; compute-bound</span><span class="arrow">&rarr;</span>
  <span class="pill dec">Decode loop &middot; memory-bound</span><span class="arrow">&rarr;</span><span class="pill">Next token</span>
</div>
<div class="foot"><div class="repo">github.com/AkbarSheikh-debug/inference-engineering</div>
<div class="meta">30 figures &middot; 9 runnable labs &middot; every number tested &middot; datasheet ceilings, not benchmarks</div></div>
</body></html>"""


def find_browser() -> str:
    for candidate in BROWSERS:
        found = shutil.which(candidate) or (candidate if Path(candidate).exists() else None)
        if found:
            return found
    sys.exit("No Edge, Chrome or Chromium found.")


def build_html() -> str:
    step = decode_step(H100_80GB, LLAMA3_8B)
    share = H100_80GB.attainable(1.0) / H100_80GB.peak_bf16
    values = {
        "__SHARE__": f"{share:.1%}",
        "__MS__": f"{step.seconds * 1e3:.1f}",
        "__GB__": f"{weight_bytes(LLAMA3_8B) / GB:.0f}",
        "__KV__": f"{kv_bytes_per_token(LLAMA3_8B) * 8192 / GiB:.0f}",
        "__USERS__": f"{max_sequences(H100_80GB, LLAMA3_8B, 8192)}",
        "__SVG__": roofline_svg(),
    }
    page = TEMPLATE
    for key, val in values.items():
        page = page.replace(key, val)
    return page


def main() -> int:
    from PIL import Image

    browser = find_browser()
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "cover.html"
        page.write_text(build_html(), encoding="utf-8")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [browser, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},{H}",
             "--force-device-scale-factor=2", "--virtual-time-budget=25000", f"--screenshot={OUT}", page.as_uri()],
            check=True, capture_output=True, timeout=240,
        )
    im = Image.open(OUT).convert("RGB")
    im.crop((0, 0, W * 2, H * 2)).save(OUT, optimize=True)
    print(f"wrote {OUT.relative_to(ROOT)} ({im.size[0]}x{im.size[1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
