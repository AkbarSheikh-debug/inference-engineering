# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
"""Draw diagrams/export/d04-roofline.svg from the specs in ie.hardware.

Pure Python, no plotting dependency. Log-log axes: arithmetic intensity
(FLOP/byte) against attainable TFLOP/s, with the A100 and H100 roofs and the
arithmetic intensity of BF16 weight matmuls at a few batch sizes.

    python tools/roofline_svg.py
"""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ie.hardware import A100_80GB, H100_80GB  # noqa: E402
from ie.roofline import weight_intensity  # noqa: E402

W, H = 900, 580
X0, X1 = 100, 860  # plot area, pixels
Y0, Y1 = 50, 470
X_DECADES, Y_DECADES = 4, 3  # x: 1..1e4 FLOP/byte, y: 1..1e3 TFLOP/s

INK, MUTED, GRID = "#1f2937", "#6b7280", "#e5e7eb"
COLORS = {A100_80GB.name: "#7c3aed", H100_80GB.name: "#0369a1"}


def fx(intensity: float) -> float:
    return X0 + math.log10(intensity) / X_DECADES * (X1 - X0)


def fy(tflops: float) -> float:
    return Y1 - math.log10(tflops) / Y_DECADES * (Y1 - Y0)


def text(x, y, s, size=13, fill=INK, anchor="start", weight="normal") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{s}</text>')


def roof_path(gpu) -> str:
    pts = [(1.0, gpu.attainable(1.0)), (gpu.ridge(), gpu.peak_bf16), (10.0**X_DECADES, gpu.peak_bf16)]
    return " ".join(f"{fx(x):.1f},{fy(y / 1e12):.1f}" for x, y in pts)


def build() -> str:
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'font-family="Segoe UI, Helvetica, Arial, sans-serif" role="img" '
        f'aria-label="Roofline plot for A100 and H100 with LLM decode and prefill operating points">',
        f'<rect x="0" y="0" width="{W}" height="{H}" rx="10" fill="#ffffff" stroke="{GRID}"/>',
        text(W / 2, 30, "Roofline: why batch-1 decode leaves the GPU almost idle", 17, INK, "middle", "bold"),
    ]
    for e in range(X_DECADES + 1):
        x = fx(10.0**e)
        out.append(f'<line x1="{x:.1f}" y1="{Y0}" x2="{x:.1f}" y2="{Y1}" stroke="{GRID}"/>')
        out.append(text(x, Y1 + 20, f"{10**e:,}", 12, MUTED, "middle"))
    for e in range(Y_DECADES + 1):
        y = fy(10.0**e)
        out.append(f'<line x1="{X0}" y1="{y:.1f}" x2="{X1}" y2="{y:.1f}" stroke="{GRID}"/>')
        out.append(text(X0 - 8, y + 4, f"{10**e:,}", 12, MUTED, "end"))
    out.append(text((X0 + X1) / 2, Y1 + 46, "arithmetic intensity (FLOP per byte moved from HBM)", 13, INK, "middle"))
    out.append(f'<text transform="translate(30,{(Y0 + Y1) / 2:.0f}) rotate(-90)" font-size="13" fill="{INK}" '
               f'text-anchor="middle">attainable performance (TFLOP/s, BF16 dense)</text>')

    for gpu in (A100_80GB, H100_80GB):
        c = COLORS[gpu.name]
        out.append(f'<polyline points="{roof_path(gpu)}" fill="none" stroke="{c}" stroke-width="3" stroke-linejoin="round"/>')
        rx = fx(gpu.ridge())
        out.append(f'<line x1="{rx:.1f}" y1="{fy(gpu.peak_bf16 / 1e12):.1f}" x2="{rx:.1f}" y2="{Y1}" '
                   f'stroke="{c}" stroke-dasharray="4 4" stroke-width="1.5"/>')

    a, h = A100_80GB, H100_80GB
    out.append(text(fx(1e4) - 6, fy(h.peak_bf16 / 1e12) - 8, f"H100: {h.peak_bf16 / 1e12:.0f} TFLOP/s", 13, COLORS[h.name], "end", "bold"))
    out.append(text(fx(1e4) - 6, fy(a.peak_bf16 / 1e12) + 18, f"A100: {a.peak_bf16 / 1e12:.0f} TFLOP/s", 13, COLORS[a.name], "end", "bold"))
    out.append(text(fx(h.ridge()) + 6, Y1 - 8, f"H100 ridge {h.ridge():.0f}", 12, COLORS[h.name]))
    out.append(text(fx(a.ridge()) - 6, Y1 - 8, f"A100 ridge {a.ridge():.0f}", 12, COLORS[a.name], "end"))
    out.append(text(fx(12), fy(1.6), "slope = HBM bandwidth", 12, MUTED))

    points = [
        (weight_intensity(1), "decode, batch 1", 12, 24, "start"),
        (weight_intensity(16), "decode, batch 16", 12, 26, "start"),
        (weight_intensity(64), "decode, batch 64", 10, 26, "start"),
        (weight_intensity(2048), "prefill, 2,048 tokens", 0, 28, "middle"),
    ]
    for inten, label, dx, dy, anchor in points:
        x, y = fx(inten), fy(h.attainable(inten) / 1e12)
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="#dc2626" stroke="#ffffff" stroke-width="2"/>')
        share = h.attainable(inten) / h.peak_bf16
        out.append(text(x + dx, y + dy, f"{label}: {share:.1%} of peak", 12, "#991b1b", anchor))

    out.append(text(X0, H - 36,
                    "BF16 weights: intensity = batch size (2 FLOPs per token per weight, 2 bytes per weight). "
                    "Points shown on the H100 roof.", 12, MUTED))
    out.append(text(X0, H - 16,
                    f"Roof slopes are HBM bandwidth: H100 {h.bandwidth / 1e12:.2f} TB/s, "
                    f"A100 {a.bandwidth / 1e12:.3f} TB/s. Datasheet peaks; real kernels reach a fraction.", 12, MUTED))
    out.append("</svg>")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    dest = ROOT / "diagrams" / "export" / "d04-roofline.svg"
    dest.write_text(build(), encoding="utf-8", newline="\n")
    print(f"wrote {dest.relative_to(ROOT)}")
