# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
"""Draw diagrams/export/d08-continuous-batching.svg from the schedules in ie.batching.

    python tools/batching_svg.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ie.batching import (  # noqa: E402
    EXAMPLE_LENGTHS as LENGTHS,
    EXAMPLE_SLOTS as SLOTS,
    continuous_schedule,
    makespan,
    static_schedule,
    utilisation,
)

INK, MUTED, GRID = "#1f2937", "#6b7280", "#e5e7eb"
BUSY, IDLE = "#2563eb", "#e5e7eb"


def build() -> str:
    static = static_schedule(LENGTHS, SLOTS)
    cont = continuous_schedule(LENGTHS, SLOTS)
    steps = max(makespan(static), makespan(cont))
    x0, cell, row_h = 90, 46, 34
    width = x0 + steps * cell + 40
    panel_h = SLOTS * row_h + 70
    height = 2 * panel_h + 90
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'font-family="Segoe UI, Helvetica, Arial, sans-serif" role="img" '
        f'aria-label="Static versus continuous batching schedule for seven requests on four slots">',
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="10" fill="#ffffff" stroke="{GRID}"/>',
        f'<text x="{width / 2}" y="28" font-size="16" font-weight="bold" fill="{INK}" text-anchor="middle">'
        f'Static versus continuous batching: 7 requests, 4 slots</text>',
    ]
    for p, (title, rows) in enumerate((("Static batching", static), ("Continuous batching", cont))):
        top = 50 + p * (panel_h + 10)
        span, util = makespan(rows), utilisation(rows)
        out.append(f'<text x="20" y="{top + 12}" font-size="14" font-weight="bold" fill="{INK}">{title}: '
                   f'{span} steps, slots busy {util:.0%}</text>')
        for s, row in enumerate(rows):
            y = top + 26 + s * row_h
            out.append(f'<text x="{x0 - 10}" y="{y + 20}" font-size="12" fill="{MUTED}" text-anchor="end">slot {s + 1}</text>')
            out.append(f'<rect x="{x0}" y="{y}" width="{span * cell}" height="{row_h - 6}" fill="{IDLE}"/>')
            for name, start, end in row:
                out.append(f'<rect x="{x0 + start * cell}" y="{y}" width="{(end - start) * cell - 2}" '
                           f'height="{row_h - 6}" rx="3" fill="{BUSY}"/>')
                out.append(f'<text x="{x0 + (start + end) * cell / 2 - 1}" y="{y + 19}" font-size="13" fill="#ffffff" '
                           f'text-anchor="middle" font-weight="bold">{name} ({end - start})</text>')
        axis_y = top + 26 + SLOTS * row_h + 6
        for t in range(0, span + 1, 5):
            out.append(f'<text x="{x0 + t * cell}" y="{axis_y + 10}" font-size="11" fill="{MUTED}" text-anchor="middle">{t}</text>')
        out.append(f'<text x="{x0 + span * cell / 2}" y="{axis_y + 26}" font-size="11" fill="{MUTED}" '
                   f'text-anchor="middle">decode steps</text>')
    out.append(f'<text x="20" y="{height - 12}" font-size="12" fill="{MUTED}">Blue: request running (decode steps). '
               f'Grey: slot idle. Toy schedule with equal-cost steps.</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    dest = ROOT / "diagrams" / "export" / "d08-continuous-batching.svg"
    dest.write_text(build(), encoding="utf-8", newline="\n")
    print(f"wrote {dest.relative_to(ROOT)}")
