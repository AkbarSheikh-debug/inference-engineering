# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
"""Documentation checks: no em-dashes in prose, and every relative link resolves.

    python tools/check_docs.py
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EM_DASH = "—"
LINK = re.compile(r"\]\(([^)\s]+)\)")


def markdown_files() -> list[Path]:
    skip = {".git", ".pytest_cache", "node_modules"}
    return [p for p in ROOT.rglob("*.md") if not skip & set(p.relative_to(ROOT).parts)]


def problems() -> list[str]:
    found: list[str] = []
    for path in markdown_files():
        rel = path.relative_to(ROOT)
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if EM_DASH in line:
                found.append(f"{rel}:{n}: em-dash in prose")
            for target in LINK.findall(line):
                if target.startswith(("http://", "https://", "#", "mailto:")):
                    continue
                raw = target.split("#")[0]
                dest = (ROOT / raw.lstrip("/") if raw.startswith("/") else path.parent / raw).resolve()
                if not dest.exists():
                    found.append(f"{rel}:{n}: broken link {target}")
    return found


if __name__ == "__main__":
    issues = problems()
    for issue in issues:
        print(issue)
    print(f"{len(issues)} problem(s) in {len(markdown_files())} markdown files")
    sys.exit(1 if issues else 0)
