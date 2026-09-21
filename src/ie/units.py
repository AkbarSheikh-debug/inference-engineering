# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
"""Unit constants. State the system once and stay in it.

Vendors quote bandwidth and FLOP/s in decimal units (TB/s, TFLOP/s) but
memory capacity in what is physically a binary quantity. Block and cache
arithmetic is naturally binary. Mixing the two silently produces 2 to 7
percent errors that still look plausible, so every conversion goes through
this module.
"""

KB = 10**3
MB = 10**6
GB = 10**9
TB = 10**12

KiB = 2**10
MiB = 2**20
GiB = 2**30


def to_gib(n_bytes: float) -> float:
    return n_bytes / GiB


def to_gb(n_bytes: float) -> float:
    return n_bytes / GB
