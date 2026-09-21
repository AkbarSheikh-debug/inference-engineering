# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
"""Storage cost of quantised weights."""


def bits_per_weight(bits: int, group_size: int | None = None, scale_bits: int = 16) -> float:
    """Effective bits per weight including per-group scale overhead.

    `group_size=None` means one scale per tensor or channel, whose cost is
    negligible. Group-wise schemes store one scale per `group_size` weights.
    """
    if group_size is None:
        return float(bits)
    return bits + scale_bits / group_size


def bytes_per_param(bits: int, group_size: int | None = None, scale_bits: int = 16) -> float:
    return bits_per_weight(bits, group_size, scale_bits) / 8.0
