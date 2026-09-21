# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
"""GPU specifications used by the roofline and capacity calculators.

Peak numbers are datasheet ceilings for dense (non-sparse) tensor-core math.
Real kernels reach a fraction of them. Treat every result derived from these
values as an upper bound on speed, not a prediction.
"""

from dataclasses import dataclass

from .units import GB, GiB, TB


@dataclass(frozen=True)
class Gpu:
    name: str
    hbm_bytes: int  # physical HBM capacity
    bandwidth: float  # HBM bandwidth, bytes/s
    peak_bf16: float  # dense BF16/FP16 tensor FLOP/s
    source: str

    def ridge(self) -> float:
        """Arithmetic intensity (FLOP/byte) where the roofline bends."""
        return self.peak_bf16 / self.bandwidth

    def attainable(self, intensity: float) -> float:
        """Roofline: attainable FLOP/s at a given arithmetic intensity."""
        return min(self.peak_bf16, self.bandwidth * intensity)


# "80 GB" GPUs carry 80 GiB of HBM: nvidia-smi reports about 81,559 MiB on an
# H100 80GB. Capacity is therefore modelled in binary units; bandwidth stays
# decimal, as published.
A100_80GB = Gpu(
    name="A100 80GB SXM",
    hbm_bytes=80 * GiB,
    bandwidth=2039 * GB,
    peak_bf16=312 * TB,
    source="NVIDIA A100 80GB datasheet (SXM: 2,039 GB/s, 312 TFLOP/s BF16 dense)",
)

H100_80GB = Gpu(
    name="H100 80GB SXM",
    hbm_bytes=80 * GiB,
    bandwidth=3350 * GB,
    peak_bf16=989 * TB,
    source="NVIDIA H100 datasheet (SXM: 3.35 TB/s, 989 TFLOP/s BF16 dense)",
)

ALL_GPUS = (A100_80GB, H100_80GB)
