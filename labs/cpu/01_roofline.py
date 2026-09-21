# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
"""Lab 01: a roofline ceiling for one decode step.

    python labs/cpu/01_roofline.py --model llama3-8b
    python labs/cpu/01_roofline.py --model llama3-8b --context 8192 --batch 1 16 64

Prints the datasheet-ceiling time per token for a model on each GPU at
several weight precisions. These are upper bounds on speed. Measured numbers
belong in labs/bench/ with the hardware recorded, never in this table.
"""

import argparse

from ie.hardware import ALL_GPUS
from ie.kv import weight_bytes
from ie.models import ALL_MODELS
from ie.quant import bytes_per_param
from ie.roofline import decode_step, ridge_batch
from ie.units import GB

PRECISIONS = (
    ("bf16", bytes_per_param(16)),
    ("int8", bytes_per_param(8)),
    ("int4 (group 128)", bytes_per_param(4, 128)),
)

MODELS = {m.name.lower(): m for m in ALL_MODELS}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="llama-3-8b", choices=sorted(MODELS))
    ap.add_argument("--batch", type=int, nargs="+", default=[1])
    ap.add_argument("--context", type=int, default=0, help="tokens of KV cache per sequence")
    args = ap.parse_args()
    model = MODELS[args.model]

    print(f"{model.name}: {model.params / 1e9:.2f}B parameters, context {args.context}\n")
    for gpu in ALL_GPUS:
        print(f"{gpu.name}: {gpu.bandwidth / 1e12:.3f} TB/s, {gpu.peak_bf16 / 1e12:.0f} TFLOP/s BF16, "
              f"ridge {gpu.ridge():.1f} FLOP/byte")
        print(f"  {'precision':18}{'weights':>10}{'batch':>7}{'ms/step':>10}{'tok/s/seq':>11}{'tok/s total':>13}  bound   ridge batch")
        for name, bpp in PRECISIONS:
            for b in args.batch:
                step = decode_step(gpu, model, batch=b, context=args.context, bytes_per_param=bpp)
                print(f"  {name:18}{weight_bytes(model, bpp) / GB:>8.2f}GB{b:>7}"
                      f"{step.seconds * 1e3:>10.2f}{1 / step.seconds:>11.0f}{b / step.seconds:>13.0f}"
                      f"  {step.bound:8}{ridge_batch(gpu, bpp):>6.0f}")
        print()


if __name__ == "__main__":
    main()
