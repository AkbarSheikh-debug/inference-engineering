# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
"""Lab 02: how big is the KV cache, and how many sequences fit?

    python labs/cpu/02_kv_calculator.py
    python labs/cpu/02_kv_calculator.py --context 32768

Also shows the counterfactual that explains grouped-query attention: the same
model with full multi-head attention (as many KV heads as query heads) and
with multi-query attention (one KV head).
"""

import argparse

from ie.hardware import ALL_GPUS
from ie.kv import kv_bytes, kv_bytes_per_token, kv_pool_bytes, max_sequences, weight_bytes
from ie.models import ALL_MODELS, LLAMA3_8B
from ie.units import GiB, KiB, to_gib


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--mem-util", type=float, default=0.9, help="fraction of HBM the engine may use")
    args = ap.parse_args()

    print("KV bytes per token = 2 x layers x KV heads x head_dim x bytes per element (BF16 = 2)\n")
    print(f"{'model':28}{'KV heads':>9}{'KiB/token':>11}{'GiB @ ' + str(args.context):>14}")
    variants = list(ALL_MODELS) + [
        LLAMA3_8B.with_kv_heads(LLAMA3_8B.n_heads, "Llama-3-8B if MHA"),
        LLAMA3_8B.with_kv_heads(1, "Llama-3-8B if MQA"),
    ]
    for m in variants:
        print(f"{m.name:28}{m.n_kv_heads:>9}{kv_bytes_per_token(m) / KiB:>11.0f}"
              f"{to_gib(kv_bytes(m, args.context)):>14.2f}")

    print(f"\nFull-context sequences that fit (BF16 weights and cache, {args.mem_util:.0%} of HBM, "
          f"context {args.context})")
    for gpu in ALL_GPUS:
        print(f"  {gpu.name} ({gpu.hbm_bytes / GiB:.0f} GiB HBM)")
        for m in variants:
            pool = kv_pool_bytes(gpu, m, mem_util=args.mem_util)
            n = max_sequences(gpu, m, args.context, mem_util=args.mem_util)
            print(f"    {m.name:26} weights {to_gib(weight_bytes(m)):6.2f} GiB, "
                  f"KV pool {to_gib(pool):6.2f} GiB, sequences {n}")

    per_tok = kv_bytes_per_token(LLAMA3_8B)
    w = weight_bytes(LLAMA3_8B)
    print(f"\nLlama-3-8B: KV bytes read per decode step equal the weight bytes when "
          f"batch x context = {w / per_tok:,.0f} tokens.")
    for batch in (8, 16, 32, 64):
        print(f"  batch {batch:>3}: context {w / per_tok / batch:>8,.0f}")


if __name__ == "__main__":
    main()
