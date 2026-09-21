# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
"""Lab 00: generate tokens with and without a KV cache, and count the difference.

    python labs/cpu/00_tiny_decoder.py --prompt-len 32 --new-tokens 32

The model is a random-weight toy (see src/ie/tinylm.py). What matters is that
the cached and uncached paths produce identical tokens and logits while doing
very different amounts of work.
"""

import argparse

import numpy as np

from ie.tinylm import TinyConfig, TinyLM


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prompt-len", type=int, default=32)
    ap.add_argument("--new-tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = TinyConfig(seed=args.seed)
    model = TinyLM(cfg)
    rng = np.random.default_rng(args.seed + 1)
    prompt = [int(t) for t in rng.integers(0, cfg.vocab, size=args.prompt_len)]

    slow = model.generate(prompt, args.new_tokens, use_cache=False)
    fast = model.generate(prompt, args.new_tokens, use_cache=True)

    assert slow.tokens == fast.tokens, "cached and uncached generation disagree"
    max_diff = float(np.max(np.abs(slow.logits - fast.logits)))

    print(f"prompt {args.prompt_len} tokens, generate {args.new_tokens} tokens, "
          f"{cfg.n_heads} query heads sharing {cfg.n_kv_heads} KV heads")
    print(f"identical tokens: yes; max |logit difference|: {max_diff:.2e}\n")
    print(f"{'':22}{'no cache':>14}{'KV cache':>14}")
    print(f"{'forward passes':22}{slow.stats.forward_calls:>14}{fast.stats.forward_calls:>14}")
    print(f"{'positions processed':22}{slow.stats.positions_processed:>14}{fast.stats.positions_processed:>14}")
    print(f"{'matmul FLOPs':22}{slow.stats.flops:>14,}{fast.stats.flops:>14,}")
    print(f"\nFLOP ratio (no cache / cache): {slow.stats.flops / fast.stats.flops:.1f}x")

    per_pass = model.params_read_per_step()
    print(f"\nWeights streamed per forward pass: {per_pass:,} parameters.")
    print(f"Prefill does {args.prompt_len} positions per pass over the weights; "
          f"each decode step does 1.")


if __name__ == "__main__":
    main()
