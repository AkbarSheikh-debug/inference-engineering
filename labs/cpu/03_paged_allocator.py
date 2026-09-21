# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
"""Lab 03: contiguous reservation versus paged blocks, on synthetic requests.

    python labs/cpu/03_paged_allocator.py

Both policies get the same KV pool. The contiguous policy reserves max_len
tokens per request up front, because it cannot know how long a request will
run. The paged policy hands out fixed-size blocks. Request lengths are drawn
from a seeded lognormal distribution: synthetic, chosen to be short-tailed
like chat traffic. It shows the mechanism, not a measurement of any engine.
"""

import argparse

from ie.paging import compare_policies


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--pool-seqs", type=int, default=64, help="pool size, in max-length sequences")
    ap.add_argument("--requests", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    r = compare_policies(args.max_len, args.block_size, args.pool_seqs, args.requests, args.seed)
    pool = r["pool_tokens"]

    print(f"pool: {pool:,} token slots; mean request {r['mean_length']:.0f} tokens, "
          f"max_len {args.max_len}, block size {args.block_size}\n")
    print(f"{'':22}{'contiguous':>14}{'paged':>10}")
    print(f"{'requests admitted':22}{r['contiguous_admitted']:>14}{r['paged_admitted']:>10}")
    print(f"{'slots holding tokens':22}{r['contiguous_used'] / pool:>14.1%}{r['paged_used'] / pool:>10.1%}")
    print(f"{'wasted inside seqs':22}{r['contiguous_waste']:>14,}{r['paged_waste']:>10,}")
    print("\nSimplification: lengths are known at admission. A real engine allocates blocks as"
          "\ndecoding proceeds and must preempt or swap a sequence when the pool runs dry.")


if __name__ == "__main__":
    main()
