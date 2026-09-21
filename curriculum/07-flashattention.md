# 07. FlashAttention: exact attention without the n-by-n matrix

**Question this answers:** attention builds a score for every pair of tokens. How can it be computed exactly without ever storing that matrix, and what does that actually save?

## The problem: the score matrix

For one head, attention computes `S = Q K^T / sqrt(d)`, a matrix with one score for every pair of positions, applies softmax to each row, then multiplies by `V`. Written naively, that is four passes over an `n x n` matrix in HBM: write `S`, read it back for the softmax, write the result `P`, read it again to multiply by `V`.

The matrix is large. In FP16, one head at `n = 65,536` needs 8 GiB for `S` alone, and at `n = 32,768` with 32 heads it is 64 GiB. It grows with the square of the context length, so it is a wall for long contexts before the KV cache of [article 03](03-kv-cache.md) even comes into it.

FlashAttention computes the same output without ever writing `S` to HBM. It splits `Q`, `K` and `V` into blocks small enough to sit in on-chip memory ([article 06](06-gpu-memory-hierarchy.md)). For each block of queries it streams over every block of keys and values, computes that block's scores on chip, folds them into a running result, and throws them away.

## The obstacle: softmax needs the whole row

Softmax subtracts the row maximum for numerical stability and divides by the row sum, so it seems to need every score of the row at once, which is exactly what blocking prevents. The fix is the **online softmax** (Milakov and Gimelshein, 2018). For each query row keep three running values: the maximum `m` so far, the running sum `l`, and the running output `o`. When a new block arrives with local maximum `m_block`:

```
m_new = max(m_old, m_block)
alpha = exp(m_old - m_new)                 # rescales everything accumulated so far
l     = l * alpha + sum(exp(s - m_new))
o     = o * alpha + sum(exp(s - m_new) * v)
```

After the last block, the answer is `o / l`. The factor `alpha` is a change of reference point: because `exp(s - m_new) = exp(s - m_old) * exp(m_old - m_new)`, every term accumulated against the old maximum moves onto the new one with a single multiplication and no loss of information.

## One row, traced

Take one row with eight scores in two blocks of four. `labs/cpu/07_flash_attention.py` runs the update:

- Block 1 is `[0.515, 0.385, 0.405, 0.435]`. The running maximum is `m = 0.515` and the running sum is `l = 3.697`.
- Block 2 is `[0.560, 0.300, 0.370, 0.320]`. Its maximum, 0.560, exceeds 0.515, so everything so far is stale and must be rescaled by `alpha = exp(0.515 - 0.560) = 0.956`. The sum becomes `l = 0.956 x 3.697 + 3.385 = 6.919`.

That is exactly the sum a single softmax over all eight scores produces, `sum(exp(s - 0.560)) = 6.919`. Not approximately: exactly.

## It really is exact

The lab runs the tiled algorithm against direct attention on random data (`n = 200`, `d = 32`, float64) for several block shapes, including ones that do not divide the sequence length, and with the key blocks visited in reverse order:

(The exact digits of the differences depend on your BLAS library and CPU; they are all around 1e-16.)

| Blocks (query x key) | Largest difference from direct attention | Largest score block held |
|---|---|---|
| 64 x 64 | 4.4e-16 | 4,096 of 40,000 |
| 37 x 29 | 5.6e-16 | 1,073 of 40,000 |
| 200 x 200 | 6.7e-16 | 40,000 of 40,000 |
| 7 x 3 | 5.6e-16 | 21 of 40,000 |
| 37 x 29, reversed order | 6.7e-16 | |

Every difference is at the level of floating-point rounding. FlashAttention is not a trade of accuracy for memory. It is the same arithmetic in a different order.

## What it saves, and what it does not

Be precise here, because it is commonly overstated. Two different quantities are involved.

**Extra memory.** Standard attention needs `O(n^2)` storage for `S`. The tiled algorithm needs only the running statistics, `O(n)` beyond the inputs and output. This is what "exact attention with O(n) memory" means.

**HBM traffic.** This shrinks by a constant factor, not an exponent. Each block of `B` query rows must stream past *every* block of keys and values, so `K` and `V` are re-read `n / B` times, and total traffic still scales as `n^2`. The published analysis gives `Theta(n^2 d^2 / M)` HBM accesses for head dimension `d` and on-chip memory `M`, against `Theta(nd + n^2)` for standard attention (Dao et al., 2022, Theorem 2). The lab counts elements moved for head dimension 128, query block 128, FP16:

| n | Standard (MiB) | Tiled (MiB) | Ratio | Score matrix per head (GiB) |
|---|---|---|---|---|
| 1,024 | 9 | 4 | 2.00 | 0.00 |
| 4,096 | 132 | 66 | 2.00 | 0.03 |
| 16,384 | 2,064 | 1,032 | 2.00 | 0.50 |
| 65,536 | 32,832 | 16,416 | 2.00 | 8.00 |

The ratio is `2B / d`, and it does not change with `n`, because both columns grow as `n^2`. A larger query block cuts traffic in proportion, but the block is bounded by on-chip capacity. So the headline benefit at these settings is not a dramatic drop in traffic. It is that the `n x n` matrix never exists, so context length is no longer limited by it, and that the remaining traffic is streamed through fast memory in one fused kernel instead of bouncing through HBM four times. This model counts elements only. It says nothing about kernel launch costs or achieved bandwidth, and this repository has no measured speedups yet.

Also note what does not change: the arithmetic. FlashAttention performs the same `O(n^2 d)` floating-point operations. It is a memory optimisation, not a FLOP optimisation.

## What about decode?

During decode there is one query per sequence, so the score "matrix" is a single row of length `n`. There is no `n x n` structure to avoid. What decode reads is the KV cache ([article 03](03-kv-cache.md)), and that read is the cost. FlashAttention's largest effect is in prefill and training, where the full matrix would otherwise exist.

## Common misconceptions

- **"FlashAttention is an approximation."** It is exact, to floating-point rounding, as the table shows.
- **"FlashAttention makes attention linear."** Memory becomes linear. Arithmetic stays quadratic, and HBM traffic stays quadratic with a smaller constant.
- **"It reduces FLOPs."** It does not. The savings come from data movement.

## Check yourself

1. Why does the running sum have to be multiplied by `alpha` when a larger maximum appears, and what would go wrong if it were not?
2. In the traffic table, why is the ratio 2.00 at every `n`? What would make it larger?
3. A colleague says FlashAttention "removes the quadratic term." State exactly which quadratic quantity it removes and which it keeps.

## Reproduce

```bash
python labs/cpu/07_flash_attention.py
```

## Sources

- Dao, Fu, Ermon, Rudra, Re, *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, NeurIPS 2022. arXiv:2205.14135.
- Milakov and Gimelshein, *Online normalizer calculation for softmax*, 2018. arXiv:1805.02867.

---

Copyright (c) 2026 Akbar Arif. Released under the [MIT License](../LICENSE). You may reuse and adapt this article as long as the copyright and license notice stays with it.
