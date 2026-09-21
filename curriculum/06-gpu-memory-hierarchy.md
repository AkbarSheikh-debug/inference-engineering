# 06. The GPU memory hierarchy: where data lives matters more than how fast you multiply

**Question this answers:** a GPU can do hundreds of trillions of operations per second, yet performance engineers spend their time arguing about where bytes sit. Why?

## Four places a number can live

Every value a GPU multiplies starts in HBM and has to be staged closer to the arithmetic units first. Each stage is faster and smaller than the one before it.

| Level | Scope | A100 | H100 | Latency (order of magnitude) |
|---|---|---|---|---|
| Registers | private to one thread | 256 KB per SM | 256 KB per SM | about one cycle |
| Shared memory and L1 | private to one SM | 192 KB per SM (up to 164 KB as shared memory) | 256 KB per SM (up to 228 KB as shared memory) | tens of nanoseconds |
| L2 cache | shared by every SM | 40 MB | 50 MB | a couple of hundred nanoseconds |
| HBM | shared by every SM | 80 GiB | 80 GiB | several hundred nanoseconds |

The A100 has 108 SMs and the H100 SXM has 132. Latencies are typical order-of-magnitude figures for this class of part; they vary by generation and access pattern, and this repository has not measured them. The capacities come from NVIDIA's architecture whitepapers.

Two facts about this table drive everything below. First, **capacity and speed trade off steeply**: the on-chip levels together hold a few tens of megabytes, while HBM holds tens of gigabytes. Second, **the L2 is one resource shared by over a hundred SMs**. To serve them all it is split into slices behind a crossbar, so any SM can reach any slice, at the cost of the crossbar being contended under heavy traffic.

## Coalescing: 32 loads, how many trips?

A warp executes one instruction across 32 threads. When that instruction is a load, each thread asks for its own address. The memory system does not serve 32 separate requests. It looks at the 32 addresses and fetches the **32-byte sectors** that contain them (four sectors make a 128-byte cache line). The sector is the smallest unit it will fetch, so the bytes moved are the number of distinct sectors touched times 32, whatever the threads actually needed.

`labs/cpu/06_gpu_memory.py` counts this for a warp of 32 threads each loading 4 bytes, as the stride between consecutive threads grows:

| Stride between threads | Sectors touched | Useful bytes | Bytes moved | Efficiency |
|---|---|---|---|---|
| 4 B | 4 | 128 | 128 | 100.0% |
| 8 B | 8 | 128 | 256 | 50.0% |
| 16 B | 16 | 128 | 512 | 25.0% |
| 32 B | 32 | 128 | 1024 | 12.5% |
| 64 B | 32 | 128 | 1024 | 12.5% |
| 128 B | 32 | 128 | 1024 | 12.5% |
| 256 B | 32 | 128 | 1024 | 12.5% |

Read the shape of that table. Efficiency falls as the stride grows, then **stops falling at 12.5 percent** once every thread owns a different sector, because 32 sectors of 32 bytes is the most a 4-byte-per-thread warp can drag in. The penalty for fully scattered 4-byte loads is therefore 8x, not 32x. The 32x figure applies only to 1-byte loads, where a fully scattered warp moves 32 sectors to use 32 bytes (3.125 percent). The lesson is not the exact number. It is that the same data requested in a different layout can cost several times the bandwidth, and bandwidth is what decode is starved of ([article 02](02-why-decode-is-slow.md)).

## Shared memory: reuse you have to ask for

The L1 and L2 caches are automatic. The hardware decides what to keep, and gives no guarantee that a value you will need again in a few microseconds is still there. Shared memory is the opposite: a fixed-size scratchpad that a kernel loads explicitly and reuses explicitly, with a guarantee that the data stays until the kernel evicts it.

The classic use is tiling a matrix multiply. To compute an output tile of `T x T` values, a thread block loads a `T`-row strip of A and a `T`-column strip of B into shared memory once, then performs `2 T^2 K` operations on them. Loaded bytes per output tile are proportional to `2 T K`, so:

```
intensity (FLOP per byte) is about  T / bytes_per_element
```

The tile size is the reuse factor. The lab multiplies two 4096 x 4096 FP16 matrices (137.4 GFLOP) and counts the bytes read as the tile grows. The closed form is checked against an explicit walk of the tile grid, including tiles that do not divide the matrix.

| Output tile | Bytes read | FLOP per byte |
|---|---|---|
| 1 x 1 (no reuse) | 256.000 GiB | 0.5 |
| 16 x 16 | 16.000 GiB | 8.0 |
| 32 x 32 | 8.000 GiB | 15.9 |
| 64 x 64 | 4.000 GiB | 31.8 |
| 128 x 128 | 2.000 GiB | 63.0 |
| 256 x 256 | 1.000 GiB | 124.1 |

Doubling the tile halves the traffic. That is the mechanism behind the "raise intensity" arrow of the roofline: it is how a kernel moves a workload to the right without changing the algorithm. Two honest caveats. The intensity here is measured against the level feeding the shared-memory tile; real kernels get further reuse from the L2, and the tile that fits is bounded by the shared-memory capacity in the table above and by how many thread blocks an SM can hold at once. So bigger is not always better.

## What this means for inference

Tiling creates reuse only where the algorithm has some. A large matrix multiply, such as prefill, has plenty: every weight is used against many tokens. A decode step at batch 1 is a matrix-vector product, where each weight is used **once**, so no tile size can raise its intensity above about 1 FLOP per byte. The only way to create reuse there is to put more tokens against each weight, which is batching ([article 05](05-continuous-batching.md)), or to verify several tokens at once ([article 08](08-speculative-decoding.md)).

The same logic explains fusion and FlashAttention ([article 07](07-flashattention.md)). Each is an exercise in keeping data at the fastest level for as long as it is needed, and in never writing an intermediate to HBM and reading it straight back.

## Common misconceptions

- **"Coalescing is a problem of old GPUs."** The sector model above is how current parts fetch global memory. A layout mismatch still multiplies the bytes moved.
- **"Shared memory is just a slower register file."** It is the only level the programmer controls explicitly, and it is shared across the threads of a block. That is what makes cooperative tiling possible.
- **"A bigger tile is always better."** It raises intensity but uses more shared memory and registers, which reduces how many thread blocks fit on an SM.

## Check yourself

1. A warp loads 4 bytes per thread with a stride of 12 bytes. Use the lab (or work it out) to say how many sectors it touches and what efficiency it gets.
2. Why does the 128 x 128 tile in the table read exactly twice as much as the 256 x 256 tile?
3. A decode step has intensity about 1 at batch 1. Which single change moves it right, and why does no tile size help?

## Reproduce

```bash
python labs/cpu/06_gpu_memory.py
```

## Sources

- NVIDIA, *NVIDIA A100 Tensor Core GPU Architecture* (whitepaper, 2020), and *NVIDIA H100 Tensor Core GPU Architecture* (whitepaper, 2022), for SM counts, register file, L1/shared memory and L2 sizes.
- NVIDIA, *CUDA C++ Best Practices Guide*, section on coalesced access to global memory, for the 32-byte sector as the unit of a memory transaction.
- Williams, Waterman, Patterson, *Roofline: an insightful visual performance model for multicore architectures*, Communications of the ACM 52(4), 2009.

---

Copyright (c) 2026 Akbar Arif. Released under the [MIT License](../LICENSE). You may reuse and adapt this article as long as the copyright and license notice stays with it.
