<!-- TITLE: Inside the Inference Machine: How LLMs Actually Run on a GPU -->
<!-- SUBTITLE: A bottleneck-first guide to inference engineering, with 30 figures, runnable code, and every number checked. -->

![Inside the Inference Machine](https://raw.githubusercontent.com/AkbarSheikh-debug/inference-engineering/main/blog/images/c00-cover.png)

*The three numbers this post explains, for Llama-3-8B on an H100. All are datasheet arithmetic, not benchmarks.*

An H100 can perform about 989 trillion BF16 operations per second. Generating one token from an 8-billion-parameter model uses 0.3 percent of that.

That is not a bug, and it is not a badly written model server. It is the central fact of LLM inference, and once you see it, almost everything else in this field stops being a pile of tricks and becomes a set of consequences.

This post is a long, from-first-principles tour of how a large language model actually runs on a GPU. It follows one prompt from the moment you press enter to the moment a token comes back, and at every step it asks the same question: **which resource is the bottleneck, and why?**

**The short version:**

- Generating text is a loop. The first pass (prefill) is limited by arithmetic. Every pass after that (decode) is limited by memory bandwidth.
- A decode step must read every weight of the model once. For Llama-3-8B that is 16 GB, which takes about 4.8 ms on an H100. The arithmetic takes 0.016 ms.
- The KV cache trades memory for compute. One 8K-token conversation on Llama-3-8B holds exactly 1 GiB, so an 80 GiB GPU serves about 57 of them.
- Batching, quantization, FlashAttention, paging, and speculative decoding each attack a specific bottleneck. Knowing which one is the whole skill.
- Every number in this post comes from small, tested Python code you can run on a laptop. No GPU required.

**A note on the numbers.** Everything here is arithmetic from published datasheets, plus small simulations. These are ceilings and mechanisms, not benchmarks. Where a result depends on an assumption, I say so. All the code is in the open-source repo:

[github.com/AkbarSheikh-debug/inference-engineering](https://github.com/AkbarSheikh-debug/inference-engineering)

It contains the articles this post is built from, nine runnable labs, and tests that fail if an article quotes a number the code does not produce. There is also a [readable site version](https://akbarsheikh-debug.github.io/inference-engineering/).

## 1. One question that organises everything

Most writing about LLM inference is a list of techniques: quantization, batching, paged attention, speculative decoding, FlashAttention. Lists are hard to remember and easy to misapply. So instead, I want to give you one question to carry through the whole post:

**Which resource is the bottleneck?**

There are only a handful of answers:

- **Memory bandwidth:** how fast bytes can move from GPU memory to the arithmetic units.
- **Memory capacity:** how many bytes fit on the card at all.
- **Compute:** how many operations per second the chip can do.
- **Launch overhead:** the fixed cost of starting work.
- **Idle time:** the GPU waiting because the scheduler has nothing ready.
- **Interconnect:** how fast GPUs can talk to each other.

Every optimization you have heard of attacks exactly one of these. Quantization attacks bandwidth and capacity. Paging attacks capacity. Continuous batching attacks idle time. Once you can name the bottleneck of a workload and show it with arithmetic, choosing the technique is close to mechanical.

Everything lives in one of five layers, and knowing which layer owns a problem tells you who can fix it:

![The five-layer inference stack](https://raw.githubusercontent.com/AkbarSheikh-debug/inference-engineering/main/diagrams/export/d02-layered-stack.png)

*The five layers of the inference stack, and what limits each one. Requests travel down, tokens travel back up.*

## 2. What actually happens between a prompt and a token

The model does not write a reply. It computes **one token**, appends it to the input, and runs again. A reply of 300 tokens is 300 trips through the same loop. Streaming is not a user-interface trick; it is how generation works.

![Request lifecycle](https://raw.githubusercontent.com/AkbarSheikh-debug/inference-engineering/main/diagrams/export/d01-request-lifecycle.png)

*One request from prompt to streamed tokens. The first trip (prefill) handles the whole prompt at once; every later trip (decode) handles one new token.*

### The six steps of one trip

**1. Tokenize.** Text becomes a list of integer ids using a fixed subword vocabulary. Llama 3 has 128,256 tokens. The model never sees characters, only ids.

**2. Embed.** Each id selects a row of a learned table, producing one vector per token. This is a lookup, not a matrix multiply, so it touches a tiny part of the weights.

**3. Run the transformer blocks.** Each block does two things to every position. Attention lets a position read from earlier positions, and the MLP transforms each position on its own. Residual connections add each result back to its input, and a normalization step (RMSNorm in Llama-style models) keeps values stable. Llama-3-8B stacks 32 of these blocks.

**4. Project to logits.** The last position's vector is multiplied by a large matrix that maps it to one score per vocabulary entry.

**5. Sample.** Turn logits into one token id. Greedy decoding takes the highest score; temperature, top-k, and top-p sampling draw from the distribution instead.

**6. Append and repeat.** The chosen token joins the sequence and the loop runs again until an end-of-sequence token or a length limit.

### Two phases with opposite personalities

The first trip is special. The prompt is already known, so the engine can push all of it through the network at once. That is **prefill**. After that, each new token depends on the previous one, so tokens must be produced one at a time. That is **decode**.

![Prefill versus decode](https://raw.githubusercontent.com/AkbarSheikh-debug/inference-engineering/main/diagrams/export/d03-prefill-decode.png)

*Prefill handles many tokens per pass over the weights and is compute-bound. Decode handles one token per sequence per pass and is memory-bound.*

The user-visible consequences are direct. **Time to first token (TTFT)** is dominated by prefill and grows with prompt length. **Inter-token latency (ITL)** is dominated by decode and depends on how fast weights and cache can be read from memory.

### What one token costs

A transformer does about two floating-point operations per parameter for each token it processes (a multiply and an add), ignoring attention's dependence on context length. An 8-billion-parameter model therefore needs about 16 GFLOP per token. It also has to read its weights: at 2 bytes per parameter in BF16 that is about 16 GB.

Those two numbers, 16 GFLOP of work and 16 GB of bytes, are the raw material for the rest of this post.

## 3. Why decode is slow on a machine that can do a petaflop

Take Llama-3-8B in BF16 on an H100. One decode step must read every weight once and do about two operations with each.

- The GPU can move data from HBM (its main memory) at 3.35 TB/s. Moving 16.06 GB takes **4.79 ms**.
- The GPU can compute at 989 TFLOP/s. Doing 16 GFLOP takes **0.016 ms**.

The memory read is roughly 300 times longer than the arithmetic. The arithmetic units spend almost the entire step waiting for weights to arrive. They are not slow. They are starving.

### Arithmetic intensity and the roofline

The ratio that captures this is **arithmetic intensity**: floating-point operations performed per byte moved from memory. For a weight matrix used by a batch of `b` tokens, each weight costs `bytes_per_param` bytes to fetch and is used for 2 operations per token:

```
intensity (FLOP per byte) = 2 x b / bytes_per_param
```

In BF16 that is simply **intensity = batch size**. One sequence decoding alone has intensity 1: one operation for every byte fetched.

A GPU has two ceilings: a compute ceiling (its peak FLOP/s) and a memory ceiling (bandwidth times intensity). Attainable performance is the lower of the two. Plotted on log axes it looks like a roof, and the intensity where the slope meets the flat part is the **ridge point**.

![Roofline for A100 and H100](https://raw.githubusercontent.com/AkbarSheikh-debug/inference-engineering/main/diagrams/export/d04-roofline.png)

*The roofline for an A100 and an H100. The ridge points are 153 and 295 FLOP per byte. A batch-1 decode step sits at intensity 1, on the far left.*

Left of the ridge a workload is **memory-bound**: buying more compute changes nothing. Right of it, **compute-bound**: buying more bandwidth changes nothing. A batch-1 decode step on an H100 uses **0.3 percent of peak** arithmetic throughput. Batch 16 reaches 5.4 percent, batch 64 reaches 21.7 percent, and only near a batch of about 295 do the weight matrices become compute-bound.

### The ceiling on tokens per second

If a step is memory-bound, its time is just the weight bytes divided by bandwidth. Here is the core of the model in the repo (`src/ie/roofline.py`):

```python
def decode_step(gpu, model, batch=1, context=0, bytes_per_param=2.0, kv_dtype_bytes=2):
    bytes_moved = weight_bytes(model, bytes_per_param) + batch * context * kv_bytes_per_token(model, kv_dtype_bytes)
    flops = 2.0 * model.params * batch
    return Step(memory_s=bytes_moved / gpu.bandwidth, compute_s=flops / gpu.peak_bf16)
```

The step takes as long as the slower of the two sides. For Llama-3-8B at batch 1, the ceilings are:

![Decode ceilings by precision](https://raw.githubusercontent.com/AkbarSheikh-debug/inference-engineering/main/blog/images/c01-decode-ceilings.png)

*Batch-1 decode ceilings. Halving the bytes per weight doubles the ceiling, because the step is limited by bytes, not arithmetic.*

An H100 tops out around 209 tokens per second per sequence in BF16, 417 in INT8, and 809 in 4-bit. An A100 gets 127, 254, and 492. **These are ceilings, not predictions.** They use datasheet peaks and ignore attention arithmetic, kernel launch overhead, and communication. Real engines land below them.

Notice what the H100 and A100 comparison says. The speedup between them (209 versus 127) tracks the bandwidth ratio (3,350 versus 2,039 GB/s), not the compute ratio (989 versus 312 TFLOP/s). A faster GPU only makes decode faster if it has more memory bandwidth.

### Prefill is the opposite

A 2,048-token prompt has an arithmetic intensity of 2,048 in BF16, far to the right of the ridge, so it is compute-bound. Its ceiling is the arithmetic time: about 33.3 ms on an H100. That is longer than one decode step (4.79 ms), but it processes 2,048 tokens instead of one.

![Prefill versus decode time](https://raw.githubusercontent.com/AkbarSheikh-debug/inference-engineering/main/blog/images/c17-prefill-vs-decode.png)

*Same GPU, same weights, opposite bottleneck. A prefill pass takes longer, but each token costs about 295 times less. That ratio is the ridge point in disguise.*

### Three ways to move a workload

Every decode optimization attacks the same equation from one side:

1. **Raise intensity by batching.** Serve many sequences per weight read, so intensity grows with batch size.
2. **Move fewer bytes.** Quantization shrinks each weight. INT8 doubles intensity at a fixed batch and halves the batch needed to reach the ridge.
3. **Get more tokens per weight read.** Speculative decoding proposes several tokens cheaply and verifies them in one pass.

Batching is nearly free while you are left of the ridge:

![Batch scaling on an H100](https://raw.githubusercontent.com/AkbarSheikh-debug/inference-engineering/main/blog/images/c02-batch-scaling.png)

*Aggregate throughput rises linearly with batch size until the ridge, then flattens. Per-sequence speed is flat until the ridge, then falls. This ignores the KV cache, which section 5 adds back in.*

Buying a GPU with more FLOP/s and the same bandwidth does nothing for a memory-bound step. That is worth remembering the next time someone quotes peak TFLOPS at you.

## 4. The machine under the model: memory hierarchy

If decode is starved for bytes, then where bytes live matters more than how fast you multiply. A number a GPU multiplies starts in HBM and must be staged closer to the arithmetic first. Each stage is smaller and faster.

- **Registers:** 256 KB per SM (streaming multiprocessor), private to a thread, about one cycle away.
- **Shared memory and L1:** up to 192 KB per SM on an A100 (256 KB on an H100), private to the SM, tens of nanoseconds.
- **L2 cache:** 40 MB on an A100, 50 MB on an H100, shared by every SM, a couple of hundred nanoseconds.
- **HBM:** 80 GiB, shared by everything, several hundred nanoseconds.

An A100 has 108 SMs and an H100 SXM has 132. The latencies are order-of-magnitude figures for this class of part; they vary by generation and access pattern, and the repo has not measured them. The capacities come from NVIDIA's architecture whitepapers.

Two facts drive everything that follows. Capacity and speed trade off steeply: the on-chip levels together hold a few tens of megabytes while HBM holds tens of gigabytes. And the L2 is one resource shared by over a hundred SMs, split into slices behind a crossbar.

### Coalescing: 32 loads, how many trips?

A warp executes one instruction across 32 threads. When it is a load, each thread asks for its own address. The memory system does not serve 32 requests. It looks at the 32 addresses and fetches the **32-byte sectors** that contain them (four sectors make a 128-byte cache line). The sector is the smallest unit it fetches, so the bytes moved equal the number of distinct sectors touched times 32.

Here is the repo's model of it (`src/ie/gpumem.py`):

```python
def warp_access(stride_bytes, elem_bytes=4, threads=32, base=0):
    sectors = set()
    for t in range(threads):
        start = base + t * stride_bytes
        sectors.update(range(start // SECTOR, (start + elem_bytes - 1) // SECTOR + 1))
    return WarpAccess(stride_bytes, len(sectors), threads * elem_bytes, len(sectors) * SECTOR)
```

![Coalescing efficiency](https://raw.githubusercontent.com/AkbarSheikh-debug/inference-engineering/main/blog/images/c09-coalescing.png)

*Efficiency of a warp of 32 threads each loading 4 bytes, as the stride between threads grows. It stops falling at 12.5 percent.*

Consecutive 4-byte loads touch 4 sectors and waste nothing. Scatter them and efficiency falls, then **plateaus at 12.5 percent** once every thread owns a different sector, because 32 sectors of 32 bytes is the most a 4-byte-per-thread warp can drag in. The penalty for fully scattered 4-byte loads is therefore **8x**, not 32x. The 32x figure applies only to 1-byte loads, where a fully scattered warp moves 32 sectors to use 32 bytes.

I mention this in detail because I got it wrong once, in my own writing, and I will come back to that at the end. The lesson is not the exact number. It is that the same data requested in a different layout can cost several times the bandwidth, and bandwidth is exactly what decode is short of.

### Shared memory and tiling: reuse you have to ask for

The L1 and L2 caches are automatic. The hardware decides what to keep, with no guarantee that a value you will need again in a few microseconds is still there. **Shared memory** is the opposite: a fixed-size scratchpad that a kernel loads explicitly and reuses explicitly.

The classic use is tiling a matrix multiply. To compute a `T x T` output tile, a thread block loads a `T`-row strip of A and a `T`-column strip of B into shared memory once, then does `2 T^2 K` operations on them. Loaded bytes per tile are proportional to `2 T K`, so intensity is about `T / bytes_per_element`. **The tile size is the reuse factor.**

![Tiling and arithmetic intensity](https://raw.githubusercontent.com/AkbarSheikh-debug/inference-engineering/main/blog/images/c10-tiling-intensity.png)

*Intensity of a 4096 x 4096 x 4096 FP16 matrix multiply as the tile grows. Doubling the tile halves the traffic. The dashed lines are the two GPUs' ridge points.*

With no reuse the multiply reads 256 GiB and achieves 0.5 FLOP per byte. A 128 x 128 tile reads 2 GiB and reaches 63 FLOP per byte. That is the mechanism behind the "move right on the roofline" arrow: a kernel raises its own intensity without changing the algorithm. Real kernels get the rest from L2 reuse and larger tiles, and bigger is not always better, because a larger tile uses more shared memory and registers, which reduces how many thread blocks fit on an SM.

### What this means for inference

Tiling creates reuse only where the algorithm has some. Prefill is a large matrix multiply, where every weight is used against many tokens. A batch-1 decode step is a **matrix-vector product**, where each weight is used **once**, so no tile size can raise its intensity above about 1 FLOP per byte. The only ways to create reuse there are to put more tokens against each weight (batching) or to verify several tokens at once (speculative decoding). We will meet both.

## 5. The KV cache: why it exists and why it fills your GPU

Attention lets each new token look at all earlier tokens. Each token is projected into a query (Q), a key (K), and a value (V). A token's output is a weighted sum of the values of earlier tokens, with weights computed from its query against their keys.

Now generate naively: at every step, feed the whole sequence through the network and read off the last position. The first thousand tokens get their K and V recomputed again and again, even though nothing about them changed. Because attention is causal, an old token's K and V depend only on tokens before it, so they are identical every time.

**The fix: compute each token's K and V once, store them, and reuse them.** That store is the KV cache.

Why K and V but not Q? A past token's query was used to produce that token's own output, and that output is already done. Only the new token needs a query, and it is compared against the stored keys and used to weight the stored values.

### How much work does it save?

For a prompt of `P` tokens and `N` generated tokens, the naive approach pushes `N x P + N(N-1)/2` token positions through the network. With a cache it pushes `P + N - 1`.

![Recompute versus cache](https://raw.githubusercontent.com/AkbarSheikh-debug/inference-engineering/main/blog/images/c05-recompute-vs-cache.png)

*Token positions processed for a 128-token prompt. At 512 generated tokens, the uncached path does 307 times more positions.*

I checked this on a real (tiny) transformer built from Llama-style parts: RMSNorm, grouped-query attention, a gated MLP, residuals, and an LM head, with random weights. With a 32-token prompt and 32 generated tokens, the uncached run pushes 1,520 positions through the network and the cached run pushes 63. The FLOP ratio is 25.3 times, with **identical tokens** and a maximum logit difference of about 3e-15. The cache is exact, not an approximation. You can run it yourself with `python labs/cpu/00_tiny_decoder.py`.

### What it costs: bytes per token

For every token and every layer the cache stores one K and one V vector per KV head:

```python
def kv_bytes_per_token(model, dtype_bytes=2):
    return 2 * model.n_layers * model.n_kv_heads * model.head_dim * dtype_bytes
```

![KV bytes per token](https://raw.githubusercontent.com/AkbarSheikh-debug/inference-engineering/main/blog/images/c03-kv-bytes-per-token.png)

*KV cache per token in BF16. The number of KV heads, not query heads, sets the cost.*

Llama-3-8B has 32 layers, 8 KV heads, and a head dimension of 128, which is **128 KiB per token**. One 8K-token conversation therefore holds exactly **1 GiB** of cache. That is why the cache, not the weights, grows with your traffic.

### Where it lives: the memory budget

On a GPU, memory divides into weights (fixed), workspace, headroom, and a pool for the cache. Whatever is left after the fixed parts is the KV pool, and the pool divided by the per-conversation cache is your concurrency.

![Where GPU memory goes](https://raw.githubusercontent.com/AkbarSheikh-debug/inference-engineering/main/diagrams/export/d05-hbm-budget.png)

*The partition of GPU memory. Weights are fixed; the KV pool is everything left, cut into fixed-size blocks.*

Take an 80 GiB GPU, let the engine use 90 percent (72 GiB), and hold Llama-3-8B in BF16. The weights take 14.96 GiB (that is 16.06 GB; note the unit conversion, I will come back to it), leaving a **57.04 GiB pool**. At 8K context and 1 GiB per conversation, that is **57 concurrent conversations**.

![HBM budget on an H100](https://raw.githubusercontent.com/AkbarSheikh-debug/inference-engineering/main/blog/images/c16-hbm-budget.png)

*The 80 GiB budget for Llama-3-8B. This ignores activation workspace, so 57 is an upper bound.*

Doubling the context halves your users, under every attention scheme:

![Users versus context length](https://raw.githubusercontent.com/AkbarSheikh-debug/inference-engineering/main/blog/images/c04-users-vs-context.png)

*Concurrent users on one 80 GiB GPU versus context length. Grouped-query attention lifts the line by 4x; it does not change its slope. Full multi-head attention at 128K context does not fit at all.*

### The bandwidth angle

Capacity is one problem. Bandwidth is another. Every decode step reads the weights once **and** each sequence's entire cache. Weights are shared by the batch; caches are not.

For Llama-3-8B, the cache read per step equals the weight read when `batch x context = 122,528` tokens. At batch 16 that happens at a context of about 7,658 tokens; at batch 64, about 1,915.

![KV crossover](https://raw.githubusercontent.com/AkbarSheikh-debug/inference-engineering/main/blog/images/c18-kv-crossover.png)

*Above the line, KV-cache reads dominate each step, and the batching gains from section 3 start to erode. This is why KV-cache quantization and sparse attention exist.*

## 6. Shrinking the cache: sharing heads and paging

The size formula has a factor you can change: the number of KV heads.

- **Multi-head attention (MHA):** every query head has its own KV head.
- **Multi-query attention (MQA, Shazeer 2019):** all query heads share one KV head.
- **Grouped-query attention (GQA, Ainslie et al. 2023):** each KV head serves a group of query heads, a middle ground.
- **Multi-head latent attention (MLA, DeepSeek-V2):** cache one small latent vector per token and rebuild keys and values from it.

![Head sharing](https://raw.githubusercontent.com/AkbarSheikh-debug/inference-engineering/main/diagrams/export/d06-head-sharing.png)

*Who shares which keys and values in MHA, GQA, MQA, and MLA.*

Llama-3-8B has 32 query heads and 8 KV heads, so each KV head serves 4 query heads. On the same 80 GiB GPU at 8K context, that is 57 users with GQA, 14 with full multi-head attention, and 456 with multi-query attention. GQA cuts the cache 4x and lets four times as many users fit. MQA cuts it 32x but gives up more model quality. Ainslie et al. report that an existing MHA checkpoint can be converted to GQA with a small fraction of the original training compute (about 5 percent in the paper). Both save memory and bandwidth; neither saves arithmetic.

### PagedAttention: fixing fragmentation

A sequence's final length is unknown when it starts, so a simple allocator reserves a contiguous slab for the maximum possible length. Most requests are far shorter, and the unused tail of every slab is wasted. The vLLM paper (Kwon et al.) profiled existing systems and found that only about 20 to 38 percent of the allocated cache memory held actual token states.

PagedAttention borrows the idea of virtual memory. Cut the pool into small fixed-size blocks, hand blocks to a sequence as it grows, and keep a per-sequence **block table** mapping logical positions to physical blocks. Waste per sequence is bounded by one partly filled block.

![Paged KV cache](https://raw.githubusercontent.com/AkbarSheikh-debug/inference-engineering/main/diagrams/export/d07-paged-kv.png)

*Two sequences and their block tables. Sequence A's 5 tokens take one full block and one quarter-full block; free blocks sit on a free list.*

The repo includes a toy allocator (`labs/cpu/03_paged_allocator.py`) that compares the two policies on the same pool:

![Paged versus contiguous](https://raw.githubusercontent.com/AkbarSheikh-debug/inference-engineering/main/blog/images/c15-paged-vs-contiguous.png)

*The same pool, admitting seeded synthetic requests. The size of the gap depends on the assumed length distribution and the 4,096-token maximum. The mechanism does not.*

Be careful with this result. The distribution is synthetic and the contiguous policy reserves a fixed maximum, so the exact 479-versus-64 gap is an artefact of those choices. What it demonstrates is the mechanism: the gap grows with the ratio of the reserved maximum to the typical length. The toy also assumes lengths are known at admission; a real engine allocates blocks as decoding proceeds and must preempt a sequence when the pool runs dry. And paging saves *fragmentation*, not total memory.

## 7. Quantization: spending fewer bytes per number

Section 3 showed that a decode step is limited by how many bytes must be read, which is the parameter count times the bytes per parameter. Quantization attacks that second factor directly. It also frees HBM for the KV cache, so it raises capacity as well as speed.

![Weight bytes by precision](https://raw.githubusercontent.com/AkbarSheikh-debug/inference-engineering/main/blog/images/c06-weight-bytes.png)

*Bytes read per decode step for Llama-3-8B. "4-bit" with group scales is really 4.125 bits per weight.*

### Number formats in one paragraph

A floating-point number spends its bits on a **sign**, an **exponent** (dynamic range), and a **mantissa** (precision). FP32 is 1/8/23. FP16 is 1/5/10. BF16 is 1/8/7: it keeps FP32's full exponent, so it covers the same range and needs no rescaling, at the cost of a shorter mantissa. FP16 has more precision but a small range (its largest finite value is 65,504), so tensors must be scaled to avoid overflow. INT8 has no exponent at all; every value is a plain integer and the range-versus-precision trade-off moves into one external number, the **scale**.

### Affine quantization

Think of a measuring gauge with a fixed number of tick marks. Each real value is rounded to the nearest tick, the spacing between ticks is the scale, and if the data does not centre on zero you slide the ticks with a zero-point:

```
q     = round(x / s) + z          (store this integer)
x_hat = s x (q - z)               (the arithmetic units reconstruct this)
```

**Symmetric** quantization fixes `z = 0`. **Asymmetric** quantization picks `z` so an interval not centred on zero uses the whole integer range. On a skewed row of eight values the two rules behave like this:

![Symmetric versus asymmetric](https://raw.githubusercontent.com/AkbarSheikh-debug/inference-engineering/main/blog/images/c08-symmetric-vs-asymmetric.png)

*INT8 round-trip error for each value of a skewed row. The zero-point improves the worst-case bound by the step ratio, 1.80x. Individual values can still favour either rule.*

The guaranteed error bound is half a step, so the zero-point improves the worst case anyone can promise by the step ratio, 1.80. Notice that the symmetric rule reconstructs 0.86 and 0.63 *better* than the asymmetric one: a coarser grid is worse on average and in the worst case, while individual values that land near a tick are almost exact. This is why testing a scheme on a handful of hand-picked weights proves nothing. The honest measurement is a distribution over the whole tensor.

There is a cost: expanding `s(q - z)` inside a matrix multiply creates a cross-term involving `z` times a sum of the other operand, which must be folded into a bias ahead of time or computed at run time. Symmetric quantization has no such term. That is why weights, whose distributions are close to zero-centred, are almost always quantized symmetrically, while skewed activations use asymmetric. The rule of thumb is symmetric for weights, asymmetric for activations, and the reason is arithmetic cost, not accuracy.

### Granularity: how many gauges

One scale for a whole matrix (**per-tensor**) is simplest, but a single large value forces a coarse step onto everything else. **Per-channel** gives each output row its own scale. **Group-wise** gives each block of, say, 128 consecutive weights its own scale.

![Quantization granularity](https://raw.githubusercontent.com/AkbarSheikh-debug/inference-engineering/main/blog/images/c07-quantization-granularity.png)

*Relative RMS error on a synthetic weight matrix with row-scale variation and 0.2 percent outliers. A mechanism demo, not a measurement of any real model.*

Two lessons. Error falls as scales get finer, because an outlier now spoils only its own row or group. And 4 bits is far less forgiving than 8: the grid has 15 levels instead of 255, so outliers hurt much more, which is why 4-bit schemes lean on group scales and on methods that protect the important weights (GPTQ and AWQ exist for exactly this). The group scale costs 16 bits per 128 weights, so "4-bit" really stores 4.125 bits per weight, which is why the INT4 weights of an 8B model are 4.14 GB, not 4.02 GB.

### What quantization does and does not do

- **Weight-only quantization** shrinks the bytes read and the memory used. Unless the kernels also do the arithmetic in low precision, weights are converted back on the fly, so the win is bandwidth and capacity, not extra FLOP/s.
- **It is not "four times faster."** It is about 3.9 times fewer bytes, so up to that much faster *if* the step is memory-bound and the kernels are efficient. Compute-bound phases such as prefill do not speed up from fewer weight bytes.
- **Accuracy is a measurement, not a formula.** This repo quotes no accuracy numbers because it has not measured any.
- **Training for it.** Rounding is a step function whose derivative is zero almost everywhere, so a naive quantization-aware training run would never update a weight. The straight-through estimator fixes this by rounding in the forward pass and pretending rounding was the identity in the backward pass.

## 8. FlashAttention: exact attention without the n-by-n matrix

For one head, attention computes `S = Q K^T / sqrt(d)`, a matrix with one score for every pair of positions, applies softmax to each row, then multiplies by `V`. Written naively, that is four passes over an `n x n` matrix in HBM: write `S`, read it back for the softmax, write the result `P`, read it again to multiply by `V`.

The matrix is large. In FP16, one head at `n = 65,536` needs 8 GiB for `S` alone, and at `n = 32,768` with 32 heads it is 64 GiB. It grows with the square of the context length, so it is a wall for long contexts before the KV cache even comes into it.

FlashAttention computes the same output without ever writing `S` to HBM. It splits Q, K, and V into blocks small enough to sit in on-chip memory. For each block of queries it streams over every block of keys and values, computes that block's scores on chip, folds them into a running result, and discards them.

### The obstacle: softmax needs the whole row

Softmax subtracts the row maximum for numerical stability and divides by the row sum, so it seems to need every score of the row at once, which is exactly what blocking prevents. The fix is the **online softmax** (Milakov and Gimelshein, 2018). Keep three running values per query row: the maximum `m` so far, the running sum `l`, and the running output `o`. When a new block arrives:

```python
m_new = np.maximum(m, s.max(axis=1))
alpha = np.exp(m - m_new)            # rescales everything accumulated so far
p = np.exp(s - m_new[:, None])
l = l * alpha + p.sum(axis=1)
o = o * alpha[:, None] + p @ v_block
```

After the last block, the answer is `o / l`. The factor `alpha` is a change of reference point: because `exp(s - m_new) = exp(s - m_old) * exp(m_old - m_new)`, every term accumulated against the old maximum moves onto the new one with a single multiplication and no loss of information.

Here is one row with eight scores in two blocks of four. Block 1 is `[0.515, 0.385, 0.405, 0.435]`, giving a running maximum `m = 0.515` and running sum `l = 3.697`. Block 2 is `[0.560, 0.300, 0.370, 0.320]`. Its maximum, 0.560, exceeds 0.515, so everything so far is stale and must be rescaled by `alpha = exp(0.515 - 0.560) = 0.956`. The sum becomes `0.956 x 3.697 + 3.385 = 6.919`, which is exactly the sum a single softmax over all eight scores produces. Not approximately. Exactly.

I ran the tiled algorithm against direct attention on random data, with block shapes that do not divide the sequence length and with the key blocks visited in reverse order. Every difference was around 1e-16, which is floating-point rounding. FlashAttention is not a trade of accuracy for memory. It is the same arithmetic in a different order.

### What it saves, and what it does not

This is commonly overstated, so be precise. Two different quantities are involved.

**Extra memory.** Standard attention needs `O(n^2)` storage for `S`. The tiled algorithm needs only the running statistics, `O(n)` beyond the inputs and output.

**HBM traffic.** This shrinks by a constant factor, not an exponent. Each block of `B` query rows must stream past *every* block of keys and values, so K and V are re-read `n / B` times and total traffic still scales as `n^2`. The published analysis gives `Theta(n^2 d^2 / M)` HBM accesses for head dimension `d` and on-chip memory `M`, against `Theta(nd + n^2)` for standard attention.

![FlashAttention storage and traffic](https://raw.githubusercontent.com/AkbarSheikh-debug/inference-engineering/main/blog/images/c11-flashattention.png)

*Left: the score matrix that no longer exists. Right: HBM traffic for head dimension 128 and query block 128. Both curves are quadratic and parallel, the gap is 2B/d = 2x at every n.*

So the headline benefit at these settings is not a dramatic drop in traffic. It is that the `n x n` matrix never exists, so context length is no longer limited by it, and the remaining traffic streams through fast memory in one fused kernel instead of bouncing through HBM four times. Also note what does not change: the arithmetic. FlashAttention performs the same `O(n^2 d)` floating-point operations. It is a memory optimization, not a FLOP optimization, and this model says nothing about kernel launch costs or achieved bandwidth.

One more thing. During decode there is one query per sequence, so the score "matrix" is a single row of length `n`. There is no `n x n` structure to avoid. What decode reads is the KV cache, and that is the cost. FlashAttention's largest effect is in prefill and training.

## 9. Speculative decoding: more tokens per read of the weights

Decoding is sequential, because token `t+1` needs token `t`. But **checking** a list of candidate tokens can be parallel: if someone hands you tokens `t+1 ... t+k`, the model can compute its own probability at all `k` positions in a single forward pass, exactly like prefill.

Speculative decoding manufactures those candidates cheaply. A small **draft** model proposes `k` tokens one at a time. The large **target** model then verifies all `k` in one pass, keeps the ones it agrees with, and corrects the first one it does not.

Why is verification cheap? A batch of `k+1` positions has arithmetic intensity about `k+1`, far to the left of the H100's ridge at about 295. The step is still memory-bound, so it costs about the same as a batch of one: 4.79 ms of weight reads for Llama-3-8B whether it verifies one position or five (ignoring KV-cache reads).

### The accept/reject rule

Write `p(x)` for the target's probability of token `x` and `q(x)` for the draft's. For each drafted position, in order: **accept** the draft token with probability `min(1, p(x) / q(x))`. On rejection, stop and draw one token from the **residual** distribution `max(0, p - q)`, normalized.

```python
def accept_probability(p_x, q_x):
    return min(1.0, p_x / q_x)

def residual_distribution(p, q):
    r = np.maximum(0.0, p - q)
    z = float(r.sum())
    return (r / z if z > 0 else p.copy()), z
```

Suppose the draft proposes *is*, *a*, *beautiful*, *day* after "The weather today", with draft probabilities 0.65, 0.50, 0.30, 0.40 and target probabilities 0.72, 0.58, 0.08, 0.20. The first two have `p >= q` and are accepted outright. *Beautiful* survives with probability `0.08 / 0.30 = 0.267`; suppose the coin rejects it. Verification stops and *day* is discarded unexamined. The rejection does not waste the step: the residual over five candidate words is *sunny* 0.673, *warm* 0.269, *cloudy* 0.058 (it sums to Z = 0.52 before normalizing). The step emits *is*, *a*, *sunny*: three tokens from one target pass.

### Why the output is exactly the target's

The obvious worry is that guessing must bias the output. It does not, and the proof is two lines. Fix a token `x` and add up the ways it can be emitted. The draft proposed `x` and it was accepted: `q(x) * min(1, p(x)/q(x)) = min(q(x), p(x))`. Or the draft proposed something else, that was rejected, and `x` came from the residual. The overall rejection probability equals the residual's normalizer `Z`, so this path contributes `max(0, p(x) - q(x))`. Add them:

```
min(q, p) + max(0, p - q) = p        for every token x
```

If `p >= q` the terms are `q` and `p - q`; if `p < q` they are `p` and 0. The emitted distribution is the target's, **for any draft model whatsoever**. A bad draft costs throughput, never correctness.

I checked this by brute force with a deliberately poor draft (only 48 percent of its proposals are accepted):

![Speculative decoding exactness](https://raw.githubusercontent.com/AkbarSheikh-debug/inference-engineering/main/blog/images/c13-speculative-exactness.png)

*Target distribution versus the frequencies emitted by the rule, with the poor draft distribution marked. The figure uses 200,000 samples; the lab runs a million and its largest gap is 0.0004.*

### How much faster?

Suppose each drafted token is accepted independently with probability `alpha`, and the draft proposes `k` tokens. The expected number of tokens per target pass is:

```
E[tokens] = 1 + alpha + alpha^2 + ... + alpha^k = (1 - alpha^(k+1)) / (1 - alpha)
```

The draft is not free. If one draft step costs a fraction `c` of a target step, one iteration costs `k c + 1` target steps, so the wall-clock speedup is `E[tokens] / (k c + 1)`.

![Speculative decoding speedup](https://raw.githubusercontent.com/AkbarSheikh-debug/inference-engineering/main/blog/images/c12-speculative-decoding.png)

*Expected tokens per target pass, and speedup if a draft step costs 5 percent of a target step, for three acceptance rates.*

At an acceptance rate of 0.7 and 4 drafted tokens you get 2.77 tokens per pass and a 2.31x speedup. Two things stand out. Acceptance rate matters far more than how far you draft: at 0.5, drafting more than a few tokens buys almost nothing and starts to cost. And a poor or expensive draft can lose outright: at acceptance 0.3, 8 drafted tokens and a draft cost of 0.3, the speedup is about 0.42, slower than plain decoding. Independent acceptance is also a simplification, and these are model arithmetic, not measurements of any engine.

Speculative decoding spends spare arithmetic to save weight reads, so it helps most when a step is memory-bound with room to spare: small batches, latency-sensitive traffic. At a large batch the step approaches the ridge point and extra verification positions compete for compute.

## 10. Scheduling: never let a finished request hold a seat

Section 3 showed that batching is the main lever on decode cost. The question is *how* a batch is assembled and torn down.

**Static batching** locks a fixed group of requests into a batch and refuses to admit anything new until every request has finished. A request that finishes early keeps its slot, and the KV cache behind it, idle and unusable, held hostage by whichever request in the batch runs longest.

**Continuous batching**, also called iteration-level batching, treats each slot as an independent resource. The scheduler rebuilds the batch on **every decode iteration**. When a request emits its end-of-sequence token it leaves immediately, and a waiting request joins on the very next iteration. The idea was introduced in the Orca system (Yu et al., OSDI 2022) and is standard in the major serving engines.

![Static versus continuous batching](https://raw.githubusercontent.com/AkbarSheikh-debug/inference-engineering/main/diagrams/export/d08-continuous-batching.png)

*Seven requests of 3, 10, 5, 6, 4, 5, and 3 decode steps on four slots. Static batching finishes in 15 steps with slots busy 60 percent of the time; continuous batching finishes in 10 steps at 90 percent.*

The gain is easy to quantify. Take four slots, each generating 40 tokens per second, and four requests that want 32, 80, 200, and 400 tokens. Under static batching all four hold their slots until the last finishes at 10 seconds, so the tokens emitted are only what each request actually generates: 32 + 80 + 200 + 400 = **712**. Under continuous batching a freed slot is refilled on the next iteration and all four slots stay busy for the whole window: 4 x 40 x 10 = **1,600**, assuming the queue does not run dry.

![Continuous batching payoff](https://raw.githubusercontent.com/AkbarSheikh-debug/inference-engineering/main/blog/images/c14-continuous-batching.png)

*Tokens emitted in a ten-second window and slot utilization. Per-request latency is unchanged. Throughput rises because finished requests no longer hold a slot idle.*

That is a factor of 2.2 on identical hardware from a scheduler change alone, and it does not come from making any single request faster. It comes from refusing to let a finished request hold a seat.

### Mixing prefill and decode

A new request cannot simply drop into a decode slot: it must first run prefill. But prefill and decode are the same pipeline, differing only in how many tokens per request pass through each matrix multiply, so a scheduler can put a new request's prompt tokens and the single decode tokens of running requests into the same launch. The trap is that one long prompt can dominate an iteration and delay every other request's next token. **Chunked prefill** fixes this: split a long prompt across several iterations and interleave each chunk with the ongoing decodes.

### The two knobs

Production schedulers bound the batch with two independent limits. **The concurrency cap** is a seat count, bounded by how many KV caches fit (57 conversations on our example GPU at 8K). **The token budget** is a work quota: the maximum number of tokens in one forward pass. With a budget of 4,096 tokens and 24 sequences decoding, decode uses 24 tokens, 0.6 percent of the budget, and three prefill chunks of 1,024 use 3,072 more, leaving 1,000 spare. Decode is nearly free in token terms and expensive in seats; prefill is the reverse. One limit cannot govern both.

Continuous batching and PagedAttention are one design, not two. Continuous batching needs a cache that can grow one sequence at a time and accept a new one into whatever space was just vacated; a contiguous slab cannot do that. And a paged cache serving static batches recovers memory and then leaves it idle. Under memory pressure the scheduler must also decide which requests to **preempt**, evicting a running request's cache to host memory or discarding it to recompute later.

## 11. Scaling out and the serving stack

Everything so far fits on one GPU. Larger models and higher traffic force you to split work across several, and there are four standard ways to do it.

![Parallelism strategies](https://raw.githubusercontent.com/AkbarSheikh-debug/inference-engineering/main/diagrams/export/d09-parallelism.png)

*Four ways to split a model across GPUs: tensor, pipeline, expert, and context parallelism, with what each one communicates.*

**Tensor parallelism** splits each layer's matrices across GPUs, at the price of an all-reduce every layer. It shards the KV cache as well as the weights, which changes the seat count in the memory budget. **Pipeline parallelism** splits the layers into stages and passes activations between them. **Expert parallelism** splits a mixture-of-experts model's experts and routes tokens between GPUs. **Context parallelism** splits the sequence itself.

A further idea is **disaggregated serving**: because prefill is compute-bound and decode is memory-bound, running them on separate pools of GPUs lets each be provisioned for its own bottleneck (see the DistServe and Splitwise papers below). These topics are outside what the repo's labs cover today, so I point you to the papers instead of quoting numbers I have not verified.

Finally, an engine such as vLLM, SGLang, or TensorRT-LLM sits in the middle of the stack, bundling everything above: it decides which requests run together, manages the memory that holds each request's state, and launches the GPU kernels that do the arithmetic.

## 12. The map: bottleneck first, technique second

Here is the whole post as one picture. Read it from the left: name your bottleneck, then read across to the techniques that attack it.

![Technique by bottleneck](https://raw.githubusercontent.com/AkbarSheikh-debug/inference-engineering/main/diagrams/export/d10-technique-bottleneck.png)

*Each technique and the bottleneck (or bottlenecks) it attacks.*

A practical way to use it:

1. **Measure the symptom.** Is the complaint time to first token, inter-token latency, throughput, or cost?
2. **Name the bound.** Compare the workload's arithmetic intensity to the ridge point. Left of it is memory-bound; right is compute-bound.
3. **Check the memory budget.** Weights plus KV cache for your context and concurrency: does it fit, and how many users does the leftover pool serve?
4. **Pick the technique that attacks that bottleneck.** Bandwidth-bound: fewer bytes or more reuse. Capacity-bound: share or page the cache. Idle GPU: continuous batching.
5. **Re-measure.** Every number here is a ceiling. The only benchmark that counts is yours.

And here is the learning order I follow. It is a dependency graph, not a ladder:

![Learning roadmap](https://raw.githubusercontent.com/AkbarSheikh-debug/inference-engineering/main/diagrams/export/d00-roadmap-dag.png)

*The dependency graph of topics. Green nodes have at least one published article in the repo; grey ones are planned.*

## 13. What checking numbers with code caught

I am writing a book on this subject, and the reason this repo exists is a rule I adopted: **every number in an article must be generated by code or cited to a primary source, and a test fails if an article quotes a number the code does not produce.**

That rule paid for itself immediately. When I re-derived numbers from code, it caught three mistakes in my own earlier draft:

- **A wrong unit conversion.** I had written that an "80 GB" GPU holds 74.5 GiB. That is wrong. GPU memory capacity is binary, and an H100 80GB reports about 81,559 MiB in `nvidia-smi`, so it carries about 80 GiB. The decimal-to-binary conversion belongs on the checkpoint size (a "16 GB" model file is 14.9 GiB), not on the card. Every downstream number, the block counts and user counts, had to be re-derived.
- **A coalescing example off by 4x.** I had a scattered warp load moving 4,096 bytes for 128 useful, a 32x penalty. The same section defined the 32-byte sector as the smallest unit fetched, which makes it 1,024 bytes and an 8x penalty. The 32x figure only holds for 1-byte loads.
- **A claim that FlashAttention makes HBM traffic linear.** It makes the *extra memory* linear. Traffic is still quadratic, with a smaller constant.

Numbers from code get corrected. Numbers typed by hand do not. If you take one habit from this post, take that one.

## 14. Run it yourself

Everything in this post is reproducible on a laptop. There is no GPU requirement.

```bash
git clone https://github.com/AkbarSheikh-debug/inference-engineering.git
cd inference-engineering
pip install -e ".[dev]"

python labs/cpu/00_tiny_decoder.py         # generate with and without a KV cache
python labs/cpu/01_roofline.py             # tokens-per-second ceilings for a model on a GPU
python labs/cpu/02_kv_calculator.py        # KV bytes per token, sequences that fit
python labs/cpu/03_paged_allocator.py      # contiguous slabs vs paged blocks
python labs/cpu/04_quantization.py         # symmetric vs asymmetric, granularity
python labs/cpu/05_continuous_batching.py  # static vs continuous, token budgets
python labs/cpu/06_gpu_memory.py           # coalescing efficiency, tiled matmul traffic
python labs/cpu/07_flash_attention.py      # tiled attention is exact; traffic model
python labs/cpu/08_speculative_decoding.py # accept/reject rule, exactness, speedup
pytest                                     # every number quoted in the articles is checked here
```

The repo is MIT licensed, so you can use, fork, and adapt any of it. If it helps your work, please cite it and link back.

- Repo: [github.com/AkbarSheikh-debug/inference-engineering](https://github.com/AkbarSheikh-debug/inference-engineering)
- Readable site: [akbarsheikh-debug.github.io/inference-engineering](https://akbarsheikh-debug.github.io/inference-engineering/)

## 15. Reading list: papers, blogs, and docs

This post stands on other people's work. Here are the sources I would read next, grouped by topic. I have linked the original pages instead of copying their figures; go look at the figures in the papers themselves, they are excellent.

### Start here: first-principles explainers

- [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) by kipply. First-principles reasoning about LLM inference performance: KV cache, FLOPs versus memory-boundness, no experiments needed.
- [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) by Horace He. Every GPU operation is compute-bound, bandwidth-bound, or overhead-bound; identify which.
- [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) by Lilian Weng. A broad survey of compression and transformer-specific techniques.
- [LLM Inference Performance Engineering: Best Practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) from Databricks. Prefill versus decode, batch size, KV cache trade-offs, and the metrics that matter.
- [All About Transformer Inference](https://jax-ml.github.io/scaling-book/inference/) from Google DeepMind's "How to Scale Your Model". Latency, KV caches, and disaggregated serving.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar, if the transformer itself is not yet second nature.

### The GPU underneath

- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM) by Simon Boehm. Coalescing, shared-memory tiling, and warp tiling, step by step.
- [Unlock GPU Performance: Global Memory Access in CUDA](https://developer.nvidia.com/blog/unlock-gpu-performance-global-memory-access-in-cuda/) from NVIDIA, on the sector as the unit of a memory transaction.
- The NVIDIA [A100](https://www.nvidia.com/en-us/data-center/a100/) and [H100](https://www.nvidia.com/en-us/data-center/h100/) pages and architecture whitepapers, for the specifications used here.
- Williams, Waterman, and Patterson, [Roofline: an insightful visual performance model for multicore architectures](https://dl.acm.org/doi/10.1145/1498765.1498785), Communications of the ACM, 2009.

### Attention, the KV cache, and memory

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017.
- Pope et al., [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102), 2022.
- Shazeer, [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150), 2019 (multi-query attention).
- Ainslie et al., [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245), 2023.
- DeepSeek-AI, [DeepSeek-V2](https://arxiv.org/abs/2405.04434), 2024 (multi-head latent attention).
- Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180), 2023, and the [vLLM announcement post](https://blog.vllm.ai/2023/06/20/vllm.html).
- Zheng et al., [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104), 2023 (prefix sharing with RadixAttention).

### FlashAttention

- Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135), 2022.
- Dao, [FlashAttention-2](https://arxiv.org/abs/2307.08691), 2023, and Shah et al., [FlashAttention-3](https://arxiv.org/abs/2407.08608), 2024.
- Milakov and Gimelshein, [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867), 2018.

### Quantization

- Dettmers et al., [LLM.int8()](https://arxiv.org/abs/2208.07339), 2022.
- Frantar et al., [GPTQ](https://arxiv.org/abs/2210.17323), 2022.
- Lin et al., [AWQ](https://arxiv.org/abs/2306.00978), 2023.
- Xiao et al., [SmoothQuant](https://arxiv.org/abs/2211.10438), 2022.

### Scheduling and serving systems

- Yu et al., [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu), OSDI 2022 (continuous batching).
- [How continuous batching enables 23x throughput in LLM inference](https://www.anyscale.com/blog/continuous-batching-llm-inference) from Anyscale.
- Agrawal et al., [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310), 2024 (chunked prefill).
- Zhong et al., [DistServe: Disaggregating Prefill and Decoding](https://arxiv.org/abs/2401.09670), 2024.
- Patel et al., [Splitwise: Efficient Generative LLM Inference Using Phase Splitting](https://arxiv.org/abs/2311.18677), 2023.
- Shoeybi et al., [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053), 2019 (tensor parallelism).

### Speculative decoding

- Leviathan, Kalman, and Matias, [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192), 2023.
- Chen et al., [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318), 2023.
- Cai et al., [Medusa](https://arxiv.org/abs/2401.10774), 2024, and Li et al., [EAGLE](https://arxiv.org/abs/2401.15077), 2024.

### Practical guides

- [Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) from NVIDIA.
- [Optimizing your LLM in production](https://huggingface.co/blog/optimize-llm) from Hugging Face.

## 16. What is next

The repo covers levels of this map in a specific order, and it is honest about what it does not cover yet. Still to come: the CPU host and the CUDA execution model, kernel fusion, tensor and pipeline parallelism in depth, the inference engines themselves, and production concerns such as routing, observability, and cost. Measured benchmarks will land in the repo only when they come with the hardware and software versions recorded.

If you found this useful, subscribe so you get the next one, and star the repo so it is easier for others to find. And tell me: **which bottleneck bites you most when serving LLMs, memory, latency, or cost?** I read every reply.

*Copyright (c) 2026 Akbar Arif. The code and text in the repository are released under the MIT License.*
