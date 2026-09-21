# Glossary

One definition per term, used the same way everywhere in the curriculum.

**Acceptance rate.** In speculative decoding, the probability that the target model accepts a token proposed by the draft. Written alpha.

**Arithmetic intensity.** Floating-point operations performed per byte moved from memory (FLOP/byte). The x-axis of a roofline.

**Batch.** The set of sequences advanced together in one forward pass. In decode, one new token per sequence.

**BF16.** Brain floating point: 16 bits, with the exponent range of FP32 and 7 mantissa bits. 2 bytes per value.

**Chunked prefill.** Splitting a long prompt's prefill across several scheduler iterations, interleaved with ongoing decodes, so one prompt cannot stall everyone else's next token.

**Coalescing.** Combining the 32 loads of a warp into as few memory transactions as possible. Consecutive addresses fall in few 32-byte sectors; scattered ones touch many.

**Compute-bound.** Limited by arithmetic throughput. Intensity lies right of the ridge point.

**Concurrency cap.** The maximum number of sequences a scheduler keeps active at once. A seat count, bounded by how many KV caches fit in memory.

**Continuous batching.** Rebuilding the batch on every decode iteration, so a finished request leaves at once and a waiting one joins on the next step. Also called iteration-level batching.

**Decode.** The phase that generates output tokens one at a time, each depending on the previous one. Typically memory-bound.

**Draft model.** In speculative decoding, the small cheap model that proposes several tokens ahead for the target model to verify.

**GiB / GB.** GiB is 2^30 bytes; GB is 10^9 bytes. GPU memory capacity in this repository is quoted in GiB, bandwidth in decimal GB/s or TB/s.

**GQA (grouped-query attention).** Attention where each key-value head is shared by a group of query heads. Between MHA (no sharing) and MQA (one shared head).

**Group-wise quantization.** Giving each block of consecutive weights (for example 128) its own scale. Costs extra bits per weight for the stored scales.

**HBM.** High-bandwidth memory: the stacked DRAM on a GPU package that holds weights and the KV cache.

**ITL (inter-token latency).** Time between consecutive output tokens for one request. Also called TPOT.

**KV cache.** Stored key and value vectors of every past token in every layer, kept so they are computed once.

**Memory-bound.** Limited by how fast data can be read from memory rather than by arithmetic. Intensity lies left of the ridge point.

**MHA (multi-head attention).** Attention where every query head has its own key-value head.

**MLA (multi-head latent attention).** Attention that caches a small latent vector per token and reconstructs keys and values from it.

**MQA (multi-query attention).** Attention where all query heads share a single key-value head.

**Online softmax.** Computing softmax over a row that arrives in blocks by keeping a running maximum, sum and output, and rescaling them when a larger maximum appears. Exact, not approximate.

**Per-channel quantization.** Giving each output row of a weight matrix its own scale.

**Preemption.** Evicting a running request's KV cache under memory pressure, by copying it to host memory or discarding it to recompute later.

**Prefill.** The phase that processes the whole prompt in parallel and fills the KV cache. Typically compute-bound.

**Ridge point.** The arithmetic intensity at which a GPU's roofline turns from bandwidth-limited to compute-limited: peak FLOP/s divided by bandwidth.

**Roofline.** A model of attainable performance as the minimum of a compute ceiling and a memory ceiling that grows with arithmetic intensity.

**Scale and zero-point.** The two numbers of affine quantization: the scale is the step between adjacent integer levels, the zero-point is the integer that represents real zero.

**Sector.** The 32-byte unit in which the GPU memory system fetches data. Four sectors make a 128-byte cache line.

**Shared memory.** A fixed-size, programmer-managed scratchpad inside each SM, loaded and reused explicitly. Distinct from the automatic L1 and L2 caches.

**SM (streaming multiprocessor).** The GPU's unit of compute, holding its own registers and shared memory. An A100 has 108; an H100 SXM has 132.

**Static batching.** A fixed group of requests run together until every one has finished, so early finishers hold their slot idle.

**Straight-through estimator.** Training trick for quantization-aware training: round in the forward pass, pretend rounding was the identity in the backward pass.

**Target model.** In speculative decoding, the full accurate model whose output distribution the procedure preserves exactly.

**Tile.** A fixed-size rectangular block of a matrix small enough to sit in shared memory. Tile size sets the reuse factor of a matrix multiply.

**Token budget.** The maximum number of tokens admitted into one forward pass. A work quota, distinct from the concurrency cap.

**TTFT (time to first token).** Time from sending a request to receiving the first output token; dominated by queueing and prefill.
