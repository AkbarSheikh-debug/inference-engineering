# 05. Continuous batching: never let a finished request hold a seat

**Question this answers:** batching raises throughput but slows each user's stream of tokens. How do production engines get the throughput without paying the latency, and why does the answer depend on the KV cache?

## The problem: head-of-line blocking

[Article 02](02-why-decode-is-slow.md) showed that serving many sequences per weight read raises arithmetic intensity, so batching is the main lever on decode cost. The question is *how* a batch is assembled and torn down.

**Static batching** locks a fixed group of requests into a batch and refuses to admit anything new until *every* request has finished. A request that finishes early keeps its slot, and the KV cache behind it, idle and unusable, held hostage by whichever request in the batch runs longest.

**Continuous batching**, also called **iteration-level batching**, treats each slot as an independent resource. The scheduler rebuilds the batch on *every decode iteration*. When a request emits its end-of-sequence token it leaves immediately, and a waiting request can join on the very next iteration. This idea was introduced in the Orca system (Yu et al., OSDI 2022) and is now standard in the major serving engines.

## The same work, two schedules

`labs/cpu/05_continuous_batching.py` schedules seven requests needing 3, 10, 5, 6, 4, 5 and 3 decode steps on four slots, with equal-cost steps.

![Static versus continuous batching schedule](../diagrams/export/d08-continuous-batching.svg)

| | Static | Continuous |
|---|---|---|
| Steps to finish all seven | 15 | 10 |
| Slots busy | 60% | 90% |

The work is identical (36 slot-steps). Only the policy changed. Under static batching, requests A, C and D finish early and sit idle until B ends at step 10; the second batch then starts and again waits for its slowest member. Continuous batching refills each slot the moment it frees.

## What the gain is worth

Take four slots, each generating 40 tokens per second, and four requests arriving together that want 32, 80, 200 and 400 tokens. They finish after 0.8, 2, 5 and 10 seconds.

- **Static:** all four hold their slots until the last finishes at t = 10 s. Tokens emitted are only what each request actually generates: 32 + 80 + 200 + 400 = **712**.
- **Continuous:** a freed slot is refilled on the next iteration, so all four slots stay busy for the whole window: 4 x 40 x 10 = **1,600** (assuming the queue does not run dry).

That is a factor of **2.2** on identical hardware from a scheduler change alone. Where does it come from? Not from making any single request faster; each request's own latency is unchanged. It comes from refusing to let a finished request hold a seat.

## Mixing prefill and decode in one batch

A new request cannot simply drop into a decode slot: it must first run **prefill**, one compute-heavy pass over its whole prompt. But prefill and decode are the same pipeline; they differ only in how many tokens per request pass through each matrix multiply in one iteration. So a scheduler can put a new request's prompt tokens *and* the single decode tokens of running requests into the same launch. Attention is handled per request against its own KV cache.

The trap is that one long prompt can dominate an iteration and delay every other request's next token. It is head-of-line blocking again, in miniature. **Chunked prefill** is the fix: split a long prompt across several iterations and interleave each chunk with the ongoing decodes, which bounds the delay any one prompt can impose on everyone else's inter-token latency.

## The two knobs that bound a batch

"As many as fit" is not a policy. Production schedulers bound the batch with two independent limits, and they bind in different regimes.

**The concurrency cap** is a seat count: the maximum number of sequences active at once. If it is too low, the GPU runs below capability and cost per token rises. If it is too high, the KV cache runs out of physical blocks and the scheduler is forced into preemption, spending its bandwidth thrashing instead of generating. The right value follows from arithmetic: divide the HBM left after weights and workspace by the per-sequence cache size. For Llama-3-8B in BF16 on an 80 GiB GPU at 8K context, [article 03](03-kv-cache.md) derives 57 sequences.

**The token budget** is the maximum number of tokens admitted into one forward pass. It is a work quota, not a seat count, and the two kinds of work it pays for are very different in size. Take a budget of 4,096 tokens with 24 sequences decoding. Each contributes one token, so decode uses 24 tokens, **0.6 percent** of the budget. Three prefill chunks of 1,024 tokens use 3,072 more, and **1,000 tokens** are spare. Decode is nearly free in token terms and expensive in seats; prefill is the reverse. One limit cannot govern both, which is why there are two.

The budget also controls how much prefill rides along with the decodes. Raise it and prompts finish sooner, but each forward pass takes longer, so every user's inter-token latency worsens. Lower it and streams stay smooth, but prompts take more iterations. That is the throughput-versus-latency trade-off, exposed as a configuration field.

## Why paging and batching are one design

Continuous batching and PagedAttention ([article 03](03-kv-cache.md)) are not independent optimisations that happen to ship together. Neither delivers its headline number without the other.

Continuous batching needs a cache that can grow one sequence at a time, shrink at unpredictable moments, and accept a new sequence into whatever space was just vacated, all between two forward passes. A contiguous per-sequence slab cannot do that; the scheduler would have to wait for a hole of the right size, which brings back the blocking. In the other direction, a paged cache serving static batches recovers the wasted memory and then leaves it idle, because nothing is admitted until the whole batch drains. The paged cache makes the scheduler's flexibility cheap, and the scheduler makes the paged cache economically meaningful.

## Under memory pressure: preemption

Admitting every arrival works only while there is spare cache. When there is not, the scheduler must decide which requests to admit, which to make wait, and, if memory is exhausted, which to **preempt**: evict a running request's cache, either by copying it to host memory or by discarding it and recomputing it later. Paged allocation is what makes this affordable, because a request's cache is already split into small blocks addressed through a block table.

## Common misconceptions

- **"Continuous batching makes each request faster."** It does not. Per-request latency is unchanged; total throughput rises because seats are never wasted.
- **"A bigger batch is always better."** A larger batch raises intensity but also lengthens each step, and each added sequence adds cache reads. The token budget and concurrency cap exist to bound that.
- **"The two limits are the same thing."** One counts sequences, the other counts tokens. A deployment can be starved of seats with tokens to spare, or the reverse.

## Check yourself

1. In the 712-versus-1,600 example, which request's latency changes under continuous batching, and by how much?
2. Why does a 4,096-token budget permit 3,072 prefill tokens even with 24 sequences decoding, and what happens to inter-token latency if you double the budget?
3. Why would a paged cache serving static batches waste the memory it just recovered?

## Reproduce

```bash
python labs/cpu/05_continuous_batching.py
python tools/batching_svg.py     # regenerates the schedule figure
```

## Sources

- Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models*, OSDI 2022.
- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, 2023. arXiv:2309.06180.

---

Copyright (c) 2026 Akbar Arif. Released under the [MIT License](../LICENSE). You may reuse and adapt this article as long as the copyright and license notice stays with it.
