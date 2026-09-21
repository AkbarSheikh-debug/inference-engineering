# 00. The roadmap: how to learn inference engineering

**Question this answers:** what do you actually need to know to understand how a large language model runs on real hardware, and in what order?

## The idea that organises everything

Most inference material is a list of techniques: quantization, batching, paged attention, speculative decoding. A list is hard to remember and easy to misapply. This curriculum is organised around one question that every technique answers:

1. **Which resource is the bottleneck?** Memory bandwidth, memory capacity, compute, launch overhead, idle GPU time, or the interconnect between GPUs.
2. **Why?** Show the arithmetic.
3. **Which technique attacks that bottleneck, and what does it cost?**

Once you can answer those three for a workload, the techniques stop being a list and become consequences. A batch-1 decode step is limited by memory bandwidth, so anything that moves fewer bytes or uses each byte more times helps. That single sentence already explains quantization, batching and speculative decoding.

## The stack you are learning

Everything in inference engineering lives in one of five layers. Knowing which layer a problem belongs to tells you who can fix it and what tools apply.

<!-- diagram:start d02-layered-stack -->
```mermaid
%% title: The five-layer inference stack
flowchart TB
    L5["<b>5 Product</b><br/>API, streaming, SLOs, cost per token<br/><i>limited by: the latency and cost trade-off</i>"]
    L4["<b>4 Serving</b><br/>routing, admission, continuous batching, prefix cache, disaggregation<br/><i>limited by: scheduling and tail latency</i>"]
    L3["<b>3 Engine</b><br/>vLLM, SGLang, TensorRT-LLM: KV memory manager, speculative decoding, graph capture<br/><i>limited by: memory management and launch overhead</i>"]
    L2["<b>2 Model and kernels</b><br/>transformer, attention variants, quantization, fused kernels<br/><i>limited by: HBM bandwidth and compute</i>"]
    L1["<b>1 Hardware</b><br/>SMs, registers, shared memory, L2, HBM, NVLink, CPU host<br/><i>limited by: bandwidth, capacity, interconnect</i>"]
    L5 -->|"requests down"| L4
    L4 --> L3
    L3 --> L2
    L2 --> L1
    L1 -.->|"tokens up"| L5
    classDef layer fill:#eff6ff,stroke:#3b82f6,color:#1e3a8a
    class L5,L4,L3,L2,L1 layer
```
<!-- diagram:end -->

## The dependency graph

Learning order is a graph, not a ladder. Quantization needs the GPU memory hierarchy and the roofline, but not attention. Paging needs the KV cache but not kernels. A green node has at least one published article (some topics inside it may still be planned); grey nodes are planned.

<!-- diagram:start d00-roadmap-dag -->
```mermaid
%% title: Learning roadmap as a dependency graph
flowchart TD
    L0["<b>0 The problem</b><br/>tokens, TTFT, ITL, cost"]
    L1["<b>1 The machine</b><br/>CPU host, GPU, memory hierarchy"]
    L2["<b>2 The model</b><br/>transformer, attention, prefill vs decode"]
    L3["<b>3 Why it is slow</b><br/>roofline, arithmetic intensity"]
    L4["<b>4 Shrink it</b><br/>quantization, FlashAttention, fusion"]
    L5["<b>5 Manage state</b><br/>KV cache, paging, prefix caching"]
    L6["<b>6 Schedule</b><br/>continuous batching, chunked prefill, speculation"]
    L7["<b>7 Scale out</b><br/>TP, PP, EP, CP, disaggregation"]
    L8["<b>8 Engines</b><br/>vLLM, SGLang, TensorRT-LLM"]
    L9["<b>9 Production</b><br/>routing, observability, cost"]
    L0 --> L1
    L0 --> L2
    L1 --> L3
    L2 --> L3
    L2 --> L5
    L3 --> L4
    L3 --> L5
    L4 --> L6
    L5 --> L6
    L6 --> L7
    L6 --> L8
    L7 --> L8
    L8 --> L9
    classDef live fill:#d1fae5,stroke:#059669,color:#064e3b
    classDef planned fill:#f3f4f6,stroke:#9ca3af,color:#374151
    class L0,L2,L3,L4,L5,L6 live
    class L1,L7,L8,L9 planned
```
<!-- diagram:end -->

## The levels

| Level | Question | Topics | Status |
|---|---|---|---|
| 0 | What is the problem? | autoregression, tokens, TTFT, ITL, throughput, cost | [01](01-request-lifecycle.md) |
| 1 | What is the machine? | CPU host, GPU, SMs, warps, registers, shared memory, L2, HBM, CUDA execution model | planned |
| 2 | What is the model doing? | forward pass, attention family (MHA, MQA, GQA, MLA), prefill vs decode | [01](01-request-lifecycle.md), [03](03-kv-cache.md) |
| 3 | Why is it slow? | roofline, arithmetic intensity, memory-bound vs compute-bound | [02](02-why-decode-is-slow.md) |
| 4 | How do we shrink it? | FP16/BF16/FP8/INT8/INT4/FP4, weight, activation and KV quantization, FlashAttention, kernel fusion | [04](04-quantization.md) (number formats, integer quantization); FlashAttention and fusion planned |
| 5 | How do we manage state? | KV cache arithmetic, PagedAttention, prefix caching | [03](03-kv-cache.md) |
| 6 | How do we schedule? | continuous batching, chunked prefill, speculative decoding | [05](05-continuous-batching.md) (continuous batching, chunked prefill); speculative decoding planned |
| 7 | How do we scale out? | tensor, pipeline, expert and context parallelism, disaggregated prefill and decode | planned |
| 8 | What runs it? | vLLM, SGLang, TensorRT-LLM, choosing an engine | planned |
| 9 | How do we run it in production? | routing, cold starts, observability, cost per token, failure modes | planned |

## Pick a path

| You are | Read in this order |
|---|---|
| An application engineer who calls an inference API | 01, 02, 04, 05, then level 9 |
| A kernel or performance engineer | 02, 04, then level 1, then 03 |
| An infrastructure or platform engineer | 01, 03, 05, then levels 7, 8 and 9 |
| A student starting from the transformer | 01 to 05 in order, then follow the graph |

## How every article is built

Each article follows the same nine steps: the question, intuition with a diagram, a derivation, the hardware view, a runnable lab, a measured result (only when one exists), a common misconception, three check-yourself questions, and primary sources.

Two rules make the numbers trustworthy. **Every number is generated by code in `src/ie/` or cited to a primary source**, and a test fails if an article quotes a number the code does not produce. **There are no measured benchmarks yet**; everything quoted is arithmetic from published specifications, and every such result is labelled a ceiling, not a forecast.

## Get started

Read [01. What happens between a prompt and the next token](01-request-lifecycle.md).

---

Copyright (c) 2026 Akbar Arif. Released under the [MIT License](../LICENSE). You may reuse and adapt this article as long as the copyright and license notice stays with it.
