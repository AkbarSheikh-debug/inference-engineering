# 08. Speculative decoding: more tokens per read of the weights

**Question this answers:** decoding is sequential, because token `t+1` needs token `t`. Can a model produce several tokens for the price of one weight read, without changing its output?

## The trick: verifying is parallel even though generating is not

[Article 02](02-why-decode-is-slow.md) showed that a decode step reads every weight to emit one token, and is limited by memory, not arithmetic. Generating tokens one after another cannot be parallelised. But **checking** a list of candidate tokens can be: if someone hands you tokens `t+1 ... t+k`, the model can compute its own probability at all `k` positions in a single forward pass, exactly like prefill.

Speculative decoding manufactures those candidates cheaply. A small **draft** model proposes `k` tokens one at a time. The large **target** model then verifies all `k` in one pass, keeps the ones it agrees with, and corrects the first one it does not.

Why is verification cheap? A batch of `k+1` positions has arithmetic intensity about `k+1` in BF16, far to the left of the H100's ridge point of about 295. The step is still memory-bound, so it costs the same as a batch of one: 4.79 ms of weight reads for Llama-3-8B on an H100 whether it verifies 1 position or 5 (ignoring KV-cache reads).

## The accept/reject rule

Write `p(x)` for the target's probability of token `x`, and `q(x)` for the draft's. For each drafted position, in order:

1. **Accept** the draft token `x` with probability `min(1, p(x) / q(x))`.
2. On **rejection**, stop verifying and draw one token from the **residual** distribution `max(0, p - q)`, normalised.

If the target likes the drafted token at least as much as the draft did (`p >= q`) it is accepted unconditionally. A coin is flipped only when the draft was over-confident, and the acceptance probability is exactly the ratio by which it overreached.

## A worked example

The draft proposes four tokens after "The weather today": *is*, *a*, *beautiful*, *day*, with `q` = 0.65, 0.50, 0.30, 0.40 and target `p` = 0.72, 0.58, 0.08, 0.20.

- *is* and *a*: `p >= q`, accepted outright.
- *beautiful*: the draft gave it 0.30, the target 0.08, so it survives with probability `0.08 / 0.30 = 0.267`. Suppose the coin rejects it.
- Verification stops. *day* is discarded unexamined, because its context is now invalid.

Rejection does not waste the step. The lab resamples from the residual over five candidate words:

| Token | target p | draft q | max(0, p - q) |
|---|---|---|---|
| sunny | 0.45 | 0.10 | 0.35 |
| warm | 0.22 | 0.08 | 0.14 |
| cloudy | 0.15 | 0.12 | 0.03 |
| beautiful | 0.08 | 0.30 | 0 (clipped) |
| nice | 0.10 | 0.40 | 0 (clipped) |

The residual sums to `Z = 0.52`, so the resampling distribution is *sunny* 0.673, *warm* 0.269, *cloudy* 0.058. The step emits *is*, *a*, *sunny*: three tokens from one target pass, where ordinary decoding needed three.

## Why the output is exactly the target's

The obvious worry is that guessing must bias the output. It does not, and the proof is two lines. Fix a token `x` and add up the ways it can be emitted.

- The draft proposed `x` and it was accepted: `q(x) * min(1, p(x)/q(x)) = min(q(x), p(x))`.
- The draft proposed something else, that was rejected, and `x` came from the residual. The overall rejection probability equals the residual's normaliser `Z`, so this path contributes `Z * max(0, p(x) - q(x)) / Z = max(0, p(x) - q(x))`.

Add them: `min(q, p) + max(0, p - q) = p`, for every `x`. If `p >= q` the terms are `q` and `p - q`; if `p < q` they are `p` and 0. The emitted distribution is the target's, **for any draft model whatsoever**. A bad draft costs throughput, never correctness.

`labs/cpu/08_speculative_decoding.py` checks this by brute force. It pushes 1,000,000 tokens through the rule using the deliberately poor draft above (only 48.0 percent of its proposals are accepted) and compares the emitted frequencies with the target:

| Token | target p | emitted |
|---|---|---|
| sunny | 0.450 | 0.450 |
| warm | 0.220 | 0.220 |
| cloudy | 0.150 | 0.150 |
| beautiful | 0.080 | 0.080 |
| nice | 0.100 | 0.100 |

In the run recorded here the largest gap is 0.0004; sampling noise at a million samples is about 0.0005.

## How much faster?

Suppose each drafted token is accepted independently with probability `alpha` (the acceptance rate, measured per draft and target pair), and the draft proposes `k` tokens. The expected number of tokens per target pass is

```
E[tokens] = 1 + alpha + alpha^2 + ... + alpha^k = (1 - alpha^(k+1)) / (1 - alpha)
```

The draft is not free. If one draft step costs a fraction `c` of a target step, one iteration costs `k c + 1` target-steps, so the wall-clock speedup is `E[tokens] / (k c + 1)`. The table uses `c = 0.05`; the simulated column of the lab agrees with the formula to within 0.01.

| alpha | k | Expected tokens per pass | Speedup at c = 0.05 |
|---|---|---|---|
| 0.5 | 1 | 1.500 | 1.43 |
| 0.5 | 4 | 1.938 | 1.61 |
| 0.5 | 8 | 1.996 | 1.43 |
| 0.7 | 1 | 1.700 | 1.62 |
| 0.7 | 4 | 2.773 | 2.31 |
| 0.7 | 8 | 3.199 | 2.28 |
| 0.9 | 1 | 1.900 | 1.81 |
| 0.9 | 4 | 4.095 | 3.41 |
| 0.9 | 8 | 6.126 | 4.38 |

Two things stand out. Acceptance rate matters far more than `k`: at `alpha = 0.5`, drafting more than a few tokens buys almost nothing and starts to cost, and the speedup at `k = 8` is lower than at `k = 4`. And a poor or expensive draft can lose outright: at `alpha = 0.3`, `k = 8`, `c = 0.3` the speedup is about 0.42, slower than plain decoding. Independent acceptance is also a simplification. Real acceptance varies with context, and these numbers are model arithmetic, not measurements of any engine.

## Where drafts come from

Nothing in the proof constrains `q`, so a draft can come from anywhere cheap. Prompt-copying (n-gram) methods need no second model and work well when the output repeats the input, as in code editing. A small companion model is the general-purpose choice. Head-based methods reuse the target's own hidden state to predict several tokens ahead (Medusa, EAGLE), which correlates their proposals with the target and raises acceptance, at the cost of a training step. Tree-based variants propose several branches and verify the whole tree in one pass.

## When it helps and when it does not

Speculative decoding spends spare arithmetic to save weight reads, so it helps when a step is memory-bound with room to spare: small batches, latency-sensitive traffic. At a large batch the weights are already shared across many sequences and the step approaches the ridge point, so extra verification positions compete for compute and the gain shrinks. It targets the sequential-dependency problem directly, which none of the other techniques in this curriculum do.

## Common misconceptions

- **"It trades accuracy for speed."** It does not. The two-line proof and the million-token check show the output distribution is the target's exactly.
- **"More drafted tokens is always better."** Past a few tokens the extra drafts are rarely accepted and still cost draft compute.
- **"It always speeds things up."** With a poorly matched or expensive draft it can be slower.

## Check yourself

1. In the worked example, why does the target's pass still produce a useful token when the third draft is rejected?
2. Using the formula, what acceptance rate gives about 2 tokens per pass at `k = 4`?
3. Why does speculative decoding help less at very large batch sizes?

## Reproduce

```bash
python labs/cpu/08_speculative_decoding.py
```

## Sources

- Leviathan, Kalman, Matias, *Fast Inference from Transformers via Speculative Decoding*, 2023. arXiv:2211.17192.
- Chen, Borgeaud, Irving, Lespiau, Sifre, Jumper, *Accelerating Large Language Model Decoding with Speculative Sampling*, 2023. arXiv:2302.01318.
- Cai et al., *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads*, 2024. arXiv:2401.10774.
- Li et al., *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty*, 2024. arXiv:2401.15077.

---

Copyright (c) 2026 Akbar Arif. Released under the [MIT License](../LICENSE). You may reuse and adapt this article as long as the copyright and license notice stays with it.
