# 04. Quantization: spending fewer bytes per number

**Question this answers:** a model trained in 16-bit arithmetic can be stored in 8 or even 4 bits and still work. How, and what exactly does it buy?

## Why it comes first

[Article 02](02-why-decode-is-slow.md) showed that a decode step is limited by how many bytes must be read, not by arithmetic. The bytes read are the parameter count times the **bytes per parameter**. Quantization attacks that second factor directly. It does not work around the memory bottleneck; it makes the thing being moved smaller. Going from BF16 (2 bytes) to INT8 (1 byte) halves the weight read, and a 4-bit format with group scales (0.516 bytes) cuts it by about 3.9x. It also frees HBM for the KV cache of [article 03](03-kv-cache.md), so it raises capacity as well as speed.

## Number formats: how bits are spent

A floating-point number spends its bits on three fields: a **sign**, an **exponent** that sets the order of magnitude (the dynamic range), and a **mantissa** that sets how many significant digits are kept (the precision). Formats differ in how they split a fixed budget between range and precision.

| Format | Bits | Sign | Exponent | Mantissa |
|---|---|---|---|---|
| FP32 | 32 | 1 | 8 | 23 |
| FP16 | 16 | 1 | 5 | 10 |
| BF16 | 16 | 1 | 8 | 7 |
| INT8 | 8 | none | none | 8-bit integer plus an external scale |

BF16 keeps FP32's full 8-bit exponent, so it covers the same range and needs no rescaling, and pays for that with a shorter mantissa. FP16 has more precision but a small range: its largest finite value is 65,504, so tensors must be scaled to avoid overflow. INT8 has no exponent at all. Every value is a plain integer, and the range-versus-precision trade-off moves into one external number, the **scale**.

## Affine quantization

Think of a measuring gauge with a fixed number of tick marks. Each real value is rounded to the nearest tick, and the spacing between ticks is the scale. If the values you need to read do not centre on zero, you slide the ticks with an offset, the zero-point.

```
q     = round(x / s) + z          (store this integer)
x_hat = s x (q - z)               (the arithmetic units reconstruct this)
```

**Symmetric** quantization fixes `z = 0` and maps `[-a, a]` onto `[-127, 127]` for INT8. **Asymmetric** quantization picks `z` so an interval that is not centred on zero, say `[-0.10, 0.86]`, uses the whole `[0, 255]` range.

## One row, both rules

`labs/cpu/04_quantization.py` quantizes eight post-activation-like values with both rules. The symmetric step is 0.006772 (it must cover `[-0.86, 0.86]`), and the asymmetric step is 0.003765 with zero-point 27.

| x | sym int | dequant | error | asym int | dequant | error |
|---|---|---|---|---|---|---|
| -0.10 | -15 | -0.1016 | 0.0016 | 0 | -0.1016 | 0.0016 |
| 0.05 | 7 | 0.0474 | 0.0026 | 40 | 0.0489 | 0.0011 |
| 0.22 | 32 | 0.2167 | 0.0033 | 85 | 0.2184 | 0.0016 |
| 0.41 | 61 | 0.4131 | 0.0031 | 136 | 0.4104 | 0.0004 |
| 0.63 | 93 | 0.6298 | 0.0002 | 194 | 0.6287 | 0.0013 |
| 0.86 | 127 | 0.8600 | 0.0000 | 255 | 0.8584 | 0.0016 |
| 0.34 | 50 | 0.3386 | 0.0014 | 117 | 0.3388 | 0.0012 |
| 0.12 | 18 | 0.1219 | 0.0019 | 59 | 0.1205 | 0.0005 |
| worst | | | 0.0033 | | | 0.0016 |

Read it carefully. The guaranteed error bound is half a step, so the zero-point improves the worst case anyone can promise by the step ratio, **1.80**. On this row the realised worst case improves by almost exactly 2x, which is luck sitting on top of the guarantee. Notice too that the symmetric column reconstructs 0.86 and 0.63 *better* than the asymmetric one: a coarser grid is worse on average and in the worst case, while individual values that land near a tick are almost exact. That is why testing a scheme on a handful of hand-picked weights proves nothing; the honest measurement is a distribution over the whole tensor.

**The zero-point is not free.** Expanding `s(q - z)` inside a matrix multiply creates a cross-term involving `z` times a sum of the other operand, which must be folded into a bias ahead of time or computed at run time. Symmetric quantization has no such term. That is why weights, whose distributions are close to zero-centred, are almost always quantized symmetrically, while skewed activations use asymmetric. The rule of thumb is symmetric for weights, asymmetric for activations, and the reason is arithmetic cost, not accuracy.

## Granularity: how many gauges

One scale for a whole matrix (**per-tensor**) is simplest, but a single large value forces a coarse step onto everything else. **Per-channel** gives each output row its own scale. **Group-wise** gives each block of, say, 128 consecutive weights its own scale. Finer granularity improves accuracy and costs more stored scales.

The lab quantizes a synthetic 256 x 1024 matrix whose rows have different magnitudes and 0.2 percent of whose entries are outliers 20 times larger. It is synthetic on purpose, to show the mechanism; it is not a measurement of any real model.

| Bits | Granularity | Relative RMS error | Bits per weight |
|---|---|---|---|
| 8 | per-tensor | 0.15251 | 8.000 |
| 8 | per-channel | 0.04125 | 8.000 |
| 8 | group of 128 | 0.01688 | 8.125 |
| 4 | per-tensor | 0.75900 | 4.000 |
| 4 | per-channel | 0.53531 | 4.000 |
| 4 | group of 128 | 0.23835 | 4.125 |

Two lessons. Error falls as scales get finer, because an outlier now spoils only its own row or group. And 4 bits is far less forgiving than 8: the grid has 15 levels instead of 255, so outliers hurt much more, which is why 4-bit schemes lean on group scales and on methods that protect the important weights. The group scale costs 16 bits per 128 weights, so "4-bit" really stores 4.125 bits per weight. That is exactly the 4.14 GB that appears in the INT4 row of [article 02](02-why-decode-is-slow.md).

## Post-training versus quantization-aware training

Everything above rounds a finished model, which is **post-training quantization** and treats rounding as damage to minimise. The alternative, **quantization-aware training**, tells the model during training that rounding is coming so it learns weights that survive it. There is a catch. Rounding is a step function, flat except for jumps, so its derivative is zero almost everywhere. Propagated honestly, every gradient reaching a quantized weight would be multiplied by zero and the model would never train. The **straight-through estimator** fixes this by rounding in the forward pass and pretending the rounding was the identity in the backward pass. The optimizer updates a full-precision shadow weight, while the loss is evaluated on its rounded image.

## What quantization does and does not do

- **Weight-only quantization** shrinks the bytes read and the memory used. Unless the kernels also do the arithmetic in low precision, the weights are converted back on the fly, so the win is bandwidth and capacity, not extra FLOP/s.
- **Activation and KV-cache quantization** are different problems. Activations vary per token and have outliers; the KV cache is written once and read many times. They get their own treatment later.
- **Accuracy is a measurement, not a formula.** Methods such as GPTQ and AWQ exist to keep accuracy at 4 bits by handling error and important weights carefully. This repository quotes no accuracy numbers until it can measure them.

## Common misconceptions

- **"INT4 means four times faster than BF16."** It means about 3.9x fewer bytes to read, so up to that much faster *if the step is memory-bound and the kernels are efficient*. Compute-bound phases such as prefill do not speed up from fewer weight bytes.
- **"Asymmetric is strictly better because the error is smaller."** It has a smaller error bound and a higher arithmetic cost. Weights use symmetric for that reason.
- **"Quantization only shrinks the model."** It also moves the roofline: fewer bytes per weight raises arithmetic intensity at any batch size.

## Check yourself

1. Why does BF16 need no rescaling to avoid overflow while FP16 does?
2. In the table above, the symmetric column is more accurate at 0.86 and 0.63. Does that mean symmetric quantization is better on this row? Why not?
3. A team quantizes to INT4 with group size 32 instead of 128. What happens to bits per weight, and what do they get in return?

## Reproduce

```bash
python labs/cpu/04_quantization.py
```

## Sources

- Rouhani et al., *Microscaling Data Formats for Deep Learning*, 2023. arXiv:2310.10537.
- Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*, 2022. arXiv:2210.17323.
- Lin et al., *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*, 2023. arXiv:2306.00978.

---

Copyright (c) 2026 Akbar Arif. Released under the [MIT License](../LICENSE). You may reuse and adapt this article as long as the copyright and license notice stays with it.
