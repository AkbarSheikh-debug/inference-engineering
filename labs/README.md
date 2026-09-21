# Labs

Runnable companions to the curriculum. Everything under `cpu/` runs on a
laptop with Python and NumPy; no GPU is needed.

```bash
pip install -e ".[dev]"
python labs/cpu/00_tiny_decoder.py
python labs/cpu/01_roofline.py --model llama-3-8b
python labs/cpu/02_kv_calculator.py
python labs/cpu/03_paged_allocator.py
python labs/cpu/04_quantization.py
python labs/cpu/05_continuous_batching.py
```

| Lab | Companion article | What it shows |
|---|---|---|
| `00_tiny_decoder.py` | [01](../curriculum/01-request-lifecycle.md), [03](../curriculum/03-kv-cache.md) | A real (tiny) decoder generating with and without a KV cache: identical output, very different work |
| `01_roofline.py` | [02](../curriculum/02-why-decode-is-slow.md) | Datasheet-ceiling time per decode step for a model on a GPU |
| `02_kv_calculator.py` | [03](../curriculum/03-kv-cache.md) | KV bytes per token, sequences that fit, MHA vs GQA vs MQA |
| `03_paged_allocator.py` | [03](../curriculum/03-kv-cache.md) | Contiguous reservation vs paged blocks on synthetic requests |
| `04_quantization.py` | [04](../curriculum/04-quantization.md) | One skewed row through both quantization rules, then a matrix at three granularities |
| `05_continuous_batching.py` | [05](../curriculum/05-continuous-batching.md) | Static vs continuous schedules, the ten-second token example, a token budget |

Two rules apply to every lab. Hardware specs and model configs come from
`src/ie/` and carry their sources. A lab that prints a *measurement* rather
than arithmetic must record the hardware, driver, and library versions with
the result; there are no measured benchmarks in this repository yet.
