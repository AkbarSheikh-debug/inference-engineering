# Primary sources

Each source is listed with the claim in this repository that rests on it. If you find a claim that is not supported by its source, please open an issue.

| Source | Supports |
|---|---|
| Vaswani et al., *Attention Is All You Need*, 2017. arXiv:1706.03762 | Transformer structure, attention over keys, values and queries |
| Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need*, 2019. arXiv:1911.02150 | Multi-query attention |
| Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, 2023. arXiv:2305.13245 | Grouped-query attention; uptraining from MHA with about 5% of original compute |
| DeepSeek-AI, *DeepSeek-V2*, 2024. arXiv:2405.04434 | Multi-head latent attention |
| Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, 2023. arXiv:2309.06180 | Paged KV cache; the share of cache memory holding actual tokens in earlier systems |
| Williams, Waterman, Patterson, *Roofline*, Communications of the ACM 52(4), 2009 | The roofline model |
| Pope et al., *Efficiently Scaling Transformer Inference*, 2022. arXiv:2211.05102 | Memory-bandwidth analysis of decode |
| Leviathan, Kalman, Matias, *Fast Inference from Transformers via Speculative Decoding*, 2023. arXiv:2211.17192 | Speculative decoding |
| Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling*, 2023. arXiv:2302.01318 | Speculative sampling (concurrent with the above) |
| Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models*, OSDI 2022 | Iteration-level (continuous) batching |
| Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*, 2022. arXiv:2210.17323 | Accuracy-preserving 4-bit weight quantization |
| Lin et al., *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*, 2023. arXiv:2306.00978 | Protecting important weights under low-bit quantization |
| Rouhani et al., *Microscaling Data Formats for Deep Learning*, 2023. arXiv:2310.10537 | Block-scaled low-bit formats |
| Touvron et al., *Llama 2*, 2023. arXiv:2307.09288 | Llama-2-7B architecture |
| Llama Team, *The Llama 3 Herd of Models*, 2024. arXiv:2407.21783 | Llama-3-8B and 70B architectures |
| NVIDIA A100 80GB datasheet | A100 SXM bandwidth (2,039 GB/s) and dense BF16 throughput (312 TFLOP/s) |
| NVIDIA H100 datasheet | H100 SXM bandwidth (3.35 TB/s) and dense BF16 throughput (989 TFLOP/s) |

Model parameter counts in `src/ie/models.py` are the published approximate totals. HBM capacity for "80 GB" GPUs is modelled as 80 GiB; `nvidia-smi` reports about 81,559 MiB on an H100 80GB.
