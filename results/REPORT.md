# Flash Evict experimental report

Exact tiled replay preserves linear memory but recomputes QK. This is not a single-traversal or zero-overhead implementation.
Dense attention still uses quadratic arithmetic. No novelty or quality-superiority claim follows from correctness alone.

## Correctness

Materialized FP32 tests cover 512, 1024, and 4096 tokens, 5/10/20% masks, GQA, causal/noncausal masks, noncontiguous inputs, and large logits. Analytic uniform-attention tests cover 8K, 32K, and 128K. A small Qwen2 model verifies prefill, physical compaction, and decoding.

## Kernel measurements

CUDA events measure warmed calls. Peak additional allocations come from `torch.cuda.memory_stats()`, with Q/K/V already allocated. Outputs, scores, and selection are included. This is one layer, not whole-model memory. Driver/context and other-process allocations are outside PyTorch allocator peaks; raw reports also record free device memory.

The first two sweeps encountered external GPU activity (including activity that resumed after a sweep began idle). Treat their timings as diagnostic, not uncontended publication measurements. Later sweeps record per-method process lists.

### benchmark.json

BF16, B=1, Hq=4, Hkv=2, D=64, retention=10%.

| Tokens | Method | Median ms | Peak additional MiB |
|---:|---|---:|---:|
| 8192 | flash_snap_w128 | 1.502 | 40.251 |
| 8192 | flash_snap_w32 | 1.185 | 16.250 |
| 8192 | fused_replay_scores | 1.433 | 4.062 |
| 8192 | fused_replay_topk | 1.579 | 4.107 |
| 8192 | torch_flash | 0.974 | 4.126 |
| 8192 | triton_attention_only | 0.852 | 4.000 |
| 32768 | flash_snap_w128 | 11.680 | 161.001 |
| 32768 | flash_snap_w32 | 10.698 | 65.000 |
| 32768 | fused_replay_scores | 19.487 | 16.250 |
| 32768 | fused_replay_topk | 19.681 | 16.426 |
| 32768 | torch_flash | 10.123 | 16.501 |
| 32768 | triton_attention_only | 10.338 | 16.000 |
| 131072 | flash_snap_w128 | 150.373 | 644.001 |
| 131072 | flash_snap_w32 | 145.649 | 260.000 |
| 131072 | fused_replay_scores | 327.163 | 65.000 |
| 131072 | fused_replay_topk | 331.336 | 66.625 |
| 131072 | torch_flash | 146.543 | 66.001 |
| 131072 | triton_attention_only | 179.398 | 64.000 |

### benchmark_contended.json

BF16, B=1, Hq=4, Hkv=2, D=64, retention=10%.

| Tokens | Method | Median ms | Peak additional MiB |
|---:|---|---:|---:|
| 8192 | flash_snap_w128 | 1.399 | 40.251 |
| 8192 | flash_snap_w32 | 1.169 | 16.250 |
| 8192 | fused_replay_scores | 1.352 | 4.062 |
| 8192 | fused_replay_topk | 1.487 | 4.107 |
| 8192 | torch_flash | 0.932 | 4.126 |
| 8192 | triton_attention_only | 0.792 | 4.000 |
| 32768 | flash_snap_w128 | 20.072 | 161.001 |
| 32768 | flash_snap_w32 | 18.915 | 65.000 |
| 32768 | fused_replay_scores | 39.661 | 16.250 |
| 32768 | fused_replay_topk | 34.836 | 16.426 |
| 32768 | torch_flash | 17.480 | 16.501 |
| 32768 | triton_attention_only | 22.095 | 16.000 |
| 131072 | flash_snap_w128 | 280.875 | 644.001 |
| 131072 | flash_snap_w32 | 278.248 | 260.000 |
| 131072 | fused_replay_scores | 560.034 | 65.000 |
| 131072 | fused_replay_topk | 580.245 | 66.625 |
| 131072 | torch_flash | 271.299 | 66.001 |
| 131072 | triton_attention_only | 316.455 | 64.000 |

### benchmark_qwen_contended.json

BF16, B=1, Hq=28, Hkv=4, D=128, retention=10%.

| Tokens | Method | Median ms | Peak additional MiB |
|---:|---|---:|---:|
| 8192 | flash_snap_w128 | 20.999 | 337.063 |
| 8192 | flash_snap_w32 | 16.925 | 169.063 |
| 8192 | fused_replay_scores | 36.039 | 56.125 |
| 8192 | fused_replay_topk | 35.957 | 56.213 |
| 8192 | torch_flash | 13.104 | 56.876 |
| 8192 | triton_attention_only | 16.919 | 56.000 |
| 32768 | flash_snap_w128 | 249.141 | 1348.251 |
| 32768 | flash_snap_w32 | 244.146 | 676.250 |
| 32768 | fused_replay_scores | 577.662 | 224.500 |
| 32768 | fused_replay_topk | 579.020 | 224.850 |
| 32768 | torch_flash | 221.885 | 227.501 |
| 32768 | triton_attention_only | 294.124 | 224.000 |

## Qwen-2.5-7B quality

Status: incomplete; 209 / 627 planned records.

The frozen exploratory pilot uses nine needle prompts (three lengths × three depths), four LongBench prompts, two GSM8K prompts, and four MMLU prompts. It is too small to establish a general quality advantage. Greedy decoding is repeated across seeds; seed spread is a reproducibility check, not independent sampling uncertainty.

Fused retention is pure top-k with no reserved recent tokens. Fused and SnapKV compress only prefill and then allow cache growth. H2O uses its existing half-recent policy and dynamically evicts during decode; equal initial retention is not equal lifetime memory. Fixed budgets 256 and 1024 directly pair fused scoring with SnapKV.

| Suite | Method | Quality mean ± seed std | Records | Mean decode s | Initial KV MiB |
|---|---|---:|---:|---:|---:|
| gsm8k | full | 100.00 ± 0.00 | 2 | 6.496 | 41.45 |
| gsm8k | fused_0.05 | 0.00 ± 0.00 | 2 | 2.026 | 2.11 |
| gsm8k | fused_0.1 | 0.00 ± 0.00 | 2 | 1.893 | 4.18 |
| gsm8k | fused_0.2 | 0.00 ± 0.00 | 2 | 8.480 | 8.31 |
| gsm8k | fused_1024 | 100.00 ± 0.00 | 2 | 6.496 | 41.45 |
| gsm8k | fused_256 | 50.00 ± 0.00 | 2 | 6.954 | 14.00 |
| gsm8k | fused_full | 100.00 ± 0.00 | 2 | 6.497 | 41.45 |
| gsm8k | h2o_0.05 | 0.00 ± 0.00 | 2 | 9.854 | 2.11 |
| gsm8k | h2o_0.1 | 0.00 ± 0.00 | 2 | 9.870 | 4.18 |
| gsm8k | snap_1024 | 100.00 ± 0.00 | 2 | 6.496 | 41.45 |
| gsm8k | snap_256 | 100.00 ± 0.00 | 2 | 6.375 | 14.00 |
| longbench | full | 67.86 ± 0.00 | 4 | 0.462 | 396.73 |
| longbench | fused_0.05 | 8.33 ± 0.00 | 4 | 0.766 | 19.87 |
| longbench | fused_0.1 | 6.25 ± 0.00 | 4 | 0.900 | 39.72 |
| longbench | fused_0.2 | 25.00 ± 0.00 | 4 | 0.822 | 79.38 |
| longbench | fused_1024 | 25.00 ± 0.00 | 4 | 0.630 | 56.00 |
| longbench | fused_256 | 7.14 ± 0.00 | 4 | 0.527 | 14.00 |
| longbench | fused_full | 67.86 ± 0.00 | 4 | 0.449 | 396.73 |
| longbench | h2o_0.05 | 29.90 ± 0.00 | 4 | 0.675 | 19.87 |
| longbench | h2o_0.1 | 38.33 ± 0.00 | 4 | 0.442 | 39.72 |
| longbench | snap_1024 | 92.65 ± 0.00 | 4 | 0.329 | 56.00 |
| longbench | snap_256 | 76.19 ± 0.00 | 4 | 0.369 | 14.00 |
| mmlu | full | 75.00 ± 0.00 | 4 | 0.037 | 25.35 |
| mmlu | fused_0.05 | 25.00 ± 0.00 | 4 | 0.035 | 1.30 |
| mmlu | fused_0.1 | 25.00 ± 0.00 | 4 | 0.035 | 2.57 |
| mmlu | fused_0.2 | 50.00 ± 0.00 | 4 | 0.035 | 5.09 |
| mmlu | fused_1024 | 75.00 ± 0.00 | 4 | 0.036 | 25.35 |
| mmlu | fused_256 | 50.00 ± 0.00 | 4 | 0.036 | 14.00 |
| mmlu | fused_full | 75.00 ± 0.00 | 4 | 0.036 | 25.35 |
| mmlu | h2o_0.05 | 50.00 ± 0.00 | 4 | 0.040 | 1.30 |
| mmlu | h2o_0.1 | 50.00 ± 0.00 | 4 | 0.040 | 2.57 |
| mmlu | snap_1024 | 75.00 ± 0.00 | 4 | 0.036 | 25.35 |
| mmlu | snap_256 | 75.00 ± 0.00 | 4 | 0.036 | 14.00 |
| needle | full | 100.00 ± 0.00 | 9 | 0.309 | 256.62 |
| needle | fused_0.05 | 0.00 ± 0.00 | 9 | 0.548 | 12.87 |
| needle | fused_0.1 | 0.00 ± 0.00 | 9 | 0.552 | 25.68 |
| needle | fused_0.2 | 0.00 ± 0.00 | 9 | 0.529 | 51.35 |
| needle | fused_1024 | 11.11 ± 0.00 | 9 | 0.422 | 56.00 |
| needle | fused_256 | 0.00 ± 0.00 | 9 | 0.531 | 14.00 |
| needle | fused_full | 100.00 ± 0.00 | 9 | 0.309 | 256.62 |
| needle | h2o_0.05 | 0.00 ± 0.00 | 9 | 0.273 | 12.87 |
| needle | h2o_0.1 | 11.11 ± 0.00 | 9 | 0.275 | 25.68 |
| needle | snap_1024 | 100.00 ± 0.00 | 9 | 0.256 | 56.00 |
| needle | snap_256 | 100.00 ± 0.00 | 9 | 0.246 | 14.00 |

Prefill wall times in raw quality records include all model layers. Baseline prefill includes the existing evaluator’s scoring diagnostics; they must not be used for an end-to-end speedup claim. Use the kernel benchmarks for latency comparisons.

Full-cache control: 19 / 19 paired generations have identical token IDs. Finite-precision kernel changes can alter greedy output; controls are retained in the summary JSON.

Historical baseline reproduction: 95 / 95 task scores and 95 / 95 token sequences match the saved original pilot on the same IDs, methods, and seeds.

