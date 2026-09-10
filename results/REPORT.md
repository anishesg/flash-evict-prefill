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

Status: incomplete; 0 / 627 planned records.

The frozen exploratory pilot uses nine needle prompts (three lengths × three depths), four LongBench prompts, two GSM8K prompts, and four MMLU prompts. It is too small to establish a general quality advantage. Greedy decoding is repeated across seeds; seed spread is a reproducibility check, not independent sampling uncertainty.

Fused retention is pure top-k with no reserved recent tokens. Fused and SnapKV compress only prefill and then allow cache growth. H2O uses its existing half-recent policy and dynamically evicts during decode; equal initial retention is not equal lifetime memory. Fixed budgets 256 and 1024 directly pair fused scoring with SnapKV.

