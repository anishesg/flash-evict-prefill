# Query recency and observation-window follow-up

Requested performance tables: in progress. The larger historical exploratory grid remains partial.

The separate full 21-task evaluation is documented in [LONGBENCH_PLAN.md](LONGBENCH_PLAN.md); its coverage and scores are written to `longbench_full/SUMMARY.md` as the campaign runs. The quality tables below retain the earlier adaptive pilot.

Quality uses the same 19 prompts (4 LongBench, 2 GSM8K, 4 MMLU, 9 needle) and three greedy repetitions. Parameters were tuned on these prompts; these results do not establish held-out benchmark quality. Quality prompts are at most 8192 tokens. The 131K measurements use synthetic inputs and do not measure answer quality at that length.

The independent decay=0.1 run reruns all 627 conditions. Each axis/pinning/control policy has 627 conditions (342 new fused conditions and 285 explicitly shared baselines). The 18-point Cartesian refinement evaluates both fixed budgets on all prompts and seeds: 114 new fused conditions plus the same 285 baselines per point.

Historical baseline token sequences reproduced: 285/285.

Reservation is inside the total budget. All compressed policies apply to every model layer, compact prompt K/V, preserve absolute RoPE positions, and let the decode cache grow. H2O-0.1 retains its original dynamic eviction behavior. Prompts shorter than a budget keep their full prompt cache; the GSM8K/MMLU pilot prompts do not undergo eviction at budget 1024.

## Quality and 131K kernel memory

Memory is peak additional PyTorch allocation for one BF16 attention layer, B=1, Hq=28, Hkv=4, D=128. Outputs, scores, selection, and masks are included; input Q/K/V and model weights are excluded.

| Policy | Budget | LongBench | GSM8K | MMLU | Needle % | Peak MiB | 131K ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| d01_r0 | 256 | 80.56 | 100.00 | 75.00 | 100.00 | 899.01 | 5571.16 |
| d01_r0 | 1024 | 90.52 | 100.00 | 75.00 | 100.00 | 899.03 | 5588.13 |
| d0.1_w128_r0 | 256 | 80.56 | 100.00 | 75.00 | 100.00 | 899.01 | 2982.97 |
| d0.1_w128_r0 | 1024 | 90.52 | 100.00 | 75.00 | 100.00 | 899.03 | 3104.67 |
| d0.1_w128_r64 | 256 | 81.25 | 100.00 | 75.00 | 100.00 | 900.22 | 3104.88 |
| d0.1_w128_r64 | 1024 | 90.52 | 100.00 | 75.00 | 100.00 | 900.25 | 3057.28 |
| SnapKV | 256 | 76.19 | 100.00 | 75.00 | 100.00 | 2705.00 | 2151.99 |
| SnapKV | 1024 | 92.65 | 100.00 | 75.00 | 100.00 | 2705.00 | 2152.68 |
| full | — | 67.86 | 100.00 | 75.00 | 100.00 | — | — |
| h2o_0.1 | — | 38.33 | 0.00 | 50.00 | 11.11 | — | — |
| original uniform fused_256 | — | 7.14 | 50.00 | 50.00 | 0.00 | — | — |
| original uniform fused_1024 | — | 25.00 | 100.00 | 75.00 | 11.11 | — | — |

## Latency at 8K, 32K, and 131K

Five warmed CUDA-event samples per measurement; medians below. The process lists before and after each measurement are in the raw JSON. Zero additional CUDA processes are required for uncontended measurements.

| Policy | Budget | 8K ms | 32K ms | 131K ms |
|---|---:|---:|---:|---:|
| d01_r0 | 256 | 20.61 | 339.14 | 5571.16 |
| d01_r0 | 1024 | 20.62 | 341.59 | 5588.13 |
| d0.1_w128_r0 | 256 | 12.04 | 187.39 | 2982.97 |
| d0.1_w128_r0 | 1024 | 12.04 | 187.34 | 3104.67 |
| d0.1_w128_r64 | 256 | 12.06 | 187.77 | 3104.88 |
| d0.1_w128_r64 | 1024 | 12.07 | 185.87 | 3057.28 |
| SnapKV | 256 | 8.83 | 129.06 | 2151.99 |
| SnapKV | 1024 | 8.84 | 129.67 | 2152.68 |

## Pareto frontier

Dominance considers all four quality scores (higher is better) and 131K additional allocation (lower is better). Latency is reported separately. No confidence intervals are inferred from repeated greedy seeds.

- Budget 256: d01_r0, d0.1_w128_r0, d0.1_w128_r64
- Budget 1024: d01_r0, d0.1_w128_r0, SnapKV

## Whole-model prefill

All 28 Qwen layers, BF16 weights, no scoring diagnostics, and immediate per-layer compaction. Synthetic token sequences measure shape-dependent cost; timings exclude tokenization, the LM head, and decode. Absolute allocator peaks include weights, caches, attention outputs, and MLP activations.

The MLP chunk size is shared by every method within a configuration; zero means the original unchunked MLP. OOM is an observed allocation failure, not a missing measurement. Rows with another CUDA process are excluded.

| Tokens | Method | Budget | MLP chunk | Median seconds | Absolute peak GiB | Incremental GiB |
|---:|---|---:|---:|---:|---:|---:|
| 8192 | full | — | 0 | 2.120 | 16.197 | 1.523 |
| 8192 | snap_1024 | 1024 | 0 | 2.147 | 15.814 | 1.141 |
| 8192 | d01_r0 | 1024 | 0 | 2.442 | 15.814 | 1.141 |
| 8192 | d0.1_w128_r0 | 1024 | 0 | 2.207 | 15.814 | 1.141 |
| 8192 | d0.1_w128_r64 | 1024 | 0 | 2.207 | 15.814 | 1.141 |
| 32768 | full | — | 0 | OOM | — | — |
| 32768 | snap_1024 | 1024 | 0 | 11.071 | 19.072 | 4.399 |
| 32768 | d01_r0 | 1024 | 0 | 16.507 | 19.072 | 4.399 |
| 32768 | d0.1_w128_r0 | 1024 | 0 | 12.316 | 19.072 | 4.399 |
| 32768 | d0.1_w128_r64 | 1024 | 0 | 12.359 | 19.072 | 4.399 |
| 131072 | full | — | 0 | OOM | — | — |
| 131072 | snap_1024 | 1024 | 0 | OOM | — | — |
| 131072 | d01_r0 | 1024 | 0 | OOM | — | — |
| 131072 | d0.1_w128_r0 | 1024 | 0 | OOM | — | — |
| 131072 | d0.1_w128_r64 | 1024 | 0 | OOM | — | — |

## Interpretation

Recency weights are exp(decay * (q - (N-1))) per token, and windowing masks exactly the final w query positions. Attention outputs use the original online softmax. Scores use its final normalizers in a tiled replay; window policies skip replay for earlier query programs. Pooling and value weighting are explicit ablations.

This is one attention-kernel launch with a second QK traversal for scored queries, followed by separate selection. It is not a single QK traversal, zero scoring cost, or O(N) dense-attention arithmetic. Live score storage is O(N); fixed-window SnapKV storage O(Nw) is also linear in N for fixed w. The measured memory ratio describes additional allocations in this implementation, not total-model memory or a novelty proof.

