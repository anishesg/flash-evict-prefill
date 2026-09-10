# Full LongBench evaluation

Status: in progress. 21 tasks, 4750 examples per method, 11 methods.

All original test examples are evaluated once with greedy decoding. Task scores use the pinned LongBench author metrics and prescribed generation lengths. Aggregate scores are equal-weight means of task scores. The English/code aggregate excludes Chinese tasks (16 tasks in the full benchmark). No full-benchmark score is emitted before every requested example is present.

Context limit: 32,768 tokens including the reserved answer budget. Official task templates use Qwen chat wrapping except the six official no-chat tasks. Middle truncation is applied at token level after wrapping, as in the earlier pilot. The pilot used an 8,192-token prompt cap; its scores are not directly comparable.

Budgets count retained prompt tokens per KV head in all 28 layers. The final prompt token is forwarded after prefix compaction; the decode cache then grows. Recency attention includes value weighting. The value-only control uses L1 value magnitude; the suffix control retains only the most recent budget tokens and has no attention sinks. SnapKV uses w=32, average-pool kernel 5, and 32 reserved recent tokens inside the budget, matching the pilot implementation.

Compatible methods share prefill, and compatible caches batch decoding. MLP chunks of 1024 tokens apply to every method. Quality-run timings and peaks include shared work and live policy banks; use the separate whole-model benchmark for performance comparisons.

| Method | Records | All-task macro | English/code macro | Macro excluding tuning prompts |
|---|---:|---:|---:|---:|
| full | 50/4750 | pending | pending | pending |
| snap_256 | 50/4750 | pending | pending | pending |
| d01_r0_256 | 50/4750 | pending | pending | pending |
| d0.1_w128_r0_256 | 50/4750 | pending | pending | pending |
| d0.1_w128_r64_256 | 50/4750 | pending | pending | pending |
| snap_1024 | 50/4750 | pending | pending | pending |
| d01_r0_1024 | 50/4750 | pending | pending | pending |
| d0.1_w128_r0_1024 | 50/4750 | pending | pending | pending |
| d0.1_w128_r64_1024 | 50/4750 | pending | pending | pending |
| value_norm_only_256 | 50/4750 | pending | pending | pending |
| uniform_recent_256 | 50/4750 | pending | pending | pending |

## Per-task quality

| Task | full | snap_256 | d01_r0_256 | d0.1_w128_r0_256 | d0.1_w128_r64_256 | snap_1024 | d01_r0_1024 | d0.1_w128_r0_1024 | d0.1_w128_r64_1024 | value_norm_only_256 | uniform_recent_256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| narrativeqa | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| qasper | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| multifieldqa_en | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| multifieldqa_zh | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| hotpotqa | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| 2wikimqa | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| musique | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| dureader | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| gov_report | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| qmsum | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| multi_news | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| vcsum | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| trec | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| triviaqa | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| samsum | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| lsht | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| passage_count | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| passage_retrieval_en | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| passage_retrieval_zh | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| lcc | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| repobench-p | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |

## Paired comparisons

Pointwise intervals (without a multiple-comparison adjustment) use 2,000 paired bootstrap resamples within each task, with tasks held fixed. They measure example variability, not variability across new tasks or model training seeds.

