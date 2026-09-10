# Flash Evict

Research prototype: exact multi-query, value-weighted KV importance without
materializing the attention matrix. Supports CUDA fp16/bf16, causal/noncausal
self-attention, GQA, and arbitrary input strides. Inference only.

**Query-recency follow-up:** decay 0.1 recovers LongBench from 25.00 to 90.52
and needle from 11.11% to 100% at budget 1024 in the completed 627-condition
pilot. Its 131K additional allocation is 899.03 MiB, with 5.59 seconds per
attention layer on A10G. The window/pinning grid and whole-model measurements
are in [the follow-up report](results/FOLLOWUP.md). This is adaptive tuning on
19 prompts, with three repeated greedy runs; quality was tested up to 8K.
Scoring uses tiled replay inside one kernel launch. See the
[original uniform-policy results](results/REPORT.md) and
[mathematical analysis](RESEARCH_NOTES.md) for the original failure and claim boundaries.

**Full LongBench follow-up:** the pending 131K window measurements are now in
the follow-up report. A separate [frozen full evaluation](results/LONGBENCH_PLAN.md)
covers all 4,750 examples in 21 tasks, 11 methods, and the value-only and suffix-only
controls. Its runner saves predictions individually and pushes each completed task.
The four reporting tests check task averaging, paired intervals, and incomplete
or duplicated evaluation records; ten adapter tests check the shared evaluation path.
Run `scripts/run_followup.py` with the kv-distill Python environment and
`PYTHONPATH=.:/home/qcb/kv-distill` to resume the sequential performance and quality campaign.

```python
from flash_evict import evict
out, importance, indices, eviction_mask = evict(
    q, k, v, budget=1024, recency_decay=0.1, reserved_recent=0)
# q: [B,Hq,N,D], k/v: [B,Hkv,N,D]; True in eviction_mask means evict.
compressed_k = k.gather(2, indices[..., None].expand(-1, -1, -1, k.shape[-1]))
```

Query weights are `exp(recency_decay * (q - (N-1)))`, computed per token.
`observation_window=128` restricts scoring to the final 128 queries and skips
the replay for earlier query programs. Both options can be combined. All query
positions still receive complete attention outputs. `reserved_recent=64`
keeps the suffix inside the total budget; `pool_kernel=5` enables a separate
SnapKV-style average-pooling ablation, and `value_weighted=False` removes the
value norm. Defaults `recency_decay=0, observation_window=0` reproduce the
historical all-query score.

The optional `StreamingController` in `flash_evict/streaming_adapter.py` compacts
each layer's stored cache immediately, while passing its full attention output
to the next layer. It has the same restricted evaluator contract as the original
adapter. Separate tests compare its cache contents and decode output with
delayed compaction.

## What is implemented

Each query-tile program computes stable online-softmax attention, then replays
its QK tiles using the final row normalizers to accumulate
`||v[k]||_1 * sum_(query, grouped query head) attention[query,k]` into FP32 scores.
Tile state lives on chip; the shared per-KV-token accumulators live in HBM.
The scoring replay is inside the attention kernel. Zero initialization and
global top-k/mask selection are separate launches. Atomic sums can differ by a
few FP32 ulps between runs; top-k tie ordering is unspecified.

This is **one attention kernel with two traversals, not single-pass scoring**.
Memory is O(B Hq N D + B Hkv N), and dense compute is O(B Hq N² D).
The implementation does not establish a first-of-its-kind or zero-overhead claim.

## Why the proposed single traversal is insufficient

For each query q, `p[q,k] = exp(s[q,k]-m[q]) / l[q]` needs the FINAL m and l.
Later KV tiles change these quantities differently for each query. After a
column has been reduced across queries, a single rescale cannot correct its
previous contribution. Keeping all query-specific partial contributions, using
an approximation, or revisiting tiles is necessary for that proposed schedule.
This is a counterexample to the supplied accumulation rule, not a general
impossibility theorem for all algorithms. Global top-k additionally needs scores
from all query programs to have completed.

## Source audit

* [LongFlow](https://github.com/yisunlp/LongFLow), cloned under
  `third_party/LongFlow`, revision `89a68d30ea28857fc8e1b229544a2cc23f52a43e`:
  `triton_kernel/attention.py` has TWO loops, stable online normalization, FP16/
  BF16 importance atomics, and a separate PyTorch argmin. Its current code differs
  from the single-traversal, unnormalized description in its paper. This project
  uses independent matrix-tile code and keeps score accumulation in FP32.
* [LongFlow paper](https://arxiv.org/abs/2603.11504) uses SnapKV for over-budget prefill.
* [MiniKV](https://supercomputing-system-ai-lab.github.io/projects/minikv/)
  already describes a two-pass Triton selective FlashAttention kernel producing
  cumulative prefill attention scores in linear memory. It is relevant prior art.
* [Triton attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
  documents the stable online-softmax recurrence used here.

SnapKV materializes an observation-window-by-context matrix, not necessarily
N×N. Its O(Nw) scoring memory is also linear in N for fixed w. H2O attention mass
can be recomputed in chunks without a resident full N×N matrix (the local
kv-distill baseline already does this). These distinctions matter for claims.

## Validation

```bash
/home/qcb/kv-distill/.venv/bin/python -m pytest -q
```

Initial A10G validation: 9 tests passed. Materialized FP32 comparisons include
512, 1024, and 4096 tokens, multiple heads/GQA, fp16/bf16, ragged tiles,
noncontiguous inputs, batch size 2, extreme logits, and zero values. Pure top-k
masks match at 5%, 10%, and 20% on the tested random inputs. Separate unweighted
checks recover H2O mass and its conservation invariant. This does not guarantee
identical masks for arbitrarily close scores under finite precision.

Extended validation: 14 tests pass, including analytic 8K/32K/128K cases,
physical cache compaction and decoding in a small Qwen2 model, and agreement
between the SnapKV benchmark implementation and the local kv-distill reference.
To include the optional Qwen adapter tests, use:

```bash
PYTHONPATH=.:/home/qcb/kv-distill /home/qcb/kv-distill/.venv/bin/python -m pytest -q
```

## Experiments

```bash
PYTHONPATH=. /home/qcb/kv-distill/.venv/bin/python scripts/benchmark.py
PYTHONPATH=. /home/qcb/kv-distill/.venv/bin/python scripts/benchmark.py \
    --heads 28 --kv-heads 4 --dim 128 --require-idle --output results/benchmark_qwen.json
PYTHONPATH=.:/home/qcb/kv-distill /home/qcb/kv-distill/.venv/bin/python scripts/evaluate.py
/home/qcb/kv-distill/.venv/bin/python scripts/report.py
/home/qcb/kv-distill/.venv/bin/python scripts/plot_quality.py
```

See [the experimental report](results/REPORT.md) and raw JSON results. GPU
sharing affected initial measurements; process lists and free-memory readings
are retained, and these timings must not be presented as exclusive-device results.
The final `benchmark_qwen.json` sweep observed no other CUDA processes before or
after any measurement. At 128K tokens (one layer, BF16, B=1, Hq=28, Hkv=4,
D=128, 10% retention), fused replay plus top-k takes 5508.32 ms and 901.23 MiB
of additional allocations; FlashAttention plus SnapKV with w=32 takes 2151.77 ms
and 2705.00 MiB. Outputs and selection are included; inputs and model weights
are excluded. Triton reports zero spills and zero global scratch.

Qwen evaluation uses the existing frozen pilot data, model revision, baseline
cache lifecycle, and metrics. The default selects a small, explicitly frozen
19-prompt exploratory subset; `--full-pilot` in a new output directory runs all
118 prompts. Partial reports clearly state incomplete record counts.

The completed pilot contains 627 conditions (19 prompts × 11 methods × 3 greedy
repetitions). All 57 uncompressed fused controls and all 285 repeated historical
baseline conditions reproduce the reference token sequences. Three repeated
greedy seeds do not provide independent quality samples. At budget 1024, fused
versus SnapKV scores are 25.00 versus 92.65 on four LongBench prompts, and 11.11%
versus 100% exact match on nine needle prompts. Quality declines after this
policy is applied; full-cache controls preserve reference generation.

The adapter is optional and leaves `/home/qcb/kv-distill` unchanged. It accepts
only that evaluator's unpadded, single-prompt, contiguous-position prefill flow;
it is not a general Transformers attention backend. Fused policies retain pure
top-k prompt entries per KV head and let the decode cache grow, matching the
prefill-only compression lifecycle of SnapKV. H2O continues dynamic eviction.
