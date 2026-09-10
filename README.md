# Flash Evict

Research prototype: exact multi-query, value-weighted KV importance without
materializing the attention matrix. Supports CUDA fp16/bf16, causal/noncausal
self-attention, GQA, and arbitrary input strides. Inference only.

```python
from flash_evict import evict
out, importance, indices, eviction_mask = evict(q, k, v, budget=256)
# q: [B,Hq,N,D], k/v: [B,Hkv,N,D]; True in eviction_mask means evict.
compressed_k = k.gather(2, indices[..., None].expand(-1, -1, -1, k.shape[-1]))
```

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
