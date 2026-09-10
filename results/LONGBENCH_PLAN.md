# Frozen full LongBench experiment

The requested evaluation contains all 4,750 examples in the 21 original LongBench tasks and 11 methods (52,250 predictions). It is separate from the adaptive four-prompt quality pilot. Report both the 21-task macro average and the 16-task English/code macro average, per-task scores, and a sensitivity analysis excluding the four tuning prompts. Use paired within-task bootstrap intervals for differences, not repeated greedy seeds.

Methods: full; SnapKV at 256 and 1024; d01_r0 at 256 and 1024; d0.1_w128_r0 at 256 and 1024; d0.1_w128_r64 at 256 and 1024; value_norm_only at 256; uniform_recent at 256. The requested recency_scores_only control is d01_r0_256: recency-weighted attention multiplied by the L1 value norm. uniform_recent is a pure suffix control without sink tokens, not a complete StreamingLLM implementation.

SnapKV uses the same local reference as the pilot: a 32-query observation window, average pooling with kernel size 5, and a reserved 32-token suffix inside the budget. Query heads sharing a KV head are averaged. Fused scoring sums these heads (a common positive factor that does not change top-k), uses decay 0.1, and weights attention mass by the KV head's L1 value norm. The r64 variant reserves 64 recent tokens inside its budget; r0 reserves none.

The model and dataset use the revisions already frozen in kv-distill. Official LongBench task templates, answer budgets, postprocessing, and metrics are loaded from `/home/qcb/kv-distill/third_party/LongBench/LongBench/`. Qwen chat wrapping is omitted for the six official no-chat tasks. Middle truncation is applied after wrapping at token level, preserving the prefix and suffix. The total context limit is 32,768, including the reserved task-specific generation budget. Prompt length averages 9,887 tokens; 94 examples are truncated. The older pilot capped prompts at 8,192 tokens, so absolute pilot scores are not directly comparable.

Every method keeps the same per-head prompt budget in all 28 layers, preserving absolute RoPE positions. The final prompt token is forwarded after prefix compaction, then the cache grows during decoding, matching the pilot's cache protocol. All methods use the same 1,024-token MLP chunk size. Greedy decoding stops at the model's generation EOS tokens and, for Samsum, the official newline stop with one minimum generated token.

Compatible policies share identical prefill outputs; all-query and window-scoring policies still run separate prefill passes. Compatible retained cache lengths batch decoding across policies. The ten adapter tests verify selection controls, shared versus independent prefill, immediate versus delayed compaction, MLP chunking, and batched versus independent decode. Finite-precision matrix multiplication can depend on batch shape. Raw quality timings share work and include live policy cache banks, so standalone latency and memory claims must use the separate benchmark scripts.

Reproduce from the repository root:

```bash
export PYTHONPATH=.:/home/qcb/kv-distill
/home/qcb/kv-distill/.venv/bin/python scripts/prepare_longbench.py
/home/qcb/kv-distill/.venv/bin/python scripts/evaluate_longbench.py --require-idle --push
/home/qcb/kv-distill/.venv/bin/python scripts/report_longbench.py
```

Tokenized inputs are ignored by Git; their hashes, source revisions, example identities, and truncation metadata are copied into the tracked run manifest. Prediction records are flushed and fsynced individually; resumed runs reject changed data, changed code, duplicate records, and incompatible protocols. The runner commits and pushes each completed task. Aggregate scores remain pending until all requested examples are present.

Performance configurations must remain distinct: the original unchunked MLP and a common 1,024-token chunked MLP. An observed OOM is a measured capacity limit. Allocator peaks come from `torch.cuda.memory_stats()` and whole-model peaks include BF16 weights. Synthetic 131K performance tests are not 131K generation quality evidence.
