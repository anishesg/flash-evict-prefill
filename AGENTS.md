You are a research engineer building a novel KV cache eviction system. Your mission: implement the first KV cache eviction policy fused into FlashAttention's prefill kernel — running inside the blockwise streaming computation without ever materializing the full N×N attention matrix.

## THE GAP YOU ARE FILLING

Every KV cache eviction method (H2O, SnapKV, PyramidKV) requires the full accumulated attention scores across all query positions. FlashAttention processes attention in blocks and never writes the full matrix to HBM. These are fundamentally incompatible.

LongFlow (arXiv:2603.11504) solved this for DECODING (single query token) by accumulating per-KV-token importance scores block by block inside a Triton kernel. But for PREFILL (multiple query tokens), they fall back to SnapKV — which materializes the full attention matrix, breaking FA's O(n) memory guarantee.

The gap: nobody has extended LongFlow's kernel to multi-query prefill. You will.

## THE APPROACH

Extend FlashAttention's prefill kernel to simultaneously:
1. Compute attention output with standard online softmax normalization (exactly as FA does)
2. Accumulate per-KV-token importance scores across ALL query blocks in SRAM
3. Emit an eviction mask at the end — all in a single kernel pass

The importance metric: ||α_{q,k} · v_k||₁ — the L1 norm of the attention-weighted value contribution, summed across all query positions. This is LongFlow's metric extended to multi-query.

Memory cost: O(n) additional scalars (one importance accumulator per KV token) — negligible vs. the KV cache itself.

## IMPLEMENTATION PLAN

### Phase 1: Understand LongFlow's kernel (1 hour)
- Clone https://github.com/yisunlp/LongFLow
- Read triton_kernel/attention.py carefully
- Understand exactly how they accumulate importance during the single-query decode pass
- Identify what changes when there are multiple query tokens

### Phase 2: Build the fused prefill kernel (3 hours)
Write a Triton kernel that:
1. Outer loop: iterate over QUERY blocks (Q_block_ptr)
2. Inner loop: iterate over KV blocks (K_block_ptr, V_block_ptr)
3. For each (Q_block, KV_block) pair:
   - Compute attention scores: S = Q_block @ K_block^T / sqrt(d)
   - Apply causal mask
   - Online softmax update (standard FA: update m_i, l_i, acc_o)
   - Compute importance contribution: for each kv position j in KV_block, accumulate:
     importance[j] += sum_over_q_in_block(alpha[q,j] * ||v_j||₁)
     where alpha[q,j] = softmax(S)[q,j]
4. After all blocks: importance[j] holds total importance for KV token j
5. Emit top-k indices and eviction mask

Key insight: the importance accumulation lives in SRAM alongside the FA accumulators. Adding it costs O(BLOCK_SIZE_KV) SRAM per block — cheap.

### Phase 3: Validate correctness (1 hour)
Compare against:
- Full attention (non-FA) with explicit importance computation
- H2O with materialized attention matrix
- Verify eviction masks match on small sequences where exact computation is possible

Test cases:
- Sequence length 512, 1024, 4096
- Multiple heads
- Various retention rates (5%, 10%, 20%)

### Phase 4: Benchmark (2 hours)
Measure:
- Latency: single-pass FA+eviction vs. FA prefill + separate SnapKV scoring pass
- Memory: peak HBM usage during prefill at 8K, 32K, 128K tokens
- Quality: apply eviction mask, run generation, compare output quality vs. baseline

The claim: equal or better eviction quality as SnapKV, with O(n) prefill memory instead of O(n·observation_window) for the scoring pass.

### Phase 5: Quality evaluation (2 hours)
- Load Qwen-2.5-7B
- Run our KV distillation baselines: full, h2o_0.05, h2o_0.1, snap_1024, snap_256
- Add our fused kernel as a new compression method
- Compare quality at same retention rates
- Target: match or exceed SnapKV quality while being fully O(n)

## ENVIRONMENT

GPU: NVIDIA A10G 24GB
Python: /home/qcb/kv-distill/.venv (has triton, torch, transformers, peft)
Reference: /home/qcb/kv-distill/third_party/ (H2O, KIVI, SnapKV, LongBench)
Prior baselines: /home/qcb/kv-distill/results/original_pilot/ (Qwen-2.5-7B scores)

## SCIENTIFIC INTEGRITY

- Verify correctness against exact (non-FA) implementation before benchmarking
- Measure actual HBM usage with torch.cuda.memory_stats()
- Test at multiple sequence lengths (not just short ones)
- Report both quality AND latency — this paper needs both to make the case
- Git commit working kernel immediately with unit tests

## THE PAPER CLAIM IF THIS WORKS

"We present the first KV cache eviction policy fused into FlashAttention's prefill kernel. Unlike prior methods which require a separate O(n·w) attention scoring pass (SnapKV) or fall back to full attention (LongFlow), our kernel computes attention output AND eviction scores in a single O(n·d) pass. We demonstrate equal or superior eviction quality to SnapKV while eliminating the scoring overhead entirely."

## START

Clone LongFlow. Read the kernel. Build the prefill extension. Verify it. Benchmark it. If it works, we have a paper.
