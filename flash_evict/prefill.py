"""Inference-only FlashAttention with exact contribution scoring by tiled replay.

One attention kernel launch, TWO QK traversals, then separate selection kernels.
No N x N tensor is allocated. Scores are FP32 atomic sums in HBM, not a global
SRAM accumulator. Query blocks retain their final softmax statistics in SRAM.
"""
import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _prefill(Q, K, V, O, I,
             qb: tl.constexpr, qh: tl.constexpr, qt: tl.constexpr, qd: tl.constexpr,
             kb: tl.constexpr, kh: tl.constexpr, kt: tl.constexpr, kd: tl.constexpr,
             vb: tl.constexpr, vh: tl.constexpr, vt: tl.constexpr, vd: tl.constexpr,
             H: tl.constexpr, HK: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
             SCALE: tl.constexpr, CAUSAL: tl.constexpr, WEIGHTED: tl.constexpr,
             SCORE: tl.constexpr, DECAY: tl.constexpr, WINDOW: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    block = tl.program_id(0)
    bh = tl.program_id(1)
    batch, head = bh // H, bh % H
    kvhead = head // (H // HK)
    rows = block * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    dims = tl.arange(0, BD)
    q = tl.load(Q + batch * qb + head * qh + rows[:, None] * qt + dims[None, :] * qd,
                (rows[:, None] < N) & (dims[None, :] < D), other=0)
    kbase = K + batch * kb + kvhead * kh
    vbase = V + batch * vb + kvhead * vh
    maximum = tl.full((BM,), -float('inf'), tl.float32)
    denominator = tl.zeros((BM,), tl.float32)
    output = tl.zeros((BM, BD), tl.float32)
    end = N
    if CAUSAL:
        end = tl.minimum(N, (block + 1) * BM)
    # Standard stable online-softmax attention. P is local to this tile.
    for start in range(0, end, BN):
        keys = start + cols
        k = tl.load(kbase + dims[:, None] * kd + keys[None, :] * kt,
                    (dims[:, None] < D) & (keys[None, :] < N), other=0)
        logits = tl.dot(q, k) * SCALE
        valid = keys[None, :] < N
        if CAUSAL:
            valid = valid & (rows[:, None] >= keys[None, :])
        logits = tl.where(valid, logits, -float('inf'))
        new_maximum = tl.maximum(maximum, tl.max(logits, 1))
        correction = tl.exp2(maximum - new_maximum)
        p = tl.exp2(logits - new_maximum[:, None])
        denominator = denominator * correction + tl.sum(p, 1)
        output = output * correction[:, None]
        v = tl.load(vbase + keys[:, None] * vt + dims[None, :] * vd,
                    (keys[:, None] < N) & (dims[None, :] < D), other=0)
        output = tl.dot(p.to(q.dtype), v, output)
        maximum = new_maximum
    tl.store(O + ((batch * H + head) * N + rows[:, None]) * D + dims[None, :],
             output / denominator[:, None], (rows[:, None] < N) & (dims[None, :] < D))
    if SCORE and (block + 1) * BM > N - WINDOW:
        # Recompute each tile using FINAL row normalizers. Summing across rows
        # before this normalization would lose the information needed to fix it.
        # Weight actual query positions, including a partially overlapping tile.
        # Earlier query programs still compute their complete attention output.
        query_weight = tl.exp(DECAY * (rows.to(tl.float32) - (N - 1)))
        query_weight = tl.where((rows >= N - WINDOW) & (rows < N), query_weight, 0.)
        for start in range(0, end, BN):
            keys = start + cols
            k = tl.load(kbase + dims[:, None] * kd + keys[None, :] * kt,
                        (dims[:, None] < D) & (keys[None, :] < N), other=0)
            logits = tl.dot(q, k) * SCALE
            valid = (keys[None, :] < N) & (rows[:, None] < N)
            if CAUSAL:
                valid = valid & (rows[:, None] >= keys[None, :])
            p = tl.exp2(tl.where(valid, logits - maximum[:, None], -float('inf'))) / denominator[:, None]
            mass = tl.sum(p * query_weight[:, None], 0)
            if WEIGHTED:
                v = tl.load(vbase + keys[:, None] * vt + dims[None, :] * vd,
                            (keys[:, None] < N) & (dims[None, :] < D), other=0)
                mass *= tl.sum(tl.abs(v.to(tl.float32)), 1)
            tl.atomic_add(I + (batch * HK + kvhead) * N + keys, mass,
                          keys < N, sem="relaxed")


def _validate(q, k, v):
    if any(t.ndim != 4 for t in (q, k, v)):
        raise ValueError("Q/K/V must have shape [batch, heads, tokens, channels]")
    b, h, n, d = q.shape
    if min(b, h, n, d, k.shape[1]) < 1 or k.shape != v.shape or k.shape[0] != b or k.shape[2:] != (n, d):
        raise ValueError("Expected nonempty self-attention Q/K/V with equal sequence and channel dimensions")
    if h % k.shape[1] or d > 256:
        raise ValueError("Query heads must be divisible by KV heads; head dimension must be <= 256")
    if any(t.device != q.device or t.dtype != q.dtype for t in (k, v)):
        raise ValueError("Q/K/V must have the same device and dtype")
    if not q.is_cuda or q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("CUDA fp16 or bf16 inputs required")
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        raise ValueError("Inference only: no backward implementation")


def prefill(q, k, v, *, causal=True, value_weighted=True, score=True,
            scale=None, block_q=32, block_k=64, recency_decay=0., observation_window=0):
    """Return (attention output, FP32 importance [B, Hkv, N]).

    Importance sums over queries and query heads sharing a KV head:
      I[b,h,k] = ||V[b,h,k]||_1 * sum_(g,q) w[q] softmax(Q K^T)[b,h,g,q,k].
    w[q] = exp(recency_decay * (q - (N-1))), optionally restricted to the
    last observation_window queries (0 means all). Attention output is unchanged.
    Window scoring skips replay in earlier query programs, saving QK work.
    With value_weighted=False, return H2O attention mass. No dropout, padding,
    cross-attention, or backward; arbitrary input strides and GQA are supported.
    FP32 atomic reduction order can cause tiny run-to-run differences. score=False
    provides the same attention implementation without replay for ablations.
    """
    _validate(q, k, v)
    recency_decay = float(recency_decay)
    if not math.isfinite(recency_decay) or recency_decay < 0:
        raise ValueError("recency_decay must be finite and nonnegative")
    if isinstance(observation_window, bool) or not isinstance(observation_window, int) or observation_window < 0:
        raise ValueError("observation_window must be a nonnegative integer; 0 means all queries")
    if block_q not in (16, 32, 64) or block_k not in (32, 64, 128):
        raise ValueError("Unsupported tile dimensions")
    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")
    b, h, n, d = q.shape
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    importance = torch.zeros((b, k.shape[1], n), dtype=torch.float32, device=q.device) if score else None
    with torch.cuda.device(q.device):
        _prefill[(triton.cdiv(n, block_q), b * h)](
            q, k, v, out, importance, *q.stride(), *k.stride(), *v.stride(),
            h, k.shape[1], n, d, scale * math.log2(math.e), causal, value_weighted,
            score, recency_decay, min(n, observation_window or n),
            block_q, block_k, max(16, triton.next_power_of_2(d)),
            num_warps=4, num_stages=2)
    return out, importance


def select_tokens(importance, budget, *, recent=0, reserved_recent=None, pool_kernel=1):
    """Return chronologically sorted kept indices and a bool KEEP mask.

    A separate GPU selection follows the attention kernel's global completion.
    Ties follow torch.topk (unspecified order). recent reserves a suffix inside
    the total budget, useful for quality ablations; default is pure top-k.
    reserved_recent is an alias for recent. Optional average pooling smooths
    scores over past-token neighbors, excluding the reserved suffix, as SnapKV.
    """
    n = importance.shape[-1]
    if reserved_recent is not None:
        if recent != 0 and recent != reserved_recent:
            raise ValueError("recent and reserved_recent disagree")
        recent = reserved_recent
    if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= n:
        raise ValueError("budget must be an integer in [1, sequence length]")
    if isinstance(recent, bool) or not isinstance(recent, int) or not 0 <= recent <= budget:
        raise ValueError("recent must be an integer in [0, budget]")
    if isinstance(pool_kernel, bool) or not isinstance(pool_kernel, int) or pool_kernel < 1 or pool_kernel % 2 != 1:
        raise ValueError("pool_kernel must be a positive odd integer")
    past = importance[..., :n-recent] if recent else importance
    if pool_kernel > 1 and budget > recent:
        past = F.avg_pool1d(past.reshape(-1, 1, past.shape[-1]), pool_kernel,
                            stride=1, padding=pool_kernel//2).reshape_as(past)
    top = past.topk(budget - recent, dim=-1).indices
    if recent:
        tail = torch.arange(n-recent, n, device=importance.device).expand(*importance.shape[:-1], recent)
        top = torch.cat((top, tail), -1)
    indices = top.sort(-1).values
    keep = torch.zeros_like(importance, dtype=torch.bool).scatter_(-1, indices, True)
    return indices, keep


def evict(q, k, v, budget, *, recent=0, reserved_recent=None, pool_kernel=1, **kwargs):
    """Return output, importance, kept indices, and bool eviction mask (True=evict)."""
    if kwargs.get("score") is False:
        raise ValueError("evict requires importance scoring")
    output, importance = prefill(q, k, v, **kwargs)
    indices, keep = select_tokens(importance, budget, recent=recent,
                                  reserved_recent=reserved_recent, pool_kernel=pool_kernel)
    return output, importance, indices, ~keep
