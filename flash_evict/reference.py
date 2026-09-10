"""Materialized, FP32 ground truth. Only use at small sequence lengths."""
import torch


def materialized(q, k, v, *, causal=True, value_weighted=True, scale=None,
                 recency_decay=0., observation_window=0):
    groups = q.shape[1] // k.shape[1]
    kr = k.float().repeat_interleave(groups, 1)
    vr = v.float().repeat_interleave(groups, 1)
    logits = q.float() @ kr.transpose(-1, -2)
    logits *= q.shape[-1] ** -0.5 if scale is None else scale
    if causal:
        positions = torch.arange(q.shape[2], device=q.device)
        logits.masked_fill_(positions[None, :] > positions[:, None], -torch.inf)
    p = logits.softmax(-1)
    output = p @ vr
    n = q.shape[2]
    positions = torch.arange(n, device=q.device)
    weights = (recency_decay * (positions.float() - (n - 1))).exp()
    if observation_window:
        weights = weights * (positions >= n - observation_window)
    mass = (p * weights[:, None]).sum(-2).reshape(q.shape[0], k.shape[1], groups, k.shape[2]).sum(2)
    if value_weighted:
        mass *= v.float().abs().sum(-1)
    return output, mass
