"""Explicit observation-window SnapKV scoring, paired with forced Flash SDPA."""
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


def flash_attention(q, k, v):
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(q,k,v,is_causal=True,enable_gqa=True)


def snap_scores(q,k,window=32):
    n=q.shape[2]
    window=min(window,n)
    groups=q.shape[1]//k.shape[1]
    # Match the local kv-distill reference: native dtype QK, FP32 softmax,
    # average grouped heads, and causal observation queries at the prompt end.
    kr=k.repeat_interleave(groups,1)
    logits=(q[:,:,-window:] @ kr.transpose(-1,-2)).float()
    logits *= q.shape[-1]**-0.5
    query_pos=torch.arange(n-window,n,device=q.device)
    key_pos=torch.arange(n,device=q.device)
    logits.masked_fill_(key_pos[None,:]>query_pos[:,None],-torch.inf)
    p=logits.softmax(-1)
    return p.sum(2).reshape(q.shape[0],k.shape[1],groups,n).mean(2)


def snap_select(scores,budget,window=32,kernel=5):
    n=scores.shape[-1]
    if n<=budget:
        idx=torch.arange(n,device=scores.device).expand(*scores.shape[:-1],n)
    else:
        window=min(window,n,budget-1)
        past=scores[...,:n-window]
        pool=F.avg_pool1d(past.flatten(0,1),kernel,stride=1,padding=kernel//2).reshape_as(past)
        top=pool.topk(budget-window,-1).indices
        tail=torch.arange(n-window,n,device=scores.device).expand(*scores.shape[:-1],window)
        idx=torch.cat((top,tail),-1).sort(-1).values
    keep=torch.zeros_like(scores,dtype=torch.bool).scatter_(-1,idx,True)
    return idx,keep


def flash_snap(q,k,v,budget,window=32):
    out=flash_attention(q,k,v)
    scores=snap_scores(q,k,window)
    idx,keep=snap_select(scores,budget,window)
    return out,scores,idx,~keep
