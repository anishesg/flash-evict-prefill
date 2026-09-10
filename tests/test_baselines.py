import pytest
import torch
from flash_evict.baselines import flash_snap,snap_scores,snap_select
from flash_evict.reference import materialized


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@torch.inference_mode()
def test_window_baseline():
    torch.manual_seed(17)
    q=torch.randn(1,6,137,32,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(1,2,137,32,device='cuda',dtype=q.dtype)
    v=torch.randn_like(k)
    out,scores,indices,mask=flash_snap(q,k,v,48,32)
    expected,_=materialized(q,k,v)
    torch.testing.assert_close(out.float(),expected,atol=.015,rtol=.015)
    assert (~mask).sum(-1).eq(48).all()
    assert not mask[...,-32:].any()
    # Full observation window reduces to accumulated H2O mass (mean of GQA heads).
    _,hh=materialized(q,k,v,value_weighted=False)
    torch.testing.assert_close(snap_scores(q,k,137),hh/3,atol=.025,rtol=.005)
    kv=pytest.importorskip('kv_distill.cache')
    _,upstream,_=kv.attention_stats(q,k,3,all_queries=False)
    assert torch.equal(scores,upstream)
    assert torch.equal(indices,kv.snap_indices(upstream,48))
