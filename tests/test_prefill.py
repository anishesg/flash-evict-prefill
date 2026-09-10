import math
import pytest
import torch
from flash_evict import prefill, evict, select_tokens
from flash_evict.reference import materialized


@pytest.fixture(autouse=True)
def precision():
    torch.manual_seed(731)
    torch.backends.cuda.matmul.allow_tf32 = False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n,h,hk,d,dtype,causal", [
    (512,4,4,64,torch.float16,True),
    (1024,4,2,64,torch.bfloat16,True),
    (4096,4,2,128,torch.bfloat16,True),
    (137,6,2,80,torch.float16,False),
    (65,4,1,32,torch.bfloat16,True),
    (1,2,1,16,torch.float16,True),
])
def test_exact(n,h,hk,d,dtype,causal):
    # Transposed tensors exercise the real model's noncontiguous layout.
    q = torch.randn((1,n,h,d),device="cuda",dtype=dtype).transpose(1,2)
    k,v = [torch.randn((1,n,hk,d),device="cuda",dtype=dtype).transpose(1,2) for _ in range(2)]
    out,importance = prefill(q,k,v,causal=causal)
    expected,scores = materialized(q,k,v,causal=causal)
    torch.testing.assert_close(out.float(),expected,atol=0.012 if dtype==torch.bfloat16 else 0.002,rtol=0.006)
    torch.testing.assert_close(importance,scores,atol=3e-3,rtol=3e-4)
    for rate in (.05,.1,.2):
        budget=max(1,math.ceil(n*rate))
        idx,keep=select_tokens(importance,budget)
        refidx,refkeep=select_tokens(scores,budget)
        assert torch.equal(keep,refkeep), (rate,(keep != refkeep).sum().item())
        assert torch.equal(idx,refidx)
    # Unweighted mode must reproduce materialized H2O mass, not weighted scores.
    _,hh = prefill(q,k,v,causal=causal,value_weighted=False)
    _,hh_ref=materialized(q,k,v,causal=causal,value_weighted=False)
    torch.testing.assert_close(hh,hh_ref,atol=3e-5,rtol=3e-4)
    torch.testing.assert_close(hh.sum(-1),torch.full_like(hh.sum(-1),n*h/hk),rtol=1e-5,atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_extreme_logits_batches_and_zero_values():
    q=torch.randn(2,4,97,64,device="cuda",dtype=torch.float16)*15
    k=torch.randn(2,2,97,64,device="cuda",dtype=torch.float16)*15
    v=torch.randn_like(k)
    out,scores,idx,mask=evict(q,k,v,17,recent=5)
    ref,refscore=materialized(q,k,v)
    torch.testing.assert_close(out.float(),ref,atol=0.004,rtol=0.006)
    torch.testing.assert_close(scores,refscore,atol=0.01,rtol=0.001)
    assert torch.isfinite(scores).all()
    assert (~mask).sum(-1).eq(17).all()
    assert not mask[...,-5:].any()
    zero,zeroscore=prefill(q,k,torch.zeros_like(v))
    assert zero.count_nonzero()==0 and zeroscore.count_nonzero()==0
    plain,_=prefill(q,k,v,score=False)
    assert torch.equal(plain,out)


def test_normalization_counterexample():
    # For key 0 the two query contributions are both 1 before seeing key 1.
    # Their final denominators differ. The column sum 2 cannot be corrected by
    # the common single-query denominator shortcut used for ranking in decode.
    logits=torch.tensor([[0.,0.],[0.,math.log(9.)]])
    normalized=logits.softmax(-1).sum(0)
    prematurely_summed=logits.exp().sum(0)
    assert torch.allclose(normalized,torch.tensor([.6,1.4]))
    assert not torch.allclose(normalized,prematurely_summed/prematurely_summed.sum()*2)
    causal_logits=torch.tensor([[math.log(9.),-torch.inf],[math.log(100.),math.log(200.)]])
    # Omitting per-query denominators can even reverse the top-1 eviction mask.
    assert causal_logits.softmax(-1).sum(0).argmax()==0
    assert causal_logits.exp().sum(0).argmax()==1


def test_selection_contract():
    scores=torch.arange(20.).reshape(1,1,20)
    idx,keep=select_tokens(scores,4,recent=2)
    assert idx.tolist()==[[[16,17,18,19]]]
    assert keep.sum()==4
    for budget in (0,21,True,2.5):
        with pytest.raises(ValueError): select_tokens(scores,budget)
    with pytest.raises(ValueError): select_tokens(scores,2,recent=3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize('n',[8192,32768,131072])
def test_long_uniform_analytic(n):
    # Q=K=0 gives a closed-form exact oracle at long lengths without N² HBM.
    # P[q,k]=1/(q+1) for k<=q; output is a running value mean and importance
    # is a reverse harmonic sum, multiplied by the value norm and GQA groups.
    q=torch.zeros(1,2,n,16,device='cuda',dtype=torch.bfloat16)
    k=torch.zeros(1,1,n,16,device='cuda',dtype=q.dtype)
    v=torch.randn_like(k)
    out,scores=prefill(q,k,v)
    positions=torch.arange(1,n+1,device='cuda',dtype=torch.float64)
    expected_out=(v.double().cumsum(2)/positions[None,None,:,None]).expand_as(out)
    harmonic=positions.reciprocal().flip(0).cumsum(0).flip(0)
    expected_scores=(v.double().abs().sum(-1)*harmonic*2).float()
    torch.testing.assert_close(out.float(),expected_out.float(),atol=.008,rtol=.008)
    torch.testing.assert_close(scores,expected_scores,atol=.002,rtol=.0003)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('n,decay,window', [
    (512,.01,0),(1024,.1,0),(4096,1.,0),
    (137,0.,32),(1024,0.,64),(4096,0.,128),
    (137,.1,65),(65,1.,128),(1,.1,32),
])
@pytest.mark.parametrize('value_weighted',[True,False])
def test_query_weighted_exact(n,decay,window,value_weighted):
    q=torch.randn(1,n,4,64,device='cuda',dtype=torch.bfloat16).transpose(1,2)
    k,v=[torch.randn(1,n,2,64,device='cuda',dtype=q.dtype).transpose(1,2) for _ in range(2)]
    kwargs=dict(recency_decay=decay,observation_window=window,value_weighted=value_weighted)
    out,scores=prefill(q,k,v,**kwargs)
    expected,ref=materialized(q,k,v,**kwargs)
    torch.testing.assert_close(out.float(),expected,atol=.012,rtol=.006)
    torch.testing.assert_close(scores,ref,atol=3e-5,rtol=4e-4)
    plain,_=prefill(q,k,v,score=False)
    assert torch.equal(out,plain)
    if not value_weighted:
        pos=torch.arange(n,device='cuda')
        weight=(decay*(pos.float()-(n-1))).exp()
        if window: weight*=pos>=n-window
        torch.testing.assert_close(scores.sum(-1),weight.sum().expand(1,2)*2,rtol=1e-5,atol=1e-4)
    budget=min(n,128);recent=min(64,budget)
    _,keep=select_tokens(scores,budget,reserved_recent=recent)
    _,reference_keep=select_tokens(ref,budget,reserved_recent=recent)
    assert torch.equal(keep,reference_keep)
    assert keep.sum(-1).eq(budget).all() and keep[...,-recent:].all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('decay,window',[(.01,0),(0.,128),(.1,65)])
def test_long_weighted_analytic(decay,window):
    n=131072
    q=torch.zeros(1,2,n,16,device='cuda',dtype=torch.bfloat16)
    k=torch.zeros(1,1,n,16,device='cuda',dtype=q.dtype)
    v=torch.ones_like(k)
    _,scores=prefill(q,k,v,recency_decay=decay,observation_window=window)
    pos=torch.arange(n,device='cuda',dtype=torch.float64)
    weight=(decay*(pos-(n-1))).exp()
    if window:weight*=pos>=n-window
    expected=(weight/(pos+1)).flip(0).cumsum(0).flip(0)*32
    torch.testing.assert_close(scores[0,0],expected.float(),atol=1e-7,rtol=5e-4)


def test_pinning_and_pooling_contract():
    from flash_evict.baselines import snap_select
    torch.manual_seed(1)
    scores=torch.randn(1,3,137)
    idx,keep=select_tokens(scores,48,reserved_recent=32,pool_kernel=5)
    expected,expected_keep=snap_select(scores,48,32,5)
    assert torch.equal(idx,expected) and torch.equal(keep,expected_keep)
    _,tail=select_tokens(scores,64,reserved_recent=64,pool_kernel=5)
    assert tail.sum()==3*64 and tail[...,-64:].all()
    for kwargs in ({'recent':True},{'pool_kernel':2},{'recent':1,'reserved_recent':2}):
        with pytest.raises(ValueError):select_tokens(scores,64,**kwargs)
