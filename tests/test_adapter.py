import pytest
import torch

pytest.importorskip('kv_distill')
from transformers import Qwen2Config,Qwen2ForCausalLM
from flash_evict.kv_distill_adapter import FusedController


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@torch.inference_mode()
@pytest.mark.parametrize('scoring,recent', [({},0),({'recency_decay':.1},5),({'observation_window':32},5)])
def test_qwen_prefill_compress_decode(scoring,recent):
    torch.manual_seed(19)
    config=Qwen2Config(vocab_size=257,hidden_size=128,intermediate_size=256,
                      num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
                      max_position_embeddings=512)
    config._attn_implementation='sdpa'
    model=Qwen2ForCausalLM(config).cuda().to(torch.bfloat16).eval()
    ids=torch.randint(0,257,(1,137),device='cuda')
    pos=torch.arange(137,device='cuda')[None]
    native=model.model(ids,position_ids=pos,use_cache=False).last_hidden_state
    ctl=FusedController(model)
    ctl.scoring=scoring;ctl.reserved_recent=recent
    ctl.prefill_backend='fused';ctl.reset();ctl.position_end=137
    fused=model.model(ids,position_ids=pos,use_cache=False).last_hidden_state
    torch.testing.assert_close(fused,native,atol=.025,rtol=.02)
    original={i:state['k'].clone() for i,state in ctl.states.items()}
    ctl.compress('fused_0.1',137)
    for i,state in ctl.states.items():
        assert state['k'].shape[2]==14
        assert set(state)=={'k','v'}
        assert state['k'].untyped_storage().nbytes()==state['k'].numel()*state['k'].element_size()
        if recent:assert torch.equal(state['k'][:,:,-recent:],original[i][:,:,-recent:])
        # Every compacted key is a key at its original (already RoPE-rotated) position.
        assert all(any(torch.equal(state['k'][0,h,j],original[i][0,h,p]) for p in range(137))
                   for h in range(2) for j in range(14))
    ctl.position_end=138
    out=model.model(ids[:,-1:],position_ids=torch.tensor([[137]],device='cuda'),use_cache=False)
    assert torch.isfinite(out.last_hidden_state).all()
    assert all(s['k'].shape[2]==15 for s in ctl.states.values())
    ctl.close()
