import pytest
import torch

pytest.importorskip('kv_distill')
from transformers import Qwen2Config,Qwen2ForCausalLM
from flash_evict.kv_distill_adapter import FusedController
from flash_evict.streaming_adapter import StreamingController,chunk_mlp


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@torch.inference_mode()
@pytest.mark.parametrize('scoring',[{'recency_decay':.1},{'observation_window':128}])
def test_immediate_layer_compaction_matches_delayed_compaction(scoring):
    torch.manual_seed(190)
    config=Qwen2Config(vocab_size=257,hidden_size=128,intermediate_size=256,
                      num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2,
                      max_position_embeddings=512)
    config._attn_implementation='sdpa'
    model=Qwen2ForCausalLM(config).cuda().to(torch.bfloat16).eval()
    ids=torch.randint(0,257,(1,137),device='cuda');pos=torch.arange(137,device='cuda')[None]
    ctl=FusedController(model);ctl.prefill_backend='fused';ctl.scoring=scoring;ctl.reserved_recent=16
    ctl.reset();ctl.position_end=137
    delayed=model.model(ids,position_ids=pos,use_cache=False).last_hidden_state
    ctl.compress('fused_48',137)
    cache={i:{k:v.clone() for k,v in state.items()} for i,state in ctl.states.items()}
    ctl.position_end=138
    decode=model.model(ids[:,-1:],position_ids=pos[:,-1:]+1,use_cache=False).last_hidden_state
    ctl.close()
    ctl=StreamingController(model,budget=48,scoring=scoring,reserved_recent=16)
    ctl.reset();ctl.position_end=137
    immediate=model.model(ids,position_ids=pos,use_cache=False).last_hidden_state
    assert torch.equal(immediate,delayed)
    for i,state in ctl.states.items():
        assert set(state)=={'k','v'}
        for key,tensor in state.items():assert torch.equal(tensor,cache[i][key])
    ctl.finish_prefill();ctl.position_end=138
    actual=model.model(ids[:,-1:],position_ids=pos[:,-1:]+1,use_cache=False).last_hidden_state
    assert torch.equal(actual,decode)
    ctl.close()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@torch.inference_mode()
@pytest.mark.parametrize('backend,method',[('full','full'),('snap','snap_48')])
def test_streaming_baselines_match_evaluator(backend,method):
    torch.manual_seed(17)
    config=Qwen2Config(vocab_size=257,hidden_size=128,intermediate_size=256,
                      num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
                      max_position_embeddings=512)
    config._attn_implementation='sdpa'
    model=Qwen2ForCausalLM(config).cuda().to(torch.bfloat16).eval()
    ids=torch.randint(0,257,(1,137),device='cuda');pos=torch.arange(137,device='cuda')[None]
    ctl=FusedController(model);ctl.reset();ctl.position_end=137
    expected=model.model(ids,position_ids=pos,use_cache=False).last_hidden_state
    ctl.compress(method,137)
    cache={i:{k:v.clone() for k,v in state.items()} for i,state in ctl.states.items()}
    ctl.close()
    ctl=StreamingController(model,backend=backend,budget=48);ctl.reset();ctl.position_end=137
    actual=model.model(ids,position_ids=pos,use_cache=False).last_hidden_state
    torch.testing.assert_close(actual,expected,atol=.025,rtol=.02)
    for i,state in ctl.states.items():
        assert set(state)=={'k','v'}
        for key,tensor in state.items():torch.testing.assert_close(tensor,cache[i][key],atol=.005,rtol=.005)
    ctl.close()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@torch.inference_mode()
def test_chunked_mlp_and_restore():
    config=Qwen2Config(vocab_size=257,hidden_size=128,intermediate_size=256,
                      num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2)
    config._attn_implementation='sdpa'
    model=Qwen2ForCausalLM(config).cuda().to(torch.bfloat16).eval()
    ids=torch.randint(0,257,(1,137),device='cuda')
    expected=model.model(ids,use_cache=False).last_hidden_state
    restore=chunk_mlp(model,31)
    actual=model.model(ids,use_cache=False).last_hidden_state
    torch.testing.assert_close(actual,expected,atol=.025,rtol=.02)
    restore()
    assert torch.equal(model.model(ids,use_cache=False).last_hidden_state,expected)
