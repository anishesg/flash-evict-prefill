import pytest
import torch

pytest.importorskip('kv_distill')
from transformers import Qwen2Config, Qwen2ForCausalLM
from flash_evict.quality_adapter import PairedController, choose_indices, decode_batch
from flash_evict.streaming_adapter import StreamingController


def tiny_model():
    config = Qwen2Config(vocab_size=257, hidden_size=128, intermediate_size=256,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        max_position_embeddings=512)
    config._attn_implementation = 'sdpa'
    return Qwen2ForCausalLM(config).cuda().to(torch.bfloat16).eval()


def test_ablation_selection_is_exact():
    k = torch.zeros(1, 2, 5, 2)
    v = torch.tensor([1., -9., 3., -7., 2.]).view(1, 1, 5, 1).expand(1, 2, 5, 2)
    norms = choose_indices(dict(backend='value_norm_only', budget=2), k, v)
    recent = choose_indices(dict(backend='uniform_recent', budget=2), k, v)
    assert norms.tolist() == [[[1, 3], [1, 3]]]
    assert recent.tolist() == [[[3, 4], [3, 4]]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('backend', ['full', 'snap', 'fused'])
@torch.inference_mode()
def test_shared_prefill_matches_independent_policy(backend):
    torch.manual_seed(73)
    model = tiny_model()
    ids = torch.randint(0, 257, (1, 137), device='cuda')
    pos = torch.arange(137, device='cuda')[None]
    scoring = dict(recency_decay=.1, observation_window=128, value_weighted=True)
    methods = [dict(name=f'{backend}_{budget}', backend=backend, budget=budget,
                    **scoring, reserved_recent=16) for budget in (32, 48)]
    paired = PairedController(model)
    paired.prepare(methods, 137)
    actual = model.model(ids, position_ids=pos, use_cache=False).last_hidden_state
    bank = paired.bank
    paired.close()
    for method in methods:
        ctl = StreamingController(model, backend=backend, budget=method['budget'],
                                  scoring=scoring, reserved_recent=16)
        ctl.reset()
        ctl.position_end = 137
        expected = model.model(ids, position_ids=pos, use_cache=False).last_hidden_state
        torch.testing.assert_close(actual, expected, atol=.001, rtol=.001)
        for layer, state in ctl.states.items():
            for key in ('k', 'v'):
                torch.testing.assert_close(bank[method['name']][layer][key], state[key], atol=.001, rtol=.001)
        ctl.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@torch.inference_mode()
def test_batched_decode_matches_independent_caches():
    torch.manual_seed(91)
    model = tiny_model()
    ids = torch.randint(0, 257, (1, 137), device='cuda')
    ctl = PairedController(model)
    methods = [dict(name=name, backend=name, budget=48) for name in ('uniform_recent', 'value_norm_only', 'snap')]
    ctl.prepare(methods, 136)
    model.model(ids[:, :-1], position_ids=torch.arange(136, device='cuda')[None], use_cache=False)
    bank = {name: {i: {k: v.clone() for k, v in state.items()} for i, state in states.items()}
            for name, states in ctl.bank.items()}
    names = [m['name'] for m in methods]
    ctl.activate(names)
    batched = decode_batch(model, ctl, int(ids[0, -1]), 137, 8, set())
    for name, actual in zip(names, batched):
        ctl.bank = {name: bank[name]}
        ctl.activate([name])
        expected = decode_batch(model, ctl, int(ids[0, -1]), 137, 8, set())[0]
        assert actual == expected
    ctl.close()
