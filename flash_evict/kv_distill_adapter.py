"""Optional bridge to the existing kv-distill Qwen2 evaluation engine.

Add /home/qcb/kv-distill to PYTHONPATH. No upstream files are modified.
Baseline methods retain the upstream behavior, including dynamic H2O decoding.
Fused methods compress once after prefill, then grow the cache like SnapKV.
"""
import math
import types
import torch
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from kv_distill.attention import Controller, forward as baseline_forward
from kv_distill.cache import gather_tokens
from .prefill import prefill, select_tokens


class FusedController(Controller):
    def __init__(self,model):
        super().__init__(model)
        self.prefill_backend="baseline"
        for attn,_ in self.originals:
            attn.forward=types.MethodType(forward,attn)

    def compress(self,method,prompt_length):
        if not method.startswith('fused_'):
            return super().compress(method,prompt_length)
        self.method=method
        suffix=method.removeprefix('fused_')
        budget=prompt_length if suffix=='full' else (
            math.ceil(prompt_length*float(suffix)) if '.' in suffix else int(suffix))
        budget=max(1,min(budget,prompt_length))
        for state in self.states.values():
            indices,_=select_tokens(state.pop('importance'),budget)
            state['k']=gather_tokens(state['k'],indices)
            state['v']=gather_tokens(state['v'],indices)
        self.mode='decode'


def forward(self,hidden_states,attention_mask=None,position_ids=None,
            past_key_value=None,output_attentions=False,use_cache=False,cache_position=None):
    ctl=self._kv_controller
    if ctl.mode!='prefill' or ctl.prefill_backend!='fused':
        return baseline_forward(self,hidden_states,attention_mask,position_ids,
                                past_key_value,output_attentions,use_cache,cache_position)
    if hidden_states.shape[0]!=1:
        raise ValueError('Evaluation adapter expects one unpadded prompt at a time')
    if past_key_value is not None or output_attentions:
        raise ValueError('Fused prefill adapter expects an empty external cache and no attention output')
    # The upstream evaluator calls model.model with contiguous absolute positions,
    # use_cache=False, and no padding. Transformers may construct a causal mask;
    # this kernel applies that causal mask internally.
    b,n,_=hidden_states.shape
    q=self.q_proj(hidden_states).view(b,n,self.num_heads,self.head_dim).transpose(1,2)
    k=self.k_proj(hidden_states).view(b,n,self.num_key_value_heads,self.head_dim).transpose(1,2)
    v=self.v_proj(hidden_states).view(b,n,self.num_key_value_heads,self.head_dim).transpose(1,2)
    cos,sin=self.rotary_emb(v,seq_len=ctl.position_end)
    q,k=apply_rotary_pos_emb(q,k,cos,sin,position_ids)
    if ctl.inference_transform is not None:
        k,v=ctl.inference_transform(self.layer_idx,k,v)
    out,importance=prefill(q,k,v)
    ctl.states[self.layer_idx]={'k':k,'v':v,'importance':importance}
    out=out.transpose(1,2).contiguous().reshape(b,n,self.hidden_size)
    return self.o_proj(out),None,past_key_value
