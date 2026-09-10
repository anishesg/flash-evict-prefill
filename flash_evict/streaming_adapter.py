"""Optional Qwen evaluation adapter that compacts each layer during prefill.

Same unpadded, single-prompt contract as kv_distill_adapter. Dense attention
outputs remain intact for the next layer; only stored K/V are compacted. This
avoids keeping all layers' full prompt caches until a final compression call.
"""
import types
import torch
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from kv_distill.cache import gather_tokens
from .baselines import flash_attention, snap_scores, snap_select
from .kv_distill_adapter import FusedController, forward as fused_forward
from .prefill import select_tokens


class StreamingController(FusedController):
    def __init__(self,model,*,backend='fused',budget=1024,scoring=None,reserved_recent=64,pool_kernel=1):
        super().__init__(model)
        if backend not in ('fused','snap','full'):raise ValueError('Unsupported backend')
        if not isinstance(budget,int) or budget<1:raise ValueError('Positive integer budget required')
        self.prefill_backend=backend
        self.cache_budget=budget
        self.scoring={} if scoring is None else dict(scoring)
        self.reserved_recent=reserved_recent;self.pool_kernel=pool_kernel
        for attn,_ in self.originals:attn.forward=types.MethodType(forward,attn)

    def finish_prefill(self):
        self.mode='decode';self.method='full'  # Standard grow-after-prefill decoding.


def chunk_mlp(model,chunk_size):
    """Bound feed-forward activation memory for inference; return a restore hook.

    This is independent of cache scoring and must be applied to all comparison
    methods. Different GEMM shapes may introduce floating-point differences.
    """
    if isinstance(chunk_size,bool) or not isinstance(chunk_size,int) or chunk_size<1:
        raise ValueError('chunk_size must be a positive integer')
    originals=[]
    def make_forward(original):
        def chunked(hidden):
            if torch.is_grad_enabled():raise RuntimeError('Chunked MLP is inference-only')
            if hidden.shape[1]<=chunk_size:return original(hidden)
            output=torch.empty_like(hidden)
            for start in range(0,hidden.shape[1],chunk_size):
                output[:,start:start+chunk_size]=original(hidden[:,start:start+chunk_size])
            return output
        return chunked
    for layer in model.model.layers:
        originals.append((layer.mlp,layer.mlp.forward))
        layer.mlp.forward=make_forward(layer.mlp.forward)
    def restore():
        for mlp,original in originals:mlp.forward=original
    return restore


def forward(self,hidden_states,attention_mask=None,position_ids=None,
            past_key_value=None,output_attentions=False,use_cache=False,cache_position=None):
    ctl=self._kv_controller
    if ctl.mode!='prefill' or ctl.prefill_backend=='fused':
        result=fused_forward(self,hidden_states,attention_mask,position_ids,past_key_value,
                             output_attentions,use_cache,cache_position)
        if ctl.mode!='prefill':return result
        state=ctl.states[self.layer_idx];n=state['k'].shape[2]
        budget=min(n,ctl.cache_budget)
        idx,_=select_tokens(state.pop('importance'),budget,
                            reserved_recent=min(budget,ctl.reserved_recent),pool_kernel=ctl.pool_kernel)
    else:
        if hidden_states.shape[0]!=1 or past_key_value is not None or output_attentions:
            raise ValueError('Streaming adapter expects one unpadded prompt and an empty external cache')
        b,n,_=hidden_states.shape
        q=self.q_proj(hidden_states).view(b,n,self.num_heads,self.head_dim).transpose(1,2)
        k=self.k_proj(hidden_states).view(b,n,self.num_key_value_heads,self.head_dim).transpose(1,2)
        v=self.v_proj(hidden_states).view(b,n,self.num_key_value_heads,self.head_dim).transpose(1,2)
        cos,sin=self.rotary_emb(v,seq_len=ctl.position_end)
        q,k=apply_rotary_pos_emb(q,k,cos,sin,position_ids)
        if ctl.inference_transform is not None:k,v=ctl.inference_transform(self.layer_idx,k,v)
        out=flash_attention(q,k,v)
        state=ctl.states[self.layer_idx]={'k':k,'v':v}
        if ctl.prefill_backend=='snap':
            idx,_=snap_select(snap_scores(q,k,32),min(n,ctl.cache_budget),32)
        else:idx=None
        out=out.transpose(1,2).contiguous().reshape(b,n,self.hidden_size)
        result=self.o_proj(out),None,past_key_value
    if idx is not None:
        state['k']=gather_tokens(state['k'],idx);state['v']=gather_tokens(state['v'],idx)
    return result
