"""Paired policies share prefill outputs, then decode independent compact caches.

Only policies with identical attention-output implementations share a prefill.
All-query and windowed fused scoring run separate prefill passes. Batching is
across policies for the same prompt, never across token positions or KV heads.
"""
import types

import torch
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from kv_distill.attention import Controller, forward as decode_forward
from kv_distill.cache import gather_tokens

from .baselines import flash_attention, snap_scores, snap_select
from .prefill import prefill, select_tokens


def default_methods():
    methods = [dict(name='full', backend='full', budget=None)]
    for budget in (256, 1024):
        methods.append(dict(name=f'snap_{budget}', backend='snap', budget=budget))
        for label, window, recent in [('d01_r0', 0, 0), ('d0.1_w128_r0', 128, 0),
                                      ('d0.1_w128_r64', 128, 64)]:
            methods.append(dict(name=f'{label}_{budget}', backend='fused', budget=budget,
                                recency_decay=.1, observation_window=window,
                                value_weighted=True, reserved_recent=recent))
    methods += [dict(name=f'{backend}_256', backend=backend, budget=256)
                for backend in ('value_norm_only', 'uniform_recent')]
    return methods


def group_key(method):
    if method['backend'] != 'fused':
        return ('baseline',)
    return ('fused', method['recency_decay'], method['observation_window'], method['value_weighted'])


def choose_indices(method, k, v, importance=None, snap=None):
    n = k.shape[2]
    budget = min(n, method['budget'] or n)
    if method['backend'] == 'full' or budget == n:
        return None
    if method['backend'] == 'uniform_recent':
        # Pure suffix control as requested; no attention sinks are reserved.
        return torch.arange(n-budget, n, device=k.device).expand(*k.shape[:2], budget)
    if method['backend'] == 'value_norm_only':
        return select_tokens(v.float().abs().sum(-1), budget)[0]
    if method['backend'] == 'snap':
        return snap_select(snap, budget, window=32, kernel=5)[0]
    return select_tokens(importance, budget, reserved_recent=min(budget, method['reserved_recent']))[0]


class PairedController(Controller):
    def __init__(self, model):
        super().__init__(model)
        self.bank = {}
        self.active_methods = []
        for attn, _ in self.originals:
            attn.forward = types.MethodType(forward, attn)

    def prepare(self, methods, length):
        if len({group_key(m) for m in methods}) != 1:
            raise ValueError('A shared prefill requires identical attention output and score settings')
        self.reset()
        self.active_methods = methods
        self.position_end = length
        for method in methods:
            self.bank[method['name']] = {}

    def activate(self, names, *, consume=True):
        """Concatenate caches along batch, requiring the same retained length."""
        sources = [self.bank[name] for name in names]
        self.states = {layer: {key: torch.cat([s[layer][key] for s in sources], 0)
                               if len(sources) > 1 else sources[0][layer][key]
                               for key in ('k', 'v')} for layer in sources[0]}
        self.mode = 'decode'
        self.method = 'full'
        if consume:
            for name in names:
                del self.bank[name]


def forward(self, hidden_states, attention_mask=None, position_ids=None,
            past_key_value=None, output_attentions=False, use_cache=False, cache_position=None):
    ctl = self._kv_controller
    if ctl.mode != 'prefill':
        return decode_forward(self, hidden_states, attention_mask, position_ids,
                              past_key_value, output_attentions, use_cache, cache_position)
    b, n, _ = hidden_states.shape
    if b != 1 or past_key_value is not None or output_attentions:
        raise ValueError('Paired prefill requires one unpadded prompt and no external cache')
    q = self.q_proj(hidden_states).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(hidden_states).view(b, n, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(hidden_states).view(b, n, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    cos, sin = self.rotary_emb(v, seq_len=ctl.position_end)
    q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
    method = ctl.active_methods[0]
    importance = snap = None
    if method['backend'] == 'fused':
        out, importance = prefill(q, k, v, **{key: method[key] for key in
                                            ('recency_decay', 'observation_window', 'value_weighted')})
    else:
        out = flash_attention(q, k, v)
        if any(m['backend'] == 'snap' and m['budget'] < n for m in ctl.active_methods):
            snap = snap_scores(q, k, window=32)
    for method in ctl.active_methods:
        idx = choose_indices(method, k, v, importance, snap)
        ctl.bank[method['name']][self.layer_idx] = {
            'k': k if idx is None else gather_tokens(k, idx),
            'v': v if idx is None else gather_tokens(v, idx)}
    out = out.transpose(1, 2).contiguous().reshape(b, n, self.hidden_size)
    return self.o_proj(out), None, past_key_value


@torch.inference_mode()
def decode_batch(model, ctl, last_token, prompt_length, max_new_tokens, stop_ids, *, suppress_first_stop=False):
    batch = next(iter(ctl.states.values()))['k'].shape[0]
    token = torch.full((batch, 1), last_token, dtype=torch.long, device=next(model.parameters()).device)
    generated = [[] for _ in range(batch)]
    finished = [False]*batch
    for step in range(max_new_tokens):
        position = prompt_length-1+step
        ctl.position_end = position+1
        pos = torch.full((batch, 1), position, dtype=torch.long, device=token.device)
        hidden = model.model(token, position_ids=pos, use_cache=False).last_hidden_state
        logits = model.lm_head(hidden[:, -1])
        if suppress_first_stop and step == 0:
            logits[:, list(stop_ids)] = -torch.inf
        token = logits.argmax(-1, keepdim=True)
        for i, token_id in enumerate(token[:, 0].tolist()):
            if not finished[i]:
                generated[i].append(token_id)
                finished[i] = token_id in stop_ids
        if all(finished):
            break
    return generated
