# Copyright (c) ModelScope Contributors. All rights reserved.
"""Adapt the official Ling3 tiny remote code without changing its parameter layout.

Both certified Transformers versions share attention and KDA adapters. Construction
metadata and RoPE lifecycle differences are isolated in ``ling3_rope``.
"""
import torch
import torch.nn.functional as F
from einops import rearrange
from functools import wraps
from importlib import import_module
from inspect import signature

_PACKING_KEYS = ('cu_seq_lens_q', 'cu_seq_lens_k', 'max_length_q', 'max_length_k', 'cu_seqlens')


def is_ling3_tiny(config):
    return (getattr(config, 'hidden_size', None), getattr(config, 'num_hidden_layers',
                                                          None), getattr(config, 'q_lora_rank',
                                                                         None)) == (1536, 24, 256)


def _packed_params(hidden_states, position_ids, attention_mask, kwargs, *, require_metadata=False):
    """Validate one boundary contract at model entry and at checkpointed leaves.

    The remote decoder drops metadata, so leaf calls must reconstruct it from
    positions. Only the model entry requires explicit metadata to disambiguate
    packing from ordinary padding.
    """
    supplied = {k: kwargs[k] for k in _PACKING_KEYS if kwargs.get(k) is not None}
    if position_ids is None:
        if supplied:
            raise ValueError('Ling3 packing requires position_ids')
        return {}
    if position_ids.ndim != 2 or position_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError('Ling3 requires 2-D integer position_ids')
    resets = position_ids.shape[0] == 1 and bool((position_ids[:, 1:] == 0).any())
    if not supplied and not resets:
        return {}
    if (hidden_states is None or position_ids.shape != hidden_states.shape[:2] or hidden_states.shape[0] != 1
            or position_ids.device != hidden_states.device):
        raise ValueError('Ling3 packing requires one flattened batch aligned with position_ids')
    if attention_mask is not None and (attention_mask.ndim != 2 or not bool((attention_mask == 1).all())):
        if supplied:
            raise ValueError('Ling3 packing does not support real padding or 4-D masks; use a 2-D all-valid mask')
        if attention_mask.ndim == 4 and bool(((position_ids[:, :-1] > 0) & (position_ids[:, 1:] == 0)).any()):
            keep = attention_mask[..., -1, :] if attention_mask.dtype == torch.bool else attention_mask[...,
                                                                                                        -1, :] > -1e4
            if bool(keep.all()):
                raise ValueError('Ling3 packed position_ids require explicit sequence metadata.')
        # A partial last-query keep mask can also represent ordinary padding.
        # Without metadata, document intent is ambiguous; do not infer 4-D packing.
        return {}
    if attention_mask is not None and (attention_mask.shape != position_ids.shape
                                       or attention_mask.device != hidden_states.device):
        raise ValueError('Ling3 packing requires an aligned 2-D keep mask')
    if require_metadata:
        if not supplied:
            raise ValueError('Ling3 packed position_ids require explicit sequence metadata.')
        if any(kwargs.get(key) is None for key in _PACKING_KEYS[:4]):
            raise ValueError('Ling3 packing requires cu_seq_lens_q/k and max_length_q/k together.')
    consecutive = (position_ids[:, 1:] == 0) | (position_ids.diff(dim=-1) == 1)
    if position_ids.shape[1] == 0 or int(position_ids[0, 0]) != 0 or bool(
        (position_ids < 0).any()) or not bool(consecutive.all()):
        raise ValueError('Ling3 packed positions must reset to zero and increase consecutively')
    from swift.utils import get_packed_seq_params

    params = get_packed_seq_params(position_ids)
    for name, value in supplied.items():
        expected = params['cu_seq_lens_q'] if name == 'cu_seqlens' else params[name]
        if isinstance(expected, torch.Tensor) and expected.ndim == 1:
            if (not isinstance(value, torch.Tensor) or value.device != expected.device or value.dtype != expected.dtype
                    or not value.is_contiguous() or not torch.equal(value, expected)):
                raise ValueError(f'Conflicting Ling3 {name}')
        else:
            if isinstance(value, torch.Tensor):
                valid = value.ndim == 0 and value.dtype in (torch.int32, torch.int64)
            else:
                valid = isinstance(value, int)
            if not valid or int(value) != int(expected):
                raise ValueError(f'Conflicting Ling3 {name}: expected an integer scalar matching the largest document')
    return params if params['cu_seq_lens_q'].numel() > 2 else {}


def patch_ling3_packing(model):
    """Validate before the remote model drops keep masks and sequence metadata."""
    if hasattr(model, '_swift_ling3_packing_handle'):
        return
    forward_signature = signature(model.forward)

    def prepare_packing(module, args, kwargs):
        call = kwargs
        if args:
            call = dict(forward_signature.bind_partial(*args, **kwargs).arguments)
            call.update(call.pop('kwargs', {}))
        inputs = call.get('input_ids')
        if inputs is None:
            inputs = call.get('inputs_embeds')
        mask = call.get('attention_mask')
        # Only the outer training input is restricted; MLA builds its own causal mask.
        if module.training and mask is not None and mask.ndim == 4:
            raise ValueError('Ling3 training does not support external 4-D attention masks; '
                             'use a 2-D padding mask or explicit padding-free metadata')
        _packed_params(inputs, call.get('position_ids'), call.get('attention_mask'), call, require_metadata=True)
        if any(call.get(k) is not None for k in _PACKING_KEYS):
            if getattr(module.config, 'num_nextn_predict_layers', 0):
                raise ValueError('Ling3 packing with MTP is not supported')
            if not getattr(getattr(module, 'model', module), '_supports_packing_metadata', False):
                raise ValueError('Ling3 packing is unavailable: the model has no packing adapters')
            if module.config._attn_implementation != 'flash_attention_2':
                raise ValueError('Ling3 packing requires the native flash_attention_2 varlen backend')
            use_cache = call.get('use_cache')
            if use_cache is None:
                use_cache = module.config.use_cache
            if call.get('past_key_values') is not None or use_cache:
                raise ValueError('Ling3 packing requires use_cache=False and no past_key_values')
        return args, kwargs

    model._swift_ling3_packing_handle = model.register_forward_pre_hook(prepare_packing, with_kwargs=True)


def _singleton_forward(original, causal_conv1d):
    """Keep singleton training on the differentiable causal-convolution path."""

    @wraps(getattr(original, '__func__', original))
    def forward(self,
                x,
                residual=None,
                mask=None,
                cache=None,
                output_final_state=False,
                cu_seqlens=None,
                chunk_indices=None,
                **kwargs):
        count = x.shape[0] if cu_seqlens is None else cu_seqlens.numel() - 1
        if not self.training or x.shape[0] * x.shape[1] != count:
            return original(
                x,
                residual=residual,
                mask=mask,
                cache=cache,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                **kwargs)
        if cache is not None or mask is not None:
            raise ValueError('Ling3 singleton training requires no historical cache or padding')
        return causal_conv1d(
            x=x,
            weight=self.weight.squeeze(1),
            bias=self.bias,
            residual=residual,
            initial_state=None,
            output_final_state=output_final_state,
            activation=self.activation,
            backend=self.backend,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            **kwargs)

    return forward


def patch_short_convolution(conv, causal_conv1d):
    """Install the singleton adapter once, without replacing parameter objects."""
    from ..patcher import patch_module_forward
    if getattr(conv, '_swift_ling3_singleton', False):
        return
    if getattr(conv, '_patched', False):
        raise ValueError('Ling3 ShortConvolution already has a conflicting patch')
    original = getattr(conv, '_old_forward', conv.forward)
    patch_module_forward(conv, _singleton_forward(original, causal_conv1d))
    conv._swift_ling3_singleton = True


def _validate_eager_causal_mask(hidden_states, mask):
    """Require an additive causal mask; a keep mask alone cannot prevent leakage."""
    batch, length = hidden_states.shape[:2]
    if mask is None:
        if length > 1:
            raise ValueError('Ling3 eager MLA requires an explicit causal mask')
        return
    if (mask.ndim != 4 or mask.shape[0] not in (1, batch) or mask.shape[-2] != length or mask.shape[-1] < length
            or not mask.is_floating_point() or mask.device != hidden_states.device):
        raise ValueError('Ling3 eager MLA requires an aligned floating-point 4-D additive causal mask')
    keys = mask.shape[-1]
    future = torch.arange(
        keys, device=mask.device)[None, :] > (
            torch.arange(length, device=mask.device)[:, None] + keys - length)
    invalid = (mask[..., future] > torch.finfo(mask.dtype).min).any() | torch.isnan(mask).any()
    if bool(invalid):
        raise ValueError('Ling3 eager MLA mask must block every future token')


def _mla_forward(self,
                 hidden_states,
                 position_embeddings,
                 attention_mask,
                 past_key_values=None,
                 cache_position=None,
                 position_ids=None,
                 *,
                 original,
                 rotary,
                 registry,
                 **kwargs):
    if 'past_key_value' in kwargs:
        if past_key_values is not None:
            raise ValueError('Specify only one past_key_value(s) argument')
        past_key_values = kwargs.pop('past_key_value')
    params = _packed_params(hidden_states, position_ids, attention_mask, kwargs)
    if params and (past_key_values is not None or kwargs.get('use_cache')):
        raise ValueError('Ling3 packed MLA does not support historical cache')
    backend = self.config._attn_implementation
    if params and backend != 'flash_attention_2':
        raise ValueError('Ling3 packed MLA requires flash_attention_2 varlen')
    if backend == 'eager':
        _validate_eager_causal_mask(hidden_states, attention_mask)
        return original(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_ids=position_ids,
            **kwargs)
    # Native registry backends use this module's is_causal flag. Eager instead
    # requires the explicit-mask guard above.
    interface = registry[backend]
    batch, length = hidden_states.shape[:2]
    q_shape = (batch, length, -1, self.qk_head_dim)
    kv_shape = (batch, length, -1, self.qk_nope_head_dim + self.v_head_dim)
    q = self.q_proj(hidden_states) if self.q_lora_rank is None else self.q_b_proj(
        self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q_pass, q_rot = torch.split(q.view(q_shape).transpose(1, 2), [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
    compressed = self.kv_a_proj_with_mqa(hidden_states)
    k_pass, k_rot = torch.split(compressed, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    kv = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(kv_shape).transpose(1, 2)
    k_pass, value = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    k_rot = k_rot.view(batch, 1, length, self.qk_rope_head_dim)
    if not self.config.rope_interleave:
        raise NotImplementedError('Ling3 non-interleaved rotary weights are not supported')
    cos, sin = position_embeddings
    q_rot, k_rot = rotary(q_rot, k_rot, cos, sin)
    key = torch.cat((k_pass, k_rot.expand(*k_pass.shape[:-1], -1)), dim=-1)
    query = torch.cat((q_pass, q_rot), dim=-1)
    if past_key_values is not None:
        key, value = past_key_values.update(key, value, self.layer_idx, {
            'sin': sin,
            'cos': cos,
            'cache_position': cache_position
        })
    padded_value = backend == 'flash_attention_2' and self.qk_head_dim != self.v_head_dim
    if padded_value:
        value = F.pad(value, [0, self.qk_head_dim - self.v_head_dim])
    kwargs.update(params)
    output, weights = interface(
        self,
        query,
        key,
        value,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        position_ids=position_ids,
        **kwargs)
    if padded_value:
        output = output[:, :, :, :self.v_head_dim]
    if self.g_proj is not None:
        gate = F.sigmoid(self.g_proj(hidden_states).float()).type_as(hidden_states)
        if self.gated_attention_proj_granularity_type == 'head_wise':
            output = output * gate[:, :, :, None]
        else:
            output = output * gate.view(batch, length, self.num_heads, self.v_head_dim)
    return self.dense(output.reshape(batch, length, -1).contiguous()), weights, past_key_values


def _padded_kda_forward(self, hidden_states, attention_mask, past_key_value, position_ids, *, chunk_kda, **kwargs):
    # Ordinary padding only. Packing metadata remains on the original path.
    """Gather valid tokens into independent sequences and scatter outputs back."""
    use_cache = kwargs.get('use_cache')
    if use_cache is None:
        use_cache = getattr(getattr(self, 'config', None), 'use_cache', False)
    if past_key_value is not None or use_cache:
        raise ValueError('Ling3 padded KDA requires use_cache=False and no historical cache')
    if any(kwargs.get(k) is not None for k in _PACKING_KEYS) or kwargs.get('cp_context') is not None:
        raise ValueError('Ling3 ordinary padding cannot be combined with packing/context parallel metadata')
    if self.mode != 'chunk':
        raise ValueError('Ling3 padded KDA training requires chunk mode')
    if (attention_mask.ndim != 2 or attention_mask.shape != hidden_states.shape[:2]
            or attention_mask.device != hidden_states.device or not bool(
                ((attention_mask == 0) | (attention_mask == 1)).all())):
        raise ValueError('Ling3 padding requires an aligned binary 2-D mask')
    mask = attention_mask.bool()
    lengths = mask.sum(-1)
    if not bool((lengths > 0).all()):
        raise ValueError('Ling3 padding cannot contain empty sequences')
    transitions = mask.to(torch.int32).diff(dim=-1)
    if bool(((transitions == 1).any(-1) & (transitions == -1).any(-1)).any()):
        raise ValueError('Ling3 padding supports left/right padding, not internal gaps')
    if position_ids is not None:
        if (position_ids.ndim != 2 or position_ids.shape[1] != mask.shape[1]
                or position_ids.shape[0] not in (1, mask.shape[0]) or position_ids.device != hidden_states.device
                or position_ids.dtype not in (torch.int32, torch.int64)):
            raise ValueError('Ling3 padding position_ids require [B,T] or [1,T] aligned integer tensors')
        if position_ids.shape[0] == 1 and mask.shape[0] > 1:
            # The model's default [1,T] positions can be shared only when
            # every valid sequence starts at physical position zero.
            if (not bool(mask[:, 0].all())
                    or not torch.equal(position_ids[0], torch.arange(mask.shape[1], device=mask.device))):
                raise ValueError('Ling3 padding broadcast position_ids require right padding and 0-based positions')
        else:
            logical = mask.long().cumsum(-1) - 1
            if not torch.equal(position_ids[mask], logical[mask]):
                raise ValueError('Ling3 padding position_ids disagree with valid-token logical positions')
    batch, length, width = hidden_states.shape
    # cu_seqlens resets convolution and recurrent state between rows.
    # Internal gaps and cache would make this compacted sequence ambiguous.
    indices = mask.reshape(-1).nonzero().flatten()
    hidden_states = hidden_states.reshape(-1, width).index_select(0, indices).unsqueeze(0)
    cu = torch.cat((lengths.new_zeros(1), lengths.cumsum(0))).contiguous()
    output, _, _ = _kda_chunk_forward(self, hidden_states, cu, chunk_kda=chunk_kda)
    output = output.new_zeros(batch * length, width).index_copy(0, indices, output.squeeze(0))
    return output.reshape(batch, length, width), None, past_key_value


def _kda_chunk_forward(self, hidden_states, cu, *, chunk_kda):
    """Share pinned chunk math between short and padded cache-free inputs."""
    q, _ = self.q_conv1d(x=self.q_proj(hidden_states), cache=None, output_final_state=False, cu_seqlens=cu)
    k, _ = self.k_conv1d(x=self.k_proj(hidden_states), cache=None, output_final_state=False, cu_seqlens=cu)
    v, _ = self.v_conv1d(x=self.v_proj(hidden_states), cache=None, output_final_state=False, cu_seqlens=cu)
    g = self.f_proj(hidden_states) if self.no_kda_lora else self.f_b_proj(self.f_a_proj(hidden_states))
    beta = self.b_proj(hidden_states).float().sigmoid()
    q, k = (rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim) for x in (q, k))
    v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)
    g = rearrange(g, '... (h d) -> ... h d', d=self.head_dim)
    output, _ = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=self.A_log,
        dt_bias=self.dt_bias,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        safe_gate=self.safe_gate,
        lower_bound=self.lower_bound,
        cu_seqlens=cu)
    gate = self.g_proj(hidden_states) if self.no_kda_lora else self.g_b_proj(self.g_a_proj(hidden_states))
    gate = rearrange(gate, '... (h d) -> ... h d', d=self.head_dim)
    return self.o_proj(rearrange(self.o_norm(output, gate), 'b t h d -> b t (h d)')), None, None


def _kda_forward(self,
                 hidden_states,
                 attention_mask=None,
                 past_key_value=None,
                 position_ids=None,
                 *,
                 original,
                 chunk_kda,
                 **kwargs):
    if attention_mask is not None and not bool((attention_mask == 1).all()):
        return _padded_kda_forward(
            self, hidden_states, attention_mask, past_key_value, position_ids, chunk_kda=chunk_kda, **kwargs)
    params = _packed_params(hidden_states, position_ids, attention_mask, kwargs)
    if params and (past_key_value is not None or kwargs.get('use_cache')):
        raise ValueError('Ling3 packed KDA does not support historical cache')
    if params:
        kwargs['cu_seqlens'] = params['cu_seq_lens_q']
    # Remote KDA selects fused recurrent at length <=64, even in training.
    # Use chunk math for autograd; preserve the other routes.
    if not self.training or hidden_states.shape[1] > 64 or past_key_value is not None:
        return original(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            position_ids=position_ids,
            **kwargs)
    if self.mode != 'chunk':
        raise ValueError('Ling3 KDA training requires chunk mode')
    output, _, _ = _kda_chunk_forward(self, hidden_states, kwargs.get('cu_seqlens'), chunk_kda=chunk_kda)
    return output, None, past_key_value


def _make_mla_forward(original, rotary, registry):

    @wraps(getattr(original, '__func__', original))
    def forward(self, *args, **kwargs):
        return _mla_forward(self, *args, original=original, rotary=rotary, registry=registry, **kwargs)

    return forward


def _make_kda_forward(original, chunk_kda):

    @wraps(getattr(original, '__func__', original))
    def forward(self, *args, **kwargs):
        return _kda_forward(self, *args, original=original, chunk_kda=chunk_kda, **kwargs)

    return forward


def _get_ling3_compat_kernel(modeling):
    """Check callable contracts before mutation; return a reason when unavailable."""
    from transformers import __version__

    from .ling3_rope import SUPPORTED_TRANSFORMERS

    if __version__ not in SUPPORTED_TRANSFORMERS:
        return None, 'Ling3 attention adapters are validated with Transformers 4.57.1 and 5.14.1 only'
    try:
        from fla.modules.conv.causal_conv1d import causal_conv1d
        signature(causal_conv1d).bind(
            x=None,
            weight=None,
            bias=None,
            residual=None,
            initial_state=None,
            output_final_state=False,
            activation=None,
            backend='triton',
            cu_seqlens=None,
            chunk_indices=None)
        signature(modeling.chunk_kda).bind(
            q=None,
            k=None,
            v=None,
            g=None,
            beta=None,
            A_log=None,
            dt_bias=None,
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=-5.0,
            cu_seqlens=None)
    except (ImportError, AttributeError, TypeError, ValueError) as error:
        return None, f'Ling3 attention adapters require the compatible FLA causal-convolution/chunk-KDA interfaces: {error}'
    return causal_conv1d, None


def install_ling3_compat(model):
    """Preflight all layers, then install instance forwards with rollback on failure."""
    from ..patcher import patch_module_forward
    if getattr(model, '_swift_ling3_compat', None):
        return
    if not is_ling3_tiny(model.config):
        raise NotImplementedError('Ling3 attention compatibility is limited to the tiny configuration')
    if getattr(model.config, 'num_nextn_predict_layers', 0):
        raise NotImplementedError('Ling3 packing with MTP is not supported')
    modeling = import_module(type(model).__module__)
    if not callable(getattr(modeling, 'apply_rotary_pos_emb_interleave', None)) or not callable(
            getattr(modeling, 'chunk_kda', None)):
        raise RuntimeError('Ling3 model does not expose the required rotary/KDA operators')
    causal_conv1d, error = _get_ling3_compat_kernel(modeling)
    if error is not None:
        raise RuntimeError(error)
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        for layer in model.model.layers:
            if layer.attention_layer_type == 'attention':
                backend = layer.attention.config._attn_implementation
                if backend != 'eager' and not callable(ALL_ATTENTION_FUNCTIONS[backend]):
                    raise RuntimeError(f'Ling3 attention backend {backend} is not callable')
    except (ImportError, AttributeError, KeyError, TypeError) as error:
        raise RuntimeError(f'Ling3 attention registry is unavailable: {error}') from error
    pending = []
    for layer in model.model.layers:
        attention = layer.attention
        if getattr(attention, '_patched', False):
            raise ValueError('Ling3 attention already has a conflicting forward patch')
        original = getattr(attention, '_old_forward', attention.forward)
        if layer.attention_layer_type == 'attention':
            if getattr(attention, 'num_key_value_groups', 1) != 1:
                raise NotImplementedError('Ling3 MLA compatibility requires equal query/key-value head counts')
            required = ('q_lora_rank', 'kv_a_proj_with_mqa', 'kv_a_layernorm', 'kv_b_proj', 'dense', 'is_causal')
            forward = _make_mla_forward(original, modeling.apply_rotary_pos_emb_interleave, ALL_ATTENTION_FUNCTIONS)
        elif layer.attention_layer_type == 'linear_attention':
            required = ('q_conv1d', 'k_conv1d', 'v_conv1d', 'A_log', 'dt_bias', 'o_norm', 'o_proj')
            forward = _make_kda_forward(original, modeling.chunk_kda)
        else:
            raise NotImplementedError(f'Unknown Ling3 attention type: {layer.attention_layer_type}')
        if not all(hasattr(attention, name) for name in required):
            raise ValueError('Ling3 attention is missing required projection/kernel capabilities')
        if layer.attention_layer_type == 'linear_attention':
            for name in ('q_conv1d', 'k_conv1d', 'v_conv1d'):
                conv = getattr(attention, name)
                if getattr(conv, '_patched', False) and not getattr(conv, '_swift_ling3_singleton', False):
                    raise ValueError('Ling3 ShortConvolution already has a conflicting patch')
        pending.append((attention, forward))
    modules = [model, model.model]
    for layer in model.model.layers:
        modules.append(layer.attention)
        if layer.attention_layer_type == 'linear_attention':
            modules.extend(getattr(layer.attention, name) for name in ('q_conv1d', 'k_conv1d', 'v_conv1d'))
    # Swift's patcher changes instance attributes only. Restore those attributes
    # if installation fails, so a model cannot remain partially adapted.
    missing = object()
    attributes = ('forward', '_old_forward', '_patched', '_swift_ling3_singleton', '_supports_packing_metadata',
                  '_swift_ling3_compat')
    snapshots = [(module, {name: module.__dict__.get(name, missing) for name in attributes}) for module in modules]
    try:
        for layer in model.model.layers:
            if layer.attention_layer_type == 'linear_attention':
                for name in ('q_conv1d', 'k_conv1d', 'v_conv1d'):
                    patch_short_convolution(getattr(layer.attention, name), causal_conv1d)
        for attention, forward in pending:
            patch_module_forward(attention, forward)
        model.model._supports_packing_metadata = True
        model._swift_ling3_compat = {
            'policy': 'attention-local-position-boundaries; native MLA dispatch; short training chunk'
        }
    except Exception:
        for module, state in snapshots:
            for name, value in state.items():
                if value is missing:
                    module.__dict__.pop(name, None)
                else:
                    setattr(module, name, value)
        raise
