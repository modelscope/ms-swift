from functools import partial
from typing import Any, Optional, Tuple
import math
import torch
import torch.distributed as dist
from torch.distributed import init_device_mesh
from transformers import PreTrainedTokenizer

from swift.llm import get_llm_model
from .utils import GatherLoss
from ...utils import get_dist_setting, get_device


# Code borrowed from deepspeed, here is why:
# 1. Reduce the dependency
# 2. The original code is complex
def _generate_layout_params(scatter_idx, seq_world_size, input):
    if scatter_idx < 2:
        bs, global_seq_len, num_local_head, head_dim = input.shape
        pre_all2all_inp_shape = [bs, seq_world_size, global_seq_len // seq_world_size, num_local_head, head_dim]
        pre_all2all_permute_idx = (1, 0, 2, 3, 4)

        post_all2all_permute_idx = (1, 2, 0, 3, 4)
        post_all2all_res_shape = [bs, global_seq_len // seq_world_size, seq_world_size * num_local_head, head_dim]
    else:
        bs, local_seq_len, num_total_head, head_dim = input.shape
        assert num_total_head % seq_world_size == 0, (f'Number of heads ({num_total_head}) must be divisible '
                                                      f'by the sequence parallel size ({seq_world_size})!')
        pre_all2all_inp_shape = [bs, local_seq_len, seq_world_size, num_total_head // seq_world_size, head_dim]
        pre_all2all_permute_idx = (2, 0, 1, 3, 4)

        post_all2all_permute_idx = (1, 0, 2, 3, 4)
        post_all2all_res_shape = [bs, seq_world_size * local_seq_len, num_total_head // seq_world_size, head_dim]

    return pre_all2all_permute_idx, pre_all2all_inp_shape, post_all2all_permute_idx, post_all2all_res_shape


def post_all2all(permute_idx, res_shape):
    """
    Post-processing function for `all2all` communication.
    """

    def post_func(input):
        if permute_idx is not None:
            input = input.permute(permute_idx).contiguous()
        output = input.reshape(res_shape).contiguous()

        return output

    return post_func


def pre_all2all_fun(permute_idx, inp_shape, input):
    """
    Pre-processing function for `all2all` communication.
    """
    input_t = input.reshape(inp_shape).contiguous()
    if permute_idx is not None:
        input_t = input_t.permute(permute_idx).contiguous()
    return input_t


def single_all_to_all(input, scatter_idx, gather_idx, group, **kwargs):
    seq_world_size = dist.get_world_size(group)
    num_heads = input.shape[2]
    if num_heads % seq_world_size != 0 and not scatter_idx < 2:
        raise NotImplementedError(f'num_heads {num_heads} cannot be split by sp world size {seq_world_size}')
    pre_all2all_permute_idx, pre_all2all_inp_shape, post_all2all_permute_idx, post_all2all_res_shape = (
        _generate_layout_params(scatter_idx, seq_world_size, input))

    input_t = pre_all2all_fun(pre_all2all_permute_idx, pre_all2all_inp_shape, input)

    post_all2all_fun = post_all2all(post_all2all_permute_idx, post_all2all_res_shape)
    output = torch.empty_like(input_t)
    dist.all_to_all_single(output, input_t, group=group)

    res = post_all2all_fun(output)
    return res


class _SeqAllToAll(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: torch.Tensor,
        scatter_idx: int,
        gather_idx: int,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        res = single_all_to_all(input, scatter_idx, gather_idx, group)
        return res

    @staticmethod
    def backward(ctx: Any, *grad_output: torch.Tensor) -> Tuple[None, torch.Tensor, None, None]:
        return None, _SeqAllToAll.apply(ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx), None, None


class DistributedAttention(torch.nn.Module):

    def __init__(
        self,
        local_attention,
        sequence_parallel,
        scatter_idx: int = 2,
        gather_idx: int = 1,
    ) -> None:
        super(DistributedAttention, self).__init__()
        self.local_attn = local_attention
        self.sequence_parallel = sequence_parallel
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor,
                *args: Any, **kwargs) -> torch.Tensor:
        query_layer = _SeqAllToAll.apply(self.sequence_parallel.sp_group, query, self.scatter_idx, self.gather_idx)
        key_layer = _SeqAllToAll.apply(self.sequence_parallel.sp_group, key, self.scatter_idx, self.gather_idx)
        value_layer = _SeqAllToAll.apply(self.sequence_parallel.sp_group, value, self.scatter_idx, self.gather_idx)
        position_ids = kwargs.pop('position_ids', None)
        if position_ids is None:
            position_ids = self.sequence_parallel.extra_kwargs['_position_ids']

        context_layer = self.local_attn(
            query_layer, key_layer, value_layer, attention_mask, *args, position_ids=position_ids, **kwargs)
        output = _SeqAllToAll.apply(self.sequence_parallel.sp_group, context_layer, self.gather_idx, self.scatter_idx)
        return output


class SequenceParallel:

    _global_inited: bool = False

    def __init__(self):
        self.sp_world_size = None
        self.dp_world_size = None
        self.rp_world_size = None
        self.world_size = None
        self.model_dtype = None
        self.tokenizer = None
        self.device_mesh = None
        self.num_heads = None
        self.causal_mask_func = None
        self.extra_kwargs = {}

    def _prepare_flash_attn(self, base_model: torch.nn.Module):
        try:
            from transformers import masking_utils

            def flash_attention_mask(batch_size,
                                     cache_position,
                                     kv_length,
                                     kv_offset=0,
                                     mask_function=masking_utils.causal_mask_function,
                                     attention_mask=None,
                                     **kwargs):
                if attention_mask is not None:
                    if attention_mask.all():
                        attention_mask = None

                return attention_mask

            masking_utils.flash_attention_mask = flash_attention_mask
            masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._global_mapping['flash_attention_2'] = flash_attention_mask

            def create_causal_mask(config, input_embeds, attention_mask, cache_position, *args, **kwargs):
                input_embeds = torch.ones(
                    (input_embeds.shape[0], input_embeds.shape[1] * self.sp_world_size, input_embeds.shape[2]),
                    dtype=input_embeds.dtype,
                    device=input_embeds.device)
                cache_position = torch.arange(0, input_embeds.shape[1], device=input_embeds.device)
                return masking_utils.origin_create_causal_mask(config, input_embeds, attention_mask, cache_position,
                                                               *args, **kwargs)

            masking_utils.origin_create_causal_mask = masking_utils.create_causal_mask
            masking_utils.create_causal_mask = create_causal_mask
        except ImportError:
            pass

        if hasattr(base_model, 'language_model'):
            text_model = base_model.language_model
        else:
            text_model = base_model

        from transformers.modeling_flash_attention_utils import is_flash_attn_available
        if is_flash_attn_available():
            # TODO this works for multi-modal models like qwen2.5-vl
            # SDPA is not supported, because we need to copy the code to our project, which will bring
            # more works for maintaining.
            from transformers import modeling_flash_attention_utils
            from transformers.modeling_flash_attention_utils import _flash_attention_forward
            _distributed_flash_attention = DistributedAttention(_flash_attention_forward, self)

            def flash_attention_forward(query_states: torch.Tensor, key_states: torch.Tensor,
                                        value_states: torch.Tensor, attention_mask: Optional[torch.Tensor], q_len,
                                        *args, **kwargs):
                return _distributed_flash_attention(query_states, key_states, value_states, attention_mask,
                                                    q_len * self.sp_world_size, *args, **kwargs)

            modeling_flash_attention_utils._flash_attention_forward = flash_attention_forward

        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        def local_flash_attn(module: torch.nn.Module, query_states, key_states, value_states, attention_mask, *args,
                             dist_attn, **kwargs):
            if module.__class__ not in [m.__class__ for m in text_model.modules()]:
                return ALL_ATTENTION_FUNCTIONS['flash_attention_2_origin'](module, query_states, key_states,
                                                                           value_states, attention_mask, *args,
                                                                           **kwargs)
            if dist_attn.local_attn is None:

                def _attention(query, key, value, *args, **kwargs):
                    query = query.transpose(1, 2)
                    key = key.transpose(1, 2)
                    value = value.transpose(1, 2)
                    if self.rp_world_size is not None and self.rp_world_size > 1:
                        from .zigzag_ring_flash_attn_varlen import zigzag_ring_flash_attn_varlen_func
                        position_ids = kwargs['position_ids']
                        cu_seqlens = self.get_cu_seqlens_from_position_ids(position_ids).to(torch.int32)
                        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
                        position_ids = self._split_packed(position_ids, cu_seqlens)
                        # position_ids = self._split(position_ids, dim=1, position_ids=position_ids)
                        mask = position_ids != -1
                        query = query.transpose(1, 2)
                        key = key.transpose(1, 2)
                        value = value.transpose(1, 2)
                        query, key, value = self._pad_qkv(query, key, value, mask)
                        output = zigzag_ring_flash_attn_varlen_func(query,
                                                                    key,
                                                                    value,
                                                                    cu_seqlens=cu_seqlens,
                                                                    max_seqlen=max_seqlen,
                                                                     causal=module.is_causal,
                                                                     dropout_p=kwargs.get('dropout', 0.0),
                                                                     softmax_scale=kwargs.get('scaling', 0.0),
                                                                     window_size=kwargs.get('sliding_window') or (-1, -1),
                                                                     group=self.rp_group)
                        return output
                    else:
                        return ALL_ATTENTION_FUNCTIONS['flash_attention_2_origin'](module, query, key,
                                                                                   value, *args, **kwargs)[0]

                dist_attn.local_attn = _attention

            return dist_attn(
                query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2), attention_mask,
                *args, **kwargs), None

        def local_sdpa_attn(module: torch.nn.Module, query_states, key_states, value_states, attention_mask, *args,
                            dist_attn, **kwargs):
            if module.__class__ not in [m.__class__ for m in text_model.modules()]:
                return ALL_ATTENTION_FUNCTIONS['sdpa_origin'](module, query_states, key_states, value_states,
                                                              attention_mask, *args, **kwargs)
            if dist_attn.local_attn is None:

                def _attention(query, key, value, *args, **kwargs):
                    query = query.transpose(1, 2)
                    key = key.transpose(1, 2)
                    value = value.transpose(1, 2)
                    if self.rp_world_size > 1:
                        raise NotImplementedError(f'SDPA does not support Ring attention!')
                    return ALL_ATTENTION_FUNCTIONS['sdpa_origin'](module, query, key, value, *args, **kwargs)[0]

                dist_attn.local_attn = _attention
            return dist_attn(
                query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2), attention_mask,
                *args, **kwargs), None

        ALL_ATTENTION_FUNCTIONS['flash_attention_2_origin'] = ALL_ATTENTION_FUNCTIONS['flash_attention_2']
        ALL_ATTENTION_FUNCTIONS['sdpa_origin'] = ALL_ATTENTION_FUNCTIONS['sdpa']
        ALL_ATTENTION_FUNCTIONS['flash_attention_2'] = partial(
            local_flash_attn, dist_attn=DistributedAttention(None, self))
        ALL_ATTENTION_FUNCTIONS['sdpa'] = partial(
            local_sdpa_attn, dist_attn=DistributedAttention(None, self))

    def _prepare_forward_hook(self, base_model: torch.nn.Module):

        def pre_forward_split_hook(_self, args, kwargs):
            input_ids = kwargs.get('input_ids', None)
            inputs_embeds = kwargs.get('inputs_embeds', None)
            position_ids = kwargs['position_ids']
            attention_mask = kwargs.get('attention_mask', None)
            if hasattr(_self, 'language_model'):
                embed_tokens = getattr(_self.language_model, 'embed_tokens', None)
            else:
                embed_tokens = getattr(_self, 'embed_tokens', None)
            input_ids, inputs_embeds, _, position_ids, attention_mask, _ = self.pad_and_split_inputs(
                input_ids, inputs_embeds, None, position_ids, attention_mask, None, embed_tokens=embed_tokens, extra_position_ids=self.extra_kwargs.get('_position_ids'))
            kwargs['input_ids'] = input_ids
            kwargs['inputs_embeds'] = inputs_embeds
            kwargs['position_ids'] = position_ids
            kwargs['attention_mask'] = attention_mask
            return args, kwargs

        base_model.register_forward_pre_hook(pre_forward_split_hook, with_kwargs=True)

    def _prepare_moe_aux_loss(self, base_model: torch.nn.Module):
        def moe_aux_loss_hook(module, args, kwargs, output):
            router_logits = getattr(output, 'router_logits', None)
            if router_logits is None:
                return output

            attention_mask = kwargs['attention_mask']
            num_layers = len(router_logits)
            sp_len = router_logits[0].shape[0]
            if isinstance(router_logits, tuple):
                compute_device = router_logits[0].device
                router_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in router_logits], dim=0)
            router_logits, _ = GatherLoss.apply(router_logits, None, self.sp_group)
            router_logits = router_logits.reshape(self.sp_world_size, num_layers, sp_len,
                                                  -1).transpose(0, 1).reshape(num_layers, self.sp_world_size * sp_len,
                                                                              -1)
            if attention_mask is not None:
                router_logits = router_logits[:, :attention_mask.shape[1], :]
            output['router_logits'] = tuple([logit.squeeze() for logit in router_logits.split(1, dim=0)])
            return output

        base_model.register_forward_hook(moe_aux_loss_hook, with_kwargs=True)

    def prepare(self, sp_size: int, model: torch.nn.Module, tokenizer: PreTrainedTokenizer):
        if self.device_mesh is not None:
            return
        self.sp_world_size = sp_size
        self.num_heads = model.config.num_key_value_heads

        llm_model = get_llm_model(model)

        if hasattr(llm_model, 'thinker'):
            base_model = llm_model.thinker.model
        else:
            base_model = llm_model.model

        if hasattr(base_model, 'language_model'):
            if hasattr(base_model.language_model, '_update_causal_mask'):
                self.causal_mask_func = base_model.language_model._update_causal_mask
        else:
            if hasattr(base_model, '_update_causal_mask'):
                self.causal_mask_func = base_model._update_causal_mask

        if not SequenceParallel._global_inited:
            self._init_device_mesh()
            self._prepare_flash_attn(base_model)
            SequenceParallel._global_inited = True

        self._prepare_forward_hook(base_model)
        if model.model_info.is_moe_model:
            self._prepare_moe_aux_loss(base_model)

        self.model_dtype = next(model.parameters()).dtype
        self.tokenizer = tokenizer

    def _pad_qkv(self, query, key, value, mask):
        mask = mask.unsqueeze(2).unsqueeze(3)
        query = query * mask
        value = value * mask
        mask = (~mask) * -1e5
        key = key + mask.to(key.dtype)
        return query, key, value

    def _pad(self, tensor, padding_value, position_ids=None, dim=1):
        """Pad tensor for sequence parallel"""
        if self.rp_world_size > 1:
            world_size = self.world_size * 2
        else:
            world_size = self.world_size

        def _do_pad(tensor):
            length = tensor.shape[dim]
            pad_num = world_size - (length % world_size)
            if pad_num == 0:
                return tensor
            if not isinstance(padding_value, torch.Tensor):
                # ids
                pad_shape = ((*tensor.shape[:dim], pad_num, *tensor.shape[dim + 1:]) if dim != -1 else
                             (*tensor.shape[:dim], pad_num))
                pad = torch.full(pad_shape, padding_value, dtype=tensor.dtype, device=tensor.device)
                tensor = torch.cat([tensor, pad], dim=dim)
            else:
                # For embeddings
                tensor = torch.cat([tensor, padding_value.unsqueeze(0).repeat(tensor.shape[0], pad_num, 1)], dim=dim)
            return tensor

        if position_ids is not None and self.rp_world_size > 1:
            cu_seqlens = self.get_cu_seqlens_from_position_ids(position_ids)
            all_tensors = []
            for i in range(len(cu_seqlens) - 1):
                if dim == 1:
                    sub_tensor = tensor[:, cu_seqlens[i]:cu_seqlens[i + 1]]
                elif dim == -1:
                    sub_tensor = tensor[..., cu_seqlens[i]:cu_seqlens[i + 1]]
                else:
                    raise NotImplementedError()
                all_tensors.append(_do_pad(sub_tensor))
            tensor = torch.cat(all_tensors, dim=dim)

        return _do_pad(tensor)

    def _unpad(self, tensor, position_ids=None):
        n_dim = tensor.dim()
        if position_ids is None:
            return tensor

        mask = position_ids >= 0
        tensor = tensor[mask]
        if tensor.dim() < n_dim:
            tensor = tensor.unsqueeze(0)
        return tensor.contiguous()

    def _gather(self, local_output, dim: int, position_ids=None):
        """Gather tensor for sequence parallel - reverse of _split"""
        if self.world_size == 1:
            return local_output

        if self.rp_world_size > 1:
            input_dim = local_output.dim()
            assert input_dim >= 2 and local_output.shape[0] == 1

            # Step 1: Gather from all sequence parallel ranks
            # Each sp_rank has its own piece, we need to gather them first
            gathered_sp = [torch.zeros_like(local_output) for _ in range(self.sp_world_size)]
            torch.distributed.all_gather(gathered_sp, local_output.contiguous(), group=self.sp_group)

            # Concatenate the sp pieces to form the complete chunk for this rp_rank
            rp_chunk = torch.cat(gathered_sp, dim=dim)

            # Step 2: Gather all rp chunks
            gathered_rp = [torch.zeros_like(rp_chunk) for _ in range(self.rp_world_size)]
            torch.distributed.all_gather(gathered_rp, rp_chunk, group=self.rp_group)

            cu_seqlens = self.get_cu_seqlens_from_position_ids(position_ids)
            all_tensor_length = []
            for i in range(len(cu_seqlens) - 1):
                length = cu_seqlens[i + 1] - cu_seqlens[i]
                padding_length = math.ceil(length / (self.world_size * 2)) * (self.world_size * 2)
                all_tensor_length.append(padding_length)

            full_output = torch.zeros([sum(all_tensor_length), *local_output.shape[2:]], device=local_output.device)
            for idx_rp, rp_tensor in enumerate(gathered_rp): # rp world size
                accumulated_length = 0
                for idx_seq, length in enumerate(all_tensor_length): # sequence number
                    local_length = length // self.rp_world_size
                    local_tensor = rp_tensor[:, accumulated_length: local_length+accumulated_length]
                    chunk_size = local_length // 2
                    left_idx = accumulated_length * self.rp_world_size + idx_rp * chunk_size
                    right_idx = accumulated_length * self.rp_world_size + (idx_rp + 1) * chunk_size
                    full_output[left_idx:right_idx] = local_tensor[:, :chunk_size]
                    left_idx = accumulated_length * self.rp_world_size + (2 * self.rp_world_size - idx_rp - 1) * chunk_size
                    right_idx = accumulated_length * self.rp_world_size + (2 * self.rp_world_size - idx_rp) * chunk_size
                    full_output[left_idx:right_idx] = local_tensor[:, chunk_size:]
                    accumulated_length += local_length

            return full_output.unsqueeze(0).contiguous()
        else:
            gathered_sp = torch.empty((local_output.shape[0] * self.sp_world_size, local_output.shape[1]),
                                       dtype=local_output.dtype,
                                       device=local_output.device)
            dist.all_gather_into_tensor(gathered_sp, local_output, group=self.sp_group)
            gathered_sp = torch.cat(gathered_sp.split(local_output.shape[0], dim=0), dim=dim)
            return gathered_sp.contiguous()

    @staticmethod
    def get_cu_seqlens_from_position_ids(position_ids: torch.LongTensor):
        position_ids = position_ids[0]
        seq_start_indices = torch.where(position_ids == 0)[0]
        seq_end_indices = torch.cat([seq_start_indices[1:], torch.tensor([len(position_ids)], device=position_ids.device)])
        seq_lengths = seq_end_indices - seq_start_indices
        cu_seqlens = torch.cumsum(torch.cat([torch.tensor([0], device=position_ids.device), seq_lengths]), dim=0)
        return cu_seqlens

    def _split_packed(self, value, cu_seqlens, dim=1):
        local_values = []
        for i in range(len(cu_seqlens) - 1):
            start, end = cu_seqlens[i], cu_seqlens[i + 1]
            if dim == 1:
                sub_value = value[:, start:end]
            elif dim == -1:
                sub_value = value[..., start:end]
            else:
                raise NotImplementedError()
            local_value = sub_value.chunk(2 * self.rp_world_size, dim=dim)
            local_values.extend(
                [
                    local_value[self.rp_rank].detach().clone(),
                    local_value[2 * self.rp_world_size - 1 - self.rp_rank].detach().clone(),
                ]
            )
        return torch.cat(local_values, dim=dim).contiguous()

    def _split(self, input, dim: int, position_ids=None):
        """Split tensor for sequence parallel"""
        if self.world_size == 1:
            return input

        if self.rp_world_size > 1:
            input_dim = input.dim()
            assert input_dim >= 2
            cu_seqlens = self.get_cu_seqlens_from_position_ids(position_ids)
            assert torch.all(cu_seqlens % (2 * self.rp_world_size) == 0)
            value_chunks = self._split_packed(input, cu_seqlens, dim=dim)
            local_value = value_chunks.chunk(self.sp_world_size, dim=dim)[self.sp_rank].contiguous()
            return local_value
        else:
            rank = self.sp_rank
            dim_size = input.size(dim)
            assert dim_size % self.sp_world_size == 0, (f'The dimension to split ({dim_size}) is not a multiple of '
                                                        f'world size ({self.sp_world_size}), cannot split tensor evenly')

            tensor_list = torch.split(input, dim_size // self.sp_world_size, dim=dim)
            output = tensor_list[rank].contiguous()
            return output

    def pad_and_split_inputs(self,
                             input_ids,
                             input_embeds,
                             labels,
                             position_ids,
                             attention_mask,
                             loss_scale,
                             embed_tokens=None,
                             extra_position_ids=None):
        """Common implementation for padding and splitting inputs
        Args:
            input_ids: input_ids
            input_embeds: input_embeds
            labels: labels
            position_ids: position_ids or position_ids for mrope
            attention_mask: attention_mask
            loss_scale: loss_scale
            embed_tokens: embed_tokens
            extra_position_ids: the real position_ids to represent the seq length information
        """
        tokenizer = self.tokenizer
        extra_position_ids = extra_position_ids if extra_position_ids is not None else position_ids
        if input_ids is not None:
            input_ids = self._pad(input_ids, padding_value=tokenizer.pad_token_id, position_ids=extra_position_ids)
        if input_embeds is not None:
            pad_emb = torch.zeros(
                (1, embed_tokens.weight.shape[-1])).to(embed_tokens.weight.device).to(embed_tokens.weight.dtype)
            input_embeds = self._pad(input_embeds, padding_value=pad_emb, position_ids=extra_position_ids)
        batch_size = input_ids.shape[
            0] if input_ids is not None else input_embeds.shape[0] if input_embeds is not None else 1
        if position_ids is not None:
            position_ids = self._pad(position_ids, padding_value=-1, position_ids=extra_position_ids, dim=-1)
        if labels is not None:
            labels = self._pad(labels, padding_value=-100, position_ids=extra_position_ids)
        if loss_scale is not None:
            loss_scale = self._pad(loss_scale, padding_value=0., position_ids=extra_position_ids)
        if extra_position_ids is not None:
            extra_position_ids = self._pad(extra_position_ids, padding_value=-1, position_ids=extra_position_ids)
            self.extra_kwargs['_position_ids'] = extra_position_ids.clone()
            if self.extra_kwargs['_position_ids'].shape[1] == 2050:
                print()
        if (input_ids is not None or input_embeds is not None) and batch_size > 1:
            # not padding_free, so not ring-attention
            inputs = input_ids if input_ids is not None else input_embeds
            attn_shape = inputs.shape[1]  # The sequence length
            if attention_mask is None:
                attention_mask = torch.ones_like(extra_position_ids)
            # no need position_ids here, because padding_free does not need attention_mask, so this is not ring-attention
            attention_mask = self._pad(attention_mask, padding_value=0)
            cache_position = torch.arange(0, attn_shape, device=inputs.device)
            # pad attention mask to 4d to avoid calculation errors
            if hasattr(self, 'causal_mask_func') and self.causal_mask_func is not None:
                attention_mask = self.causal_mask_func(attention_mask, inputs.to(self.model_dtype), cache_position,
                                                       None, None)
        if input_ids is not None:
            input_ids = self._split(input_ids, dim=1, position_ids=extra_position_ids)
        if input_embeds is not None:
            input_embeds = self._split(input_embeds, dim=1, position_ids=extra_position_ids)
        if labels is not None:
            labels = torch.roll(labels, shifts=-1, dims=-1)
            labels = self._split(labels, dim=-1, position_ids=extra_position_ids)
        if loss_scale is not None:
            loss_scale = torch.roll(loss_scale, shifts=-1, dims=-1)
            loss_scale = self._split(loss_scale, dim=-1, position_ids=extra_position_ids)
        
        if position_ids is not None:
            position_ids = self._split(position_ids, dim=-1, position_ids=extra_position_ids)

        return input_ids, input_embeds, labels, position_ids, attention_mask, loss_scale

    def _init_device_mesh(self):
        """Initialize device mesh for sequence parallel"""
        rank, local_rank, world_size, local_world_size = get_dist_setting()
        self.dp_world_size = world_size // self.sp_world_size
        rp_world_size = self.sp_world_size // self.num_heads
        if rp_world_size <= 1:
            # Create device mesh: (dp_world_size, sp_world_size)
            self.device_mesh = init_device_mesh(
                get_device().split(':')[0],
                mesh_shape=(self.dp_world_size, self.sp_world_size),
                mesh_dim_names=('data', 'sequence'))
            self.rp_world_size = rp_world_size
            self.world_size = self.sp_world_size
        else:
            self.sp_world_size = self.num_heads
            self.rp_world_size = rp_world_size
            self.world_size = self.rp_world_size * self.sp_world_size
            # Create device mesh: (dp_world_size, rp_world_size, sp_world_size)
            self.device_mesh = init_device_mesh(
                get_device().split(':')[0],
                mesh_shape=(self.dp_world_size, self.rp_world_size, self.sp_world_size),
                mesh_dim_names=('data', 'ring', 'sequence'))

    @property
    def sp_group(self):
        """Return the sequence parallel group"""
        return self.device_mesh['sequence'].get_group() if self.device_mesh else None

    @property
    def sp_rank(self):
        """Return the sequence parallel rank"""
        return dist.get_rank(self.device_mesh['sequence'].get_group()) if self.device_mesh else 0

    @property
    def dp_group(self):
        """Return the data parallel group"""
        return self.device_mesh['data'].get_group() if self.device_mesh else None

    @property
    def dp_rank(self):
        """Return the data parallel rank"""
        return dist.get_rank(self.device_mesh['data'].get_group()) if self.device_mesh else 0

    @property
    def rp_group(self):
        """Return the data parallel group"""
        return self.device_mesh['ring'].get_group() if self.device_mesh and 'ring' in self.device_mesh.mesh_dim_names else None

    @property
    def rp_rank(self):
        """Return the data parallel rank"""
        return dist.get_rank(self.device_mesh['ring'].get_group()) if self.device_mesh and 'ring' in self.device_mesh.mesh_dim_names else -1

    def pad_and_split_extra_inputs(self, inputs):
        """Common input preparation function"""
        position_ids = inputs.get('_position_ids')
        if position_ids is None:
            position_ids = inputs.get('position_ids')
        if 'labels' in inputs:
            labels = inputs['labels']
            _, _, labels, _, _, _ = self.pad_and_split_inputs(None, None, labels, None, None, None, extra_position_ids=position_ids)
            inputs['labels'] = labels
        if 'loss_scale' in inputs:
            loss_scale = inputs['loss_scale']
            _, _, _, _, _, loss_scale = self.pad_and_split_inputs(None, None, None, None, None, loss_scale, extra_position_ids=position_ids)
            inputs['loss_scale'] = loss_scale
