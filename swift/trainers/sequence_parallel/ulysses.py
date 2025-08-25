from functools import partial
from typing import Any, Optional, Tuple

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
        sequence_process_group: dist.ProcessGroup,
        scatter_idx: int = 2,
        gather_idx: int = 1,
    ) -> None:
        super(DistributedAttention, self).__init__()
        self.local_attn = local_attention
        self.spg = sequence_process_group
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor,
                *args: Any, **kwargs) -> torch.Tensor:
        query_layer = _SeqAllToAll.apply(self.spg, query, self.scatter_idx, self.gather_idx)
        key_layer = _SeqAllToAll.apply(self.spg, key, self.scatter_idx, self.gather_idx)
        value_layer = _SeqAllToAll.apply(self.spg, value, self.scatter_idx, self.gather_idx)
        position_ids = kwargs.pop('position_ids', None)
        if position_ids is not None:
            shape0 = position_ids.shape[0]
            position_ids_output = torch.empty((shape0 * dist.get_world_size(self.spg), position_ids.shape[1]),
                                              dtype=position_ids.dtype,
                                              device=position_ids.device)
            dist.all_gather_into_tensor(position_ids_output, position_ids, group=self.spg)
            position_ids = torch.cat(position_ids_output.split(shape0, dim=0), dim=1)
        context_layer = self.local_attn(
            query_layer, key_layer, value_layer, attention_mask, *args, position_ids=position_ids, **kwargs)
        output = _SeqAllToAll.apply(self.spg, context_layer, self.gather_idx, self.scatter_idx)
        return output


class SequenceParallel:

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

        from transformers.modeling_flash_attention_utils import is_flash_attn_available
        if is_flash_attn_available():
            # TODO this works for multi-modal models like qwen2.5-vl
            # SDPA is not supported, because we need to copy the code to our project, which will bring
            # more works for maintaining.
            from transformers import modeling_flash_attention_utils
            from transformers.modeling_flash_attention_utils import _flash_attention_forward
            _distributed_flash_attention = DistributedAttention(_flash_attention_forward, self.sp_group)

            def flash_attention_forward(query_states: torch.Tensor, key_states: torch.Tensor,
                                        value_states: torch.Tensor, attention_mask: Optional[torch.Tensor], q_len,
                                        *args, **kwargs):
                return _distributed_flash_attention(query_states, key_states, value_states, attention_mask,
                                                    q_len * self.sp_world_size, *args, **kwargs)

            modeling_flash_attention_utils._flash_attention_forward = flash_attention_forward

        if hasattr(base_model, 'language_model'):
            text_model = base_model.language_model
            if hasattr(base_model.language_model, '_update_causal_mask'):
                self.causal_mask_func = base_model.language_model._update_causal_mask
        else:
            text_model = base_model
            if hasattr(base_model, '_update_causal_mask'):
                self.causal_mask_func = base_model._update_causal_mask

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
                        from .zigzag_ring_flash_attn import zigzag_ring_flash_attn_func
                        output = zigzag_ring_flash_attn_func(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
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

        if 'flash_attention_2_origin' not in ALL_ATTENTION_FUNCTIONS:
            ALL_ATTENTION_FUNCTIONS['flash_attention_2_origin'] = ALL_ATTENTION_FUNCTIONS['flash_attention_2']
            ALL_ATTENTION_FUNCTIONS['sdpa_origin'] = ALL_ATTENTION_FUNCTIONS['sdpa']
            ALL_ATTENTION_FUNCTIONS['flash_attention_2'] = partial(
                local_flash_attn, dist_attn=DistributedAttention(None, self.sp_group))
            ALL_ATTENTION_FUNCTIONS['sdpa'] = partial(
                local_sdpa_attn, dist_attn=DistributedAttention(None, self.sp_group))

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
                input_ids, inputs_embeds, None, position_ids, attention_mask, None, embed_tokens=embed_tokens)
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
        self._init_device_mesh()

        llm_model = get_llm_model(model)

        if hasattr(llm_model, 'thinker'):
            base_model = llm_model.thinker.model
        else:
            base_model = llm_model.model

        self._prepare_flash_attn(base_model)
        self._prepare_forward_hook(base_model)
        if model.model_info.is_moe_model:
            self._prepare_moe_aux_loss(base_model)

        self.model_dtype = next(model.parameters()).dtype
        self.tokenizer = tokenizer

    def _pad(self, tensor, padding_value, dim=-1):
        """Pad tensor for sequence parallel"""
        length = tensor.shape[dim]
        if length % self.world_size == 0:
            return tensor

        pad_num = self.world_size - (length % self.world_size)
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

    def _gather(self, local_output, dim: int):
        """Gather tensor for sequence parallel - reverse of _split"""
        if self.world_size == 1:
            return local_output

        if self.rp_world_size > 1:
            input_dim = local_output.dim()
            assert input_dim >= 2

            batch_size, local_seq_len, *rest = local_output.shape

            # Step 1: Gather from all sequence parallel ranks
            # Each sp_rank has its own piece, we need to gather them first
            gathered_sp = [torch.zeros_like(local_output) for _ in range(self.sp_world_size)]
            torch.distributed.all_gather(gathered_sp, local_output.contiguous(), group=self.sp_group)

            # Concatenate the sp pieces to form the complete chunk for this rp_rank
            rp_chunk = torch.cat(gathered_sp, dim=dim)

            # Step 2: Gather all rp chunks
            gathered_rp = [torch.zeros_like(rp_chunk) for _ in range(self.rp_world_size)]
            torch.distributed.all_gather(gathered_rp, rp_chunk, group=self.rp_group)

            # Step 3: Reconstruct the original tensor by unfolding the Z-pattern
            # We need to separate each rp chunk into two parts as per the original pattern
            full_seq_len = local_seq_len * self.world_size
            chunk_size = full_seq_len // (2 * self.rp_world_size)

            # Initialize the full output tensor
            full_output = torch.zeros([batch_size, full_seq_len, *rest], device=local_output.device)

            # Place each chunk in its correct position
            for i in range(self.rp_world_size):
                # In the original split, rank i got chunk[i] and chunk[2*rp_world_size-i-1]
                # Now we need to place them back
                full_output[:, i * chunk_size:(i + 1) * chunk_size] = gathered_rp[i][:, :chunk_size]
                full_output[:, (2 * self.rp_world_size - i - 1) * chunk_size:(2 * self.rp_world_size - i) * chunk_size] = \
                gathered_rp[i][:, chunk_size:]

            return full_output.contiguous()
        else:
            gathered_sp = torch.empty((local_output.shape[0] * self.sp_world_size, local_output.shape[1]),
                                       dtype=local_output.dtype,
                                       device=local_output.device)
            dist.all_gather_into_tensor(gathered_sp, local_output, group=self.sp_group)
            gathered_sp = torch.cat(gathered_sp.split(local_output.shape[0], dim=0), dim=dim)
            return gathered_sp.contiguous()

    def _split(self, input, dim: int):
        """Split tensor for sequence parallel"""
        if self.world_size == 1:
            return input

        if self.rp_world_size > 1:
            input_dim = input.dim()
            assert input_dim >= 2
            batch_size, seq_len, *rest = input.shape

            value_chunks = input.chunk(2 * self.rp_world_size, dim=dim)

            local_value = torch.cat(
                [value_chunks[self.rp_rank], value_chunks[2 * self.rp_world_size - self.rp_rank - 1]], dim=dim
            ).chunk(self.sp_world_size, dim=dim)[self.sp_rank]

            new_shape = [batch_size, seq_len // self.world_size] + rest
            return local_value.reshape(new_shape).contiguous()
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
                             embed_tokens=None):
        """Common implementation for padding and splitting inputs"""
        tokenizer = self.tokenizer
        if input_ids is not None:
            input_ids = self._pad(input_ids, padding_value=tokenizer.pad_token_id, dim=-1)
        if input_embeds is not None:
            pad_emb = torch.zeros(
                (1, embed_tokens.weight.shape[-1])).to(embed_tokens.weight.device).to(embed_tokens.weight.dtype)
            input_embeds = self._pad(input_embeds, padding_value=pad_emb, dim=1)
        batch_size = input_ids.shape[
            0] if input_ids is not None else input_embeds.shape[0] if input_embeds is not None else 1
        if position_ids is not None:
            position_ids = self._pad(position_ids, padding_value=0, dim=-1)
        if (input_ids is not None or input_embeds is not None) and batch_size > 1:
            inputs = input_ids if input_ids is not None else input_embeds
            attn_shape = inputs.shape[1]  # The sequence length
            if attention_mask is None:
                attention_mask = torch.ones_like(position_ids)
            attention_mask = self._pad(attention_mask, padding_value=0, dim=-1)
            cache_position = torch.arange(0, attn_shape, device=inputs.device)
            # pad attention mask to 4d to avoid calculation errors
            if hasattr(self, 'causal_mask_func') and self.causal_mask_func is not None:
                attention_mask = self.causal_mask_func(attention_mask, inputs.to(self.model_dtype), cache_position,
                                                       None, None)
        if input_ids is not None:
            input_ids = self._split(input_ids, dim=1)
        if input_embeds is not None:
            input_embeds = self._split(input_embeds, dim=1)
        if position_ids is not None:
            position_ids = self._split(position_ids, dim=-1)
        if labels is not None:
            labels = self._pad(labels, padding_value=-100, dim=-1)
            labels = torch.roll(labels, shifts=-1, dims=-1)
            labels = self._split(labels, dim=-1)

        if loss_scale is not None:
            loss_scale = self._pad(loss_scale, padding_value=0., dim=-1)
            loss_scale = torch.roll(loss_scale, shifts=-1, dims=-1)
            loss_scale = self._split(loss_scale, dim=-1)

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
        if 'labels' in inputs:
            labels = inputs['labels']
            _, _, labels, _, _, _ = self.pad_and_split_inputs(None, None, labels, None, None, None)
            inputs['labels'] = labels
        if 'loss_scale' in inputs:
            loss_scale = inputs['loss_scale']
            _, _, _, _, _, loss_scale = self.pad_and_split_inputs(None, None, None, None, None, loss_scale)
            inputs['loss_scale'] = loss_scale
