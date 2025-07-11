from functools import partial
from types import MethodType
from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
from packaging import version

from swift.llm import get_llm_model
from .base import CommonSequenceParallel
from .utils import (SequenceParallelDispatcher, SequenceParallelSampler, _get_per_token_logps_grpo,
                    _get_train_sampler_grpo, _prepare_inputs, _prepare_inputs_grpo, get_common_dataloader,
                    get_per_token_logps, loss_scale_sp_func, old_policy_grpo, setup_compute_acc,
                    split_by_mini_batches_grpo)

assert version.parse(torch.__version__) >= version.parse('2.0.0')
torch._dynamo.config.capture_dynamic_output_shape_ops = True

torch_compile_options = {
    'epilogue_fusion': True,
    'max_autotune': False,
    'shape_padding': True,
    'trace.enabled': False,
    'triton.cudagraphs': False,
}


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


class Ulysses(CommonSequenceParallel):

    def __init__(self):
        super().__init__()
        self.split_in_forward = None
        self.causal_mask_func = None

    def init_sequence_parallel(self, size):
        if self._inited:
            return
        self._inited = True
        self.sp_world_size = size
        self._init_device_mesh()

        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS['flash_attention_2_origin'] = ALL_ATTENTION_FUNCTIONS['flash_attention_2']
        ALL_ATTENTION_FUNCTIONS['sdpa_origin'] = ALL_ATTENTION_FUNCTIONS['sdpa']

        def local_flash_attn(module: torch.nn.Module, query_states, key_states, value_states, attention_mask, *args,
                             dist_attn, **kwargs):
            if dist_attn.local_attn is None:

                def _attention(query, key, value, *args, **kwargs):
                    query = query.transpose(1, 2)
                    key = key.transpose(1, 2)
                    value = value.transpose(1, 2)
                    return ALL_ATTENTION_FUNCTIONS['flash_attention_2_origin'](module, query, key, value, *args,
                                                                               **kwargs)[0]

                dist_attn.local_attn = _attention

            return dist_attn(
                query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2), attention_mask,
                *args, **kwargs), None

        def local_sdpa_attn(module: torch.nn.Module, query_states, key_states, value_states, attention_mask, *args,
                            dist_attn, **kwargs):
            if dist_attn.local_attn is None:

                def _attention(query, key, value, *args, **kwargs):
                    query = query.transpose(1, 2)
                    key = key.transpose(1, 2)
                    value = value.transpose(1, 2)
                    return ALL_ATTENTION_FUNCTIONS['sdpa_origin'](module, query, key, value, *args, **kwargs)[0]

                dist_attn.local_attn = _attention
            return dist_attn(
                query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2), attention_mask,
                *args, **kwargs), None

        ALL_ATTENTION_FUNCTIONS['flash_attention_2'] = partial(
            local_flash_attn, dist_attn=DistributedAttention(None, self.sp_group))
        ALL_ATTENTION_FUNCTIONS['sdpa'] = partial(local_sdpa_attn, dist_attn=DistributedAttention(None, self.sp_group))

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

    def prepare_model(self, model, tokenizer):

        def pre_forward_split_hook(_self, args, kwargs):
            input_ids = kwargs.get('input_ids', None)
            inputs_embeds = kwargs.get('inputs_embeds', None)
            position_ids = kwargs['position_ids']
            attention_mask = kwargs.get('attention_mask', None)
            if hasattr(_self, 'language_model'):
                embed_tokens = getattr(_self.language_model, 'embed_tokens', None)
            else:
                embed_tokens = getattr(_self, 'embed_tokens', None)
            _input_ids, inputs_embeds, _, position_ids, attention_mask, _ = self.pad_and_split_inputs(
                input_ids, inputs_embeds, None, position_ids, attention_mask, None, embed_tokens=embed_tokens)
            kwargs['input_ids'] = _input_ids
            kwargs['inputs_embeds'] = inputs_embeds
            kwargs['position_ids'] = position_ids
            kwargs['attention_mask'] = attention_mask
            return args, kwargs

        llm_model = get_llm_model(model)

        if hasattr(llm_model, 'thinker'):
            base_model = llm_model.thinker.model
        else:
            base_model = llm_model.model
        if hasattr(base_model, 'language_model'):
            self.causal_mask_func = base_model.language_model._update_causal_mask
        else:
            self.causal_mask_func = base_model._update_causal_mask
        base_model.register_forward_pre_hook(pre_forward_split_hook, with_kwargs=True)
        self.model_dtype = next(model.parameters()).dtype
        self.tokenizer = tokenizer

    def get_dataloader(self, trainer, dataset, batch_size, skip_batches: int = 0):
        return get_common_dataloader(
            self,
            trainer,
            dataset,
            batch_size,
            SequenceParallelSampler,
            SequenceParallelDispatcher,
            skip_batches=skip_batches)

    def prepare_trainer(self, trainer):
        # TODO hack methods, not cool
        if trainer.train_dataset is None:
            raise ValueError('Trainer: training requires a train_dataset.')

        trainer.ulysses = self
        if trainer.__class__.__name__ == 'Seq2SeqTrainer':
            trainer._origin_prepare_inputs = trainer._prepare_inputs
            trainer._prepare_inputs = MethodType(partial(_prepare_inputs, sp_instance=self), trainer)
            trainer.compute_loss_func = partial(loss_scale_sp_func, sp_instance=self)

        elif trainer.__class__.__name__ == 'DPOTrainer':
            trainer._origin_prepare_inputs = trainer._prepare_inputs
            trainer._prepare_inputs = MethodType(partial(_prepare_inputs, sp_instance=self), trainer)
            trainer.get_per_token_logps = partial(get_per_token_logps, sp_instance=self)

        elif trainer.__class__.__name__ == 'GRPOTrainer':
            import trl
            assert version.parse(trl.__version__) >= version.parse('0.18.0')
            trainer.ulysses = self
            trainer.args.gradient_accumulation_steps = trainer.args.gradient_accumulation_steps * self.sp_world_size
            trainer.old_policy = MethodType(partial(old_policy_grpo, sp_instance=self), trainer)
            trainer._get_train_sampler = MethodType(partial(_get_train_sampler_grpo, sp_instance=self), trainer)
            trainer._prepare_inputs = MethodType(partial(_prepare_inputs_grpo, sp_instance=self), trainer)
            trainer._get_per_token_logps = MethodType(partial(_get_per_token_logps_grpo, sp_instance=self), trainer)
            trainer.split_by_mini_batches = MethodType(partial(split_by_mini_batches_grpo, sp_instance=self), trainer)

        setup_compute_acc(self)
