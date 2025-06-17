import math
import os
from contextlib import contextmanager
from functools import partial
from types import MethodType
from typing import Any, Dict, Iterator, List, Optional, Tuple

import datasets
import numpy as np
import torch
import torch.distributed as dist
import trl
from packaging import version
from torch.distributed.device_mesh import init_device_mesh
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Sampler
from trl.extras.profiling import profiling_decorator

from swift.llm import DataLoaderDispatcher, DataLoaderShard, get_llm_model, to_device
from swift.utils import get_current_device, get_device, get_dist_setting, seed_worker
from .base import CommonSequenceParallel
from .utils import (
    GatherLoss, ChunkedCrossEntropyLoss, loss_scale_sp_func, _prepare_inputs,
    SequenceParallelSampler, SequenceParallelDispatcher, setup_compute_acc, get_common_dataloader,
    get_per_token_logps
)

assert version.parse(torch.__version__) >= version.parse('2.0.0')
torch._dynamo.config.capture_dynamic_output_shape_ops = True


torch_compile_options = {
    'epilogue_fusion': True,
    'max_autotune': False,
    'shape_padding': True,
    'trace.enabled': False,
    'triton.cudagraphs': False,
}


@profiling_decorator
def _prepare_inputs_grpo(self, generation_batch):
    ulysses = self.ulysses
    mode = 'train' if self.model.training else 'eval'
    if mode == 'train':
        # changes : `* ulysses.sp_world_size`
        generate_every = self.args.steps_per_generation * self.num_iterations * ulysses.sp_world_size
        if self._step % generate_every == 0 or self._buffered_inputs is None:
            generation_batch = self._generate_and_score_completions(generation_batch)
            self._buffered_inputs = generation_batch  # < this is the change
        # changes : `* ulysses.sp_world_size`
        inputs = self._buffered_inputs[self._step % (self.args.steps_per_generation * ulysses.sp_world_size)]
        self._step += 1
    else:
        inputs = self._generate_and_score_completions(generation_batch)
    return inputs


def old_policy(self):
    ulysses = self.ulysses
    # changes: `* ulysses.sp_world_size`
    return (self.num_iterations > 1
            or self.args.steps_per_generation * ulysses.sp_world_size > self.args.gradient_accumulation_steps)



@contextmanager
def padding_free_context(self, model: torch.nn.Module):
    ctx = {}

    def _padding_free_input_hook(module, args, kwargs):
        attention_mask = kwargs['attention_mask']
        ctx['padding_left'] = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if 'input_ids' in kwargs and kwargs.get('input_ids') is not None:
            kwargs['position_ids'] = torch.arange(kwargs['input_ids'].shape[1]).unsqueeze(0).repeat(
                kwargs['input_ids'].shape[0], 1).to(kwargs['input_ids'].dtype).to(kwargs['input_ids'].device)
            kwargs['input_ids'] = kwargs['input_ids'][attention_mask.bool()].unsqueeze(0)
        else:
            kwargs['position_ids'] = torch.arange(kwargs['inputs_embeds'].shape[1]).unsqueeze(0).repeat(
                kwargs['inputs_embeds'].shape[0], 1).to(torch.int64).to(kwargs['inputs_embeds'].device)
            kwargs['inputs_embeds'] = kwargs['inputs_embeds'][attention_mask.bool()].unsqueeze(0)
        kwargs['position_ids'] = kwargs['position_ids'][attention_mask.bool()].unsqueeze(0)
        kwargs.pop('attention_mask', None)
        return args, kwargs

    def _padding_free_output_hook(module, args, kwargs, result):
        position_ids = kwargs['position_ids']
        seq_lengths = []
        pos = position_ids[0]
        resets = torch.where(pos[1:] < pos[:-1])[0] + 1

        max_length = 0
        if len(resets) == 0:
            # Only one sequence in this batch item
            seq_lengths = [pos.max().item() + 1]
        else:
            # Multiple sequences
            start = 0
            for end in resets:
                seq_lengths.append(end - start)
                start = end
            seq_lengths.append(pos.shape[0] - start)

        max_length = max(seq_lengths)
        logits = result.logits.squeeze(0)
        unpacked_logits = []

        start = 0
        for length in seq_lengths:
            seq_state = logits[start:start + length]
            padding = torch.zeros((max_length - length)).to(logits.dtype).to(logits.device)
            if ctx['padding_left']:
                seq_state = torch.cat((padding, seq_state), dim=0)
            else:
                seq_state = torch.cat((seq_state, padding), dim=0)
            unpacked_logits.append(seq_state)
            start += length
        result.logits = torch.stack(unpacked_logits, dim=0)
        return result

    llm_model = get_llm_model(model)

    if self.padding_free:
        remove_handle1 = llm_model.model.register_forward_pre_hook(
            _padding_free_input_hook, with_kwargs=True, prepend=True)
        # cannot unpack here
        llm_model._unpack_output = _padding_free_output_hook
        llm_model._pack_input = _padding_free_input_hook
    yield
    if self.padding_free:
        remove_handle1.remove()


@profiling_decorator
def _get_per_token_logps(self, model, inputs):
    from trl.trainer.utils import selective_log_softmax
    ulysses = self.ulysses
    # original logits to keep
    logits_to_keep = inputs['logits_to_keep']
    input_ids = inputs['input_ids']
    inputs = {
        k: v
        for k, v in inputs.items() if k not in [
            'logits_to_keep', 'completion_mask', 'ref_per_token_logps', 'advantages', 'old_per_token_logps',
            'truncated_mask'
        ]
    }

    with self._template_context(self.template), padding_free_context(self, model):
        output = model(**inputs)
        logits = output.logits
    # original sequence length sharded
    origin_length = input_ids.shape[-1]
    if self.padding_free:
        _origin_logits_to_keep = logits_to_keep
        # if padding_free, calculate all logits tokens
        logits_to_keep = inputs['attention_mask'].sum()
        # packing again
        input_ids = input_ids[inputs['attention_mask'].bool()].unsqueeze(0)
        # set origin length to all logits length
        origin_length = inputs['attention_mask'].sum()
    # split input_ids to labels
    _, _, labels, _, _, _ = ulysses.pad_and_split_inputs(None, None, input_ids.clone(), None, None, None)

    shape1 = logits.shape[1]
    labels = torch.where(labels == -100, self.tokenizer.pad_token_id, labels)
    # calculate padding size of ulysses for example, 9 to 10 if sp=2
    padding_size = shape1 * ulysses.sp_world_size - origin_length
    # left shift one token to leave the last token
    logits_to_keep_padded = logits_to_keep + padding_size + 1

    # ckip logits_to_keep
    logits_to_keep_sharded = max(
        min(logits_to_keep_padded - (ulysses.sp_world_size - ulysses.sp_rank - 1) * shape1, shape1), 0)
    if logits_to_keep_sharded != 0:
        logits_kept = logits[:, -logits_to_keep_sharded:, :]
        logits_kept = logits_kept / self.temperature
        labels_kept = labels[:, -logits_to_keep_sharded:]
    else:
        logits_kept = logits[:, logits.shape[1]:, :]
        logits_kept = logits_kept / self.temperature
        labels_kept = labels[:, labels.shape[1]:]
    # how many padding tokens
    # for example:
    # aaaa bbbb cccc dddd
    # if logits_to_keep+padding_size+1 = 10
    # then bb cccc dddd will calculate selective_log_softmax
    # other tokens will be padded with 0.
    left_padding_len = shape1 - logits_to_keep_sharded
    per_token_logps = selective_log_softmax(logits_kept, labels_kept)
    _padding_logps = (
        torch.zeros((per_token_logps.shape[0], left_padding_len)).to(per_token_logps.device).to(per_token_logps.dtype))
    per_token_logps_padded = torch.cat((_padding_logps, per_token_logps), dim=1)
    _padding_labels = (torch.zeros((labels.shape[0], left_padding_len)).to(labels.device).to(labels.dtype))
    labels_padded = torch.cat((_padding_labels, labels_kept), dim=1)
    per_token_logps, _ = GatherLoss.apply(per_token_logps_padded, labels_padded, ulysses.sp_group, 1)
    if padding_size > 0:
        per_token_logps = per_token_logps[:, :-padding_size]
    if self.padding_free:
        llm_model = get_llm_model(model)
        output.logits = per_token_logps
        # unpack output after sp logps have been calculated
        _, inputs = llm_model._pack_input(None, None, inputs)
        per_token_logps = llm_model._unpack_output(None, None, inputs, output).logits
        delattr(llm_model, '_unpack_output')
        delattr(llm_model, '_pack_input')
        logits_to_keep = _origin_logits_to_keep
    # ignore the last token
    return per_token_logps[:, -logits_to_keep - 1:-1]


def split_by_mini_batches(self, inputs, advantages):
    ulysses = self.ulysses
    inputs_len = len(inputs)
    output = [None] * ulysses.sp_world_size
    # gather inputs within a sp group
    dist.all_gather_object(output, inputs, group=ulysses.sp_group)
    output = [p for sublist in output for p in sublist]
    inputs = output

    rank, local_rank, world_size, local_world_size = get_dist_setting()
    start_rank = (rank // ulysses.sp_world_size) * ulysses.sp_world_size
    process_slice = slice(
        start_rank * inputs_len,
        (start_rank + ulysses.sp_world_size) * inputs_len,
    )

    advantages = advantages[process_slice]

    mode = 'train' if self.model.training else 'eval'
    bs = self.args.per_device_train_batch_size if mode == 'train' else self.args.per_device_eval_batch_size
    spg = self.args.steps_per_generation * ulysses.sp_world_size if mode == 'train' else 1
    if mode == 'eval':
        # TODO only take the first bs rows, because eval does not support loop
        inputs = inputs[:bs]
        advantages = advantages[:bs]
    assert len(inputs) == bs * spg, f'Expected {bs * spg} inputs, got {len(inputs)}'
    spg_chunks = [inputs[i * bs:(i + 1) * bs] for i in range(spg)]
    # Split advantages by spg chunks
    advantage_chunks = torch.chunk(advantages, spg)

    return spg_chunks, advantage_chunks


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

    def get_dataloader(self, trainer, dataset, batch_size):
        return get_common_dataloader(self, trainer, dataset, batch_size, SequenceParallelSampler, SequenceParallelDispatcher)

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
            assert version.parse(trl.__version__) >= version.parse('0.18.0')
            trainer.ulysses = self
            trainer.args.gradient_accumulation_steps = trainer.args.gradient_accumulation_steps * self.sp_world_size
            trainer.old_policy = MethodType(old_policy, trainer)
            trainer._prepare_inputs = MethodType(_prepare_inputs_grpo, trainer)
            trainer._get_per_token_logps = MethodType(_get_per_token_logps, trainer)
            trainer.split_by_mini_batches = MethodType(split_by_mini_batches, trainer)

            class DataloaderWrap:

                def __init__(self, dataloader):
                    self.dataloader = dataloader

                def __getattr__(self, item):
                    return getattr(self.dataloader, item)

                def __len__(wrapped):
                    return len(wrapped.dataloader) * self.sp_world_size

                def __iter__(self):
                    yield from self.dataloader

            def get_train_dataloader(trainer):
                dataloader = trainer.get_origin_train_dataloader()
                return DataloaderWrap(dataloader)

            trainer.get_origin_train_dataloader = trainer.get_train_dataloader
            trainer.get_train_dataloader = MethodType(get_train_dataloader, trainer)

        setup_compute_acc(self)
