# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import os
from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple

import datasets
import numpy as np
import torch
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Sampler

from swift.llm import DataLoaderDispatcher, DataLoaderShard, get_llm_model
from swift.utils import get_current_device, get_dist_setting, seed_worker
from .base import SequenceParallel

if TYPE_CHECKING:
    try:
        from ..rlhf_trainer import GRPOTrainer
        from ..rlhf_trainer.grpo_trainer import InputsType
    except ImportError:
        pass
# Conditional import for profiling decorator
try:
    from trl.extras.profiling import profiling_decorator
except ImportError:
    # Fallback if trl is not available
    def profiling_decorator(func):
        return func


try:
    from trl.trainer.utils import entropy_from_logits
except ImportError:
    from ..rlhf_trainer.utils import entropy_from_logits


class GatherLoss(torch.autograd.Function):
    """Gather loss from sequence group"""

    @staticmethod
    def forward(ctx, loss, labels, process_group, gather_idx=None):
        """
        Args:
            loss: loss tensor after splitting
            labels: labels tensor after splitting
            process_group: the sequence parallel group
            gather_idx: gather the tensors on this dim
        """
        ctx.process_group = process_group
        shape0 = labels.shape[0]
        ctx.scatter_shape = labels.shape[gather_idx or 0]
        ctx.gather_idx = gather_idx or 0
        world_size = dist.get_world_size(group=process_group)  # the sp world size
        output = torch.empty((shape0 * world_size, *loss.shape[1:]), dtype=loss.dtype, device=loss.device)
        # gather all from sp group
        dist.all_gather_into_tensor(output, loss, group=process_group)
        if gather_idx is not None:
            output = torch.cat(output.split(shape0, dim=0), dim=gather_idx)
        labels_output = torch.empty((shape0 * world_size, *labels.shape[1:]), dtype=labels.dtype, device=labels.device)
        dist.all_gather_into_tensor(labels_output, labels, group=process_group)
        if gather_idx is not None:
            labels_output = torch.cat(labels_output.split(shape0, dim=0), dim=gather_idx)
        return output, labels_output

    @staticmethod
    def backward(ctx, *grad_output):
        _grad = grad_output[0] * dist.get_world_size(group=ctx.process_group)
        return _grad.split(
            ctx.scatter_shape, dim=ctx.gather_idx)[dist.get_rank(ctx.process_group)].contiguous(), None, None, None


class ChunkedCrossEntropyLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, labels, chunk_size):
        ctx.save_for_backward(logits, labels)
        ctx.chunk_size = chunk_size

        losses = []
        for i in range(math.ceil(logits.shape[0] / chunk_size)):
            l_start = i * chunk_size
            l_end = min((i + 1) * chunk_size, logits.shape[0])
            logits_chunk = logits[l_start:l_end]
            labels_chunk = labels[l_start:l_end]
            loss_fct = CrossEntropyLoss(reduction='none')
            loss_chunk = loss_fct(logits_chunk, labels_chunk)
            losses.append(loss_chunk)
            del logits_chunk
            del labels_chunk
        all_losses = torch.cat(losses)
        return all_losses

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        logits, labels = ctx.saved_tensors
        chunk_size = ctx.chunk_size

        for i in range(math.ceil(logits.shape[0] / chunk_size)):
            l_start = i * chunk_size
            l_end = min((i + 1) * chunk_size, logits.shape[0])
            logits_chunk = logits[l_start:l_end].detach().requires_grad_(True)
            labels_chunk = labels[l_start:l_end]
            loss_fct = CrossEntropyLoss(reduction='none')
            with torch.enable_grad():
                loss_chunk = loss_fct(logits_chunk, labels_chunk)
                grad_output_chunk = grad_outputs[0][l_start:l_end]
                _loss_chunk = (loss_chunk * grad_output_chunk).sum()
                grad_chunk = torch.autograd.grad(_loss_chunk, logits_chunk, retain_graph=False)[0]
                logits[l_start:l_end] = grad_chunk

        return logits, None, None


def loss_scale_sp_func(outputs, labels, loss_scale=None, num_items_in_batch=None, sp_instance=None) -> torch.Tensor:
    """Common loss function for sequence parallel training"""
    if hasattr(outputs, 'logits'):
        logits = outputs.logits
    else:
        logits = outputs
    device = logits.device

    if labels.shape[1] > logits.shape[1]:
        _, _, labels, _, _, _ = sp_instance.pad_and_split_inputs(None, None, labels, None, None, None)
    if loss_scale is not None and loss_scale.shape[1] > logits.shape[1]:
        _, _, _, _, _, loss_scale = sp_instance.pad_and_split_inputs(None, None, None, None, None, loss_scale)
    logits = logits.view(-1, logits.shape[-1])

    labels = labels.flatten().to(device)
    sploss_parallel_size = int(os.environ.get('CELOSS_PARALLEL_SIZE', '0'))
    if sploss_parallel_size > 0:
        loss = ChunkedCrossEntropyLoss.apply(logits, labels, sploss_parallel_size)
    else:
        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(logits, labels)
    if loss_scale is not None:
        loss_scale = loss_scale.flatten().to(device)
        loss = (loss_scale * loss)
    loss, labels = GatherLoss.apply(loss, labels, sp_instance.sp_group)
    mask = (labels != -100)
    total_loss = loss[mask].sum()
    if num_items_in_batch is None:
        total_loss = total_loss / mask.sum()
    else:
        total_loss = total_loss / num_items_in_batch
    return total_loss


def _prepare_inputs(self, inputs, sp_instance):
    """Common input preparation function"""
    if 'labels' in inputs:
        labels = inputs['labels']
        _, _, labels, _, _, _ = sp_instance.pad_and_split_inputs(None, None, labels, None, None, None)
        inputs['labels'] = labels
    return self._origin_prepare_inputs(inputs)


class SequenceParallelSampler(Sampler):
    """Sampler for sequence parallel training"""

    def __init__(self, sp_instance, dataset, shuffle: bool = True, seed=None, round_up: bool = True) -> None:
        self.sp_instance = sp_instance
        rank = dist.get_rank(sp_instance.device_mesh['data'].get_group())
        world_size = sp_instance.device_mesh['data'].size()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.shuffle = shuffle
        assert seed is not None
        self.seed = seed
        self.epoch = 0
        self.round_up = round_up

        if self.round_up:
            self.num_samples = math.ceil(len(self.dataset) / world_size)
            self.total_size = self.num_samples * self.world_size
        else:
            self.num_samples = math.ceil((len(self.dataset) - rank) / world_size)
            self.total_size = len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        if self.round_up:
            indices = (indices * int(self.total_size / len(indices) + 1))[:self.total_size]

        indices = indices[self.rank:self.total_size:self.world_size]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class SequenceParallelDispatcher(DataLoaderDispatcher):
    """Dispatcher for sequence parallel training"""

    def __init__(self, dataloader, sp_instance, device=None, skip_batches: int = 0):
        super().__init__(dataloader)
        self.sp_instance = sp_instance
        self.device = device
        self.skip_batches = skip_batches

    @property
    def rank(self):
        return self.sp_instance.dp_rank if dist.is_initialized() else 0

    @property
    def world_size(self):
        return self.sp_instance.dp_world_size if dist.is_initialized() else 1

    @property
    def group(self):
        return self.sp_instance.dp_group if dist.is_initialized() else 1


def setup_compute_acc(sp_instance):
    """Setup compute_acc function for sequence parallel training"""
    from swift.plugin import metric
    from swift.trainers import mixin
    compute_acc_origin = metric.compute_acc

    def compute_acc(preds, labels, *args, **kwargs) -> Dict[str, List[float]]:
        # Gather preds and labels across the sp group
        if isinstance(preds, np.ndarray):
            preds = torch.from_numpy(preds).to(get_current_device())
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels).to(get_current_device())
        if labels.shape[1] > preds.shape[1]:
            _, _, labels, _, _, _ = sp_instance.pad_and_split_inputs(None, None, labels, None, None, None)
        shape0 = preds.shape[0]
        preds_output = torch.empty((shape0 * sp_instance.sp_world_size, preds.shape[1]),
                                   dtype=preds.dtype,
                                   device=preds.device)
        dist.all_gather_into_tensor(preds_output, preds, group=sp_instance.sp_group)
        preds_output = torch.cat(preds_output.split(shape0, dim=0), dim=1)
        shape0 = labels.shape[0]
        labels_output = torch.empty((shape0 * sp_instance.sp_world_size, labels.shape[1]),
                                    dtype=labels.dtype,
                                    device=labels.device)
        dist.all_gather_into_tensor(labels_output, labels, group=sp_instance.sp_group)
        labels_output = torch.cat(labels_output.split(shape0, dim=0), dim=1)
        # roll back to fit compute_acc
        labels_output = torch.roll(labels_output, shifts=1, dims=1)
        return compute_acc_origin(preds_output, labels_output, *args, **kwargs)

    metric.compute_acc = compute_acc
    mixin.compute_acc = compute_acc


# For DPO
def get_per_token_logps(logits: torch.FloatTensor,
                        labels: torch.LongTensor,
                        label_pad_token_id=-100,
                        sp_instance=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Common DPO per-token logps function for sequence parallel training"""
    if labels.shape[1] > logits.shape[1]:
        _, _, labels, _, _, _ = sp_instance.pad_and_split_inputs(None, None, labels, None, None, None)
    loss_mask = labels != label_pad_token_id
    labels = labels.clone()  # No need to shift, pad and split has shifted the inputs.
    labels[~loss_mask] = 0
    labels = labels.to(logits.device)
    loss_mask = loss_mask.to(logits.device)
    mean_logits = logits.mean(-1)
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    total_per_token_logps, total_loss_mask = GatherLoss.apply(per_token_logps, loss_mask, sp_instance.sp_group, 1)

    world_size = dist.get_world_size(group=sp_instance.sp_group)
    total_mean_logits = mean_logits.new_empty((mean_logits.shape[0], mean_logits.shape[1] * world_size))
    dist.all_gather_into_tensor(total_mean_logits, mean_logits, group=sp_instance.sp_group)
    total_per_token_logps[~total_loss_mask] = 0
    return total_per_token_logps, total_mean_logits, total_loss_mask


def get_common_dataloader(sp_instance,
                          trainer,
                          dataset,
                          batch_size,
                          sampler_class,
                          dispatcher_class,
                          skip_batches: int = 0):
    """Common dataloader creation function"""
    data_collator = trainer.data_collator
    if isinstance(dataset, datasets.Dataset):
        dataset = trainer._remove_unused_columns(dataset, description='training')
    else:
        data_collator = trainer._get_collator_with_removed_columns(data_collator, description='training')

    if hasattr(dataset, '__len__'):
        sampler = sampler_class(sp_instance, dataset, seed=42)
        dataloader_params = {
            'batch_size': batch_size,
            'collate_fn': data_collator,
            'num_workers': trainer.args.dataloader_num_workers,
            'pin_memory': trainer.args.dataloader_pin_memory,
            'persistent_workers': trainer.args.dataloader_persistent_workers,
        }

        if not isinstance(dataset, torch.utils.data.IterableDataset):
            if skip_batches > 0:
                from accelerate.data_loader import SkipBatchSampler
                sampler = SkipBatchSampler(sampler, skip_batches=skip_batches * batch_size)
            dataloader_params['sampler'] = sampler
            dataloader_params['drop_last'] = trainer.args.dataloader_drop_last
            dataloader_params['worker_init_fn'] = partial(
                seed_worker, num_workers=trainer.args.dataloader_num_workers, rank=trainer.args.process_index)

        return DataLoaderShard(dataset, device=trainer.accelerator.device, **dataloader_params)
    else:
        dataloader_params = {
            'collate_fn': data_collator,
            'num_workers': trainer.args.dataloader_num_workers,
            'pin_memory': trainer.args.dataloader_pin_memory,
            'persistent_workers': trainer.args.dataloader_persistent_workers,
            'prefetch_factor': trainer.args.dataloader_prefetch_factor
        }
        if dist.is_initialized() and dataloader_params['prefetch_factor']:
            dataloader_params['prefetch_factor'] = dataloader_params['prefetch_factor'] * dist.get_world_size()
        dataloader = DataLoader(dataset, batch_size=batch_size, **dataloader_params)
        dataloader = dispatcher_class(dataloader, sp_instance, trainer.accelerator.device, skip_batches=skip_batches)
        return dataloader


# GRPO related functions
@profiling_decorator
def _prepare_inputs_grpo(self, generation_batch, sp_instance):
    """Common GRPO input preparation function for sequence parallel training"""
    mode = 'train' if self.model.training else 'eval'
    if mode == 'train':
        # changes : `* sp_instance.sp_world_size`
        generate_every = self.args.steps_per_generation * self.num_iterations * sp_instance.sp_world_size
        if self._step % generate_every == 0 or self._buffered_inputs is None:
            generation_batch = self._generate_and_score_completions(generation_batch)
            self._buffered_inputs = generation_batch  # < this is the change
        # changes : `* sp_instance.sp_world_size`
        inputs = self._buffered_inputs[self._step % (self.args.steps_per_generation * sp_instance.sp_world_size)]
        self._step += 1
    else:
        inputs = self._generate_and_score_completions(generation_batch)
    return inputs


def _get_train_sampler_grpo(self, dataset=None, sp_instance=None):
    """Get train sampler for GRPO sequence parallel training"""
    try:
        from trl.trainer.grpo_trainer import RepeatSampler
    except ImportError:
        raise ImportError('trl is required for GRPO training. Please install it with: pip install trl')

    if dataset is None:
        dataset = self.train_dataset
    return RepeatSampler(
        data_source=dataset,
        mini_repeat_count=self.num_generations,
        batch_size=self.args.generation_batch_size // self.num_generations,
        repeat_count=self.num_iterations * self.args.steps_per_generation * sp_instance.sp_world_size,
        shuffle=self.shuffle_dataset,
        seed=self.args.seed,
    )


def old_policy_grpo(self, sp_instance):
    """Old policy for GRPO sequence parallel training"""
    # changes: `* sp_instance.sp_world_size`
    return (self.num_iterations > 1
            or self.args.steps_per_generation * sp_instance.sp_world_size > self.args.gradient_accumulation_steps)


def split_by_mini_batches_grpo(self, inputs, advantages, sp_instance):
    """Split by mini batches for GRPO sequence parallel training"""
    inputs_len = len(inputs)
    output = [None] * sp_instance.sp_world_size
    # gather inputs within a sp group
    dist.all_gather_object(output, inputs, group=sp_instance.sp_group)
    output = [p for sublist in output for p in sublist]
    inputs = output

    rank, local_rank, world_size, local_world_size = get_dist_setting()
    start_rank = (rank // sp_instance.sp_world_size) * sp_instance.sp_world_size
    process_slice = slice(
        start_rank * inputs_len,
        (start_rank + sp_instance.sp_world_size) * inputs_len,
    )

    advantages = advantages[process_slice]

    mode = 'train' if self.model.training else 'eval'
    bs = self.args.per_device_train_batch_size if mode == 'train' else self.args.per_device_eval_batch_size
    spg = self.args.steps_per_generation * sp_instance.sp_world_size if mode == 'train' else 1
    if mode == 'eval':
        # TODO only take the first bs rows, because eval does not support loop
        inputs = inputs[:bs]
        advantages = advantages[:bs]
    assert len(inputs) == bs * spg, f'Expected {bs * spg} inputs, got {len(inputs)}'
    spg_chunks = [inputs[i * bs:(i + 1) * bs] for i in range(spg)]
    # Split advantages by spg chunks
    advantage_chunks = torch.chunk(advantages, spg)

    return spg_chunks, advantage_chunks


@contextmanager
def padding_free_context_grpo(self, model: torch.nn.Module, sp_instance):
    """Padding free context for GRPO sequence parallel training"""
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
        has_entropies = hasattr(result, 'entropies') and result.entropies is not None
        if has_entropies:
            entropies = result.entropies.squeeze(0)

        unpacked_logits = []
        unpacked_entropies = [] if has_entropies else None
        start = 0

        for length in seq_lengths:
            seq_state = logits[start:start + length]
            padding = torch.zeros((max_length - length, ), dtype=logits.dtype, device=logits.device)
            if ctx['padding_left']:
                seq_state = torch.cat((padding, seq_state), dim=0)
            else:
                seq_state = torch.cat((seq_state, padding), dim=0)
            unpacked_logits.append(seq_state)

            if has_entropies:
                ent_state = entropies[start:start + length]
                if ctx['padding_left']:
                    ent_state = torch.cat((padding, ent_state), dim=0)
                else:
                    ent_state = torch.cat((ent_state, padding), dim=0)
                unpacked_entropies.append(ent_state)
            start += length

        result.logits = torch.stack(unpacked_logits, dim=0)
        if has_entropies:
            result.entropies = torch.stack(unpacked_entropies, dim=0)
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
def _get_per_token_logps_and_entropies_grpo(
        self: 'GRPOTrainer',
        model: torch.nn.Module,
        inputs: 'InputsType',
        sp_instance: SequenceParallel,
        compute_entropy: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Get per token logps for GRPO sequence parallel training"""
    try:
        from trl.trainer.utils import selective_log_softmax
    except ImportError:
        raise ImportError('trl is required for GRPO training. Please install it with: pip install trl')

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

    with self._template_context(self.template), padding_free_context_grpo(self, model, sp_instance):
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
    _, _, labels, _, _, _ = sp_instance.pad_and_split_inputs(None, None, input_ids.clone(), None, None, None)

    shape1 = logits.shape[1]
    labels = torch.where(labels == -100, self.processing_class.pad_token_id, labels)
    # calculate padding size for example, 9 to 10 if sp=2
    padding_size = shape1 * sp_instance.sp_world_size - origin_length
    # left shift one token to leave the last token
    logits_to_keep_padded = logits_to_keep + padding_size + 1

    # skip logits_to_keep
    logits_to_keep_sharded = max(
        min(logits_to_keep_padded - (sp_instance.sp_world_size - sp_instance.sp_rank - 1) * shape1, shape1), 0)
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
    entropies = None
    _padding_logps = torch.zeros((per_token_logps.shape[0], left_padding_len),
                                 device=per_token_logps.device,
                                 dtype=per_token_logps.dtype)

    per_token_logps_padded = torch.cat((_padding_logps, per_token_logps), dim=1)

    _padding_labels = torch.zeros((labels.shape[0], left_padding_len), device=labels.device, dtype=labels.dtype)
    labels_padded = torch.cat((_padding_labels, labels_kept), dim=1)
    per_token_logps, _ = GatherLoss.apply(per_token_logps_padded, labels_padded, sp_instance.sp_group, 1)
    if compute_entropy:
        entropies = entropy_from_logits(logits_kept)
        entropies_padded = torch.cat((_padding_logps, entropies), dim=1)
        entropies, _ = GatherLoss.apply(entropies_padded, labels_padded, sp_instance.sp_group, 1)

    if padding_size > 0:
        per_token_logps = per_token_logps[:, :-padding_size]
    if self.padding_free:
        llm_model = get_llm_model(model)
        output.logits = per_token_logps
        output.entropies = entropies
        # unpack output after sp logps have been calculated
        _, inputs = llm_model._pack_input(None, None, inputs)
        output = llm_model._unpack_output(None, None, inputs, output)
        per_token_logps = output.logits
        entropies = output.entropies
        delattr(llm_model, '_unpack_output')
        delattr(llm_model, '_pack_input')
        logits_to_keep = _origin_logits_to_keep
        per_token_logps = per_token_logps[:, -logits_to_keep - 1:-1]
        if compute_entropy:
            entropies = entropies[:, -logits_to_keep - 1:-1]
    # ignore the last token
    return per_token_logps, entropies
