# Copyright (c) ModelScope Contributors. All rights reserved.
import gc
import torch
from accelerate.utils import gather as hf_gather
from accelerate.utils import gather_object as hf_gather_object
from contextlib import nullcontext
from dataclasses import dataclass
from megatron.core import mpu
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.optimizer import ChainedOptimizer
from typing import Any, Optional

from swift.dataloader import DataLoaderDispatcher
from swift.megatron.utils import get_batch_on_this_cp_rank, get_packed_seq_params
from swift.utils import empty_cache, get_current_device, get_logger, to_device

logger = get_logger()


def get_batch_on_this_pp_rank(args, data, vp_stage=None):
    if args.task_type == 'causal_lm':
        data['labels'] = torch.roll(data['labels'], -1, dims=-1)
        if 'loss_scale' in data:
            data['loss_scale'] = torch.roll(data['loss_scale'], -1, dims=-1)
    batch = to_device(data, get_current_device(), non_blocking=True)
    if args.pipeline_model_parallel_size == 1:
        return batch
    is_pp_last_stage = mpu.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage)
    if not is_pp_last_stage:
        batch['labels'] = None
        if 'loss_scale' in batch:
            batch['loss_scale'] = None

    return batch


def gather(tensor, group: Optional[torch.distributed.ProcessGroup] = None):
    if group is None:
        return hf_gather(tensor)
    size = torch.distributed.get_world_size(group=group)
    output = [torch.empty_like(tensor) for _ in range(size)]
    torch.distributed.all_gather(output, tensor, group=group, async_op=False)

    return torch.cat(output, dim=0)


def gather_object(object: Any, group: Optional[torch.distributed.ProcessGroup] = None):
    if group is None:
        return hf_gather_object(object)
    size = torch.distributed.get_world_size(group=group)
    output_objects = [None for _ in range(size)]
    torch.distributed.all_gather_object(output_objects, object, group=group)
    return [x for y in output_objects for x in y]


# code borrowed from verl
@torch.no_grad()
def load_megatron_model_to_gpu(models, load_grad=True, load_frozen_params=True):
    for model_chunk in models:
        if isinstance(model_chunk, DDP):
            model_chunk_all_buffers = [model_chunk.buffers, model_chunk.expert_parallel_buffers]
            for buffers in model_chunk_all_buffers:
                for buffer in buffers:
                    # sometimes, we don't want to load grad for pure inference
                    if load_grad and hasattr(buffer, 'grad_data_size'):
                        current_storage_size = buffer.grad_data.storage().size()
                        if current_storage_size == 0 or current_storage_size == buffer.grad_data_size:
                            buffer.grad_data.storage().resize_(buffer.grad_data_size)
                            buffer.grad_data.zero_()
                        else:
                            # Non-standard layers (e.g. GatedDeltaNet) may have grad
                            # buffers with mismatched storage size; skip resize and
                            # zero in-place with current storage.
                            buffer.grad_data.zero_()

                    if buffer.param_data.storage().size() == 0:
                        buffer.param_data.storage().resize_(buffer.param_data_size)
                        # copy data from cpu to cuda
                        buffer.param_data.copy_(buffer.param_data.cpu_data, non_blocking=True)

            if load_frozen_params:
                device_id = get_current_device()
                for param in model_chunk.module.parameters():
                    if not param.requires_grad and param.device.type == 'cpu':
                        param.data = param.data.to(device_id, non_blocking=True)
        else:
            # we need this for ref module
            device_id = get_current_device()
            for _, param in model_chunk.named_parameters():
                param.data = param.data.to(device_id, non_blocking=True)
                if param.grad is not None:
                    param.grad = param.grad.to(device_id, non_blocking=True)
    gc.collect()
    empty_cache()


@torch.no_grad()
def offload_megatron_model_to_cpu(models):
    """
    In megatron, the model and optimizer storage are:
    - bf16 parameter data chunked in model parallel group
    - fp32 grad chunked in model parallel group
    - fp32 main_parameter chunked in model and dp group
    - fp32 optimizer state chunked in model and dp group

    When using LoRA, frozen base model parameters are NOT managed by DDP buffers.
    They must be offloaded separately via direct param iteration.
    """
    for model_chunk in models:
        if isinstance(model_chunk, DDP):
            model_chunk_all_buffers = [model_chunk.buffers, model_chunk.expert_parallel_buffers]
            for buffers in model_chunk_all_buffers:
                for buffer in buffers:
                    # offload parameters
                    if buffer.param_data.storage().size() > 0:
                        existing = getattr(buffer.param_data, 'cpu_data', None)
                        if existing is None:
                            buffer.param_data.cpu_data = torch.empty(
                                buffer.param_data.size(),
                                dtype=buffer.param_data.dtype,
                                device='cpu',
                                pin_memory=True,
                            )
                            buffer.param_data_size = buffer.param_data.storage().size()
                        else:
                            assert existing.shape == buffer.param_data.shape, (
                                f'cpu_data shape {tuple(existing.shape)} != '
                                f'param_data shape {tuple(buffer.param_data.shape)}; '
                                'reallocating would reintroduce the 2x peak.')
                            assert existing.dtype == buffer.param_data.dtype, (
                                f'cpu_data dtype {existing.dtype} != '
                                f'param_data dtype {buffer.param_data.dtype}; '
                                'reallocating would reintroduce the 2x peak.')
                        # Synchronous D2H copy into the preexisting pinned
                        # buffer; must complete before resize_(0) frees the
                        # GPU storage.
                        buffer.param_data.cpu_data.copy_(buffer.param_data.data, non_blocking=False)
                        buffer.param_data.storage().resize_(0)

                    assert buffer.param_data_size == buffer.param_data.cpu_data.storage().size()

                    if buffer.grad_data.storage().size() > 0:
                        # if the grad_data size is already zero, we assume that it is already offloaded
                        buffer.grad_data_size = buffer.grad_data.storage().size()
                        buffer.grad_data.storage().resize_(0)

            for param in model_chunk.module.parameters():
                if not param.requires_grad and param.device.type != 'cpu':
                    param.data = param.data.to('cpu', non_blocking=True)
        else:
            # we need this for ref module
            for _, param in model_chunk.named_parameters():
                param.data = param.data.to('cpu', non_blocking=True)
                if param.grad is not None:
                    param.grad = param.grad.to('cpu', non_blocking=True)
    gc.collect()
    empty_cache()


@torch.no_grad()
def load_megatron_copy_params(optimizers):
    """
    Load optimizer parameters back to GPU. Handles ChainedOptimizer.

    Args:
        optimizers: Optimizer or ChainedOptimizer instance.
    """

    def _iter_opts(opt):
        if isinstance(opt, ChainedOptimizer):
            return opt.chained_optimizers
        return [opt]

    def load_tensor_to_gpu(tensor):
        if tensor is None:
            return
        device_id = get_current_device()
        tensor.data = tensor.data.to(device_id, non_blocking=True)

    def load_group_to_gpu(group):
        if group is None:
            return

        if isinstance(group, list):
            for param_group in group:
                if isinstance(param_group, list):
                    for param in param_group:
                        load_tensor_to_gpu(param)
                else:
                    load_tensor_to_gpu(param_group)
        else:
            load_tensor_to_gpu(group)

    # Load all parameter groups to GPU for each underlying optimizer

    for _opt in _iter_opts(optimizers):
        if hasattr(_opt, 'shard_fp32_from_float16_groups'):
            load_group_to_gpu(_opt.shard_fp32_from_float16_groups)


@torch.no_grad()
def offload_megatron_copy_params(optimizers):
    """
    Offload optimizer parameters to CPU. Supports both Megatron optimizers
    and `ChainedOptimizer`, which wraps a list of underlying optimizers.

    Args:
        optimizers: The optimizer or ChainedOptimizer instance.
    """

    def _iter_opts(opt):
        if isinstance(opt, ChainedOptimizer):
            return opt.chained_optimizers
        return [opt]

    def offload_tensor_to_cpu(tensor):
        if tensor is None:
            return
        tensor.data = tensor.data.to('cpu', non_blocking=True)

    def offload_group_to_cpu(group):
        if group is None:
            return

        if isinstance(group, list):
            for param_group in group:
                if isinstance(param_group, list):
                    for param in param_group:
                        offload_tensor_to_cpu(param)
                else:
                    offload_tensor_to_cpu(param_group)
        else:
            offload_tensor_to_cpu(group)

    # Offload all parameter groups to CPU for each underlying optimizer

    for _opt in _iter_opts(optimizers):
        if hasattr(_opt, 'shard_fp32_from_float16_groups'):
            offload_group_to_cpu(_opt.shard_fp32_from_float16_groups)


@torch.no_grad()
def load_megatron_optimizer(optimizers):

    def _iter_opts(opt):
        if isinstance(opt, ChainedOptimizer):
            return opt.chained_optimizers
        return [opt]

    for _opt in _iter_opts(optimizers):
        load_megatron_copy_params(_opt)
        if _opt.optimizer is not None:
            # if we are using HybridDeviceOptimizer, we need to only move gpu optimizer state to gpu
            if hasattr(_opt.optimizer, '_move_new_state_to_right_device'):
                _opt.optimizer._move_new_state_to_right_device()
            else:
                opt_state_dict_values = _opt.optimizer.state.values()
                for v in opt_state_dict_values:
                    if 'exp_avg' in v:
                        v['exp_avg'] = v['exp_avg'].to(get_current_device(), non_blocking=True)
                    if 'exp_avg_sq' in v:
                        v['exp_avg_sq'] = v['exp_avg_sq'].to(get_current_device(), non_blocking=True)
        gc.collect()
        empty_cache()


@torch.no_grad()
def offload_megatron_optimizer(optimizers):

    def _iter_opts(opt):
        if isinstance(opt, ChainedOptimizer):
            return opt.chained_optimizers
        return [opt]

    for _opt in _iter_opts(optimizers):
        offload_megatron_copy_params(_opt)
        # worker may hold zero parameter when enabling custom pipeline layout
        if _opt.optimizer is not None:
            # HybridDeviceOptimizer: offload all sub-optimizer states to CPU
            hdo = _opt.optimizer
            if all(hasattr(hdo, attr) for attr in ('sub_optimizers', 'inner_param_to_orig_param', 'state')):
                for optimizer in hdo.sub_optimizers:
                    for param, state in optimizer.state.items():
                        for k, v in state.items():
                            if not isinstance(v, torch.Tensor):
                                continue
                            orig_param = hdo.inner_param_to_orig_param.get(param, param)
                            hdo.state[orig_param][k] = state[k] = v.to('cpu')
            else:
                opt_state_dict_values = _opt.optimizer.state.values()
                for v in opt_state_dict_values:
                    if 'exp_avg' in v:
                        v['exp_avg'] = v['exp_avg'].to('cpu', non_blocking=True)
                    if 'exp_avg_sq' in v:
                        v['exp_avg_sq'] = v['exp_avg_sq'].to('cpu', non_blocking=True)
        gc.collect()
        empty_cache()


def log_gpu_memory(prefix: str = '', info_once: bool = False):
    log_msg = (f'{prefix} GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB allocated, '
               f'{torch.cuda.memory_reserved() / 1024**3:.2f}GB reserved')
    if info_once:
        logger.info_once(log_msg, hash_id=prefix)
    else:
        logger.info(log_msg)


@dataclass
class TrainerState:
    should_save: bool = False
    should_eval: bool = False
    should_log: bool = False

    iteration: int = 0
    consumed_train_samples: int = 0
    # compat transformers
    max_steps: Optional[int] = None

    best_metric: Optional[float] = None
    best_global_step: Optional[int] = None
    last_model_checkpoint: Optional[str] = None
    best_model_checkpoint: Optional[str] = None

    @property
    def global_step(self) -> int:
        return self.iteration


class MegatronDataLoaderDispatcher(DataLoaderDispatcher):

    @property
    def group(self):
        return mpu.get_data_parallel_group()


def build_streaming_dataloader(args, dataset, collate_fn):
    base_dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.dataloader_pin_memory,
        collate_fn=collate_fn,
        batch_size=args.micro_batch_size,
        prefetch_factor=args.dataloader_prefetch_factor if args.dataloader_num_workers > 0 else None,
        persistent_workers=args.dataloader_persistent_workers if args.dataloader_num_workers > 0 else False,
    )
    return MegatronDataLoaderDispatcher(base_dataloader)


_NPU_ATTENTION_MASK_2D_MODEL_TYPES = {'qwen3_5', 'qwen3_5_moe'}


def _should_use_npu_generated_attention_mask(args) -> bool:
    from transformers.utils import is_torch_npu_available
    if not is_torch_npu_available():
        return False
    if args.task_type != 'causal_lm' or args.padding_free:
        return False
    if getattr(args, 'attention_backend', None) == 'local':
        return False
    return bool(getattr(args, 'use_flash_attn', False))


def _prepare_npu_generated_attention_mask(batch, *, keep_attention_mask_2d: bool) -> None:
    if keep_attention_mask_2d:
        attention_mask = batch.get('attention_mask')
        if 'attention_mask_2d' not in batch and attention_mask is not None:
            batch['attention_mask_2d'] = (~attention_mask).sum(dim=(1, 2)) > 0
    else:
        batch.pop('attention_mask_2d', None)
    batch['attention_mask'] = None


def prepare_batch(args, data, vp_stage=None):
    """Prepare a micro-batch for Megatron forward: PP slicing, packed_seq_params, CP slicing.

    Extracted from BaseMegatronTrainer._prepare_batch for reuse in ray workers.
    """
    batch = get_batch_on_this_pp_rank(args, data, vp_stage=vp_stage)
    seq_lens = batch.pop('seq_lens', None)
    # Consider compatibility and security.
    num_samples = batch.pop('num_samples', None)
    if seq_lens is not None:
        if num_samples is not None:
            assert num_samples == len(seq_lens), (
                f"'num_samples' ({num_samples}) is inconsistent with len(seq_lens) ({len(seq_lens)}).")
        num_samples = len(seq_lens)
    text_position_ids = batch.pop('text_position_ids', None)
    if text_position_ids is None:
        text_position_ids = batch.get('position_ids')
    if _should_use_npu_generated_attention_mask(args):
        _prepare_npu_generated_attention_mask(
            batch, keep_attention_mask_2d=getattr(args, 'model_type', None) in _NPU_ATTENTION_MASK_2D_MODEL_TYPES)
    else:
        batch.pop('attention_mask_2d', None)
    if args.padding_free and text_position_ids is not None:
        batch['packed_seq_params'] = get_packed_seq_params(text_position_ids)
        if seq_lens is not None:
            batch['packed_seq_params'].seq_lens = torch.tensor(seq_lens, device=text_position_ids.device)
        if num_samples is not None:
            batch['packed_seq_params'].num_samples = num_samples
    batch = get_batch_on_this_cp_rank(args, batch)
    return batch


def compute_per_token_logps_fn(model, args, data_iterator, temperature=1.0, no_grad=True, enable_routing_replay=False):
    """Forward pass → logits → temperature-scaled per-token logps.

    Returns:
        (per_token_logps, routing_topk_idx) — either may be None on non-last PP stages.
    """
    from swift.megatron.utils import (RouterReplayHelper, forward_step_helper, get_local_topk_idx_for_current_rank,
                                      get_router_replay_data, set_router_replay_data)
    from .vocab_parallel_utils import compute_logps_and_entropy_from_logits

    data = prepare_batch(args, next(data_iterator))
    data.pop('loss_scale', None)
    labels = data.get('labels')

    routing_topk_idx = None
    global_topk_idx = data.pop('routed_experts', None)
    if enable_routing_replay and RouterReplayHelper.is_replay_forward_action(model.config):
        assert global_topk_idx is not None, 'When router_replay_mode = R3, routed_experts must be in data'
        routing_topk_idx = get_local_topk_idx_for_current_rank(global_topk_idx, model.config,
                                                               data.get('packed_seq_params'))
        set_router_replay_data(routing_topk_idx, model.config)

    data_for_forward = {k: v for k, v in data.items() if k != 'labels'}
    context = torch.no_grad() if no_grad else nullcontext()
    is_training = model.training
    if is_training:
        model.eval()
    try:
        with context:
            output_tensor = forward_step_helper(model, data_for_forward)
    finally:
        if is_training:
            model.train()

    if enable_routing_replay and RouterReplayHelper.is_r2_record_action(model.config):
        routing_topk_idx = get_router_replay_data(model.config)

    if labels is None or output_tensor is None:
        return None, routing_topk_idx

    if temperature != 1.0:
        output_tensor.div_(temperature)
    per_token_logps, _ = compute_logps_and_entropy_from_logits(output_tensor, labels)

    packed_seq_params = data.get('packed_seq_params')
    if packed_seq_params is not None:
        num_samples = packed_seq_params.seq_lens.shape[0]
    else:
        input_ids = data.get('input_ids')
        num_samples = input_ids.shape[0] if input_ids is not None else labels.shape[0]

    if args.context_parallel_size > 1:
        per_token_logps = reconstruct_tensor_cp(args.context_parallel_size, per_token_logps, packed_seq_params,
                                                num_samples)
    return per_token_logps, routing_topk_idx


def reconstruct_tensor_cp(cp_size, tensor, packed_seq_params, num_samples):
    """In CP mode, all_gather and reconstruct full tensor sequences."""
    cp_rank = mpu.get_context_parallel_rank()

    # All-gather across CP ranks
    output_list = [torch.empty_like(tensor) for _ in range(cp_size)]
    torch.distributed.all_gather(output_list, tensor.contiguous(), group=mpu.get_context_parallel_group())
    output_list[cp_rank] = tensor

    if packed_seq_params is not None:
        cu_seqlens_full = packed_seq_params.cu_seqlens_q
        cu_seqlens_cp = cu_seqlens_full // cp_size

        # Calculate total packed length
        total_packed_len = cu_seqlens_full[num_samples].item()
        output_full = tensor.new_zeros(1, total_packed_len)

        # Reconstruct each sequence
        for i in range(num_samples):
            start_full = cu_seqlens_full[i].item()
            end_full = cu_seqlens_full[i + 1].item()
            seq_len = end_full - start_full

            # Length of each chunk after CP split
            chunk_len = seq_len // cp_size
            half_chunk = chunk_len // 2

            # Concatenate from each CP rank's output (load-balanced split)
            for j in range(cp_size):
                o = output_list[j][0]
                start_cp = cu_seqlens_cp[i].item()
                o0 = o[start_cp:start_cp + half_chunk]
                o1 = o[start_cp + half_chunk:start_cp + chunk_len]

                # Place back to full sequence
                output_full[0, start_full + j * half_chunk:start_full + (j + 1) * half_chunk] = o0
                output_full[0, end_full - (j + 1) * half_chunk:end_full - j * half_chunk] = o1
    else:
        # non-padding_free mode: [batch_size, seq_len/cp_size] -> [batch_size, seq_len]
        # Each CP rank has chunks split with load-balanced pattern (2*cp_size chunks)
        batch_size = tensor.shape[0]
        seq_len_per_cp = tensor.shape[1]
        full_seq_len = seq_len_per_cp * cp_size
        output_full = tensor.new_zeros(batch_size, full_seq_len)

        # Each CP rank j holds chunks j and (2*cp_size - j - 1) from the original 2*cp_size split
        # Reconstruct the full sequence by placing chunks back in correct positions
        chunk_len = full_seq_len // (2 * cp_size)

        for j in range(cp_size):
            o = output_list[j]
            # This rank holds 2 chunks: chunk j and chunk (2*cp_size - j - 1)
            half_len = seq_len_per_cp // 2
            o0 = o[:, :half_len]
            o1 = o[:, half_len:]

            # Place chunk j at position j * chunk_len
            output_full[:, j * chunk_len:(j + 1) * chunk_len] = o0
            reverse_idx = 2 * cp_size - j - 1
            output_full[:, reverse_idx * chunk_len:(reverse_idx + 1) * chunk_len] = o1

    return output_full
