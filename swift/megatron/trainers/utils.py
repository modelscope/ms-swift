# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import torch
from megatron.core import mpu
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.utils import get_batch_on_this_cp_rank as mcore_get_batch_on_this_cp_rank
from megatron.training import get_args, get_timers
from megatron.training.training import (cuda_graph_capture, cuda_graph_set_manual_hooks,
                                        get_tensor_shapes_adjust_fn_for_distillation, has_nvidia_modelopt)
from megatron.training.utils import (logical_and_across_model_parallel_group,
                                     reduce_max_stat_across_model_parallel_group, unwrap_model)

from swift.llm import get_packed_seq_params as _get_packed_seq_params
from swift.llm import to_device
from swift.utils import get_current_device


def get_swift_datasets_provider(train_dataset, val_dataset):

    def swift_datasets_provider(train_val_test_num_samples):
        nonlocal val_dataset
        args = get_args()
        data_parallel_size = mpu.get_data_parallel_world_size()
        step_batch_size = args.micro_batch_size * data_parallel_size
        # To avoid errors caused by the validation set being insufficient to complete a single step.
        if val_dataset is not None and hasattr(val_dataset, '__len__') and len(val_dataset) < step_batch_size:
            val_dataset = None
        return train_dataset, val_dataset, None

    return swift_datasets_provider


# Code borrowed from NVIDIA/Megatron-LM
def get_batch_on_this_tp_rank(data_iterator):
    args = get_args()

    data = next(data_iterator)
    is_finished = data.pop('is_finished', False)
    data['labels'] = torch.roll(data['labels'], -1, dims=-1)
    if 'loss_scale' in data:
        data['loss_scale'] = torch.roll(data['loss_scale'], -1, dims=-1)
    batch = to_device(data, 'cuda', non_blocking=True)
    if args.pipeline_model_parallel_size == 1:
        pass
    elif mpu.is_pipeline_first_stage():
        batch['labels'] = None
        batch['loss_scale'] = None
    elif mpu.is_pipeline_last_stage():
        batch['input_ids'] = None
    else:
        for key in ('input_ids', 'labels', 'loss_scale'):
            batch[key] = None

    if is_finished:
        args.train_iters = args.curr_iteration + 1

    return batch


def get_packed_seq_params(position_ids: torch.Tensor) -> PackedSeqParams:
    params = _get_packed_seq_params(position_ids)
    return PackedSeqParams(
        cu_seqlens_q=params['cu_seq_lens_q'],
        cu_seqlens_kv=params['cu_seq_lens_k'],
        max_seqlen_q=params['max_length_q'],
        max_seqlen_kv=params['max_length_k'],
        qkv_format='thd')


def _split_tokens(tokens, cu_seqlens):
    assert tokens.shape[-2] == 1, f'tokens.shape: {tokens.shape}'  # [..., 1, L]
    new_tokens = []
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
    for i in range(cu_seqlens.shape[0] - 1):
        val = tokens[..., cu_seqlens[i]:cu_seqlens[i + 1]]
        val = val.view(
            *tokens.shape[:-1],
            2 * cp_size,
            val.shape[-1] // (2 * cp_size),
        )
        index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device='cpu',
                             pin_memory=True).cuda(non_blocking=True)
        val = val.index_select(-2, index)
        new_tokens.append(val.view(*tokens.shape[:-1], -1))
    return torch.cat(new_tokens, dim=-1)


def _split_tokens_decoder_input(tokens, cu_seqlens):
    assert tokens.shape[1] == 1, f'tokens.shape: {tokens.shape}'  # [L, 1, E]
    new_tokens = []
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
    for i in range(cu_seqlens.shape[0] - 1):
        val = tokens[cu_seqlens[i]:cu_seqlens[i + 1], ...]
        val = val.view(
            2 * cp_size,
            val.shape[0] // (2 * cp_size),
            *tokens.shape[1:],
        )
        index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device='cpu',
                             pin_memory=True).cuda(non_blocking=True)
        val = val.index_select(0, index)
        new_tokens.append(val.view(-1, *tokens.shape[1:]))
    return torch.cat(new_tokens, dim=0)


def get_batch_on_this_cp_rank(batch: Dict[str, Any]):
    """Slice batch input along sequence dimension into multiple chunks,
    which are parallelized across GPUs in a context parallel group.
    """

    # With causal masking, each token only attends to its prior tokens. Simply split
    # sequence into CP chunks can result in severe load imbalance. That's to say, chunks
    # at the end of sequence have bigger workload than others. To address this issue,
    # we split sequence into 2*CP ranks. Assuming CP=2, we then get 4 chunks, chunk_0
    # and chunk_3 are assigned to GPU0, chunk_1 and chunk_2 are assigned to GPU1, so
    # that we can get balanced workload among GPUs in a context parallel group.
    cp_size = mpu.get_context_parallel_world_size()
    if cp_size > 1:
        args = get_args()
        keys = ['labels', 'attention_mask', 'position_ids', 'loss_scale']
        if args.model_meta.is_multimodal:
            keys.append('decoder_input')
        else:
            keys.append('input_ids')
        packed_seq_params = batch.get('packed_seq_params')
        if packed_seq_params is None:
            return mcore_get_batch_on_this_cp_rank(batch)
        for key, val in batch.items():
            if key not in keys:
                continue
            if val is not None:
                if key == 'decoder_input':
                    batch[key] = _split_tokens_decoder_input(val, packed_seq_params.cu_seqlens_q)
                else:
                    batch[key] = _split_tokens(val, packed_seq_params.cu_seqlens_q)

    return batch


def get_batch(data_iterator):
    """Generate a batch."""
    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)
    args = get_args()
    num_samples = batch.pop('num_samples')
    text_position_ids = batch.pop('text_position_ids', None)
    if text_position_ids is None:
        text_position_ids = batch.get('position_ids')
    if args.padding_free and text_position_ids is not None:
        batch['packed_seq_params'] = get_packed_seq_params(text_position_ids)
        batch['packed_seq_params'].num_samples = num_samples
    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)
    return batch


def train_step(forward_step_func, data_iterator, model, optimizer, opt_param_scheduler, config):
    # borrowed from Megatron-LM 0.13.rc2
    """Single training step."""
    args = get_args()
    timers = get_timers()

    # CUDA Graph capturing only executes once, when it's the first training iteration.
    if args.curr_iteration == args.iteration and args.external_cuda_graph:
        cuda_graph_capture(model, config, args)

        # Set grad to zero.
        for model_chunk in model:
            model_chunk.zero_grad_buffer()
        optimizer.zero_grad()

        # Collect garbage and empty unused memory.
        gc.collect()
        torch.cuda.empty_cache()

    rerun_state_machine = get_rerun_state_machine()
    while rerun_state_machine.should_run_forward_backward(data_iterator):
        # Set grad to zero.
        for model_chunk in model:
            model_chunk.zero_grad_buffer()
        optimizer.zero_grad()

        if has_nvidia_modelopt:
            # [ModelOpt]: Pipeline-parallel Distillation stacks student and teacher tensors
            adjust_tensor_shapes_fn = get_tensor_shapes_adjust_fn_for_distillation(model, args.seq_length,
                                                                                   args.micro_batch_size,
                                                                                   args.decoder_seq_length)
        else:
            adjust_tensor_shapes_fn = None

        # Forward pass.
        forward_backward_func = get_forward_backward_func()
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=get_num_microbatches(),
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=args.decoder_seq_length,
            forward_only=False,
            adjust_tensor_shapes_fn=adjust_tensor_shapes_fn,
        )
    should_checkpoint, should_exit, exit_code = rerun_state_machine.should_checkpoint_and_exit()
    if should_exit:
        return {}, True, should_checkpoint, should_exit, exit_code, None, None

    # Empty unused memory.
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    # Vision gradients.
    if args.vision_pretraining and args.vision_pretraining_type == 'dino':
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.cancel_gradients_last_layer(args.curr_iteration)

    # Update parameters.

    timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
    timers('optimizer').stop()

    # when freezing sub-models we may have a mixture of successful and unsucessful ranks,
    # so we must gather across mp ranks
    update_successful = logical_and_across_model_parallel_group(update_successful)
    # grad_norm and num_zeros_in_grad will be None on ranks without trainable params,
    # so we must gather across mp ranks
    grad_norm = reduce_max_stat_across_model_parallel_group(grad_norm)
    if args.log_num_zeros_in_grad:
        num_zeros_in_grad = reduce_max_stat_across_model_parallel_group(num_zeros_in_grad)

    # Vision momentum.
    if args.vision_pretraining and args.vision_pretraining_type == 'dino':
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.update_momentum(args.curr_iteration)

    # Update learning rate.
    if update_successful:
        increment = get_num_microbatches() * args.micro_batch_size * args.data_parallel_size
        opt_param_scheduler.step(increment=increment)
        skipped_iter = 0
    else:
        skipped_iter = 1

    # Empty unused memory.
    if args.empty_unused_memory_level >= 2:
        torch.cuda.empty_cache()

    # Set the manual hooks when CUDA Graphs are enabled.
    if args.curr_iteration == args.iteration and args.external_cuda_graph:
        if args.use_distributed_optimizer and args.overlap_param_gather:
            cuda_graph_set_manual_hooks(model)

    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        # Average loss across microbatches.
        loss_reduced = {}

        for key in losses_reduced[0].keys():
            val = [x[key].view(-1) for x in losses_reduced]
            if val[0].numel() == 2:
                if args.sft:
                    # in mcore the normalization happens on micro batch instead of global
                    val = torch.vstack(val)
                    val = val[:, 0] / val[:, 1]
                    val = val.mean()
                    torch.distributed.all_reduce(val, group=mpu.get_data_parallel_group(with_context_parallel=True))
                    val /= torch.distributed.get_world_size(
                        group=mpu.get_data_parallel_group(with_context_parallel=True))
                    loss_reduced[key] = val
                else:
                    # there is one dict per microbatch. in new reporting, we average
                    # over the total number of tokens across the global batch.
                    val = torch.vstack(val).sum(dim=0)
                    torch.distributed.all_reduce(val, group=mpu.get_data_parallel_group(with_context_parallel=True))
                    loss_reduced[key] = val[0] / val[1]
            elif val[0].numel() == 1:
                # legacy behavior, we average over the number of microbatches
                val = torch.cat(val).mean()
                loss_reduced[key] = val
            else:
                raise ValueError(f'Invalid value shape: {val[0].shape} for key {key}')
        return (
            loss_reduced,
            skipped_iter,
            should_checkpoint,
            should_exit,
            exit_code,
            grad_norm,
            num_zeros_in_grad,
        )
    return {}, skipped_iter, should_checkpoint, should_exit, exit_code, grad_norm, num_zeros_in_grad
