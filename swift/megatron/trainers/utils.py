# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import torch
from megatron.core import mpu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import get_batch_on_this_cp_rank as mcore_get_batch_on_this_cp_rank
from megatron.training import get_args

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
    position_ids = batch['position_ids']
    if position_ids.ndim == 3:
        text_position_ids = position_ids[0]
        batch['position_ids'] = position_ids[1:]
    else:
        text_position_ids = position_ids
    if args.padding_free and text_position_ids is not None:
        batch['packed_seq_params'] = get_packed_seq_params(text_position_ids)
        batch['packed_seq_params'].num_samples = num_samples
    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)
    return batch
