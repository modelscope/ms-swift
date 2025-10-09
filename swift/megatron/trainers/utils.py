# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import torch
from megatron.core import mpu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import get_batch_on_this_cp_rank as mcore_get_batch_on_this_cp_rank
from megatron.training import get_args

from swift.llm import get_packed_seq_params as _get_packed_seq_params
from swift.llm import to_device


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
def get_batch_on_this_tp_rank(data, vp_stage=None):
    args = get_args()

    if args.task_type == 'causal_lm':
        data['labels'] = torch.roll(data['labels'], -1, dims=-1)
        if 'loss_scale' in data:
            data['loss_scale'] = torch.roll(data['loss_scale'], -1, dims=-1)
    batch = to_device(data, 'cuda', non_blocking=True)
    if args.pipeline_model_parallel_size == 1:
        return batch
    if not mpu.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp_stage):
        batch['input_ids'] = None
    if not mpu.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage):
        batch['labels'] = None
        batch['loss_scale'] = None

    return batch


def get_packed_seq_params(position_ids: torch.Tensor) -> PackedSeqParams:
    params = _get_packed_seq_params(position_ids)
    return PackedSeqParams(
        cu_seqlens_q=params['cu_seq_lens_q'],
        cu_seqlens_kv=params['cu_seq_lens_k'],
        max_seqlen_q=params['max_length_q'],
        max_seqlen_kv=params['max_length_k'],
        qkv_format='thd')


def split_cp_inputs(inputs: torch.Tensor, cu_seqlens: torch.Tensor, dim: int):
    if dim < 0:
        dim = (dim + inputs.ndim) % inputs.ndim
    new_inputs = []
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
    for i in range(cu_seqlens.shape[0] - 1):
        slices = [slice(None)] * inputs.ndim
        slices[dim] = slice(cu_seqlens[i], cu_seqlens[i + 1])
        val = inputs[slices]
        view_shape = (*inputs.shape[:dim], 2 * cp_size, val.shape[dim] // (2 * cp_size), *inputs.shape[dim + 1:])
        val = val.view(view_shape)
        index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device='cpu',
                             pin_memory=True).cuda(non_blocking=True)
        val = val.index_select(dim, index)
        view_shape = (*inputs.shape[:dim], -1, *inputs.shape[dim + 1:])
        new_inputs.append(val.view(view_shape))
    return torch.cat(new_inputs, dim=dim)


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
            if args.task_type == 'seq_cls' and key == 'labels':
                continue
            if val is not None:
                batch[key] = split_cp_inputs(val, packed_seq_params.cu_seqlens_q, -1)

    return batch
