# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Optional

import torch
from megatron.core import mpu


def reduce_max_stat_across_model_parallel_group(stat: float) -> float:
    """
    Ranks without an optimizer will have no grad_norm or num_zeros_in_grad stats.
    We need to ensure the logging and writer rank has those values.
    This function reduces a stat tensor across the model parallel group.

    We use an all_reduce max since the values have already been summed across optimizer ranks where possible
    """
    stat = torch.tensor([stat], dtype=torch.float32, device=torch.cuda.current_device())
    torch.distributed.all_reduce(stat, op=torch.distributed.ReduceOp.MAX, group=mpu.get_model_parallel_group())
    return stat.item()


def logical_and_across_model_parallel_group(input: bool) -> bool:
    """
    This function gathers a bool value across the model parallel group
    """
    input = int(bool(input))
    input = torch.tensor([input], dtype=torch.int, device=torch.cuda.current_device())
    torch.distributed.all_reduce(input, op=torch.distributed.ReduceOp.MIN, group=mpu.get_model_parallel_group())
    return bool(input.item())


def split_cp_inputs(inputs: torch.Tensor, cu_seqlens: Optional[torch.Tensor], dim: int):
    if dim < 0:
        dim = (dim + inputs.ndim) % inputs.ndim
    new_inputs = []
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
    for i in range(1 if cu_seqlens is None else (cu_seqlens.shape[0] - 1)):
        if cu_seqlens is None:
            val = inputs
        else:
            slices = [slice(None)] * inputs.ndim
            slices[dim] = slice(cu_seqlens[i], cu_seqlens[i + 1])
            val = inputs[tuple(slices)]
        view_shape = (*inputs.shape[:dim], 2 * cp_size, val.shape[dim] // (2 * cp_size), *inputs.shape[dim + 1:])
        val = val.view(view_shape)
        index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device='cpu',
                             pin_memory=True).cuda(non_blocking=True)
        val = val.index_select(dim, index)
        view_shape = (*inputs.shape[:dim], -1, *inputs.shape[dim + 1:])
        new_inputs.append(val.view(view_shape))
    return torch.cat(new_inputs, dim=dim)
