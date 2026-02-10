# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
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
