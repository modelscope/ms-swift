# Copyright (c) ModelScope Contributors. All rights reserved.
"""Pure functions for GKD loss computation, extracted from MegatronGKDTrainer."""
import torch
import torch.nn.functional as F
from megatron.core import mpu


def align_vocab_size(student_logits, teacher_logits):
    stu_vocab = student_logits.shape[-1]
    tea_vocab = teacher_logits.shape[-1]
    if stu_vocab == tea_vocab:
        return student_logits, teacher_logits
    if stu_vocab < tea_vocab:
        student_logits = F.pad(student_logits, (0, tea_vocab - stu_vocab), 'constant', 0)
        student_logits[..., stu_vocab:] = teacher_logits[..., stu_vocab:]
    else:
        teacher_logits = F.pad(teacher_logits, (0, stu_vocab - tea_vocab), 'constant', 0)
        teacher_logits[..., tea_vocab:] = student_logits[..., tea_vocab:]
    return student_logits, teacher_logits


def vocab_parallel_topk(logits: torch.Tensor, k: int) -> tuple:
    """Global top-k from vocab-parallel sharded logits. TP=1 → plain torch.topk."""
    tp_size = mpu.get_tensor_model_parallel_world_size()
    if tp_size == 1:
        return torch.topk(logits, k=k, dim=-1)

    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp_group = mpu.get_tensor_model_parallel_group()
    partition_vocab_size = logits.shape[-1]

    local_topk_vals, local_topk_ids = torch.topk(logits, k=k, dim=-1)
    local_topk_ids = local_topk_ids + tp_rank * partition_vocab_size

    gathered_vals = [torch.empty_like(local_topk_vals) for _ in range(tp_size)]
    gathered_ids = [torch.empty_like(local_topk_ids) for _ in range(tp_size)]
    torch.distributed.all_gather(gathered_vals, local_topk_vals, group=tp_group)
    torch.distributed.all_gather(gathered_ids, local_topk_ids, group=tp_group)

    all_vals = torch.cat(gathered_vals, dim=-1)
    all_ids = torch.cat(gathered_ids, dim=-1)
    global_topk_vals, sel = torch.topk(all_vals, k=k, dim=-1)
    global_topk_ids = torch.gather(all_ids, dim=-1, index=sel)
    return global_topk_vals, global_topk_ids


def tp_gather_topk(logits: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather logits at global top-k indices with TP-aware partitioning."""
    tp_size = mpu.get_tensor_model_parallel_world_size()
    if tp_size == 1:
        return torch.gather(logits, dim=-1, index=indices)

    tp_rank = mpu.get_tensor_model_parallel_rank()
    partition_vocab_size = logits.shape[-1]
    vocab_start = tp_rank * partition_vocab_size

    in_range = (indices >= vocab_start) & (indices < vocab_start + partition_vocab_size)
    local_indices = (indices - vocab_start).clamp(0, partition_vocab_size - 1)
    gathered = torch.gather(logits, dim=-1, index=local_indices)
    gathered = gathered.masked_fill(~in_range, float('-inf'))

    gathered_for_reduce = gathered.detach()
    torch.distributed.all_reduce(
        gathered_for_reduce, op=torch.distributed.ReduceOp.MAX, group=mpu.get_tensor_model_parallel_group())
    return torch.where(in_range, gathered, gathered_for_reduce)


def cp_reduce(total_loss, num_valid, *, cp_size):
    """Normalize total_loss by num_valid with CP all-reduce when cp_size > 1."""
    num_valid_f = num_valid.float() if isinstance(num_valid, torch.Tensor) else torch.tensor(
        float(num_valid), device=total_loss.device)
    if cp_size > 1:
        torch.distributed.all_reduce(
            num_valid_f, op=torch.distributed.ReduceOp.SUM, group=mpu.get_context_parallel_group())
        torch.distributed.all_reduce(
            total_loss, op=torch.distributed.ReduceOp.SUM, group=mpu.get_context_parallel_group())
    if num_valid_f == 0:
        return total_loss * 0
    return total_loss / num_valid_f
