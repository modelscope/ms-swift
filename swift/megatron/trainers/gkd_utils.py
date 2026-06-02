# Copyright (c) ModelScope Contributors. All rights reserved.
"""Pure functions for GKD loss computation, extracted from MegatronGKDTrainer."""
import torch
import torch.nn.functional as F
from megatron.core import mpu
from typing import Optional

from .vocab_parallel_utils import vocab_parallel_kl_div, vocab_parallel_log_softmax


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


def jsd_topk(student_logits, teacher_topk_logprobs, teacher_topk_indices, mask, beta, temperature):
    """JSD on teacher's top-k distribution."""
    s_topk = tp_gather_topk(student_logits, teacher_topk_indices)
    s_topk.div_(temperature)
    t_topk = teacher_topk_logprobs / temperature

    s_topk_masked = s_topk[mask]
    t_topk_masked = t_topk[mask]

    if s_topk_masked.numel() == 0:
        return student_logits.new_zeros(())

    t_log_p = F.log_softmax(t_topk_masked, dim=-1)
    s_log_p = F.log_softmax(s_topk_masked, dim=-1)
    t_p = torch.exp(t_log_p)

    if beta == 0:
        jsd = (t_p * (t_log_p - s_log_p)).sum(dim=-1)
    elif beta == 1:
        s_p = torch.exp(s_log_p)
        jsd = (s_p * (s_log_p - t_log_p)).sum(dim=-1)
    else:
        s_p = torch.exp(s_log_p)
        m_log_p = torch.log(beta * t_p + (1 - beta) * s_p + 1e-10)
        jsd = beta * (t_p * (t_log_p - m_log_p)).sum(-1) + (1 - beta) * (s_p * (s_log_p - m_log_p)).sum(-1)

    return jsd.sum()


def generalized_jsd_loss(
    student_logits: torch.Tensor,
    teacher_logits: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    beta: float = 0.5,
    temperature: float = 1.0,
    chunk_size: int = 512,
    teacher_topk_logprobs: Optional[torch.Tensor] = None,
    teacher_topk_indices: Optional[torch.Tensor] = None,
    cp_size: int = 1,
) -> torch.Tensor:
    if labels is not None:
        mask = labels != -100
        local_num_valid = mask.sum()
    else:
        mask = None
        local_num_valid = torch.tensor(
            student_logits.shape[0] * student_logits.shape[1], dtype=torch.long, device=student_logits.device)
    num_valid = local_num_valid.float()

    if cp_size > 1:
        torch.distributed.all_reduce(
            num_valid, op=torch.distributed.ReduceOp.SUM, group=mpu.get_context_parallel_group())

    if num_valid == 0:
        return (student_logits.sum() * 0).reshape(())

    if teacher_topk_logprobs is not None and teacher_topk_indices is not None:
        if mask is None:
            mask = torch.ones(student_logits.shape[:2], dtype=torch.bool, device=student_logits.device)
        total_loss = jsd_topk(student_logits, teacher_topk_logprobs, teacher_topk_indices, mask, beta, temperature)
        if cp_size > 1:
            torch.distributed.all_reduce(
                total_loss, op=torch.distributed.ReduceOp.SUM, group=mpu.get_context_parallel_group())
        return total_loss / num_valid

    student_logits, teacher_logits = align_vocab_size(student_logits, teacher_logits)

    if mask is not None:
        student_logits_masked = student_logits[mask]
        teacher_logits_masked = teacher_logits[mask]
    else:
        student_logits_masked = student_logits.view(-1, student_logits.size(-1))
        teacher_logits_masked = teacher_logits.view(-1, teacher_logits.size(-1))
    del student_logits, teacher_logits
    student_logits_masked.div_(temperature)
    teacher_logits_masked.div_(temperature)

    local_num_valid_int = local_num_valid.item()
    total_loss = student_logits_masked.new_zeros(())

    if beta != 0 and beta != 1:
        beta_t = torch.tensor(beta, dtype=student_logits_masked.dtype, device=student_logits_masked.device)
        log_beta = torch.log(beta_t)
        log_1_minus_beta = torch.log1p(-beta_t)
    else:
        beta_t = log_beta = log_1_minus_beta = None

    for start_idx in range(0, local_num_valid_int, chunk_size):
        end_idx = min(start_idx + chunk_size, local_num_valid_int)
        s_chunk = student_logits_masked[start_idx:end_idx]
        t_chunk = teacher_logits_masked[start_idx:end_idx]

        s_log_probs = vocab_parallel_log_softmax(s_chunk)
        t_log_probs = vocab_parallel_log_softmax(t_chunk)
        del s_chunk, t_chunk

        if beta == 0:
            jsd_chunk = vocab_parallel_kl_div(s_log_probs, t_log_probs)
        elif beta == 1:
            jsd_chunk = vocab_parallel_kl_div(t_log_probs, s_log_probs)
        else:
            mixture_log_probs = torch.logsumexp(
                torch.stack([s_log_probs + log_1_minus_beta, t_log_probs + log_beta]),
                dim=0,
            )
            kl_teacher = vocab_parallel_kl_div(mixture_log_probs, t_log_probs)
            kl_student = vocab_parallel_kl_div(mixture_log_probs, s_log_probs)
            del mixture_log_probs
            jsd_chunk = beta_t * kl_teacher + (1 - beta_t) * kl_student
            del kl_teacher, kl_student

        total_loss = total_loss + jsd_chunk.sum()
        del jsd_chunk, s_log_probs, t_log_probs

    del student_logits_masked, teacher_logits_masked

    if cp_size > 1:
        torch.distributed.all_reduce(
            total_loss, op=torch.distributed.ReduceOp.SUM, group=mpu.get_context_parallel_group())

    return total_loss / num_valid
