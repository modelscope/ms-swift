# Copyright (c) ModelScope Contributors. All rights reserved.
"""Megatron-specific GKD utilities: TP-aware gather/topk, CP reduce and teacher CP slicing."""
import torch
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from mcore_bridge import split_cp_inputs
from megatron.core import mpu
from megatron.core.tensor_parallel.mappings import (copy_to_tensor_model_parallel_region,
                                                    gather_from_sequence_parallel_region)

from swift.rlhf_trainers.gkd_loss import TeacherOutput, jsd_loss
from .vocab_parallel_utils import vocab_parallel_kl_div, vocab_parallel_log_softmax


@dataclass
class TeacherHiddenStates:
    hidden_states: torch.Tensor
    labels: torch.Tensor


@contextmanager
def gkd_hidden_states_context(model, enabled=True):
    """Skip only the final projection; keep the teacher/student PP forwards intact."""
    model = getattr(model, 'language_model', model)
    if not enabled or not model.post_process:
        yield
        return
    # Still call the module so distributed-optimizer parameter-gather hooks run.
    layer = model.output_layer
    had_override = 'forward' in layer.__dict__
    original = layer.forward
    layer.forward = lambda hidden_states, *args, **kwargs: (hidden_states, None)
    try:
        yield
    finally:
        if had_override:
            layer.forward = original
        else:
            del layer.forward


@lru_cache(maxsize=1)
def _get_liger_jsd():
    from liger_kernel.chunked_loss.jsd_loss import LigerFusedLinearJSDFunction

    class VocabParallelJSD(LigerFusedLinearJSDFunction):

        @staticmethod
        def distillation_loss_fn(student_logits, teacher_logits, beta=0.5, target=None, ignore_index=-100):
            dtype = torch.promote_types(student_logits.dtype, torch.float32)
            return jsd_loss(
                student_logits.to(dtype), teacher_logits.to(dtype), beta, vocab_parallel_log_softmax,
                vocab_parallel_kl_div)

    return VocabParallelJSD


def chunked_gkd_loss(student, teacher_output, labels, student_model, teacher_model, beta, temperature, chunk_size):
    """Full-vocabulary JSD from hidden states, with TP/SP gradient routing and CP-local masks."""
    student_model = getattr(student_model, 'language_model', student_model)
    teacher_model = getattr(teacher_model, 'language_model', teacher_model)

    def prepare_hidden(hidden, model):
        # The model's postprocess returns [batch, sequence / SP, hidden].
        if mpu.get_tensor_model_parallel_world_size() > 1:
            if model.config.sequence_parallel:
                hidden = gather_from_sequence_parallel_region(hidden.transpose(0, 1).contiguous())
                hidden = hidden.transpose(0, 1).contiguous()
            else:
                # Keep the backward input contiguous for M-Core's in-place TP reduction.
                hidden = copy_to_tensor_model_parallel_region(hidden)
        return hidden

    def output_weight(model):
        if model.share_embeddings_and_output_weights:
            return model.shared_embedding_or_output_weight()
        return model.output_layer.weight

    student = prepare_hidden(student, student_model)
    with torch.no_grad():
        teacher = prepare_hidden(teacher_output.hidden_states, teacher_model)
    s_mask, t_mask = labels != -100, teacher_output.labels != -100
    num_valid = s_mask.sum()
    # Inputs are already masked. Liger uses these dummy labels only for normalization (hard-loss weight is zero).
    targets = torch.zeros_like(labels[s_mask])
    loss = _get_liger_jsd().apply(student[s_mask], output_weight(student_model), teacher[t_mask],
                                  output_weight(teacher_model), targets, student_model.output_layer.bias,
                                  teacher_model.output_layer.bias, 0., 1., beta, -100, temperature, False, chunk_size,
                                  False)
    # Liger returns a token mean; Megatron normalizes the summed loss across micro-batches and CP/DP.
    return loss * num_valid, num_valid


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


def cp_slice_teacher_output(teacher_output: TeacherOutput,
                            packed_seq_params=None,
                            cp_partition_mode: str = 'zigzag') -> TeacherOutput:
    cp_size = mpu.get_context_parallel_world_size()
    if cp_size == 1 or teacher_output.labels is None:
        return teacher_output
    cu_seqlens = getattr(packed_seq_params, 'cu_seqlens_q', None)
    kwargs = {}
    if cp_partition_mode == 'contiguous':
        kwargs['cp_partition_mode'] = 'contiguous'

    def _slice(x, dim):
        if x is None:
            return None
        return split_cp_inputs(x, cu_seqlens, dim, **kwargs)

    return TeacherOutput(
        full_logits=_slice(teacher_output.full_logits, 1),
        topk_logprobs=_slice(teacher_output.topk_logprobs, 1),
        topk_indices=_slice(teacher_output.topk_indices, 1),
        labels=_slice(teacher_output.labels, 1),
    )


def cp_reduce(total_loss, num_valid, *, cp_size):
    """Normalize total_loss by num_valid with CP all-reduce when cp_size > 1."""
    num_valid_f = num_valid.float() if isinstance(num_valid, torch.Tensor) else torch.tensor(
        float(num_valid), device=total_loss.device)
    if cp_size > 1:
        torch.distributed.all_reduce(
            num_valid_f, op=torch.distributed.ReduceOp.SUM, group=mpu.get_context_parallel_group())
        torch.distributed.all_reduce(
            total_loss, op=torch.distributed.ReduceOp.SUM, group=mpu.get_context_parallel_group())
    # Avoid the host-device sync from ``if num_valid_f == 0`` (num_valid_f is a GPU scalar,
    # so a Python bool() forces a .item() stream sync every step). Keep the zero-valid
    # guard on-device with torch.where: num_valid==0 -> loss 0, else total_loss/num_valid.
    safe = torch.where(num_valid_f == 0, torch.ones_like(num_valid_f), num_valid_f)
    loss = total_loss / safe
    return torch.where(num_valid_f == 0, torch.zeros_like(loss), loss)
