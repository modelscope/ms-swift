# Copyright (c) ModelScope Contributors. All rights reserved.
"""Megatron-specific GKD utilities: TP-aware gather/topk, CP reduce and teacher CP slicing."""
import torch
import torch.nn.functional as F
from contextlib import contextmanager
from dataclasses import dataclass
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


def get_gkd_language_model(model):
    model = getattr(model, 'language_model', model)
    if model.post_process:
        from mcore_bridge.model.gpt_model import GPTModel
        from megatron.core.tensor_parallel.layers import ColumnParallelLinear

        if (type(model.output_layer) is not ColumnParallelLinear or getattr(
                getattr(model, '_forward_output_layer', None), '__func__', None) is not GPTModel._forward_output_layer):
            raise ValueError('gkd_loss_chunk_size requires an unmodified mcore-bridge output projection '
                             '(output-layer adapters and custom logit transformations are unsupported)')
        if model.config.defer_embedding_wgrad_compute:
            raise ValueError('gkd_loss_chunk_size does not support defer_embedding_wgrad_compute')
    return model


@contextmanager
def gkd_hidden_states_context(model):
    """Skip only the final projection; keep the teacher/student PP forwards intact."""
    model = get_gkd_language_model(model)
    if not model.post_process:
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


class _ChunkedJSD(torch.autograd.Function):
    """Like fused linear distillation losses, retain gradients instead of [tokens, vocab] activations."""

    @staticmethod
    def forward(ctx, student, weight, bias, teacher, teacher_weight, teacher_bias, beta, temperature, chunk_size):
        loss_dtype = torch.promote_types(student.dtype, torch.float32)
        grad_student = torch.empty_like(student)
        grad_weight = torch.zeros_like(weight, dtype=loss_dtype) if weight.requires_grad else None
        grad_bias = torch.zeros_like(bias, dtype=loss_dtype) if bias is not None and bias.requires_grad else None
        total = torch.zeros((), dtype=loss_dtype, device=student.device)
        with torch.enable_grad():
            w = weight.detach().requires_grad_(weight.requires_grad)
            b = bias.detach().requires_grad_(bias.requires_grad) if bias is not None else None
            for start in range(0, student.shape[0], chunk_size):
                end = start + chunk_size
                s = student[start:end].detach().requires_grad_(True)
                with torch.no_grad():
                    t_logits = F.linear(teacher[start:end], teacher_weight, teacher_bias).to(loss_dtype) / temperature
                s_logits = F.linear(s, w, b).to(loss_dtype) / temperature
                loss = jsd_loss(s_logits, t_logits, beta, vocab_parallel_log_softmax, vocab_parallel_kl_div, chunk_size)
                inputs = [s] + ([w] if grad_weight is not None else []) + ([b] if grad_bias is not None else [])
                grads = iter(torch.autograd.grad(loss, inputs))
                grad_student[start:end] = next(grads)
                if grad_weight is not None:
                    grad_weight.add_(next(grads))
                if grad_bias is not None:
                    grad_bias.add_(next(grads))
                total.add_(loss.detach())
                del s_logits, t_logits, loss
        ctx.save_for_backward(grad_student, grad_weight, grad_bias)
        return total

    @staticmethod
    def backward(ctx, grad_output):
        grads = tuple(g * grad_output if g is not None else None for g in ctx.saved_tensors)
        return *grads, None, None, None, None, None, None


def chunked_gkd_loss(student, teacher_output, labels, student_model, teacher_model, beta, temperature, chunk_size):
    """Full-vocabulary JSD from hidden states, with TP/SP gradient routing and CP-local masks."""
    student_model = get_gkd_language_model(student_model)
    teacher_model = get_gkd_language_model(teacher_model)

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
    if num_valid != t_mask.sum():
        raise ValueError('Student and teacher must have the same number of response tokens')
    student_weight, teacher_weight = output_weight(student_model), output_weight(teacher_model)
    if student_weight.shape[0] != teacher_weight.shape[0]:
        raise ValueError('Chunked GKD requires matching teacher/student vocabulary partitions')
    loss = _ChunkedJSD.apply(student[s_mask], student_weight, student_model.output_layer.bias, teacher[t_mask],
                             teacher_weight, teacher_model.output_layer.bias, beta, temperature, chunk_size)
    return loss, num_valid


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
