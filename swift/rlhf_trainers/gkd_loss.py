# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared GKD loss utilities across HF / Megatron / Ray backends."""
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Tuple

# ---------------------------------------------------------------------------
# Data types — shared across all backends
# ---------------------------------------------------------------------------


class DataSource(str, Enum):
    """Data source for GKD training."""
    DATASET = 'dataset'
    STUDENT = 'student'
    TEACHER = 'teacher'


@dataclass
class TeacherOutput:
    """Unified container for teacher model outputs from all three sources:
    local full-vocab, local top-k, and external API top-k.
    """
    full_logits: Optional[torch.Tensor] = None
    topk_logprobs: Optional[torch.Tensor] = None
    topk_indices: Optional[torch.Tensor] = None
    opsd_teacher_labels: Optional[torch.Tensor] = None

    @property
    def is_topk_mode(self) -> bool:
        return self.topk_logprobs is not None and self.topk_indices is not None

    def validate(self):
        if self.full_logits is None and not self.is_topk_mode:
            raise ValueError('TeacherOutput must provide either full_logits or '
                             '(topk_logprobs, topk_indices). Got neither.')

    def select(self, mask: torch.Tensor) -> 'TeacherOutput':
        """Select active positions by boolean mask."""
        return TeacherOutput(
            full_logits=self.full_logits[mask] if self.full_logits is not None else None,
            topk_logprobs=self.topk_logprobs[mask] if self.topk_logprobs is not None else None,
            topk_indices=self.topk_indices[mask] if self.topk_indices is not None else None,
        )

    def to_topk(self, k: int, topk_fn=None) -> 'TeacherOutput':
        """Convert full logits to topk representation."""
        if self.is_topk_mode:
            return self
        fn = topk_fn or (lambda logits, k: torch.topk(logits, k=k, dim=-1))
        vals, ids = fn(self.full_logits, k)
        return TeacherOutput(
            topk_logprobs=vals,
            topk_indices=ids,
            opsd_teacher_labels=self.opsd_teacher_labels,
        )


# ---------------------------------------------------------------------------
# Default primitives (standard PyTorch, no TP/CP)
# ---------------------------------------------------------------------------


def default_log_softmax(logits: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits, dim=-1)


def default_kl_div(input_log_probs: torch.Tensor, target_log_probs: torch.Tensor) -> torch.Tensor:
    """KL(target || input), returns per-position scalar [N]."""
    return (torch.exp(target_log_probs) * (target_log_probs - input_log_probs)).sum(-1)


def default_gather(logits: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return torch.gather(logits, dim=-1, index=indices)


# ---------------------------------------------------------------------------
# jsd_loss — the single place to modify KL/JSD computation logic
# ---------------------------------------------------------------------------


def jsd_loss(
    s_logits: torch.Tensor,
    t_logits: torch.Tensor,
    beta: float,
    log_softmax_fn: Callable = default_log_softmax,
    kl_div_fn: Callable = default_kl_div,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Chunked JSD between student and teacher.

    This is THE function for JSD math. To customize KL computation,
    modify the loop body below or inject custom log_softmax_fn / kl_div_fn.

    Args:
        s_logits: [N, D] student logits (temperature-scaled)
        t_logits: [N, D] teacher logits (temperature-scaled)
        beta: JSD interpolation (0=forward KL, 1=reverse KL, 0<beta<1=JSD)
        log_softmax_fn: (logits [C, D]) -> log_probs [C, D]
        kl_div_fn: (input_log [C, D], target_log [C, D]) -> per_position [C]
        chunk_size: chunk size for memory efficiency

    Returns:
        Scalar — unnormalized total JSD (caller normalizes by num_valid).
    """
    N = s_logits.size(0)
    # N may be 0 when a CP rank's partition has no valid tokens;
    # returning zero lets cp_reduce still all-reduce without hanging.
    if N == 0:
        return s_logits.new_zeros(())

    total = s_logits.new_zeros(())

    if beta != 0 and beta != 1:
        beta_t = torch.tensor(beta, dtype=s_logits.dtype, device=s_logits.device)
        log_beta = torch.log(beta_t)
        log_1_minus_beta = torch.log1p(-beta_t)
    else:
        beta_t = log_beta = log_1_minus_beta = None

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        s_log = log_softmax_fn(s_logits[start:end])
        t_log = log_softmax_fn(t_logits[start:end])

        if beta == 0:
            jsd = kl_div_fn(s_log, t_log)
        elif beta == 1:
            jsd = kl_div_fn(t_log, s_log)
        else:
            m_log = torch.logsumexp(torch.stack([s_log + log_1_minus_beta, t_log + log_beta]), dim=0)
            jsd = beta_t * kl_div_fn(m_log, t_log) + (1 - beta_t) * kl_div_fn(m_log, s_log)

        total = total + jsd.sum()

    return total


# ---------------------------------------------------------------------------
# Internal vocab alignment
# ---------------------------------------------------------------------------


def _align_vocab(student_logits: torch.Tensor, teacher_logits: torch.Tensor):
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


# ---------------------------------------------------------------------------
# extract_active — unified mask extraction (OPSD / non-OPSD)
# ---------------------------------------------------------------------------


def extract_active(
    student_logits: torch.Tensor,
    teacher_output: TeacherOutput,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, TeacherOutput, torch.Tensor]:
    """Extract active positions from student logits and teacher output.

    Args:
        student_logits: [B, S, V] or [1, T, V]
        teacher_output: TeacherOutput with same leading dims
        labels: [B, S] or [1, T], -100 for inactive (must be pre-shifted)

    Returns:
        (student_active [N, V], teacher_active TeacherOutput [N, ...], num_valid tensor)
    """
    opsd_labels = teacher_output.opsd_teacher_labels
    if opsd_labels is not None:
        s_mask = labels != -100
        t_mask = opsd_labels != -100
        assert s_mask.sum() == t_mask.sum(), (f'OPSD label count mismatch: student={s_mask.sum().item()}, '
                                              f'teacher={t_mask.sum().item()}. '
                                              'Student and teacher must share the same response tokens.')
        s_active = student_logits[s_mask]
        t_active = teacher_output.select(t_mask)
        if t_active.is_topk_mode:
            uncovered = torch.isinf(t_active.topk_logprobs).all(dim=-1)
            if uncovered.any():
                keep = ~uncovered
                s_active = s_active[keep]
                t_active = t_active.select(keep)
        return s_active, t_active, torch.tensor(int(s_active.shape[0]), device=labels.device)

    mask = labels != -100
    if teacher_output.is_topk_mode:
        uncovered = torch.isinf(teacher_output.topk_logprobs).all(dim=-1)
        if uncovered.any():
            mask = mask & ~uncovered
    return student_logits[mask], teacher_output.select(mask), mask.sum()


# ---------------------------------------------------------------------------
# gkd_loss — full pipeline: mask → prepare → jsd
# ---------------------------------------------------------------------------


def gkd_loss(
    student_logits: torch.Tensor,
    teacher_output: TeacherOutput,
    labels: torch.Tensor,
    beta: float,
    temperature: float,
    gather_fn: Callable = default_gather,
    log_softmax_fn: Callable = default_log_softmax,
    kl_div_fn: Callable = default_kl_div,
    chunk_size: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Full GKD loss pipeline. Returns (total_loss, num_valid).

    Caller is responsible for normalization (e.g. simple division for HF,
    CP all-reduce + division for Megatron).

    Args:
        student_logits: [B, S, V] student model logits
        teacher_output: TeacherOutput (full_logits or topk)
        labels: [B, S], pre-shifted, -100 for inactive positions
        beta: JSD interpolation coefficient
        temperature: temperature scaling
        gather_fn: (logits[N,V], indices[N,K]) -> [N,K], for topk gather
        log_softmax_fn: logits -> log_probs (may be TP-aware for full-vocab)
        kl_div_fn: (input_log, target_log) -> per_position KL (may be TP-aware)
        chunk_size: chunk size for memory efficiency

    Returns:
        (total_loss, num_valid) — unnormalized total and count of valid positions.
    """
    teacher_output.validate()
    s_active, t_active, num_valid = extract_active(student_logits, teacher_output, labels)

    if t_active.is_topk_mode:
        s_logits = gather_fn(s_active, t_active.topk_indices)
        t_logits = t_active.topk_logprobs
        lsf, kdf = default_log_softmax, default_kl_div
    else:
        s_logits = s_active
        t_logits = t_active.full_logits
        s_logits, t_logits = _align_vocab(s_logits, t_logits)
        lsf, kdf = log_softmax_fn, kl_div_fn

    s_logits = s_logits / temperature
    t_logits = t_logits / temperature

    total = jsd_loss(s_logits, t_logits, beta, lsf, kdf, chunk_size)
    return total, num_valid


# ---------------------------------------------------------------------------
# build_opsd_teacher_data — shared OPSD teacher data construction
# ---------------------------------------------------------------------------


def build_opsd_teacher_data(inputs, strip_assistant=False):
    """Build teacher data for OPSD by replacing the last user message with teacher_prompt.

    Args:
        inputs: list of data dicts, each may contain 'teacher_prompt'
        strip_assistant: if True, remove trailing assistant message before replacement

    Returns:
        List of teacher data dicts, or None if teacher_prompt is not in all inputs.
    """
    if not all('teacher_prompt' in d and d['teacher_prompt'] for d in inputs):
        return None
    result = []
    for data in inputs:
        item = {k: v for k, v in data.items() if k != 'teacher_prompt'}
        messages = [dict(m) for m in data.get('messages', [])]
        if strip_assistant and messages and messages[-1]['role'] == 'assistant':
            messages.pop()
        for msg in reversed(messages):
            if msg['role'] == 'user':
                msg['content'] = data['teacher_prompt']
                break
        item['messages'] = messages
        result.append(item)
    return result
