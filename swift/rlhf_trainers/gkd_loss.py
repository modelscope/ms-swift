# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared GKD loss utilities across HF / Megatron / Ray backends."""
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Data types — shared across all backends
# ---------------------------------------------------------------------------


class DataSource(str, Enum):
    """Data source for GKD training."""
    DATASET = 'dataset'
    STUDENT = 'student'
    TEACHER = 'teacher'  # deprecated, pre-sample before training


@dataclass
class TeacherOutput:
    """Unified container for teacher model outputs from all three sources:
    local full-vocab, local top-k, and external API top-k.
    """
    full_logits: Optional[torch.Tensor] = None
    topk_logprobs: Optional[torch.Tensor] = None
    topk_indices: Optional[torch.Tensor] = None
    # Log-probability assigned by the teacher to the actually observed response
    # token at each position. This keeps the teacher-student gap exact even when
    # the distillation loss itself only retains the teacher's top-k support.
    target_logprobs: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None

    @property
    def is_topk_mode(self) -> bool:
        return self.topk_logprobs is not None and self.topk_indices is not None

    def to_device(self, device) -> 'TeacherOutput':
        """Move all tensor fields to ``device`` in place (Ray: teacher_output is
        collated on the CPU driver, moved to the GPU worker before forward)."""
        for name in ('full_logits', 'topk_logprobs', 'topk_indices', 'target_logprobs', 'labels'):
            v = getattr(self, name)
            if isinstance(v, torch.Tensor):
                setattr(self, name, v.to(device))
        return self

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
            target_logprobs=self.target_logprobs[mask] if self.target_logprobs is not None else None,
            labels=self.labels[mask] if self.labels is not None else None,
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
            target_logprobs=self.target_logprobs,
            labels=self.labels,
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
        t_logits: [N, D] teacher logits/logps (temperature-scaled)
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

    Uses ``teacher_output.labels`` (always present) to derive the
    teacher mask, and ``labels`` for the student mask. When non-OPSD the two are
    identical so the result is equivalent to masking by student labels alone.

    Args:
        student_logits: [B, S, V] or [1, T, V]
        teacher_output: TeacherOutput with same leading dims
        labels: [B, S] or [1, T], -100 for inactive (must be pre-shifted)

    Returns:
        (student_active [N, V], teacher_active TeacherOutput [N, ...], num_valid tensor)
    """
    t_labels = teacher_output.labels
    s_mask = labels != -100
    if t_labels is not None:
        # Teacher labels are always set (equals student labels when non-OPSD).
        # OPSD: teacher scores a different prompt, so its label mask differs in
        # position but must have the same count of valid response tokens.
        # Non-OPSD: teacher labels == student labels, so masks are identical.
        t_mask = t_labels != -100
        assert s_mask.sum() == t_mask.sum(), (f'Label count mismatch: student={s_mask.sum().item()}, '
                                              f'teacher={t_mask.sum().item()}. '
                                              'Student and teacher must share the same response tokens.')
    else:
        # Fallback for PP non-last stage placeholders (TeacherOutput() with all None).
        t_mask = s_mask
    s_active = student_logits[s_mask]
    t_active = teacher_output.select(t_mask)
    if t_active.is_topk_mode:
        uncovered = torch.isinf(t_active.topk_logprobs).all(dim=-1)
        if uncovered.any():
            keep = ~uncovered
            s_active = s_active[keep]
            t_active = t_active.select(keep)
    return s_active, t_active, torch.tensor(int(s_active.shape[0]), device=labels.device)


@torch.no_grad()
def gkd_monitoring_stats(
    student_logits: torch.Tensor,
    teacher_output: TeacherOutput,
    labels: torch.Tensor,
    *,
    full_vocab_topk: int = 16,
    student_topk_fn: Callable = torch.topk,
    teacher_topk_fn: Callable = torch.topk,
    gather_fn: Callable = default_gather,
    target_logprob_fn: Optional[Callable] = None,
) -> Dict[str, torch.Tensor]:
    """Return additive GKD diagnostics over active response tokens.

    ``topk_overlap`` follows the standard token-level definition
    ``|TopK(student) intersect TopK(teacher)| / K``. In a top-k teacher path,
    K is the retained teacher width; for full-vocabulary distillation it defaults
    to 16.

    ``teacher_student_gap`` is computed on the observed response token exactly as
    ``log p_teacher(y_t) - log p_student(y_t)``. The returned values are sums and
    counts so callers can aggregate them correctly across DP/CP ranks.
    """
    s_active, t_active, num_valid = extract_active(student_logits, teacher_output, labels)
    zero = student_logits.new_zeros((), dtype=torch.float32)
    if int(num_valid.item()) == 0:
        return {
            'topk_overlap_sum': zero,
            'topk_overlap_count': zero,
            'teacher_student_gap_sum': zero,
            'teacher_student_gap_count': zero,
        }

    if t_active.is_topk_mode:
        k = min(t_active.topk_indices.shape[-1], s_active.shape[-1])
        teacher_topk_ids = t_active.topk_indices[..., :k]
    else:
        k = min(full_vocab_topk, s_active.shape[-1], t_active.full_logits.shape[-1])
        _, teacher_topk_ids = teacher_topk_fn(t_active.full_logits, k)

    _, student_topk_ids = student_topk_fn(s_active, k)
    overlap_count = (teacher_topk_ids.unsqueeze(-1) == student_topk_ids.unsqueeze(-2)).any(dim=-1).sum(dim=-1)
    overlap_sum = (overlap_count.float() / k).sum()

    if t_active.labels is not None:
        active_target_ids = t_active.labels.long()
    else:
        active_target_ids = labels[labels != -100].long()
        if active_target_ids.numel() != s_active.shape[0]:
            # A legacy top-k path can omit entire uncovered rows without carrying
            # teacher labels. The overlap metric is still valid, but an exact gap
            # cannot be aligned to the remaining observed tokens.
            return {
                'topk_overlap_sum': overlap_sum.float(),
                'topk_overlap_count': num_valid.float(),
                'teacher_student_gap_sum': zero,
                'teacher_student_gap_count': zero,
            }
    if target_logprob_fn is None:
        # Avoid materializing a second full-vocabulary log-probability tensor in
        # the common non-TP path.
        student_target_logits = gather_fn(s_active.float(), active_target_ids.unsqueeze(-1)).squeeze(-1)
        student_target_logprobs = student_target_logits - torch.logsumexp(s_active.float(), dim=-1)
    else:
        student_target_logprobs = target_logprob_fn(s_active, active_target_ids)

    if t_active.target_logprobs is not None:
        teacher_target_logprobs = t_active.target_logprobs.float()
        gap_mask = torch.isfinite(teacher_target_logprobs)
    elif t_active.full_logits is not None:
        if target_logprob_fn is None:
            teacher_target_logits = gather_fn(t_active.full_logits.float(), active_target_ids.unsqueeze(-1)).squeeze(-1)
            teacher_target_logprobs = teacher_target_logits - torch.logsumexp(t_active.full_logits.float(), dim=-1)
        else:
            teacher_target_logprobs = target_logprob_fn(t_active.full_logits, active_target_ids)
        gap_mask = torch.ones_like(teacher_target_logprobs, dtype=torch.bool)
    else:
        # Compatibility fallback for top-k tensors produced by older paths. It
        # is exact only where the observed token is present in the retained set.
        matches = t_active.topk_indices == active_target_ids.unsqueeze(-1)
        gap_mask = matches.any(dim=-1)
        match_pos = matches.float().argmax(dim=-1, keepdim=True)
        teacher_target_logprobs = torch.gather(t_active.topk_logprobs.float(), -1, match_pos).squeeze(-1)

    gap = teacher_target_logprobs - student_target_logprobs
    return {
        'topk_overlap_sum': overlap_sum.float(),
        'topk_overlap_count': num_valid.float(),
        'teacher_student_gap_sum': gap.masked_fill(~gap_mask, 0).sum().float(),
        'teacher_student_gap_count': gap_mask.sum().float(),
    }


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
