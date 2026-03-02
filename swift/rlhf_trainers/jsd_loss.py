"""Unified JSD (Jensen-Shannon Divergence) loss implementation for GKD training.

This module provides a memory-efficient, chunked JSD loss computation that supports:
1. Full vocabulary mode: Uses complete logits from both models
2. Top-K mode with local teacher: Extracts top-k from teacher logits
3. Top-K mode with API: Uses pre-computed teacher logprobs and indices

The implementation uses chunked processing to reduce peak memory usage.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def compute_jsd_loss(
    student_logits: torch.Tensor,
    teacher_logits: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    beta: float = 0.5,
    temperature: float = 1.0,
    chunk_size: int = 256,
    topk: Optional[int] = None,
    teacher_topk_logprobs: Optional[torch.Tensor] = None,
    teacher_topk_indices: Optional[torch.Tensor] = None,
    log_softmax_fn=None,
    kl_div_fn=None,
) -> torch.Tensor:
    """Compute JSD loss with unified chunked processing for memory efficiency.

    This function handles all three modes in a unified way:
    - Full vocab mode: teacher_logits provided, topk=None
    - Top-K local mode: teacher_logits provided, topk specified
    - Top-K API mode: teacher_topk_logprobs and teacher_topk_indices provided

    Args:
        student_logits: Student model logits [batch, seq_len, vocab_size]
        teacher_logits: Teacher model logits [batch, seq_len, vocab_size], None for API mode
        labels: Token labels for masking [batch, seq_len], -100 for ignored positions
        beta: JSD interpolation coefficient (0=Forward KL, 0.5=JSD, 1=Reverse KL)
        temperature: Temperature for softmax scaling
        chunk_size: Chunk size for memory-efficient processing
        topk: Number of top-k logits to use. None for full vocabulary mode.
        teacher_topk_logprobs: Pre-computed teacher log probs [batch, seq_len, topk] (API mode)
        teacher_topk_indices: Pre-computed teacher token indices [batch, seq_len, topk] (API mode)
        log_softmax_fn: Optional custom log_softmax function (e.g., for vocab parallel)
        kl_div_fn: Optional custom KL div function (e.g., for vocab parallel)

    Returns:
        Scalar loss value
    """
    # Determine mode
    use_api_mode = teacher_topk_logprobs is not None and teacher_topk_indices is not None
    use_topk = topk is not None or use_api_mode

    # Build mask
    if labels is not None:
        mask = labels != -100
    else:
        mask = torch.ones(student_logits.shape[:2], dtype=torch.bool, device=student_logits.device)

    num_valid = mask.sum()
    if num_valid == 0:
        return student_logits.new_zeros(())

    # Dispatch to appropriate mode
    if use_api_mode:
        return _compute_topk_api_loss(student_logits, teacher_topk_logprobs, teacher_topk_indices, mask, num_valid,
                                      beta, temperature)
    elif use_topk:
        return _compute_topk_local_loss_chunked(student_logits, teacher_logits, mask, num_valid, beta, temperature,
                                                topk, chunk_size)
    else:
        return _compute_full_vocab_loss_chunked(student_logits, teacher_logits, mask, num_valid, beta, temperature,
                                                chunk_size, log_softmax_fn, kl_div_fn)


def _compute_topk_jsd(
    teacher_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    student_logits: torch.Tensor,
    student_log_probs: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Compute JSD on top-k distribution.

    Args:
        teacher_probs: Teacher probabilities [*, topk]
        teacher_log_probs: Teacher log probabilities [*, topk]
        student_logits: Student logits at top-k positions [*, topk]
        student_log_probs: Student log probabilities [*, topk]
        beta: JSD interpolation coefficient

    Returns:
        JSD values [*] (reduced over topk dimension)
    """
    if beta == 0:
        # Forward KL: KL(teacher || student)
        return (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    elif beta == 1:
        # Reverse KL: KL(student || teacher)
        student_probs = F.softmax(student_logits, dim=-1)
        return (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)
    else:
        # Full JSD with mixture distribution
        student_probs = F.softmax(student_logits, dim=-1)
        mixture_probs = beta * teacher_probs + (1 - beta) * student_probs
        mixture_log_probs = torch.log(mixture_probs + 1e-10)
        kl_teacher = (teacher_probs * (teacher_log_probs - mixture_log_probs)).sum(dim=-1)
        kl_student = (student_probs * (student_log_probs - mixture_log_probs)).sum(dim=-1)
        return beta * kl_teacher + (1 - beta) * kl_student


def _compute_topk_api_loss(
    student_logits: torch.Tensor,
    teacher_topk_logprobs: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    mask: torch.Tensor,
    num_valid: torch.Tensor,
    beta: float,
    temperature: float,
) -> torch.Tensor:
    """Compute Top-K JSD loss using pre-computed API logprobs.

    This mode is already memory-efficient since teacher logprobs are pre-computed
    and only top-k values are stored.
    """
    # Apply temperature to student logits
    student_logits_scaled = student_logits / temperature

    # Get teacher probs from log probs
    teacher_probs = torch.exp(teacher_topk_logprobs)

    # Gather student logits at teacher's top-k positions
    student_topk_logits = torch.gather(student_logits_scaled, dim=-1, index=teacher_topk_indices)
    del student_logits_scaled
    student_topk_log_probs = F.log_softmax(student_topk_logits, dim=-1)

    # Compute JSD
    jsd = _compute_topk_jsd(teacher_probs, teacher_topk_logprobs, student_topk_logits, student_topk_log_probs, beta)

    # Apply mask and compute mean
    jsd_masked = jsd * mask.float()
    return jsd_masked.sum() / num_valid


def _compute_topk_local_loss_chunked(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    num_valid: torch.Tensor,
    beta: float,
    temperature: float,
    topk: int,
    chunk_size: int,
) -> torch.Tensor:
    """Compute Top-K JSD loss with local teacher using chunked processing.

    Processes the sequence in chunks along the sequence dimension to avoid
    keeping full vocab-size tensors in memory simultaneously.
    """
    seq_len = student_logits.shape[1]
    total_loss = student_logits.new_zeros(())

    for start_idx in range(0, seq_len, chunk_size):
        end_idx = min(start_idx + chunk_size, seq_len)

        chunk_mask = mask[:, start_idx:end_idx]
        if chunk_mask.sum() == 0:
            continue

        # Get logits chunks and apply temperature
        student_chunk = student_logits[:, start_idx:end_idx, :] / temperature
        teacher_chunk = teacher_logits[:, start_idx:end_idx, :] / temperature

        # Get top-k from teacher chunk, then release teacher chunk
        teacher_topk_logits, topk_indices = torch.topk(teacher_chunk, k=topk, dim=-1)
        del teacher_chunk

        teacher_probs = F.softmax(teacher_topk_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_topk_logits, dim=-1)
        del teacher_topk_logits

        # Gather student logits at top-k positions, then release student chunk
        student_topk_logits = torch.gather(student_chunk, dim=-1, index=topk_indices)
        del student_chunk, topk_indices

        student_log_probs = F.log_softmax(student_topk_logits, dim=-1)

        # Compute JSD and accumulate
        jsd = _compute_topk_jsd(teacher_probs, teacher_log_probs, student_topk_logits, student_log_probs, beta)
        jsd_masked = jsd * chunk_mask.float()
        total_loss = total_loss + jsd_masked.sum()

        del jsd, jsd_masked, student_topk_logits, student_log_probs, teacher_probs, teacher_log_probs

    return total_loss / num_valid


def _compute_full_vocab_loss_chunked(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    num_valid: torch.Tensor,
    beta: float,
    temperature: float,
    chunk_size: int,
    log_softmax_fn,
    kl_div_fn=None,
) -> torch.Tensor:
    """Compute full vocabulary JSD loss with chunked processing.

    Supports custom log_softmax and kl_div functions for vocab-parallel computation.
    """
    # Use default implementations if not provided
    if log_softmax_fn is None:

        def log_softmax_fn(x):
            return F.log_softmax(x, dim=-1)

    if kl_div_fn is None:

        def kl_div_fn(p, q):
            return F.kl_div(p, q, reduction='none', log_target=True)

    # Apply temperature and masking to flatten valid tokens
    student_logits_masked = (student_logits / temperature)[mask]
    teacher_logits_masked = (teacher_logits / temperature)[mask]
    del student_logits, teacher_logits

    num_valid_int = num_valid.item() if isinstance(num_valid, torch.Tensor) else int(num_valid)
    total_loss = student_logits_masked.new_zeros(())

    # Precompute beta tensors if needed
    if beta != 0 and beta != 1:
        beta_t = torch.tensor(beta, dtype=student_logits_masked.dtype, device=student_logits_masked.device)
        log_beta = torch.log(beta_t)
        log_1_minus_beta = torch.log1p(-beta_t)
    else:
        beta_t = log_beta = log_1_minus_beta = None

    for start_idx in range(0, num_valid_int, chunk_size):
        end_idx = min(start_idx + chunk_size, num_valid_int)
        s_chunk = student_logits_masked[start_idx:end_idx]
        t_chunk = teacher_logits_masked[start_idx:end_idx]

        s_log_probs = log_softmax_fn(s_chunk)
        t_log_probs = log_softmax_fn(t_chunk)
        del s_chunk, t_chunk

        if beta == 0:
            jsd_chunk = kl_div_fn(s_log_probs, t_log_probs)
        elif beta == 1:
            jsd_chunk = kl_div_fn(t_log_probs, s_log_probs)
        else:
            mixture_log_probs = torch.logsumexp(
                torch.stack([s_log_probs + log_1_minus_beta, t_log_probs + log_beta]),
                dim=0,
            )
            kl_teacher = kl_div_fn(mixture_log_probs, t_log_probs)
            kl_student = kl_div_fn(mixture_log_probs, s_log_probs)
            del mixture_log_probs
            jsd_chunk = beta_t * kl_teacher + (1 - beta_t) * kl_student
            del kl_teacher, kl_student

        total_loss = total_loss + jsd_chunk.sum()
        del jsd_chunk, s_log_probs, t_log_probs

    del student_logits_masked, teacher_logits_masked
    return total_loss / num_valid
