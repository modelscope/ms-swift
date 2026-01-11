# Copyright (c) Alibaba, Inc. and its affiliates.
"""Vocabulary-parallel utilities for Tensor Parallelism.

This module provides utilities for computing log_softmax, entropy, KL divergence,
and other operations across vocab-parallel sharded tensors in Tensor Parallelism (TP).

When using TP, the vocabulary dimension is sharded across TP ranks. These utilities
correctly handle the distributed computation by:
1. Finding global max via all_reduce (for numerical stability)
2. Computing sum of exp via all_reduce (for normalization)
3. All-reducing partial sums for final results
"""

from typing import Optional, Tuple

import torch
from megatron.core import mpu


def vocab_parallel_log_softmax(logits: torch.Tensor) -> torch.Tensor:
    """Compute log_softmax across vocab-parallel sharded logits.

    When using Tensor Parallelism, vocab is sharded across TP ranks.
    This function correctly computes log_softmax by:
    1. Finding global max via all_reduce
    2. Computing sum of exp via all_reduce
    3. Computing log_softmax using the global statistics

    Args:
        logits: Logits tensor [..., partition_vocab_size]

    Returns:
        log_softmax tensor [..., partition_vocab_size]
    """
    tp_size = mpu.get_tensor_model_parallel_world_size()

    if tp_size == 1:
        return torch.nn.functional.log_softmax(logits, dim=-1)

    tp_group = mpu.get_tensor_model_parallel_group()

    # Step 1: Find global max for numerical stability
    logits_max = logits.max(dim=-1, keepdim=True)[0]
    torch.distributed.all_reduce(logits_max, op=torch.distributed.ReduceOp.MAX, group=tp_group)

    # Step 2: Compute exp(logits - max) and sum across all TP ranks
    exp_logits = torch.exp(logits - logits_max)
    sum_exp = exp_logits.sum(dim=-1, keepdim=True)
    torch.distributed.all_reduce(sum_exp, op=torch.distributed.ReduceOp.SUM, group=tp_group)

    # Step 3: Compute log_softmax
    log_softmax = logits - logits_max - torch.log(sum_exp)

    return log_softmax


def vocab_parallel_entropy(log_probs: torch.Tensor, chunk_size: int = 512) -> torch.Tensor:
    """Compute entropy from pre-computed vocab-parallel sharded log probabilities.

    When using Tensor Parallelism, vocab is sharded across TP ranks.
    This function correctly computes entropy by:
    1. Computing partial entropy = -sum(exp(log_p) * log_p) on each rank's partition
    2. All-reducing the partial entropies to get the global sum.

    Entropy is computed in chunks to reduce memory usage.

    Args:
        log_probs: Pre-computed log probabilities tensor [..., partition_vocab_size]
        chunk_size: Number of tokens to process per chunk (default: 512)

    Returns:
        Entropy tensor [...] (scalar per position)
    """
    tp_group = mpu.get_tensor_model_parallel_group()
    tp_size = mpu.get_tensor_model_parallel_world_size()

    # Flatten all but the last dimension for chunked processing
    original_shape = log_probs.shape[:-1]
    vocab_size = log_probs.shape[-1]
    log_probs_flat = log_probs.view(-1, vocab_size)  # [total_tokens, partition_vocab_size]
    total_tokens = log_probs_flat.shape[0]

    entropies_list = []
    for start_idx in range(0, total_tokens, chunk_size):
        end_idx = min(start_idx + chunk_size, total_tokens)
        log_probs_chunk = log_probs_flat[start_idx:end_idx]  # [chunk_size, partition_vocab_size]

        # Compute partial entropy on this rank's vocab partition
        # entropy = -sum(p * log_p) = -sum(exp(log_p) * log_p)
        probs = torch.exp(log_probs_chunk)
        partial_entropy = -(probs * log_probs_chunk).sum(dim=-1)  # [chunk_size]

        # All-reduce to get global entropy if using TP
        if tp_size > 1:
            torch.distributed.all_reduce(partial_entropy, op=torch.distributed.ReduceOp.SUM, group=tp_group)

        entropies_list.append(partial_entropy)

    # Concatenate all chunks and reshape back
    entropies = torch.cat(entropies_list, dim=0)
    entropies = entropies.view(original_shape)

    return entropies


def vocab_parallel_kl_div(input_log_probs: torch.Tensor, target_log_probs: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence for vocab-parallel sharded log probabilities.

    KL(target || input) = sum(target_prob * (target_log_prob - input_log_prob))
                        = sum(exp(target_log_prob) * (target_log_prob - input_log_prob))

    Since both log_probs are sharded across TP, we compute the partial sum
    on each rank and then all_reduce to get the global sum.

    Args:
        input_log_probs: Input log probabilities [..., partition_vocab_size]
        target_log_probs: Target log probabilities [..., partition_vocab_size]

    Returns:
        KL divergence per position [...], already reduced across TP
    """
    tp_group = mpu.get_tensor_model_parallel_group()

    # Compute partial KL on this rank's vocab partition
    target_probs = torch.exp(target_log_probs)
    partial_kl = (target_probs * (target_log_probs - input_log_probs)).sum(dim=-1)

    if mpu.get_tensor_model_parallel_world_size() > 1:
        tp_group = mpu.get_tensor_model_parallel_group()
        torch.distributed.all_reduce(partial_kl, op=torch.distributed.ReduceOp.SUM, group=tp_group)

    return partial_kl


def vocab_parallel_gather_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    log_probs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Gather log probabilities for target labels from vocab-parallel logits.

    When using TP, each rank only has a partition of the vocabulary. This function:
    1. Computes log_softmax with vocab-parallel support (if log_probs not provided)
    2. Gathers log probs for target tokens
    3. All-reduces to combine results from all TP ranks

    Args:
        logits: Logits tensor [batch, seq, partition_vocab_size]
        labels: Token labels [batch, seq], -100 for masked positions
        log_probs: Pre-computed log_softmax (optional, to avoid recomputation)

    Returns:
        per_token_logps: [batch, seq] log probabilities for target tokens
    """
    tp_size = mpu.get_tensor_model_parallel_world_size()
    tp_rank = mpu.get_tensor_model_parallel_rank()

    # Compute log_probs if not provided
    if log_probs is None:
        log_probs = vocab_parallel_log_softmax(logits)

    # Get the local vocab range for this TP rank
    partition_vocab_size = log_probs.shape[-1]
    if tp_size > 1:
        vocab_start = tp_rank * partition_vocab_size
        vocab_end = vocab_start + partition_vocab_size
        # Check which labels fall within this TP rank's vocab partition
        local_labels = labels - vocab_start
        # Mask for labels within local vocab range
        in_range_mask = (labels >= vocab_start) & (labels < vocab_end)
        # Clamp local_labels to valid range for gather
        local_labels = local_labels.clamp(min=0, max=partition_vocab_size - 1)
    else:
        local_labels = labels
        in_range_mask = torch.ones_like(labels, dtype=torch.bool)
        local_labels = local_labels.clamp(min=0)

    # Gather log probs for target tokens
    gathered_logps = torch.gather(log_probs, dim=-1, index=local_labels.unsqueeze(-1)).squeeze(-1)

    # For TP: only the rank that owns the target token has the correct log prob
    # Other ranks have incorrect values, so we need to zero them out
    if tp_size > 1:
        gathered_logps = gathered_logps * in_range_mask.float()
        # All-reduce to sum contributions from all ranks
        # (only one rank has non-zero value for each token)
        torch.distributed.all_reduce(
            gathered_logps, op=torch.distributed.ReduceOp.SUM, group=mpu.get_tensor_model_parallel_group())

    # Apply loss mask (labels == -100 are masked)
    loss_mask = labels != -100
    per_token_logps = gathered_logps * loss_mask.float()

    return per_token_logps


def compute_logps_and_entropy_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    compute_entropy: bool = False,
    entropy_chunk_size: int = 512,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Compute per-token log probabilities and optionally entropy from logits.

    This is a unified method that efficiently computes both logps and entropy
    in a single pass when both are needed, sharing the log_softmax computation.

    Note: In Megatron, labels are already shifted (via torch.roll in get_batch_on_this_tp_rank),
    so logits and labels are already aligned. No additional shift is needed here.

    Args:
        logits: Logits tensor [batch, seq, partition_vocab_size] or [1, total_tokens, partition_vocab_size]
        labels: Token labels [batch, seq] or [1, total_tokens], -100 for masked positions
        compute_entropy: Whether to compute entropy (default: False)
        entropy_chunk_size: Chunk size for entropy computation (default: 512)

    Returns:
        Tuple of:
            - per_token_logps: [batch, seq] or [1, total_tokens] log probabilities for target tokens
            - per_token_entropy: Same shape as per_token_logps, or None if compute_entropy=False
    """
    # Compute log_softmax (shared for both logps and entropy)
    log_probs = vocab_parallel_log_softmax(logits)

    # Gather logps for target tokens
    per_token_logps = vocab_parallel_gather_logps(logits, labels, log_probs=log_probs)

    # Compute entropy if requested (reuse log_probs to avoid redundant computation)
    per_token_entropy = None
    if compute_entropy:
        per_token_entropy = vocab_parallel_entropy(log_probs, chunk_size=entropy_chunk_size)

    return per_token_logps, per_token_entropy
