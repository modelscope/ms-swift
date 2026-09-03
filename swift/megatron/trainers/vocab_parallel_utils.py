# Copyright (c) ModelScope Contributors. All rights reserved.
"""Vocabulary-parallel utilities for Tensor Parallelism.

This module provides utilities for computing log_softmax, entropy, KL divergence,
and other operations across vocab-parallel sharded tensors in Tensor Parallelism (TP).

When using TP, the vocabulary dimension is sharded across TP ranks. These utilities
correctly handle the distributed computation by:
1. Finding global max via all_reduce (for numerical stability)
2. Computing sum of exp via a differentiable all_reduce (for normalization)
3. All-reducing partial sums for final results

Note on gradients: ``torch.distributed.all_reduce`` is invisible to autograd, so
whether a raw call is correct depends on what consumes the reduced value. See
``_AllReduceAcrossVocabShards`` and the comments on each reduction below.
"""

import torch
from megatron.core import mpu, tensor_parallel
from typing import Optional, Tuple


class _AllReduceAcrossVocabShards(torch.autograd.Function):
    """All-reduce a value that every vocab shard both contributes to and depends on.

    A raw ``torch.distributed.all_reduce`` is not part of the autograd graph, so each
    rank's backward only ever sees its own partial derivative. That is what we want
    when the reduced value is consumed by replicated math -- the vocab dimension is
    already gone, every rank runs the identical remaining computation, and its partial
    derivative is therefore the total one (entropy and KL below).

    It is wrong when every rank's output still depends on the reduced value, as for
    the ``sum(exp)`` normalizer of log_softmax: a target token lives on a single shard,
    so the other ranks' incoming gradient is zero and the ``-softmax`` half of the
    gradient silently vanishes on them. Summing the partial derivatives recovers it.
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group) -> torch.Tensor:
        ctx.group = group
        tensor = tensor.clone()
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        grad_output = grad_output.contiguous().clone()
        torch.distributed.all_reduce(grad_output, op=torch.distributed.ReduceOp.SUM, group=ctx.group)
        return grad_output, None


def vocab_parallel_log_softmax(logits: torch.Tensor) -> torch.Tensor:
    """Compute log_softmax across vocab-parallel sharded logits.

    When using Tensor Parallelism, vocab is sharded across TP ranks.
    This function correctly computes log_softmax by:
    1. Finding global max via all_reduce
    2. Computing sum of exp via a differentiable all_reduce
    3. Computing log_softmax using the global statistics

    Both the forward values and the backward gradients match a single-rank
    ``torch.log_softmax`` over the full vocabulary.

    Args:
        logits: Logits tensor [..., partition_vocab_size]

    Returns:
        log_softmax tensor [..., partition_vocab_size]
    """
    tp_size = mpu.get_tensor_model_parallel_world_size()

    if tp_size == 1:
        return torch.nn.functional.log_softmax(logits, dim=-1)

    tp_group = mpu.get_tensor_model_parallel_group()

    # Step 1: Find global max for numerical stability. This is a constant shift and is
    # computed without grad: the local max is not the global one, so keeping it in the
    # graph would scatter a bogus gradient onto each rank's local argmax position.
    with torch.no_grad():
        logits_max = logits.max(dim=-1, keepdim=True)[0]
        torch.distributed.all_reduce(logits_max, op=torch.distributed.ReduceOp.MAX, group=tp_group)

    # Step 2: Compute exp(logits - max) and sum across all TP ranks. Every rank's
    # log_softmax divides by this sum, so backward must all-reduce the partials too.
    exp_logits = torch.exp(logits - logits_max)
    sum_exp = _AllReduceAcrossVocabShards.apply(exp_logits.sum(dim=-1, keepdim=True), tp_group)

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

        # All-reduce to get global entropy if using TP. A raw all_reduce is correct
        # here: the vocab dimension is fully reduced away, so everything downstream is
        # replicated across the TP group and each rank's incoming gradient is already
        # the total derivative. Routing this through an autograd-aware reduce would
        # sum it tp_size times instead.
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

    # As in vocab_parallel_entropy, a raw all_reduce is the correct gradient behaviour
    # here because the per-position KL is consumed by replicated math.
    if mpu.get_tensor_model_parallel_world_size() > 1:
        tp_group = mpu.get_tensor_model_parallel_group()
        torch.distributed.all_reduce(partial_kl, op=torch.distributed.ReduceOp.SUM, group=tp_group)

    return partial_kl


def vocab_parallel_gather_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Gather log probabilities for target labels from vocab-parallel logits.

    Uses Megatron's vocab-parallel cross entropy so backward retains the softmax
    gradient on every TP vocab shard.

    Args:
        logits: Logits tensor [batch, seq, partition_vocab_size]
        labels: Token labels [batch, seq], -100 for masked positions

    Returns:
        per_token_logps: [batch, seq] log probabilities for target tokens
    """
    safe_labels = labels.masked_fill(labels == -100, 0)
    per_token_logps = -tensor_parallel.vocab_parallel_cross_entropy(vocab_parallel_logits=logits, target=safe_labels)
    return per_token_logps * (labels != -100)


def compute_logps_and_entropy_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    compute_entropy: bool = False,
    entropy_chunk_size: int = 512,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Compute per-token log probabilities and optionally entropy from logits.

    Log probabilities use Megatron's vocab-parallel cross entropy for a correct
    distributed backward. The full log_softmax is computed only when entropy is requested.

    Note: In Megatron, labels are already shifted (via torch.roll in get_batch_on_this_tp_rank),
    so logits and labels are already aligned. No additional shift is needed here.

    Temperature scaling should be applied by the caller before invoking this function,
    so that this function remains a pure computation without side effects on the input.

    Args:
        logits: Logits tensor [batch, seq, partition_vocab_size] or [1, total_tokens, partition_vocab_size].
                Should be pre-scaled by temperature if needed.
        labels: Token labels [batch, seq] or [1, total_tokens], -100 for masked positions
        compute_entropy: Whether to compute entropy (default: False)
        entropy_chunk_size: Chunk size for entropy computation (default: 512)

    Returns:
        Tuple of:
            - per_token_logps: [batch, seq] or [1, total_tokens] log probabilities for target tokens
            - per_token_entropy: Same shape as per_token_logps, or None if compute_entropy=False
    """
    per_token_entropy = None
    if compute_entropy:
        log_probs = vocab_parallel_log_softmax(logits)
        per_token_entropy = vocab_parallel_entropy(log_probs, chunk_size=entropy_chunk_size)

    per_token_logps = vocab_parallel_gather_logps(logits, labels)
    return per_token_logps, per_token_entropy
