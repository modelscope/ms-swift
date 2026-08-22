# Copyright (c) ModelScope Contributors. All rights reserved.
import math
import torch
import torch.distributed as dist
from typing import Dict, Tuple


def compute_m2po_log_ratio(
    per_token_logps: torch.Tensor,
    old_per_token_logps: torch.Tensor,
    rollout_per_token_logps: torch.Tensor = None,
) -> torch.Tensor:
    """Compute ``log(pi_current / pi_behavior)`` with an on-policy fallback."""
    behavior_logps = rollout_per_token_logps \
        if rollout_per_token_logps is not None else old_per_token_logps
    if behavior_logps is None:
        raise ValueError('M2PO requires rollout_per_token_logps or old_per_token_logps.')
    if per_token_logps.shape != behavior_logps.shape:
        raise ValueError('Current and behavior-policy log probabilities must have identical shapes, got '
                         f'{per_token_logps.shape} and {behavior_logps.shape}.')
    return per_token_logps - behavior_logps


def _distributed_context(process_group=None) -> Tuple[int, int]:
    if not dist.is_available() or not dist.is_initialized():
        return 1, 0
    return dist.get_world_size(process_group), dist.get_rank(process_group)


def _all_gather_variable(values: torch.Tensor, process_group=None) -> Tuple[torch.Tensor, int]:
    """Gather one-dimensional tensors with different lengths in process-group rank order."""
    world_size, rank = _distributed_context(process_group)
    if world_size == 1:
        return values, 0

    local_count = torch.tensor([values.numel()], dtype=torch.long, device=values.device)
    gathered_counts = [torch.zeros_like(local_count) for _ in range(world_size)]
    dist.all_gather(gathered_counts, local_count, group=process_group)
    counts = torch.cat(gathered_counts)
    max_count = int(counts.max().item())
    local_offset = int(counts[:rank].sum().item())
    if max_count == 0:
        return values.new_empty(0), local_offset

    padded = values.new_zeros(max_count)
    padded[:values.numel()] = values
    gathered_values = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered_values, padded, group=process_group)
    return torch.cat([rank_values[:count]
                      for rank_values, count in zip(gathered_values, counts.tolist())]), local_offset


def compute_m2po_mask(
    log_ratio: torch.Tensor,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    m2_threshold: float = 0.04,
    process_group=None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the final-paper M2PO token mask and batch diagnostics.

    M2PO constrains only the two quadrants where PPO clipping would be active:
    ``advantage > 0, ratio > 1`` and ``advantage < 0, ratio < 1``. The largest
    squared behavior-policy log-ratio outliers are removed until the second
    moment of the remaining trust-region tokens is at most ``m2_threshold``.

    When distributed training is initialized, the selection is computed across
    the supplied process group. Every rank therefore uses the same batch-level
    threshold while receiving a mask aligned with its local tokens.
    """
    if not math.isfinite(m2_threshold) or m2_threshold < 0:
        raise ValueError(f'm2_threshold must be finite and non-negative, got {m2_threshold}.')
    if log_ratio.shape != completion_mask.shape or log_ratio.shape != advantages.shape:
        raise ValueError('log_ratio, completion_mask, and advantages must have identical shapes, got '
                         f'{log_ratio.shape}, {completion_mask.shape}, and {advantages.shape}.')

    valid_mask = completion_mask.bool()
    world_size, _ = _distributed_context(process_group)

    with torch.no_grad():
        detached_log_ratio = log_ratio.detach()
        invalid_flag = (~torch.isfinite(detached_log_ratio) & valid_mask).any().to(dtype=torch.int32)
        if world_size > 1:
            dist.all_reduce(invalid_flag, op=dist.ReduceOp.MAX, group=process_group)
        if invalid_flag.item():
            raise ValueError('M2PO received non-finite behavior-policy log-ratios on valid completion tokens.')

        detached_advantages = advantages.detach()
        trust_region_mask = valid_mask & (((detached_advantages > 0) & (detached_log_ratio > 0))
                                          | ((detached_advantages < 0) & (detached_log_ratio < 0)))

        flat_trust_indices = torch.nonzero(trust_region_mask.reshape(-1), as_tuple=False).squeeze(-1)
        local_values = detached_log_ratio.float().square().reshape(-1)[flat_trust_indices]
        global_values, local_offset = _all_gather_variable(local_values, process_group)

        global_valid_count = valid_mask.sum().to(dtype=torch.long)
        if world_size > 1:
            dist.all_reduce(global_valid_count, op=dist.ReduceOp.SUM, group=process_group)

        trust_count = global_values.numel()
        keep_count = trust_count
        keep_global = torch.ones(trust_count, dtype=torch.bool, device=global_values.device)
        if trust_count:
            sorted_values, order = torch.sort(global_values)
            prefix_counts = torch.arange(1, trust_count + 1, dtype=sorted_values.dtype, device=sorted_values.device)
            prefix_means = torch.cumsum(sorted_values, dim=0) / prefix_counts
            keep_count = int((prefix_means <= m2_threshold).sum().item())
            keep_global.zero_()
            keep_global[order[:keep_count]] = True

        local_keep = keep_global[local_offset:local_offset + local_values.numel()]
        final_mask = valid_mask.clone()
        final_mask.reshape(-1)[flat_trust_indices] = local_keep

        zero = global_values.new_zeros(())
        m2_before = global_values.mean() if trust_count else zero
        m2_after = global_values[keep_global].mean() if keep_count else zero
        valid_count = int(global_valid_count.item())
        masked_count = trust_count - keep_count

        metrics = {
            'm2_before': m2_before,
            'm2_after': m2_after,
            'masked_fraction': zero.new_tensor(masked_count / valid_count if valid_count else 0.0),
            'trust_region_fraction': zero.new_tensor(trust_count / valid_count if valid_count else 0.0),
            'valid_count': zero.new_tensor(valid_count),
            'trust_region_count': zero.new_tensor(trust_count),
            'kept_trust_region_count': zero.new_tensor(keep_count),
        }

    return final_mask, metrics


def compute_m2po_token_loss(
    log_ratio: torch.Tensor,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    m2_threshold: float = 0.04,
    process_group=None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Return the unreduced final-paper M2PO loss and its non-differentiable mask."""
    m2po_mask, metrics = compute_m2po_mask(
        log_ratio=log_ratio,
        completion_mask=completion_mask,
        advantages=advantages,
        m2_threshold=m2_threshold,
        process_group=process_group,
    )
    # Keep the exponent finite even for a non-trust-region outlier. The mask is
    # still computed from the unclamped log-ratio, so this does not weaken M2.
    clamped_log_ratio = torch.clamp(log_ratio, min=-20, max=20)
    stable_log_ratio = log_ratio + (clamped_log_ratio - log_ratio).detach()
    ratio = torch.exp(stable_log_ratio)
    per_token_loss = torch.where(m2po_mask, -ratio * advantages, torch.zeros_like(advantages))
    return per_token_loss, m2po_mask, metrics
