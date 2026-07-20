# Copyright (c) ModelScope Contributors. All rights reserved.

import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from swift.utils import nanstd


def compute_advantages(
    rewards_per_func: torch.Tensor,
    reward_weights: torch.Tensor,
    num_generations: int,
    advantage_estimator: str = 'grpo',
    scale_rewards: str = 'group',
    kl_in_reward: bool = False,
    beta: float = 0.0,
    kl_values: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute advantages from per-function rewards.

    This is a pure tensor function suitable for all backends (HF, Megatron, Ray).
    Input tensors should already be gathered across all processes.

    Produces the **per-sequence** base advantage from rewards. The OPD-RL teacher signal
    is *not* injected here: it is a per-token signal applied later (when the base
    advantage is broadcast to ``[B, T]`` while writing it onto the batch). See
    :func:`compute_teacher_logratio` (the advantage signal) and
    :func:`compute_teacher_kl_per_token` (the k3 monitoring metric).

    Ref model KL (``kl_in_reward``) is subtracted from rewards **before** advantage
    normalization — standard GRPO/PPO regularization that prevents the policy from
    drifting too far from the reference model.

    Args:
        rewards_per_func: ``[N, n_funcs]`` per-function reward matrix.
        reward_weights: ``[n_funcs]`` weighting tensor.
        num_generations: ``K`` — completions per prompt.
        advantage_estimator: ``'grpo'``, ``'rloo'``, or ``'reinforce_plus_plus'``.
        scale_rewards: ``'batch'``, ``'group'``, ``'none'``, or ``'gdpo'``.
        kl_in_reward: Subtract ref model KL from rewards (pre-normalization).
        beta: Ref model KL penalty coefficient.
        kl_values: ``[N]`` ref model KL values (required when ``kl_in_reward=True``).

    Returns:
        ``(advantages, rewards)`` both ``[N]``.
    """
    rewards = (rewards_per_func * reward_weights.unsqueeze(0)).nansum(dim=1)

    if kl_in_reward and beta != 0.0 and kl_values is not None:
        rewards = rewards - beta * kl_values

    K = num_generations
    grouped = rewards.view(-1, K)
    group_mean = grouped.mean(dim=1).repeat_interleave(K)

    if advantage_estimator == 'rloo' and K > 1:
        advantages = rewards * K / (K - 1) - group_mean * K / (K - 1)
    else:
        advantages = rewards - group_mean

    std: Optional[torch.Tensor] = None
    if advantage_estimator == 'reinforce_plus_plus':
        if scale_rewards == 'batch':
            std = advantages.std().expand_as(advantages) if advantages.numel() > 1 else torch.zeros_like(advantages)
        elif scale_rewards == 'group':
            std = (advantages.view(-1, K).std(dim=1).repeat_interleave(K) if K > 1 else torch.zeros_like(advantages))
    else:
        if scale_rewards == 'batch':
            std = rewards.std().expand_as(rewards) if rewards.numel() > 1 else torch.zeros_like(rewards)
        elif scale_rewards == 'group':
            std = grouped.std(dim=1).repeat_interleave(K) if K > 1 else torch.zeros_like(rewards)
        elif scale_rewards == 'gdpo':
            n_funcs = rewards_per_func.shape[1]
            normalized_list = []
            for i in range(n_funcs):
                r_i = rewards_per_func[:, i].view(-1, K)
                g_mean = torch.nanmean(r_i, dim=1, keepdim=True)
                g_std = nanstd(r_i, dim=1, keepdim=True) + 1e-8
                norm_i = torch.nan_to_num((r_i - g_mean) / g_std, nan=0.0)
                normalized_list.append(reward_weights[i] * norm_i.view(-1))
            summed = sum(normalized_list)
            advantages = (summed - summed.mean()) / (summed.std() + 1e-8)
            std = None

    if std is not None and scale_rewards != 'none':
        advantages = advantages / (std + 1e-4)

    return advantages, rewards


def compute_advantages_dynamic(
    rewards_per_func: torch.Tensor,
    reward_weights: torch.Tensor,
    prompt_ids: List[str],
    request_ids: List[str],
    advantage_estimator: str = 'grpo',
    scale_rewards: str = 'group',
    kl_in_reward: bool = False,
    beta: float = 0.0,
    kl_values: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Request-aware advantage computation for dynamic sample counts.

    Groups rewards by ``prompt_id`` (via ``request_id`` deduplication) and
    computes advantages within each group. Supports variable numbers of
    completions per prompt (multi-turn scenarios).

    Like :func:`compute_advantages`, this returns only the per-sequence base advantage;
    the OPD-RL teacher signal is applied per-token later (see
    :func:`compute_teacher_logratio`).

    Input tensors should already be gathered across all processes.

    Args:
        rewards_per_func: ``[N, n_funcs]`` per-function reward matrix.
        reward_weights: ``[n_funcs]`` weighting tensor.
        prompt_ids: Length-N list of prompt identifiers.
        request_ids: Length-N list of request identifiers (may have duplicates).
        advantage_estimator: ``'grpo'``, ``'rloo'``, or ``'reinforce_plus_plus'``.
        scale_rewards: ``'batch'``, ``'group'``, ``'none'``.
        kl_in_reward: Subtract ref model KL from rewards (pre-normalization).
        beta: Ref model KL penalty coefficient.
        kl_values: ``[N]`` ref model KL values.

    Returns:
        ``(advantages, rewards)`` both ``[N]`` (with duplicate entries for repeated request_ids).
    """
    device = rewards_per_func.device
    rewards = (rewards_per_func * reward_weights.unsqueeze(0)).nansum(dim=1)

    if kl_in_reward and beta != 0.0 and kl_values is not None:
        rewards = rewards - beta * kl_values

    # Deduplicate by request_id (keep last occurrence)
    seen = {}
    for idx, rid in enumerate(request_ids):
        seen[rid] = idx
    unique_indices = torch.tensor(sorted(seen.values()), device=device)
    unique_request_ids = [request_ids[i] for i in unique_indices.cpu()]
    unique_prompt_ids = [prompt_ids[i] for i in unique_indices.cpu()]
    unique_rewards = rewards[unique_indices]

    # Group by prompt_id
    prompt_to_indices: Dict[str, List[int]] = {}
    for idx, pid in enumerate(unique_prompt_ids):
        prompt_to_indices.setdefault(pid, []).append(idx)

    prompt_means = torch.zeros(len(unique_rewards), device=device)
    for pid, idxs in prompt_to_indices.items():
        idx_t = torch.tensor(idxs, device=device)
        prompt_means[idx_t] = unique_rewards[idx_t].mean()

    if advantage_estimator == 'rloo':
        request_advantages = torch.zeros_like(unique_rewards)
        for pid, idxs in prompt_to_indices.items():
            K = len(idxs)
            idx_t = torch.tensor(idxs, device=device)
            r_group = unique_rewards[idx_t]
            if K > 1:
                request_advantages[idx_t] = r_group * K / (K - 1) - r_group.mean() * K / (K - 1)
            else:
                request_advantages[idx_t] = r_group - r_group.mean()
    else:
        request_advantages = unique_rewards - prompt_means

    # Normalize
    if advantage_estimator == 'reinforce_plus_plus':
        if scale_rewards == 'batch':
            adv_std = (request_advantages.std() if request_advantages.numel() > 1 else torch.tensor(0.0, device=device))
            prompt_stds = torch.full_like(request_advantages, adv_std)
        elif scale_rewards == 'group':
            prompt_stds = torch.zeros(len(unique_rewards), device=device)
            for pid, idxs in prompt_to_indices.items():
                idx_t = torch.tensor(idxs, device=device)
                adv_group = request_advantages[idx_t]
                prompt_stds[idx_t] = adv_group.std() if len(idxs) > 1 else 0.0
        else:
            prompt_stds = None
    else:
        if scale_rewards == 'batch':
            r_std = unique_rewards.std() if unique_rewards.numel() > 1 else torch.tensor(0.0, device=device)
            prompt_stds = torch.full_like(unique_rewards, r_std)
        elif scale_rewards == 'group':
            prompt_stds = torch.zeros(len(unique_rewards), device=device)
            for pid, idxs in prompt_to_indices.items():
                idx_t = torch.tensor(idxs, device=device)
                r_group = unique_rewards[idx_t]
                prompt_stds[idx_t] = r_group.std() if len(idxs) > 1 else 0.0
        else:
            prompt_stds = None

    if prompt_stds is not None and scale_rewards != 'none':
        request_advantages = request_advantages / (prompt_stds + 1e-4)

    # Map back to original order
    rid_to_idx = {rid: idx for idx, rid in enumerate(unique_request_ids)}
    indices_in_unique = torch.tensor([rid_to_idx[r] for r in request_ids], device=device)
    advantages = request_advantages[indices_in_unique]

    return advantages, rewards


def compute_teacher_kl_per_token(
    teacher_per_token_logps: torch.Tensor,
    policy_per_token_logps: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-token teacher KL (OPD-RL) via the k3 estimator -- **monitoring only**.

    ``teacher`` and ``policy`` logps are token-in-token-out on the *same* sampled tokens
    (the teacher logp on the student-sampled token).
    It is the magnitude of the reverse KL between student and teacher and a good "how far
    is the student from the teacher" gauge -- it should *decrease* over training.

    Args:
        teacher_per_token_logps: ``[B, T]`` teacher logp on sampled tokens.
        policy_per_token_logps: ``[B, T]`` student (old) logp on the same tokens.
        completion_mask: ``[B, T]`` response-token mask.

    Returns:
        ``[B, T]`` per-token teacher KL (masked outside the response).
    """
    d = teacher_per_token_logps - policy_per_token_logps
    # Mask before exp so padding sentinel values cannot overflow and produce inf * 0.
    d = d.masked_fill(~completion_mask.bool(), 0.0)
    per_token = torch.exp(d) - d - 1
    return per_token


def compute_teacher_logratio(
    teacher_per_token_logps: torch.Tensor,
    policy_per_token_logps: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-token signed teacher log-ratio (OPD-RL) -- the k1 reverse-KL estimator.

    This is the correct OPD-RL policy-gradient signal (PG OPD): the negative single-sample
    reverse-KL estimate used as a reward, ``r_t = teacher_logp(y_t) - student_logp(y_t)``

    Args:
        teacher_per_token_logps: ``[B, T]`` teacher logp on sampled tokens.
        policy_per_token_logps: ``[B, T]`` student (old) logp on the same tokens.
        completion_mask: ``[B, T]`` response-token mask.

    Returns:
        ``[B, T]`` per-token signed log-ratio (masked outside the response).
    """
    d = teacher_per_token_logps - policy_per_token_logps
    return d.masked_fill(~completion_mask.bool(), 0.0)


def expand_advantage_to_per_token(
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    teacher_per_token_logps: Optional[torch.Tensor] = None,
    policy_per_token_logps: Optional[torch.Tensor] = None,
    teacher_kl_coef: float = 0.0,
) -> torch.Tensor:
    """Expand the per-sequence base advantage ``[B]`` to per-token ``[B, T]``.

    Broadcasting the per-sequence advantage to per-token happens *here* (at batch
    construction) rather than in the loss, so the OPD-RL teacher signal can be added
    per token: ``adv_t = base_adv + coef * (teacher_logp_t - student_logp_t)`` .
    Without a teacher this is a plain broadcast.

    Args:
        advantages: ``[B]`` per-sequence base advantage.
        completion_mask: ``[B, T]`` response-token mask (defines the token frame).
        teacher_per_token_logps: ``[B, T]`` teacher logp on sampled tokens (OPD-RL).
        policy_per_token_logps: ``[B, T]`` student (old) logp on the same tokens.
        teacher_kl_coef: Coefficient for the per-token teacher signal.

    Returns:
        ``[B, T]`` per-token advantage.
    """
    per_token_adv = advantages.unsqueeze(1).expand_as(completion_mask).clone()
    if teacher_per_token_logps is not None and teacher_kl_coef != 0.0:
        signed = compute_teacher_logratio(teacher_per_token_logps, policy_per_token_logps, completion_mask)
        per_token_adv = per_token_adv + teacher_kl_coef * signed
    return per_token_adv


def apply_rlsd_reweight(
    base_advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    teacher_per_token_logps: torch.Tensor,
    policy_per_token_logps: torch.Tensor,
    lam: float,
    clip_range: float,
    negative_only: bool = False,
) -> torch.Tensor:
    """RLSD (Self-Distilled RLVR) token-level advantage reweighting.

    Redistributes the per-sequence GRPO advantage *inside* each trajectory using the
    teacher-vs-student log-prob gap, without ever flipping the sign of the environment
    reward (the reweight is strictly positive). Mirrors ``_build_stgca_advantages`` in
    the reference implementation (RLSD/verl/workers/actor/dp_opsd_actor.py)::

        delta_t  = stop_grad(logP_T(y_t) - logP_S(y_t))
        w_t      = exp(sign(A) * delta_t)
        reweight = (1 - lam) + lam * clip(w_t, 1 - clip_range, 1 + clip_range)
        A_hat_t  = A * stop_grad(reweight)

    ``lam`` mixes between pure GRPO (``lam=0`` -> plain broadcast) and full RLSD reweighting
    (``lam=1``). The teacher is the "informed self": the same policy conditioned on the
    ground-truth answer, scoring the *same* sampled tokens (``teacher_per_token_logps``).
    ``policy_per_token_logps`` is the student side (the batch-time old logps).

    Args:
        base_advantages: ``[B]`` per-sequence GRPO advantage.
        completion_mask: ``[B, T]`` response-token mask.
        teacher_per_token_logps: ``[B, T]`` teacher logp on sampled tokens.
        policy_per_token_logps: ``[B, T]`` student (old) logp on the same tokens.
        lam: Effective mixing weight (already schedule-adjusted).
        clip_range: Evidence-weight clip epsilon ``eps_w``.
        negative_only: When True, only reweight sequences with ``A < 0`` (incorrect
            responses); ``A >= 0`` sequences keep pure GRPO advantages.

    Returns:
        ``[B, T]`` per-token reweighted advantage (masked outside the response).
    """
    mask = completion_mask
    base_bt = base_advantages.unsqueeze(1).expand_as(mask)
    delta = (teacher_per_token_logps.detach() - policy_per_token_logps.detach()) * mask
    sign_a = torch.sign(base_bt)
    weights = torch.exp(sign_a * delta) * mask
    clipped = torch.clamp(weights, min=1.0 - clip_range, max=1.0 + clip_range)
    reweight = (1.0 - lam) + lam * clipped
    if negative_only:
        seq_neg = (base_advantages < 0).float().unsqueeze(1)
        reweight = seq_neg * reweight + (1.0 - seq_neg)
    return base_bt * reweight.detach() * mask


def _sdar_agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str) -> torch.Tensor:
    """Masked loss aggregation mirroring verl ``agg_loss`` (verl/trainer/ppo/core_algos.py).

    Supported modes: ``token-mean`` (default), ``seq-mean-token-sum``, ``seq-mean-token-mean``.
    """
    if loss_agg_mode == 'token-mean':
        return (loss_mat * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
    if loss_agg_mode == 'seq-mean-token-sum':
        seq_losses = (loss_mat * loss_mask).sum(dim=-1)
        return seq_losses.mean()
    if loss_agg_mode == 'seq-mean-token-mean':
        seq_losses = (loss_mat * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1).clamp(min=1.0)
        return seq_losses.mean()
    raise ValueError(f'Unknown loss_agg_mode: {loss_agg_mode}')


def compute_sdar_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gate_beta: float = 5.0,
    loss_agg_mode: str = 'token-mean',
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """SDAR (Self-Distilled Agentic RL) confidence-gated teacher distillation loss.

    Exact port of ``compute_sdar_loss`` in the reference (SDAR/verl/trainer/ppo/sdar_utils.py).
    Token-level gated distillation where the gate is derived from the teacher-vs-student
    log-prob gap, so tokens where the teacher is more confident receive a stronger
    distillation signal::

        delta_t = logP_T(y_t) - logP_S(y_t)
        g_t     = sigmoid(gate_beta * delta_t)                 # detached, no grad
        L_SDAR  = agg( g_t * (logP_T(y_t) - logP_S(y_t)) )     # student keeps grad

    The gate ``g_t`` and the teacher log-probs are detached, so gradients flow only through
    the student log-probs. This auxiliary loss is *added* to the GRPO policy loss
    (``loss = policy_loss + sdar_loss_coef * L_SDAR``); it does not modify the advantage.

    Args:
        student_log_probs: ``[B, T]`` current-policy logP_theta(y_t | x, y_<t) (retains grad).
        teacher_log_probs: ``[B, T]`` teacher logP(y_t | x, r, y_<t) on the same sampled tokens.
            The teacher sees skill-augmented / privileged input ``r``; frozen (no grad).
        response_mask: ``[B, T]`` mask for valid response tokens.
        gate_beta: sigmoid gate temperature; higher = sharper gating.
        loss_agg_mode: aggregation mode (default ``token-mean``, matching the reference).

    Returns:
        ``(loss, metrics)`` where ``loss`` is a scalar and ``metrics`` holds gating statistics.
    """
    teacher_log_probs = teacher_log_probs.detach()

    delta_t = teacher_log_probs - student_log_probs.detach()

    gate = torch.sigmoid(gate_beta * delta_t).detach()

    kl_per_token = teacher_log_probs - student_log_probs

    gated_kl = gate * kl_per_token

    loss = _sdar_agg_loss(gated_kl, response_mask, loss_agg_mode)

    with torch.no_grad():
        mask_sum = response_mask.sum().clamp(min=1)
        gate_mean = (gate * response_mask).sum() / mask_sum
        gate_active = ((gate > 0.5).float() * response_mask).sum() / mask_sum
        gap_mean = (delta_t * response_mask).sum() / mask_sum

    metrics = {
        'sdar/gate_mean': gate_mean.item(),
        'sdar/gate_active_ratio': gate_active.item(),
        'sdar/teacher_gap_mean': gap_mean.item(),
        'sdar/loss': loss.detach().item(),
    }

    return loss, metrics


@dataclass
class RewardMetrics:
    """Reward statistics for logging."""
    reward_mean: float
    reward_std: float
    frac_reward_zero_std: float
    per_func_mean: Dict[str, float]
    per_func_std: Dict[str, float]


def compute_reward_metrics(
    rewards: torch.Tensor,
    rewards_per_func: torch.Tensor,
    reward_func_names: List[str],
    num_generations: int,
    scale_rewards: str,
) -> RewardMetrics:
    """Compute reward statistics for monitoring.
    Args:
        rewards: ``[N]`` scalar rewards (already weighted).
        rewards_per_func: ``[N, n_funcs]`` per-function rewards.
        reward_func_names: Names of reward functions.
        num_generations: ``K``.
        scale_rewards: Scaling strategy (affects std computation).
    Returns:
        :class:`RewardMetrics` with all statistics.
    """
    group_rewards = rewards.view(-1, num_generations)
    reward_mean = group_rewards.mean(-1).mean().item()

    if scale_rewards in ('group', 'none', 'gdpo'):
        reward_std = group_rewards.std(-1).mean().item() if num_generations > 1 else 0.0
    elif scale_rewards == 'batch':
        reward_std = rewards.std().item() if rewards.numel() > 1 else 0.0
    else:
        reward_std = 0.0

    if num_generations > 1:
        is_std_zero = torch.isclose(group_rewards.std(dim=1), torch.zeros_like(group_rewards.std(dim=1)))
    else:
        is_std_zero = torch.ones(group_rewards.size(0), dtype=torch.bool, device=group_rewards.device)
    frac_zero_std = is_std_zero.float().mean().item()

    per_func_mean = {}
    per_func_std = {}
    for i, name in enumerate(reward_func_names):
        col = rewards_per_func[:, i]
        per_func_mean[name] = torch.nanmean(col).item()
        valid = col[~torch.isnan(col)]
        per_func_std[name] = valid.std().item() if valid.numel() > 1 else 0.0

    return RewardMetrics(
        reward_mean=reward_mean,
        reward_std=reward_std,
        frac_reward_zero_std=frac_zero_std,
        per_func_mean=per_func_mean,
        per_func_std=per_func_std,
    )
