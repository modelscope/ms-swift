"""Loss configuration for the SFT path.

Explicit loss assembly so the GA-correctness contract is visible in the dev layer,
not inherited implicitly from twinkle's default.

Why reduction='sum':
  legacy swift computes ``loss = outputs.loss.sum() / num_items_in_batch`` (SUM with a
  single token denominator across the whole gradient-accumulation window). twinkle's SUM
  path matches this: calculate_loss accumulates loss.sum() + num_tokens across micro-batches
  and clip_grad_norm divides the grad by the total token count. This makes GA=k/bs=1
  gradient-equivalent to GA=1/bs=k. reduction='mean' would instead weight each micro-batch
  equally (denominator = gradient_accumulation_steps), which is only correct when every
  micro-batch has the same token count. So SFT must use SUM.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, Optional

from twinkle.loss import Loss

if TYPE_CHECKING:
    from swift.dev.config import RLHFConfig
    from swift.dev.model import TrainableModel

# Losses that read ``outputs['embeddings']`` (the pooled, L2-normalized sentence vector) rather
# than logits. Keys are swift loss names as they appear in TrainConfig.loss_type.
EMBEDDING_LOSS_TYPES = ('infonce', 'cosine_similarity', 'contrastive', 'online_contrastive')

# Reranker (cross-encoder) losses: score query-document pairs from logits, no pair interleaving.
RERANKER_LOSS_TYPES = ('pointwise_reranker', 'listwise_reranker')

# Sequence-classification problem types; picks the loss objective (MSE / CE / BCE).
PROBLEM_TYPES = ('regression', 'single_label_classification', 'multi_label_classification')


def configure_loss(model: TrainableModel,
                   *,
                   loss_type: str = 'cross_entropy',
                   reduction: str = 'sum',
                   enable_channel_loss: bool = False,
                   dft: bool = False,
                   **kwargs) -> None:
    """Set the SFT loss on ``model`` with an explicit reduction (default 'sum').

    Args:
        model: a twinkle-derived Model (has set_loss).
        loss_type: swift loss name resolved via the unified naming layer (SFT default
            'cross_entropy'). Non-CE losses (grpo/dpo/...) resolve too but belong to
            their own recipes; SFT keeps CE.
        reduction: 'sum' (default; GA-correct, aligns legacy) or 'mean'.
        enable_channel_loss: Report token-level loss grouped by the dataset's sample-level ``channel`` field.
        dft: Apply DFT entropy weighting before total and channel aggregation.
    """
    from swift.dev.naming import resolve_loss

    if loss_type != 'cross_entropy':
        raise NotImplementedError(f"SFT configure_loss only supports 'cross_entropy', got {loss_type!r}")
    loss_cls = resolve_loss('channel' if enable_channel_loss else loss_type)
    model.set_loss(loss_cls(reduction=reduction, dft=dft, **kwargs))


def configure_embedding_loss(model: TrainableModel,
                             *,
                             loss_type: str = 'infonce',
                             mrl_dims: Optional[Dict[int, float]] = None,
                             **kwargs) -> None:
    """Set the embedding (contrastive) loss on ``model``.

    Separate from :func:`configure_loss` because the two have incompatible contracts: SFT's CE takes
    ``reduction`` and normalizes by token count, whereas these losses score whole sentences and
    report ``num_tokens=0`` (no per-token normalization), so passing a reduction would be rejected.

    Applied on BOTH backends, unlike SFT. Megatron computes CE internally and ignores set_loss for
    causal_lm, but under ``task='embedding'`` its scheduler pools to ``[n_seqs, hidden]`` and calls
    ``loss_instance`` explicitly -- and MegatronModel.set_loss additionally binds ``process_group``
    to the DP group so InfonceLoss's in-batch all-gather cannot deadlock earlier PP stages.

    Args:
        model: a twinkle-derived Model (has set_loss).
        loss_type: one of :data:`EMBEDDING_LOSS_TYPES`.
        mrl_dims: Matryoshka ``{dim: weight}``. ``None`` trains the full width only. Rejected by
            ``cosine_similarity``, whose absolute-similarity target has no per-prefix reading.
        **kwargs: forwarded to the loss constructor (temperature, margin, distance_metric, ...).
    """
    from swift.dev.naming import resolve_loss

    if loss_type not in EMBEDDING_LOSS_TYPES:
        raise NotImplementedError(f'configure_embedding_loss supports {list(EMBEDDING_LOSS_TYPES)}, '
                                  f'got {loss_type!r}. Reranker losses score query-document pairs '
                                  'from logits and belong to a reranker recipe.')
    loss_cls = resolve_loss(loss_type)
    if mrl_dims is not None:
        kwargs['mrl_dims'] = mrl_dims
    # Pass an INSTANCE: twinkle's construct_class returns an instance unchanged, so constructing here
    # keeps the mrl_dims validation (EmbeddingLoss.__init__ raises for cosine_similarity) at the dev
    # call site rather than deep inside twinkle.
    model.set_loss(loss_cls(**kwargs))


def configure_reranker_loss(model: TrainableModel, *, loss_type: str = 'pointwise_reranker', **kwargs) -> None:
    """Set a reranker (cross-encoder) loss on ``model``.

    Reads ``outputs['logits']`` (a per-pair relevance score), so it is separate from the embedding
    losses. Works on both backends: transformers rides a num_labels=1 SequenceClassification head;
    Megatron maps reranker to the bridge's seq_cls head (num_labels=1). The last-valid-token pooling
    that produces ``[n_seqs, 1]`` is done by the head (transformers) or the processor (Megatron)
    before this loss runs.

    Args:
        model: a twinkle-derived Model (has set_loss).
        loss_type: one of :data:`RERANKER_LOSS_TYPES`.
        **kwargs: forwarded to the loss constructor (e.g. temperature for listwise).
    """
    from swift.dev.naming import resolve_loss

    if loss_type not in RERANKER_LOSS_TYPES:
        raise NotImplementedError(f'configure_reranker_loss supports {list(RERANKER_LOSS_TYPES)}, '
                                  f'got {loss_type!r}.')
    loss_cls = resolve_loss(loss_type)
    model.set_loss(loss_cls(**kwargs))


def configure_seq_cls_loss(model: TrainableModel, *, problem_type: str, num_labels: int, **kwargs) -> None:
    """Set the sequence-classification loss on ``model``.

    ``problem_type`` is REQUIRED (not inferred): it selects the objective (regression -> MSE,
    single_label -> CE, multi_label -> BCE), matching HF/legacy numerics. The head has already
    reduced logits to ``[B, num_labels]`` (transformers head / Megatron processor pooling) before
    this loss runs.

    Args:
        model: a twinkle-derived Model (has set_loss).
        problem_type: one of :data:`PROBLEM_TYPES`.
        num_labels: class count (1 for regression), used for the CE reshape and the MSE squeeze.
    """
    from swift.dev.naming import resolve_loss

    if problem_type not in PROBLEM_TYPES:
        raise ValueError(f'problem_type must be one of {list(PROBLEM_TYPES)}, got {problem_type!r}. '
                         'It is required (not inferred) so the training objective is explicit.')
    loss_cls = resolve_loss('seq_cls')
    model.set_loss(loss_cls(problem_type=problem_type, num_labels=num_labels, **kwargs))


# rlhf_type -> the twinkle loss name it maps onto. Most are same-named in twinkle's torch_loss_mapping;
# the exceptions are: 'kto' (no standalone loss yet -> the DPO family's paired 'kto_pair' variant), and
# 'ppo' (whose POLICY loss is the same clipped surrogate as GRPO -> 'grpo'; its critic is a separate
# value loss set by configure_ppo_value_loss, not here).
_RLHF_LOSS_NAME = {
    'grpo': 'grpo',
    'dpo': 'dpo',
    'cpo': 'cpo',
    'orpo': 'orpo',
    'simpo': 'simpo',
    'gkd': 'gkd',
    'rm': 'reward',
    'kto': 'dpo',
    'ppo': 'grpo',
}


class _AdvancedGKDLoss(Loss):
    """GKD with the optional supervised CE term used for dataset-sourced batches."""

    require_logits = True

    def __init__(self, base_loss, *, sft_alpha: float):
        self.base_loss = base_loss
        self.sft_alpha = sft_alpha
        self.require_logps = getattr(base_loss, 'require_logps', True)
        self.require_entropy = getattr(base_loss, 'require_entropy', False)

    def __call__(self, inputs, outputs, *, apply_sft_loss=False, **kwargs):
        import torch.nn.functional as F
        from twinkle.data_format import LossOutput

        result = self.base_loss(inputs, outputs, **kwargs)
        loss = result['loss']
        if apply_sft_loss and self.sft_alpha > 0:
            labels = inputs['labels']
            logits = outputs['logits']
            if logits.shape[1] != labels.shape[1]:
                logits = logits[:, -labels.shape[1]:]
            sft_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100, reduction='mean')
            loss = loss + self.sft_alpha * sft_loss
        return LossOutput(loss=loss, num_tokens=result['num_tokens'])


class _ConfiguredGRPOLoss(Loss):
    """Apply dev-only GRPO controls around a twinkle policy loss."""

    def __init__(
        self,
        base_loss,
        *,
        importance_sampling_level: str,
        delta: Optional[float],
        top_entropy_quantile: float,
        log_entropy: bool,
        rollout_importance_sampling_mode: Optional[str],
        rollout_importance_sampling_threshold: float,
        log_rollout_offpolicy_metrics: bool,
        off_policy_sequence_mask_delta: Optional[float],
        loss_type: str,
        fipo_decay_rate: float,
        fipo_clip_range: Optional[float],
        fipo_clip_high_only: bool,
        fipo_safety_threshold: Optional[float],
    ):
        self.base_loss = base_loss
        self.importance_sampling_level = importance_sampling_level
        self.delta = delta
        self.top_entropy_quantile = top_entropy_quantile
        self.log_entropy = log_entropy
        self.rollout_importance_sampling_mode = rollout_importance_sampling_mode
        self.rollout_importance_sampling_threshold = rollout_importance_sampling_threshold
        self.log_rollout_offpolicy_metrics = log_rollout_offpolicy_metrics
        self.off_policy_sequence_mask_delta = off_policy_sequence_mask_delta
        self.loss_type = loss_type
        self.fipo_gamma = 2**(-1 / fipo_decay_rate)
        self.fipo_clip_range = fipo_clip_range
        self.fipo_clip_high_only = fipo_clip_high_only
        self.fipo_safety_threshold = fipo_safety_threshold
        self.require_logps = True
        self.require_logits = getattr(base_loss, 'require_logits', False)
        self.require_entropy = log_entropy or top_entropy_quantile < 1.0

    def _importance_weights(self, logps, old_logps, mask):
        import torch

        log_ratio = torch.clamp(logps - old_logps, min=-20.0, max=20.0)
        if self.importance_sampling_level == 'token':
            log_weights = log_ratio
        else:
            sequence = ((log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).unsqueeze(-1)
            if self.importance_sampling_level == 'sequence':
                log_weights = sequence
            elif self.importance_sampling_level == 'sequence_token':
                log_weights = logps - logps.detach() + sequence.detach()
            else:
                raise ValueError(f'Unknown importance_sampling_level={self.importance_sampling_level!r}.')
        return log_ratio, torch.exp(log_weights)

    def _rollout_weights(self, log_ratio, mask):
        import torch

        ratio = torch.exp(torch.clamp(log_ratio, min=-20.0, max=20.0))
        mode = self.rollout_importance_sampling_mode
        threshold = self.rollout_importance_sampling_threshold
        if mode == 'token_truncate':
            return torch.clamp(ratio, max=threshold)
        if mode == 'token_mask':
            return torch.where(ratio <= threshold, ratio, torch.zeros_like(ratio))
        sequence = torch.exp((torch.log(ratio.clamp(min=1e-10)) * mask).sum(-1)
                             / mask.sum(-1).clamp(min=1.0))
        if mode == 'sequence_truncate':
            return torch.clamp(sequence, max=threshold).unsqueeze(-1).expand_as(ratio)
        if mode == 'sequence_mask':
            return ratio * (sequence <= threshold).unsqueeze(-1)
        return ratio

    def _policy_loss(self, ratio, advantages, logps):
        """Preserve legacy dual-clip ordering: PPO clipping sees the raw ratio."""
        import torch

        if self.delta is None or self.loss_type not in {'grpo', 'dapo', 'fipo', 'bnpo', 'dr_grpo'}:
            return self.base_loss._compute_per_token_loss(ratio, advantages, logps)
        clipped_ratio = torch.clamp(
            ratio,
            1 - self.base_loss.epsilon,
            1 + self.base_loss.epsilon_high,
        )
        dual_clipped_ratio = torch.clamp(ratio, max=self.delta)
        return -torch.min(dual_clipped_ratio * advantages, clipped_ratio * advantages)

    def _fipo_weights(self, log_ratio, ratio, advantages, mask):
        import torch

        future_delta = log_ratio.masked_fill(~mask, 0.0)
        if self.delta is not None:
            future_delta = torch.where(ratio > self.delta, torch.zeros_like(future_delta), future_delta)
        seq_len = future_delta.shape[1]
        positions = torch.arange(seq_len, device=log_ratio.device).unsqueeze(1)
        future_kl = torch.zeros_like(future_delta)
        for start in range(0, seq_len, 128):
            end = min(seq_len, start + 128)
            block_positions = torch.arange(start, end, device=log_ratio.device).unsqueeze(0)
            distance = block_positions - positions
            decay = torch.pow(
                torch.as_tensor(self.fipo_gamma, dtype=log_ratio.dtype, device=log_ratio.device),
                distance.clamp(min=0))
            decay = decay * (distance >= 0).to(log_ratio.dtype)
            future_kl += torch.matmul(future_delta[:, start:end], decay.t())
        weights = torch.exp(future_kl.masked_fill(~mask, 0.0))
        if self.fipo_clip_range:
            lower = 1.0 if self.fipo_clip_high_only else 1.0 - self.fipo_clip_range
            weights = torch.clamp(weights, min=lower, max=1.0 + self.fipo_clip_range)
        if self.fipo_safety_threshold is not None:
            unsafe = (advantages < 0) & (ratio > self.fipo_safety_threshold)
            weights = torch.where(unsafe, torch.clamp(weights, min=0.8, max=1.0), weights)
        return weights.detach()

    def __call__(  # noqa: C901
        self,
        inputs,
        outputs,
        *,
        old_logps=None,
        ref_logps=None,
        advantages=None,
        rollout_logps=None,
        truncated=None,
        **kwargs,
    ):
        import torch
        from twinkle.data_format import LossOutput

        labels = inputs['labels']
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        logps = outputs.get('logps')
        if logps is None:
            raise RuntimeError('Configured GRPO loss requires outputs["logps"].')
        alignment_mask = labels.ne(getattr(self.base_loss, 'ignore_index', -100))
        device, dtype = logps.device, logps.dtype
        old = (logps.detach() if old_logps is None else self.base_loss._pad_and_align_to_batch(
            old_logps, alignment_mask, device, dtype))
        advantages = self.base_loss._pad_and_align_to_batch(advantages, alignment_mask, device, dtype)
        mask = alignment_mask.clone()

        channel_loss = {}
        entropy_mask = None
        if self.require_entropy:
            entropies = outputs.get('entropies')
            if entropies is None:
                raise RuntimeError('Entropy logging/filtering requires outputs["entropies"].')
            if self.log_entropy:
                count = alignment_mask.sum().detach().float()
                channel_loss['entropy'] = torch.stack(
                    ((entropies * alignment_mask).sum().detach().float(), count))
            if self.top_entropy_quantile < 1.0 and alignment_mask.any():
                threshold = torch.quantile(
                    entropies[alignment_mask].float(), 1.0 - self.top_entropy_quantile)
                entropy_mask = entropies.ge(threshold)

        if truncated is not None:
            truncated_mask = torch.as_tensor(truncated, dtype=torch.bool, device=device).reshape(-1, 1)
            mask = mask & ~truncated_mask

        log_ratio, ratio = self._importance_weights(logps, old, mask)
        per_token_loss = self._policy_loss(ratio, advantages, logps)
        if self.loss_type == 'fipo':
            per_token_loss = per_token_loss * self._fipo_weights(log_ratio, ratio, advantages, mask)
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        beta = float(getattr(self.base_loss, 'beta', 0.0))
        if beta > 0.0 and ref_logps is not None:
            ref = self.base_loss._pad_and_align_to_batch(ref_logps, alignment_mask, device, dtype)
            ref_delta = torch.clamp(ref - logps, min=-20.0, max=20.0)
            per_token_kl = torch.clamp(torch.exp(ref_delta) - ref_delta - 1.0, min=-10.0, max=10.0)
            per_token_loss = per_token_loss + beta * per_token_kl

        rollout = None
        if rollout_logps is not None:
            rollout = self.base_loss._pad_and_align_to_batch(rollout_logps, alignment_mask, device, dtype)
            rollout_log_ratio = old - rollout
            if self.log_rollout_offpolicy_metrics:
                count = mask.sum().detach().float()
                channel_loss['rollout_log_ratio'] = torch.stack(
                    ((rollout_log_ratio.abs() * mask).sum().detach().float(), count))
            if self.rollout_importance_sampling_mode is not None:
                per_token_loss = per_token_loss * self._rollout_weights(rollout_log_ratio, mask)
        elif self.rollout_importance_sampling_mode is not None:
            raise ValueError('rollout_importance_sampling_mode requires rollout_logps from the sampler.')

        if self.off_policy_sequence_mask_delta is not None:
            old_policy = rollout if rollout is not None else old
            sequence_delta = ((old_policy - logps) * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
            sequence_advantage = (advantages * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
            keep = ~((sequence_delta > self.off_policy_sequence_mask_delta) & (sequence_advantage < 0))
            mask = mask & keep.unsqueeze(-1)

        if self.loss_type == 'fipo':
            loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            loss = self.base_loss._aggregate_loss(per_token_loss, mask, **kwargs)
        result = LossOutput(loss=loss, num_tokens=0)
        if channel_loss:
            result['channel_loss'] = channel_loss
        return result


class _AdvancedGRPOLoss(Loss):
    """GRPO plus optional SDAR distillation and CHORD's auxiliary SFT objective."""

    def __init__(self, base_loss, *, sdar_loss_coef: float = 0.0, sdar_gate_beta: float = 5.0):
        self.base_loss = base_loss
        self.sdar_loss_coef = sdar_loss_coef
        self.sdar_gate_beta = sdar_gate_beta
        self.require_logps = True
        self.require_logits = True
        self.require_entropy = getattr(base_loss, 'require_entropy', False)

    @staticmethod
    def _slice_batch(values, end):
        if values is None:
            return None
        try:
            return values[:end]
        except (TypeError, KeyError):
            return values

    def __call__(self, inputs, outputs, *, chord_count=0, chord_mu=0.0, chord_phi=False, teacher_logps=None, **kwargs):
        import torch
        import torch.nn.functional as F
        from twinkle.data_format import LossOutput

        labels = inputs['labels']
        batch_size = labels.shape[0]
        rl_count = batch_size - int(chord_count)
        if rl_count < 1:
            raise ValueError('advanced GRPO batches must contain at least one rollout sample.')
        rl_inputs = dict(inputs)
        rl_inputs['labels'] = labels[:rl_count]
        rl_outputs = {
            name: self._slice_batch(value, rl_count)
            for name, value in outputs.items()
        }
        rl_kwargs = {
            name: self._slice_batch(value, rl_count)
            for name, value in kwargs.items()
        }
        result = self.base_loss(rl_inputs, rl_outputs, **rl_kwargs)
        loss = result['loss'] * (1.0 - float(chord_mu))

        if teacher_logps is not None and self.sdar_loss_coef > 0:
            from swift.rl_core.advantage import compute_sdar_loss

            student_logps = rl_outputs['logps']
            response_mask = rl_inputs['labels'].ne(-100)
            teacher = self.base_loss._pad_and_align_to_batch(
                teacher_logps, response_mask, student_logps.device, student_logps.dtype)
            sdar_loss, _ = compute_sdar_loss(student_logps, teacher, response_mask, self.sdar_gate_beta)
            loss = loss + self.sdar_loss_coef * sdar_loss

        if chord_count:
            chord_logits = outputs['logits'][rl_count:]
            chord_labels = labels[rl_count:]
            token_loss = F.cross_entropy(
                chord_logits.reshape(-1, chord_logits.shape[-1]),
                chord_labels.reshape(-1),
                ignore_index=-100,
                reduction='none')
            valid = chord_labels.reshape(-1).ne(-100)
            if chord_phi:
                probability = torch.exp(-token_loss.detach())
                token_loss = token_loss * probability * (1.0 - probability)
            chord_loss = token_loss[valid].mean() if valid.any() else token_loss.sum() * 0.0
            loss = loss + float(chord_mu) * chord_loss
        output = LossOutput(loss=loss, num_tokens=0)
        if result.get('channel_loss') is not None:
            output['channel_loss'] = result['channel_loss']
        return output


def configure_rlhf_loss(model: TrainableModel, rlhf_config: 'RLHFConfig') -> None:
    """Set the RLHF/RL loss on ``model`` from ``rlhf_config.rlhf_type`` and its hyperparameters.

    Peer of :func:`configure_loss`, for the RLHF recipes (run_grpo / run_dpo / run_gkd / run_ppo). The
    heavy losses themselves already live in twinkle (GRPO family, the DPO family, GKD) plus RewardLoss
    added alongside; this only maps the dev Config onto the right constructor, so a recipe never
    hand-picks a loss class or its argument names.

    Only the fields each algorithm actually reads are forwarded, so twinkle's own defaults stand for
    the rest (e.g. an unset ``beta`` leaves the loss default rather than being overwritten with None).
    ``beta`` in particular is normally pre-filled per-algorithm by process.py::_derive_rlhf_beta.

    rlhf_type='ppo' sets only the POLICY loss (the shared clipped surrogate, GRPOLoss, with
    epsilon=cliprange and KL applied in the loop's reward shaping rather than the loss). Its critic is
    a separate value model whose loss is set by :func:`configure_ppo_value_loss`.
    """
    from swift.dev.naming import resolve_loss

    rlhf_type = rlhf_config.rlhf_type
    if rlhf_type not in _RLHF_LOSS_NAME:
        raise ValueError(f'Unknown rlhf_type={rlhf_type!r}; expected one of {sorted(_RLHF_LOSS_NAME)}.')

    loss_name = _RLHF_LOSS_NAME[rlhf_type]
    grpo_loss_type = 'grpo'
    if rlhf_type == 'grpo' and rlhf_config.loss_type:
        grpo_loss_type = rlhf_config.loss_type[0]
        loss_name = {'dapo': 'bnpo', 'fipo': 'grpo'}.get(grpo_loss_type, grpo_loss_type)
    loss_cls = resolve_loss(loss_name)
    loss = loss_cls(**_rlhf_loss_kwargs(rlhf_type, rlhf_config))
    configured_grpo = bool(
        rlhf_type == 'grpo' and (grpo_loss_type == 'fipo' or rlhf_config.importance_sampling_level != 'token'
                                  or rlhf_config.delta is not None or rlhf_config.top_entropy_quantile < 1.0
                                  or rlhf_config.log_entropy or rlhf_config.overlong_filter
                                  or rlhf_config.rollout_importance_sampling_mode
                                  or rlhf_config.log_rollout_offpolicy_metrics
                                  or rlhf_config.off_policy_sequence_mask_delta is not None))
    if configured_grpo:
        loss = _ConfiguredGRPOLoss(
            loss,
            importance_sampling_level=rlhf_config.importance_sampling_level,
            delta=rlhf_config.delta,
            top_entropy_quantile=rlhf_config.top_entropy_quantile,
            log_entropy=rlhf_config.log_entropy,
            rollout_importance_sampling_mode=rlhf_config.rollout_importance_sampling_mode,
            rollout_importance_sampling_threshold=rlhf_config.rollout_importance_sampling_threshold,
            log_rollout_offpolicy_metrics=rlhf_config.log_rollout_offpolicy_metrics,
            off_policy_sequence_mask_delta=rlhf_config.off_policy_sequence_mask_delta,
            loss_type=grpo_loss_type,
            fipo_decay_rate=rlhf_config.fipo_decay_rate,
            fipo_clip_range=rlhf_config.fipo_clip_range,
            fipo_clip_high_only=rlhf_config.fipo_clip_high_only,
            fipo_safety_threshold=rlhf_config.fipo_safety_threshold)
    if rlhf_type == 'grpo' and (rlhf_config.chord_sft_dataset or rlhf_config.sdar_loss_coef > 0):
        loss = _AdvancedGRPOLoss(
            loss, sdar_loss_coef=rlhf_config.sdar_loss_coef, sdar_gate_beta=rlhf_config.sdar_gate_beta)
    elif rlhf_type == 'gkd' and rlhf_config.sft_alpha > 0:
        loss = _AdvancedGKDLoss(loss, sft_alpha=rlhf_config.sft_alpha)
    model.set_loss(loss)


def configure_ppo_value_loss(value_model: TrainableModel, rlhf_config: 'RLHFConfig') -> None:
    """Set PPO's clipped value-regression loss on the critic (a seq_cls num_labels=1 value model).

    Separate from :func:`configure_rlhf_loss` because PPO trains two models with two objectives: the
    policy (clipped surrogate, set by configure_rlhf_loss) and the critic (this value loss). Forwards
    ``cliprange_value`` and ``vf_coef``; the loop supplies ``returns``/``old_values`` per step.
    """
    from swift.dev.naming import resolve_loss

    loss_cls = resolve_loss('ppo_value')
    value_model.set_loss(loss_cls(cliprange_value=rlhf_config.cliprange_value, vf_coef=rlhf_config.vf_coef))


def _rlhf_loss_kwargs(rlhf_type: str, rlhf_config: 'RLHFConfig') -> Dict[str, Any]:
    """The constructor kwargs for one rlhf_type's loss, forwarding only the fields it reads.

    Split into an online (policy-gradient / distillation) and a preference/pairwise half so each stays
    a short, single-purpose mapping rather than one long branch.
    """
    if rlhf_type in ('grpo', 'gkd', 'ppo'):
        return _online_loss_kwargs(rlhf_type, rlhf_config)
    return _preference_loss_kwargs(rlhf_type, rlhf_config)


def _online_loss_kwargs(rlhf_type: str, rlhf_config: 'RLHFConfig') -> Dict[str, Any]:
    """kwargs for the on-policy losses (GRPO/PPO clip params, GKD's temperature); grpo/gkd read beta."""
    kwargs: Dict[str, Any] = {}
    if rlhf_type == 'gkd':
        if rlhf_config.beta is not None:
            kwargs['beta'] = rlhf_config.beta
        kwargs['temperature'] = rlhf_config.temperature
        return kwargs
    if rlhf_type == 'ppo':
        # PPO's policy loss is the shared clipped surrogate; the clip range is `cliprange`, and its KL
        # is applied as a reward penalty in the loop (beta=0 here, GRPOLoss's default) to avoid
        # double-counting.
        kwargs['epsilon'] = rlhf_config.cliprange
        return kwargs
    # grpo: KL is either folded into the reward or added by GRPOLoss, never both.
    calculate_kl = rlhf_config.calculate_KL is not False
    if rlhf_config.beta is not None and not rlhf_config.kl_in_reward and calculate_kl:
        kwargs['beta'] = rlhf_config.beta
    kwargs['epsilon'] = rlhf_config.epsilon
    if rlhf_config.epsilon_high is not None:
        kwargs['epsilon_high'] = rlhf_config.epsilon_high
    return kwargs


def _preference_loss_kwargs(rlhf_type: str, rlhf_config: 'RLHFConfig') -> Dict[str, Any]:
    """kwargs for the preference / pairwise losses (dpo/kto/simpo/cpo/orpo/rm)."""
    beta = rlhf_config.beta
    kwargs: Dict[str, Any] = {}
    # dpo/kto/simpo/cpo take beta straight through; orpo folds it into lambda_orpo and rm has none.
    if beta is not None and rlhf_type in ('dpo', 'kto', 'simpo', 'cpo'):
        kwargs['beta'] = beta
    if rlhf_type in ('dpo', 'kto'):
        # dev stores loss_type as a list (legacy CLI accepts several); the twinkle DPO family takes a
        # single variant. kto rides the DPO family's paired 'kto_pair' variant.
        kwargs['loss_type'] = ('kto_pair' if rlhf_type == 'kto' else
                               (rlhf_config.loss_type[0] if rlhf_config.loss_type else 'sigmoid'))
        kwargs['label_smoothing'] = rlhf_config.label_smoothing
    elif rlhf_type == 'simpo':
        kwargs['gamma'] = rlhf_config.simpo_gamma
    elif rlhf_type == 'cpo':
        # twinkle CPOLoss names the behaviour-cloning weight bc_coef; dev keeps legacy's cpo_alpha.
        kwargs['bc_coef'] = rlhf_config.cpo_alpha
    elif rlhf_type == 'orpo' and beta is not None:
        # ORPO has no reference model and no beta: its single weight is lambda_orpo, which legacy/TRL
        # carry in the beta slot -- so the derived beta maps onto lambda_orpo here.
        kwargs['lambda_orpo'] = beta
    elif rlhf_type == 'rm' and rlhf_config.center_rewards_coefficient is not None:
        kwargs['center_rewards_coefficient'] = rlhf_config.center_rewards_coefficient
    return kwargs
