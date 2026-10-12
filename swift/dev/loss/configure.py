# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from swift.dev.config import RLConfig, TrainConfig
    from swift.dev.model import TrainableModel

# Losses that read ``outputs['embeddings']`` (the pooled, L2-normalized sentence vector) rather
# than logits. Keys are swift loss names as they appear in TrainConfig.loss_type.
EMBEDDING_LOSS_TYPES = ('infonce', 'cosine_similarity', 'contrastive', 'online_contrastive')

# Reranker (cross-encoder) losses: score query-document pairs from logits, no pair interleaving.
RERANKER_LOSS_TYPES = ('pointwise_reranker', 'listwise_reranker')

# Sequence-classification problem types; picks the loss objective (MSE / CE / BCE).
PROBLEM_TYPES = ('regression', 'single_label_classification', 'multi_label_classification')


def liger_fused_ce_enabled(train_config: TrainConfig) -> bool:
    """True when the run asked for Liger's fused-linear-cross-entropy loss path.

    Both switches must be on: ``use_liger_kernel`` (the master Liger flag) and the op-level
    ``liger_kernel_config['fused_linear_cross_entropy']``. This is the single source of truth shared
    by :func:`configure_loss` (which selects the loss) and the recipe (which sets the loop's forward
    task to ``'fused_lm_ce'`` so TransformersFusedCEPatch skips the lm_head GEMM and stashes the head).
    The two must agree: a fused loss under task='causal_lm' silently degrades to unfused CE, while a
    standard CE under task='fused_lm_ce' would read hidden states as if they were logits.
    """
    return bool(train_config.use_liger_kernel
                and (train_config.liger_kernel_config or {}).get('fused_linear_cross_entropy'))


def configure_loss(model: TrainableModel,
                   *,
                   loss_type: str = 'cross_entropy',
                   reduction: str = 'sum',
                   enable_channel_loss: bool = False,
                   dft: bool = False,
                   fused_linear_cross_entropy: bool = False,
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
        fused_linear_cross_entropy: select Liger's fused lm_head+CE loss (see
            :func:`liger_fused_ce_enabled`). The caller must also run the forward under
            task='fused_lm_ce'; the recipe wires both from that one predicate.
    """
    from swift.dev.utils.naming import resolve_loss
    if loss_type != 'cross_entropy':
        raise NotImplementedError(f"SFT configure_loss only supports 'cross_entropy', got {loss_type!r}")
    if fused_linear_cross_entropy:
        # The fused kernel computes CE itself, so the channel-grouped and DFT-weighted aggregations
        # (which wrap the standard CE) do not apply -- refuse rather than silently drop them.
        if enable_channel_loss or dft:
            raise ValueError('fused_linear_cross_entropy replaces the standard CE path and cannot be combined '
                             'with enable_channel_loss or enable_dft_loss.')
        loss_cls = resolve_loss('liger_fused_linear_cross_entropy')
        model.set_loss(loss_cls(reduction=reduction, **kwargs))
        return
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
    from swift.dev.utils.naming import resolve_loss

    if loss_type not in EMBEDDING_LOSS_TYPES:
        raise NotImplementedError(f'configure_embedding_loss supports {list(EMBEDDING_LOSS_TYPES)}, '
                                  f'got {loss_type!r}. Reranker losses score query-document pairs '
                                  'from logits and belong to a reranker recipe.')
    loss_cls = resolve_loss(loss_type)
    if mrl_dims is not None:
        kwargs['mrl_dims'] = mrl_dims
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
    from swift.dev.utils.naming import resolve_loss

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
    from swift.dev.utils.naming import resolve_loss

    if problem_type not in PROBLEM_TYPES:
        raise ValueError(f'problem_type must be one of {list(PROBLEM_TYPES)}, got {problem_type!r}. '
                         'It is required (not inferred) so the training objective is explicit.')
    loss_cls = resolve_loss('seq_cls')
    model.set_loss(loss_cls(problem_type=problem_type, num_labels=num_labels, **kwargs))


_RLHF_LOSS_NAME = {
    'grpo': 'grpo',
    'dpo': 'dpo',
    'cpo': 'cpo',
    'orpo': 'orpo',
    'simpo': 'simpo',
    'gkd': 'gkd',
    'rm': 'reward',
    'kto': 'kto',
    'ppo': 'grpo',
    # Self-distillation: OPSD (single privileged teacher) and MOPD (K weighted teachers). RFT is absent
    # on purpose -- it does plain cross-entropy SFT on the filtered set (configure_loss), not an RL loss.
    'opsd': 'opsd',
    'mopd': 'mopd',
}

## 这里还有一个rlhf字符没清理干净，应该改为rl
def configure_rlhf_loss(model: TrainableModel, rl_config: 'RLConfig') -> None:
    """Set the RLHF/RL loss on ``model`` from ``rl_config.rl_type`` and its hyperparameters.

    Peer of :func:`configure_loss`, for the RLHF recipes (run_grpo / run_dpo / run_gkd / run_ppo). The
    heavy losses themselves live in twinkle (the GRPO family, the DPO family, GKD, RewardLoss); this only
    maps the dev Config onto the right constructor and forwards the fields it reads, so a recipe never
    hand-picks a loss class or its argument names.

    The GRPO advanced controls (sequence-level IS, dual-clip, entropy filter/logging, overlong filtering,
    truncated-IS rollout correction, off-policy sequence masking, FIPO future-KL weighting, SDAR
    distillation and CHORD's auxiliary SFT) are constructor parameters of twinkle's ``GRPOLoss`` -- mapped
    in :func:`_online_loss_kwargs`. twinkle therefore owns both the numerics AND the mutual-exclusion guard
    that refuses to pair those controls with a variant whose objective lives in an overridden
    ``_reduce_loss`` (REAL / DRO / DPPO / GeoMean / ClipCov / KLCov), raising with the offending control
    names; dev keeps no duplicate of that guard.

    Only the fields each algorithm actually reads are forwarded, so twinkle's own defaults stand for the
    rest (e.g. an unset ``beta`` leaves the loss default rather than being overwritten with None). ``beta``
    in particular is normally pre-filled per-algorithm by process.py::_derive_rlhf_beta.

    rl_type='ppo' sets only the POLICY loss (the shared clipped surrogate, GRPOLoss, with epsilon=cliprange
    and KL applied in the loop's reward shaping rather than the loss). Its critic is a separate value model
    whose loss is set by :func:`configure_ppo_value_loss`.
    """
    from swift.dev.utils.naming import resolve_loss

    rl_type = rl_config.rl_type
    if rl_type not in _RLHF_LOSS_NAME:
        raise ValueError(f'Unknown rl_type={rl_type!r}; expected one of {sorted(_RLHF_LOSS_NAME)}.')

    loss_name = _RLHF_LOSS_NAME[rl_type]
    grpo_loss_type = 'grpo'
    if rl_type == 'grpo' and rl_config.loss_type:
        grpo_loss_type = rl_config.loss_type[0]
        # dapo reuses BNPO's batch-normalised aggregation; fipo reuses the plain GRPO surrogate and layers
        # its future-KL weighting on top (see GRPOLoss._configured_grpo). Both resolve to a base class here
        # while ``grpo_loss_type`` still carries the real variant into the kwargs mapping.
        loss_name = {'dapo': 'bnpo', 'fipo': 'grpo'}.get(grpo_loss_type, grpo_loss_type)
    loss_cls = resolve_loss(loss_name)
    model.set_loss(loss_cls(**_rlhf_loss_kwargs(rl_type, rl_config, grpo_loss_type)))


def configure_ppo_value_loss(value_model: TrainableModel, rl_config: 'RLConfig') -> None:
    """Set PPO's clipped value-regression loss on the critic (a seq_cls num_labels=1 value model).

    Separate from :func:`configure_rlhf_loss` because PPO trains two models with two objectives: the
    policy (clipped surrogate, set by configure_rlhf_loss) and the critic (this value loss). Forwards
    ``cliprange_value`` and ``vf_coef``; the loop supplies ``returns``/``old_values`` per step.
    """
    from swift.dev.utils.naming import resolve_loss

    loss_cls = resolve_loss('ppo_value')
    value_model.set_loss(loss_cls(cliprange_value=rl_config.cliprange_value, vf_coef=rl_config.vf_coef))


def configure_rlhf_metrics(model: TrainableModel, rl_config: 'RLConfig') -> None:
    """Register the algorithm-specific twinkle.metric component on ``model`` for this rl_type.

    Peer of :func:`configure_rlhf_loss`, for the metric side. Every recipe's per-step loss already flows
    through twinkle's default ``LossMetric`` (loss / grad_norm / loss_<channel>), which the loop's unified
    record merge surfaces; this adds ONLY the algorithm-specific component the loss cannot express -- the
    GRPO family's ratio/clip/entropy stats, the DPO family's rewards/margins/accuracies, PPO's policy
    stats. Registered on the model (not held by the loop) so twinkle accumulates it automatically inside
    ``forward_backward`` from the same forward kwargs the loss sees (``old_logps``/``advantages`` for GRPO,
    ``ref_logps`` for DPO), and ``model.calculate_metric`` returns it.

    MUST be called AFTER ``configure_optimizer``: twinkle's ``add_metric`` appends onto the active
    optimizer group's metric list, which does not exist until the group is built.

    Metric classes are passed as CLASS objects, never name strings -- a str reaching twinkle's
    ``construct_class`` is routed to ``Plugin.load_plugin`` (hub download), which is exactly what
    ``test_no_name_string_is_handed_to_twinkles_loader`` forbids.

    rl_types with no algorithm component (deliberate no-ops):
      - ``gkd`` / ``opsd`` / ``mopd``: the training loss IS the teacher-student divergence (JSD / k3), so
        ``LossMetric`` already surfaces it; a separate distill component would recompute the same quantity.
      - ``kto``: unpaired (one completion + a binary label), so it does not fit DPOMetric's interleaved
        chosen/rejected contract.
      - ``rm``: a num_labels=1 seq_cls score head, not a paired LM objective.
    """
    from twinkle.metric import CISPOMetric, DPOMetric, GRPOMetric, GSPOMetric, PPOMetric

    rl_type = rl_config.rl_type
    if rl_type == 'grpo':
        # Mirror configure_rlhf_loss's variant resolution: the metric's clip statistic must match the
        # loss's (GSPO clips a per-sequence geometric-mean ratio; CISPO clips unconditionally).
        grpo_loss_type = rl_config.loss_type[0] if rl_config.loss_type else 'grpo'
        metric_cls = {'gspo': GSPOMetric, 'cispo': CISPOMetric}.get(grpo_loss_type, GRPOMetric)
        kwargs: Dict[str, Any] = {'epsilon': rl_config.epsilon, 'temperature': rl_config.distill_temperature}
        if rl_config.epsilon_high is not None:
            kwargs['epsilon_high'] = rl_config.epsilon_high
        model.add_metric(metric_cls, is_training=True, **kwargs)
    elif rl_type == 'ppo':
        # PPO's policy loss is the shared clipped surrogate with clip range ``cliprange`` (see
        # configure_rlhf_loss); PPOMetric reports the same token-level ratio/clip stats.
        model.add_metric(
            PPOMetric, is_training=True, epsilon=rl_config.cliprange, temperature=rl_config.distill_temperature)
    elif rl_type in ('dpo', 'cpo', 'orpo', 'simpo'):
        # Paired preference objectives: DPOMetric surfaces logps/chosen|rejected and (with a reference)
        # rewards/margins|accuracies. Forward beta only when set so twinkle's default stands otherwise.
        kwargs = {'beta': rl_config.beta} if rl_config.beta is not None else {}
        model.add_metric(DPOMetric, is_training=True, **kwargs)
    # gkd / opsd / mopd / kto / rm: no algorithm-specific component (see the docstring).


def configure_ppo_value_metric(value_model: TrainableModel, rl_config: 'RLConfig') -> None:
    """Register ``PPOValueMetric`` on PPO's critic (peer of :func:`configure_ppo_value_loss`).

    Surfaces value_mean / return_mean / value_clip_ratio / explained_variance from the same
    ``returns``/``old_values`` the value loss reads. MUST be called after the critic's
    ``configure_optimizer`` (see :func:`configure_rlhf_metrics`).
    """
    from twinkle.metric import PPOValueMetric

    value_model.add_metric(PPOValueMetric, is_training=True, epsilon=rl_config.cliprange_value)


def configure_embedding_metric(model: TrainableModel) -> None:
    """Register twinkle's ``EmbeddingMetric`` on an embedding (InfoNCE/contrastive) model.

    The embedding counterpart of :func:`configure_rlhf_metrics`: the contrastive loss alone is an opaque
    scalar, so this surfaces the geometry that actually shows an embedding model converging -- ``pos_sim``
    (anchor-positive cosine, should RISE) and ``neg_sim`` (anchor vs in-batch negatives, should stay flat or
    FALL), the head signal twinkle's own embedding cookbook reads. twinkle auto-registers ``LossMetric`` +
    ``Accuracy`` + ``TrainMetric`` on every model, but ``Accuracy`` is a no-op here (embeddings carry no class
    label), so without this the record would hold only ``loss``.

    MUST be called AFTER ``configure_optimizer`` (twinkle's ``add_metric`` appends onto the active optimizer
    group's metric list, which does not exist until the group is built) -- the same contract as
    :func:`configure_rlhf_metrics`, which is why the recipe threads it through ``TrainAssembly.fit``'s
    ``configure_metric`` hook rather than the (pre-optimizer) ``configure_loss`` callable.

    Passed as a CLASS object, never a name string (a str reaching twinkle's ``construct_class`` is routed to
    ``Plugin.load_plugin`` -- the hub download ``test_no_name_string_is_handed_to_twinkles_loader`` forbids).
    ``EmbeddingMetric`` also reports ``loss``/``grad_norm``; for embeddings those coincide with ``LossMetric``'s
    (per-sequence losses report ``num_tokens=0``, so ``LossMetric`` falls back to the same ``total/count`` mean),
    and ``_record_step`` re-pins the authoritative ``loss`` after the merge.
    """
    from twinkle.metric import EmbeddingMetric

    model.add_metric(EmbeddingMetric, is_training=True)


def _rlhf_loss_kwargs(rl_type: str, rl_config: 'RLConfig', grpo_loss_type: str = 'grpo') -> Dict[str, Any]:
    """The constructor kwargs for one rl_type's loss, forwarding only the fields it reads.

    Split into an online (policy-gradient / distillation) and a preference/pairwise half so each stays
    a short, single-purpose mapping rather than one long branch. ``grpo_loss_type`` is the resolved GRPO
    variant (grpo/sapo/dr_grpo/...), needed because a variant's own hyperparameters are read only by its
    loss subclass.
    """
    if rl_type in ('grpo', 'gkd', 'ppo', 'opsd', 'mopd'):
        return _online_loss_kwargs(rl_type, rl_config, grpo_loss_type)
    return _preference_loss_kwargs(rl_type, rl_config)


def _online_loss_kwargs(rl_type: str, rl_config: 'RLConfig', grpo_loss_type: str = 'grpo') -> Dict[str, Any]:
    """kwargs for the on-policy losses (GRPO/PPO clip params, GKD's temperature); grpo/gkd read beta."""
    kwargs: Dict[str, Any] = {}
    if rl_type in ('opsd', 'mopd'):
        # The sampled-token k3 surrogate reads ONLY the divergence direction. v1 deliberately applies no
        # distillation temperature (it distils on the model's native log-probs) and carries no
        # reference-KL term, so neither is forwarded to the loss -- OPSDLoss.__call__ reads neither.
        # ``beta`` is rejected at validate time for opsd/mopd rather than dropped here in silence;
        # ``distill_temperature`` is NOT rejected because it still drives the rollout sampling fallback
        # (recipe/_distill.py::distill_sampling_params) -- it just never reaches the divergence.
        kwargs['reverse'] = rl_config.opsd_reverse
        return kwargs
    if rl_type == 'gkd':
        if rl_config.beta is not None:
            kwargs['beta'] = rl_config.beta
        kwargs['temperature'] = rl_config.distill_temperature
        # Auxiliary supervised CE mixed into the distillation loss on dataset-sourced rounds; 0 (default)
        # keeps pure distillation. The recipe enables it per round via GKDLoss.__call__(apply_sft_loss=True).
        kwargs['sft_alpha'] = rl_config.sft_alpha
        return kwargs
    if rl_type == 'ppo':
        # PPO's policy loss is the shared clipped surrogate; the clip range is `cliprange`, and its KL
        # is applied as a reward penalty in the loop (beta=0 here, GRPOLoss's default) to avoid
        # double-counting.
        kwargs['epsilon'] = rl_config.cliprange
        return kwargs
    # grpo: KL is either folded into the reward or added by GRPOLoss, never both.
    calculate_kl = rl_config.calculate_KL is not False
    if rl_config.beta is not None and not rl_config.kl_in_reward and calculate_kl:
        kwargs['beta'] = rl_config.beta
    kwargs['epsilon'] = rl_config.epsilon
    if rl_config.epsilon_high is not None:
        kwargs['epsilon_high'] = rl_config.epsilon_high
    # Variant-only hyperparameters: SAPO's soft-gate temperatures and DR-GRPO's length normaliser are read
    # solely by their own loss subclass, so forward them only for the matching variant -- passing them to
    # the base GRPOLoss would land in ``**kwargs`` and be dropped silently (the W20/W23 dead-param trap).
    if grpo_loss_type == 'sapo':
        kwargs['tau_pos'] = rl_config.tau_pos
        kwargs['tau_neg'] = rl_config.tau_neg
    elif grpo_loss_type == 'dr_grpo':
        kwargs['max_completion_length'] = rl_config.max_completion_length
    elif grpo_loss_type == 'real':
        # REAL reduces each sequence to a scalar score and splits generation groups, so it needs the group
        # size to reshape the batch; its soft-constraint temperature is real_tau.
        kwargs['real_tau'] = rl_config.real_tau
        kwargs['num_generations'] = rl_config.num_generations
    elif grpo_loss_type == 'dro':
        # DRO's quadratic log-ratio penalty strength; read only by DROLoss (it validates >0).
        kwargs['dro_beta'] = rl_config.dro_beta
    elif grpo_loss_type in ('dppo_tv', 'dppo_kl'):
        # DPPO's divergence threshold IS the shared epsilon/epsilon_high (verl reuses clip_ratio_low/high
        # as clip_divergence_low/high), already forwarded above; the only DPPO-specific knob is the
        # truncated-IS stability bound, read solely by the DPPO losses.
        kwargs['clip_ratio_c'] = rl_config.clip_ratio_c
    elif grpo_loss_type == 'clip_cov':
        # Clip-Cov's covariance-window bounds and the fraction of tokens to zero; read only by ClipCovLoss.
        kwargs['clip_cov_ratio'] = rl_config.clip_cov_ratio
        kwargs['clip_cov_lb'] = rl_config.clip_cov_lb
        kwargs['clip_cov_ub'] = rl_config.clip_cov_ub
    elif grpo_loss_type == 'kl_cov':
        # KL-Cov's selected-token fraction and the extra KL coefficient; read only by KLCovLoss.
        kwargs['kl_cov_ratio'] = rl_config.kl_cov_ratio
        kwargs['ppo_kl_coef'] = rl_config.ppo_kl_coef
    # 'gpg' (pure REINFORCE, -logp*adv) and 'geo_mean' (sequence geometric mean) read no extra knobs beyond
    # the shared epsilon/epsilon_high/beta forwarded above, so they need no variant branch here.

    # -- advanced on-policy controls (twinkle GRPOLoss constructor knobs) -----------------------------
    # Forwarded for every GRPO variant. Each defaults to "off", so a plain run leaves GRPOLoss's
    # ``_advanced`` / ``_distill_wrap`` False and takes the unchanged base surrogate path. twinkle owns the
    # numerics and the guard that refuses to pair these with a ``_reduce_loss``-overriding variant.
    kwargs['loss_type'] = grpo_loss_type
    importance_sampling_level = rl_config.importance_sampling_level
    if grpo_loss_type == 'gspo' and importance_sampling_level == 'token':
        # GSPO is sequence-level IS by definition, and GSPOLoss pins that default. The configured pipeline
        # recomputes the ratio from ``importance_sampling_level`` (it does not call the base override), so
        # promote an otherwise-default 'token' to keep GSPO sequence-level when another control turns the
        # pipeline on; an explicit 'sequence' / 'sequence_token' (GSPO-token) is honored as-is.
        importance_sampling_level = 'sequence'
    kwargs['importance_sampling_level'] = importance_sampling_level
    kwargs['delta'] = rl_config.delta
    kwargs['top_entropy_quantile'] = rl_config.top_entropy_quantile
    kwargs['log_entropy'] = rl_config.log_entropy
    kwargs['overlong_filter'] = rl_config.overlong_filter
    kwargs['rollout_importance_sampling_mode'] = rl_config.rollout_importance_sampling_mode
    kwargs['rollout_importance_sampling_threshold'] = rl_config.rollout_importance_sampling_threshold
    kwargs['log_rollout_offpolicy_metrics'] = rl_config.log_rollout_offpolicy_metrics
    kwargs['off_policy_sequence_mask_delta'] = rl_config.off_policy_sequence_mask_delta
    kwargs['fipo_decay_rate'] = rl_config.fipo_decay_rate
    kwargs['fipo_clip_range'] = rl_config.fipo_clip_range
    kwargs['fipo_clip_high_only'] = rl_config.fipo_clip_high_only
    kwargs['fipo_safety_threshold'] = rl_config.fipo_safety_threshold
    # SDAR (confidence-gated teacher distillation) and CHORD (auxiliary SFT rows appended to the batch)
    # slice the batch and read logits/logps around the policy loss. ``enable_chord`` is derived from the
    # CHORD dataset being configured; the recipe appends the rows and passes chord_count/mu/phi per step.
    kwargs['sdar_loss_coef'] = rl_config.sdar_loss_coef
    kwargs['sdar_gate_beta'] = rl_config.sdar_gate_beta
    kwargs['enable_chord'] = bool(rl_config.chord_sft_dataset)

    if rl_config.enable_sampling_replay:
        # The base GRPOLoss validates beta==0 / entropy_coef==0 and switches its __call__ to require the
        # forward-replayed logps. The model forward reads the flag off the loss instance, which (with the
        # dev wrappers gone) is this same GRPOLoss, so the constructor kwarg is the single source of truth.
        kwargs['enable_sampling_replay'] = True
    return kwargs


def _preference_loss_kwargs(rl_type: str, rl_config: 'RLConfig') -> Dict[str, Any]:
    """kwargs for the preference / pairwise losses (dpo/kto/simpo/cpo/orpo/rm)."""
    beta = rl_config.beta
    kwargs: Dict[str, Any] = {}
    # dpo/kto/simpo/cpo take beta straight through; orpo folds it into lambda_orpo and rm has none.
    if beta is not None and rl_type in ('dpo', 'kto', 'simpo', 'cpo'):
        kwargs['beta'] = beta
    if rl_type == 'kto':
        # KTO is the UNPAIRED objective (KTOLoss): a single completion per row + a binary label, anchored
        # on the detached z_KL the loop estimates from a mismatched KL batch. It reads beta and the two
        # imbalance-correcting weights only -- no loss_type/label_smoothing/sft_weight (DPO-family paired
        # knobs) and no reference_free form (both the log-ratio and z_KL need a reference).
        kwargs['desirable_weight'] = rl_config.desirable_weight
        kwargs['undesirable_weight'] = rl_config.undesirable_weight
    elif rl_type == 'dpo':
        # DPOLoss takes a single variant OR a list combined as a weighted sum (MPO), so forward the whole
        # list rather than just its head; twinkle validates every item and the loss_weights alignment.
        kwargs['loss_type'] = rl_config.loss_type or ['sigmoid']
        kwargs['label_smoothing'] = rl_config.label_smoothing
        # DPO-family knobs the twinkle DPOLoss reads: the DiscoPOP gate temperature, the f-divergence that
        # reshapes the preference margin plus its alpha coefficient. Their dev defaults equal twinkle's, so
        # forwarding them unconditionally is a no-op for a plain sigmoid run.
        kwargs['discopop_tau'] = rl_config.discopop_tau
        kwargs['f_divergence_type'] = rl_config.f_divergence_type
        kwargs['f_alpha_divergence_coef'] = rl_config.f_alpha_divergence_coef
        # ld_alpha (length-desensitization) and loss_weights (MPO) default to None; forward only when set so
        # twinkle's own defaults stand rather than being overwritten with None.
        if rl_config.ld_alpha is not None:
            kwargs['ld_alpha'] = rl_config.ld_alpha
        if rl_config.loss_weights is not None:
            kwargs['loss_weights'] = rl_config.loss_weights
        # DPOLoss also honors ``reference_free`` (drop the reference log-ratio) and an NLL term on the
        # chosen response to curb likelihood displacement; dev carries the latter as ``rpo_alpha`` (RPO's
        # NLL weight), which maps onto twinkle's ``sft_weight``. Both were declared-but-unforwarded dead
        # params (W21). ``reference_free`` is wired together with run_dpo._build_reference returning None
        # so the loss takes its reference-free branch instead of the zero-loss fallthrough.
        kwargs['reference_free'] = rl_config.reference_free
        if rl_config.rpo_alpha:
            kwargs['sft_weight'] = rl_config.rpo_alpha
    elif rl_type == 'simpo':
        kwargs['gamma'] = rl_config.simpo_gamma
    elif rl_type == 'cpo':
        # twinkle CPOLoss names the behaviour-cloning weight bc_coef; dev keeps legacy's cpo_alpha.
        kwargs['bc_coef'] = rl_config.cpo_alpha
    elif rl_type == 'orpo' and beta is not None:
        # ORPO has no reference model and no beta: its single weight is lambda_orpo, which legacy/TRL
        # carry in the beta slot -- so the derived beta maps onto lambda_orpo here.
        kwargs['lambda_orpo'] = beta
    elif rl_type == 'rm' and rl_config.center_rewards_coefficient is not None:
        kwargs['center_rewards_coefficient'] = rl_config.center_rewards_coefficient
    return kwargs
