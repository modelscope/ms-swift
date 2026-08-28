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
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

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
                   **kwargs) -> None:
    """Set the SFT loss on ``model`` with an explicit reduction (default 'sum').

    Args:
        model: a twinkle-derived Model (has set_loss).
        loss_type: swift loss name resolved via the unified naming layer (SFT default
            'cross_entropy'). Non-CE losses (grpo/dpo/...) resolve too but belong to
            their own recipes; SFT keeps CE.
        reduction: 'sum' (default; GA-correct, aligns legacy) or 'mean'.
    """
    from swift.dev.naming import resolve_loss

    if loss_type != 'cross_entropy':
        raise NotImplementedError(f"SFT configure_loss only supports 'cross_entropy', got {loss_type!r}")
    loss_cls = resolve_loss(loss_type)
    model.set_loss(loss_cls(reduction=reduction, **kwargs))


def configure_embedding_loss(model: TrainableModel,
                             *,
                             loss_type: str = 'infonce',
                             mrl_dims: Optional[Union[Dict[int, float], str]] = None,
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
        mrl_dims: Matryoshka ``{dim: weight}``; a JSON string is parsed (legacy passes
            ``'{"32": 1.0}'``). ``None`` trains the full width only. Rejected by
            ``cosine_similarity``, whose absolute-similarity target has no per-prefix reading.
        **kwargs: forwarded to the loss constructor (temperature, margin, distance_metric, ...).
    """
    from swift.dev.naming import resolve_loss

    if loss_type not in EMBEDDING_LOSS_TYPES:
        raise NotImplementedError(f'configure_embedding_loss supports {list(EMBEDDING_LOSS_TYPES)}, '
                                  f'got {loss_type!r}. Reranker losses score query-document pairs '
                                  'from logits and belong to a reranker recipe.')
    loss_cls = resolve_loss(loss_type)
    parsed = _parse_mrl_dims(mrl_dims)
    if parsed is not None:
        kwargs['mrl_dims'] = parsed
    # Pass an INSTANCE: twinkle's construct_class returns an instance unchanged, so constructing here
    # keeps the mrl_dims validation (EmbeddingLoss.__init__ raises for cosine_similarity) at the dev
    # call site rather than deep inside twinkle.
    model.set_loss(loss_cls(**kwargs))


def _parse_mrl_dims(mrl_dims: Optional[Union[Dict[int, float], str]]) -> Optional[Dict[int, float]]:
    """Normalize mrl_dims to ``{int: float}``, mirroring legacy's parse.

    Legacy swift.trainers.arguments (arguments.py:264-266) accepts either a dict or a JSON string
    and coerces to ``{int(k): float(v)}`` -- JSON object keys are always strings, so a raw
    ``json.loads`` would yield str keys and every ``dim > hidden_size`` comparison would raise.
    """
    if not mrl_dims:
        return None
    if isinstance(mrl_dims, str):
        from swift.dev.utils import json_parse_to_dict
        mrl_dims = json_parse_to_dict(mrl_dims)
    return {int(k): float(v) for k, v in mrl_dims.items()}


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

    loss_cls = resolve_loss(_RLHF_LOSS_NAME[rlhf_type])
    model.set_loss(loss_cls(**_rlhf_loss_kwargs(rlhf_type, rlhf_config)))


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
    # grpo
    if rlhf_config.beta is not None:
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
