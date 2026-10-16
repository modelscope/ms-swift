"""The outcome-reward channel shared by the online-RL recipes (GRPO / PPO / RFT)."""
from __future__ import annotations
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from swift.dev.recipe.rl.auxiliary import reward_group, spec_world_size

if TYPE_CHECKING:
    from swift.dev.config import DistributedConfig, ModelConfig, RLConfig, TemplateConfig


def build_reward_scorers(model_config: ModelConfig, template_config: TemplateConfig, rl_config: RLConfig,
                         distributed_config: DistributedConfig, *, sampler_model: Optional[str],
                         backend: str) -> Tuple[List[Any], List[str]]:
    """Resolve the online-RL outcome channel -- ``--orm`` rules + ``--reward_model`` models -- into ONE scorer list.

    Every scorer meets the same ``func(completions, **columns)`` contract
    (:class:`~swift.dev.rewards.scorer.RewardScorer`), so the loop scores rules and reward models together
    through :func:`~swift.dev.rewards.reward.score_completions` rather than down a rule path and a separate
    reward-model path. Rules come first, then each reward model, so ``--orm_weights`` aligns to the order the
    channel was declared in. Returns ``(scorers, names)``.

    A reward model is classified by :func:`~swift.dev.rewards.scorer.resolve_reward_model` from its real
    ``task_type`` and built by the SAME shared factory the inference/synthesis path uses
    (:func:`~swift.dev.rewards.scorer.build_model_reward`), on its own ``reward_<i>`` DeviceGroup sized by
    ``--orm_parallel_spec``. The group names and world sizes here MUST match what the run body planned
    through ``plan_rl_device_groups(auxiliary_groups=...)``. Only a SCALAR (seq_cls/reranker) reward model is
    placeable in online RL today: a generative or api judge needs the judge config surface and the tp-aware
    DeviceGroup placement that phase③ unifies, so it is rejected here rather than silently mis-scored as a
    scalar head.
    """
    from swift.dev.builders import frozen_auxiliary_distributed_config
    from swift.dev.rewards.reward import get_reward_funcs
    from swift.dev.rewards.scorer import build_model_reward, resolve_reward_model

    scorers, names = get_reward_funcs(rl_config.orm, rl_config) if rl_config.orm else ([], [])
    scorers, names = list(scorers), list(names)

    reward_models = list(rl_config.reward_model or [])
    if not reward_models:
        return scorers, names

    count = len(reward_models)

    def _aligned(values, default, field):
        resolved = [default] * count if values is None else list(values)
        if len(resolved) != count:
            raise ValueError(f'{field} must contain exactly one value per reward_model.')
        return resolved

    model_types = _aligned(rl_config.reward_model_type, None, 'reward_model_type')
    revisions = _aligned(rl_config.reward_model_revision, None, 'reward_model_revision')
    template_names = _aligned(rl_config.reward_template, None, 'reward_template')
    adapters = _aligned(rl_config.reward_adapters or None, None, 'reward_adapters')

    for index, (model_id, model_type, revision, template_name, adapter) in enumerate(
            zip(reward_models, model_types, revisions, template_names, adapters)):
        spec = resolve_reward_model(model_id, adapter, sampler_model, backend)
        if spec is None or spec.kind != 'scalar':
            kind = 'none' if spec is None else spec.kind
            raise ValueError(
                f'--reward_model[{index}]={model_id!r} resolves to a {kind!r} reward, but online RL (grpo/rft) '
                'places only a SCALAR (seq_cls/reranker) reward model today. A generative or api judge reward '
                'is served by the inference/synthesis path (best-of-n ``swift infer``); online-RL judge '
                'placement lands with the phase③ auxiliary-placement unification.')
        spec.model_type, spec.revision, spec.template_name = model_type, revision, template_name
        spec.parallel_spec = rl_config.orm_parallel_spec
        scorers.append(
            build_model_reward(
                spec,
                model_config=model_config,
                template_config=template_config,
                distributed_config=frozen_auxiliary_distributed_config(
                    distributed_config, spec_world_size(spec.parallel_spec), parallel_spec=spec.parallel_spec),
                remote_group=reward_group(index)))
        names.append(spec.display_name)
    return scorers, names
