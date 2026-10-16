"""The outcome-reward channel shared by the online-RL recipes (GRPO / PPO / RFT)."""
from __future__ import annotations
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from swift.dev.recipe.rl.auxiliary import reward_group, spec_world_size

if TYPE_CHECKING:
    from swift.dev.config import DistributedConfig, ModelConfig, RewardConfig, TemplateConfig


def split_reward_channel(items: Optional[List[Any]]) -> Tuple[List[Any], List[str]]:
    """Split one ``--orm``/``--prm`` channel into ``(rule_specs, model_ids)``.

    The one splitter both the run bodies' DeviceGroup PLANNING and the channel BUILD call, so the two
    cannot drift on which item is a model: an item is a rule when it is a registered name, a class, or
    a callable (:func:`swift.dev.rewards.reward.is_registered_reward`), and a reward-MODEL id otherwise
    -- the same grammar every consumer of these channels resolves with.
    """
    from swift.dev.rewards.reward import is_registered_reward

    items = list(items or [])
    rule_specs = [item for item in items if is_registered_reward(item)]
    model_ids = [item for item in items if not is_registered_reward(item)]
    return rule_specs, model_ids


def _aligned(values: Optional[List[Any]], count: int, field: str) -> List[Any]:
    """One value per model id, or raise. The companion lists are positional with the model subsequence."""
    resolved = [None] * count if values is None else list(values)
    if len(resolved) != count:
        raise ValueError(f'{field} must contain exactly one value per model id in the channel.')
    return resolved


def build_reward_scorers(model_config: ModelConfig, template_config: TemplateConfig, reward_config: RewardConfig,
                         distributed_config: DistributedConfig, *, sampler_model: Optional[str],
                         backend: str) -> Tuple[List[Any], List[str]]:
    """Resolve the online-RL outcome channel -- the ``--orm`` rules + model ids -- into ONE scorer list.

    Every scorer meets the same ``func(completions, **columns)`` contract
    (:class:`~swift.dev.rewards.scorer.RewardScorer`), so the loop scores rules and reward models together
    through :func:`~swift.dev.rewards.reward.score_completions` rather than down a rule path and a separate
    reward-model path. The rules keep their typed order first, then each model id in typed order, so
    ``--orm_weights`` aligns to the order the channel was declared in. Returns ``(scorers, names)``.

    A model id is classified by :func:`~swift.dev.rewards.scorer.resolve_reward_model` from its real
    ``task_type`` and built by the SAME shared factory the inference/synthesis path uses
    (:func:`~swift.dev.rewards.scorer.build_model_reward`), on its own ``reward_<i>`` DeviceGroup sized by
    ``--orm_parallel_spec``. The group names and world sizes here MUST match what the run body planned
    through ``plan_rl_device_groups(auxiliary_groups=...)`` (both sides call :func:`split_reward_channel`).
    Each model's loading details come from the ``orm_adapters``/``orm_model_type``/``orm_model_revision``/
    ``orm_template`` family, positional with the model subsequence. Only a SCALAR (seq_cls/reranker) reward
    model is placeable in online RL today: a generative or api judge needs the judge config surface and the
    tp-aware DeviceGroup placement that phase③ unifies, so it is rejected here rather than silently
    mis-scored as a scalar head.
    """
    from swift.dev.builders import frozen_auxiliary_distributed_config
    from swift.dev.rewards.reward import get_reward_funcs
    from swift.dev.rewards.scorer import build_model_reward, resolve_reward_model

    rule_specs, model_ids = split_reward_channel(reward_config.orm)
    scorers, names = get_reward_funcs(rule_specs, reward_config) if rule_specs else ([], [])
    scorers, names = list(scorers), list(names)
    if not model_ids:
        return scorers, names

    model_types = _aligned(reward_config.orm_model_type, len(model_ids), 'orm_model_type')
    revisions = _aligned(reward_config.orm_model_revision, len(model_ids), 'orm_model_revision')
    template_names = _aligned(reward_config.orm_template, len(model_ids), 'orm_template')
    adapters = _aligned(reward_config.orm_adapters, len(model_ids), 'orm_adapters')

    for index, (model_id, model_type, revision, template_name, adapter) in enumerate(
            zip(model_ids, model_types, revisions, template_names, adapters)):
        spec = resolve_reward_model(model_id, adapter, sampler_model, backend)
        if spec is None or spec.kind != 'scalar':
            kind = 'none' if spec is None else spec.kind
            raise ValueError(
                f'--orm model item {model_id!r} resolves to a {kind!r} reward, but online RL (grpo/rft) '
                'places only a SCALAR (seq_cls/reranker) reward model today. A generative or api judge reward '
                'is served by the inference/synthesis path (best-of-n ``swift infer``); online-RL judge '
                'placement lands with the phase③ auxiliary-placement unification.')
        spec.model_type, spec.revision, spec.template_name = model_type, revision, template_name
        spec.parallel_spec = reward_config.orm_parallel_spec
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
