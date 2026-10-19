# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from swift.dev2.config import DistributedConfig, ModelConfig, QuantizeConfig, TemplateConfig
    from swift.dev2.rewards.scorer import RewardSpec

#: Upper bound on a judge's generation: the verdict is a short analysis plus a ``Reward: <score>`` line.
JUDGE_MAX_NEW_TOKENS = 512


def build_reward_model(spec: 'RewardSpec', *, model_config: 'ModelConfig',
                       template_config: 'TemplateConfig', distributed_config: 'DistributedConfig',
                       rollout_config: Any, backend: str = 'vllm', sampler: Any = None,
                       engine_args: Optional[dict] = None, remote_group: Optional[str] = None,
                       judge_template: Optional[str] = None,
                       quantize_config: Optional['QuantizeConfig'] = None) -> Any:
    """One resolved reward spec -> ONE scorer, whatever its serving kind.

    ``scalar`` builds the frozen seq-cls model (its own DeviceGroup when ``remote_group`` is given,
    with the elasticity inputs ``remote_class`` recorded attached to the handle); the judge kinds
    (``api`` / ``generative_reuse`` / ``generative_independent``) share the judge branch, whose
    engine starts from the same SHAPED engine args the sampler got (rollout-derived knobs + the
    caller's overrides). Rule scorers are NOT built here -- they resolve locally in the recipe
    and never see a device.
    """
    if spec.kind == 'api':
        from swift.dev2.rewards.scorer import ApiJudgeReward

        return ApiJudgeReward(
            spec.model_id, model=(engine_args or {}).get('model'), judge_template=judge_template)

    if spec.kind != 'scalar':
        from swift.dev2.builders import to_sampling_params
        from swift.dev2.rewards.scorer import GenerativeJudgeReward

        judge_params = to_sampling_params(None, temperature=0.0, max_tokens=JUDGE_MAX_NEW_TOKENS)
        if spec.kind == 'generative_reuse':
            return GenerativeJudgeReward(
                sampler, judge_template=judge_template, adapter_path=spec.adapter, params=judge_params)
        judge_sampler = _build_judge_sampler(
            spec, model_config, template_config, backend, _shaped_engine_args(backend, rollout_config, engine_args),
            distributed_config, quantize_config, remote_group)
        return GenerativeJudgeReward(
            judge_sampler, judge_template=judge_template, adapter_path=spec.adapter,
            params=judge_params, owns_sampler=True)

    from swift.dev2.rewards.scorer import RewardModel

    frozen, template = _build_frozen_reward_model(
        spec, model_config, template_config, distributed_config=distributed_config,
        remote_group=remote_group)
    # placement_key carries the group name out (the elastic registration reads it); the other two
    # are the actor-native elasticity inputs remote_class recorded at construction.
    return RewardModel(
        frozen, template, placement_key=remote_group,
        worker_factory=getattr(frozen, '_worker_factory', None),
        device_mesh=getattr(frozen, 'device_mesh', None))


def _build_frozen_reward_model(spec: 'RewardSpec', model_config: 'ModelConfig',
                               template_config: 'TemplateConfig', *,
                               distributed_config: Optional['DistributedConfig'] = None,
                               remote_group: Optional[str] = None):
    """One frozen forward model plus its own template -> ``(model, template)`` (the scalar-RM pair).

    The real ``task_type`` / ``num_labels`` come from the checkpoint's metadata rather than a
    hard-coded head; ``template_name`` overrides the policy's template and ``max_length`` is cleared
    so the frozen model is not truncated to the policy's budget; an optional frozen LoRA is attached.
    """
    from swift.dev2.builders import build_model, build_template, load_model_processor
    from swift.dev2.builders.model import frozen_auxiliary_distributed_config
    from swift.dev2.config import DistributedConfig
    # TODO(dev2): bridge until the recipe/common batch migrates configure_frozen_adapter
    from swift.dev.recipe.common.assembly import configure_frozen_adapter
    # TODO(dev2): bridge until the model-registration batch migrates get_model_info_meta
    from swift.model import get_model_info_meta

    model_info, _ = get_model_info_meta(spec.model_id, model_type=spec.model_type, revision=spec.revision)
    reward_model_config = copy.copy(model_config)
    reward_model_config.model = spec.model_id
    reward_model_config.model_type = spec.model_type or model_info.model_type
    reward_model_config.model_revision = spec.revision
    reward_model_config.task_type = model_info.task_type
    reward_model_config.num_labels = model_info.num_labels

    _, processor = load_model_processor(reward_model_config)
    reward_template_config = copy.copy(template_config)
    reward_template_config.template = spec.template_name
    reward_template_config.max_length = None
    reward_template = build_template(
        reward_template_config, processor, task_type=reward_model_config.task_type)
    reward_template.max_length = None

    from twinkle import DeviceMesh

    device_mesh = DeviceMesh.from_spec(spec.parallel_spec) if spec.parallel_spec else None
    dist = distributed_config or DistributedConfig(mode='local')
    if spec.parallel_spec:
        dist = frozen_auxiliary_distributed_config(dist, device_mesh.world_size, parallel_spec=spec.parallel_spec)
    reward_model = build_model(
        reward_model_config, dist, device_mesh=device_mesh, remote_group=remote_group)
    reward_model = configure_frozen_adapter(
        reward_model, reward_template, [spec.adapter] if spec.adapter else [], role='reward')
    if getattr(reward_template, 'use_model', False):
        reward_template.model = getattr(reward_model, 'model', reward_model)
    return reward_model, reward_template


def _build_judge_sampler(spec: 'RewardSpec', model_config: 'ModelConfig', template_config: 'TemplateConfig',
                         backend: str, engine_args: Optional[Dict[str, Any]],
                         distributed_config: Optional['DistributedConfig'],
                         quantize_config: Optional['QuantizeConfig'],
                         remote_group: Optional[str] = None) -> Any:
    """Build a standalone generative judge engine for a judge that cannot reuse the sampler.

    A copy of the run's ModelConfig repointed at the judge's own model with a generation ``task_type``
    (never a pooling one). The judge's own ``parallel_spec`` drives BOTH its DP mesh and its engine
    tensor-parallel width so the mesh's tp dim is real; with no spec it inherits the run's engine args
    and a DP-only mesh. Under ray it lands in its channel's own reward DeviceGroup.
    """
    from swift.dev2.builders import build_parallel_mesh, build_sampler, build_template, load_model_processor

    judge_backend = backend if backend in {'vllm', 'sglang', 'transformers'} else 'vllm'
    judge_model_config = copy.copy(model_config)
    judge_model_config.model = spec.model_id
    judge_model_config.task_type = spec.task_type or 'causal_lm'
    _, processor = load_model_processor(judge_model_config)
    judge_template = build_template(template_config, processor)
    if spec.parallel_spec is not None:
        from twinkle import DeviceMesh

        judge_mesh = DeviceMesh.from_spec(spec.parallel_spec)
        # Both vLLM and SGLang read the engine's TP from engine_args['tensor_parallel_size']; setting
        # it from the same spec that sized the DeviceGroup's gpus_per_worker keeps the two in agreement.
        judge_engine_args = dict(engine_args or {})
        judge_engine_args['tensor_parallel_size'] = judge_mesh.tp_world_size or 1
    else:
        judge_mesh = build_parallel_mesh(distributed_config, none_if_single_rank=True)
        judge_engine_args = engine_args
    return build_sampler(
        judge_model_config,
        backend=judge_backend,
        engine_args=judge_engine_args,
        device_mesh=judge_mesh,
        template=judge_template,
        adapters=[spec.adapter] if spec.adapter else None,
        remote_group=remote_group,
        quantize_config=quantize_config)


def _shaped_engine_args(backend: str, rollout_config: Any, engine_args: Optional[dict]) -> dict:
    """A judge engine's base args: the same SHAPED form the sampler got, then the caller's overrides."""
    from swift.dev2.builders.sampler import build_engine_args

    shaped = dict(build_engine_args(backend, None, rollout_config))
    shaped.update(engine_args or {})
    return shaped
