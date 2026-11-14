# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
import copy
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from swift.dev.config import DistributedConfig, ModelConfig, QuantizeConfig, TemplateConfig
    from swift.dev.rewards.scorer import RewardSpec

#: Upper bound on a judge's generation: the verdict is a short analysis plus a ``Reward: <score>`` line.
JUDGE_MAX_NEW_TOKENS = 512


def build_reward_model(spec: 'RewardSpec', *, model_config: 'ModelConfig',
                       template_config: 'TemplateConfig', distributed_config: 'DistributedConfig',
                       rollout_config: Any, backend: str = 'vllm', sampler: Any = None,
                       engine_args: Optional[dict] = None, remote_group: Optional[str] = None,
                       worker_group: Optional[str] = None, judge_template: Optional[str] = None,
                       quantize_config: Optional['QuantizeConfig'] = None) -> Any:
    """One resolved reward spec -> ONE scorer worker (``worker(trajectories) -> scores``), whatever its kind.

    ``scalar`` builds the frozen seq-cls model on ``remote_group`` and wraps it in a ``FunctionWorker``;
    the judge kinds become a ``GenerativeWorker`` over the sampler (``generative_reuse``) or over a
    judge engine of their own on ``remote_group`` (``generative_independent``), whose engine starts
    from the same SHAPED engine args the sampler got (rollout-derived knobs + the caller's overrides);
    ``api`` is a ``FunctionWorker`` around :class:`ApiJudgeReward`. The worker itself is placed on
    ``worker_group`` (a CPU group: it only calls the handles it holds). Rule scorers are NOT built
    here -- :func:`swift.dev.recipe.utils.resolve_reward_scorers` wraps them directly.
    """
    from twinkle.reward import FunctionWorker

    if spec.kind == 'api':
        from swift.dev.rewards.scorer import ApiJudgeReward

        return FunctionWorker(
            ApiJudgeReward(spec.model_id, model=(engine_args or {}).get('model'), judge_template=judge_template),
            remote_group=worker_group)

    if spec.kind != 'scalar':
        import functools

        from twinkle.reward import GenerativeWorker

        from swift.dev.builders import to_sampling_params
        from swift.dev.rewards.scorer import DEFAULT_JUDGE_TEMPLATE, judge_trajectory, parse_judge_score

        judge_sampler = sampler
        if spec.kind != 'generative_reuse':
            judge_sampler = _build_judge_sampler(
                spec, model_config, template_config, backend,
                _shaped_engine_args(backend, rollout_config, engine_args), distributed_config, quantize_config,
                remote_group)
        return GenerativeWorker(
            judge_sampler,
            functools.partial(judge_trajectory, judge_template or DEFAULT_JUDGE_TEMPLATE),
            parse_judge_score,
            sampling_params=to_sampling_params(None, temperature=0.0, max_tokens=JUDGE_MAX_NEW_TOKENS),
            adapter_path=spec.adapter,
            remote_group=worker_group)

    from swift.dev.rewards.scorer import RewardModel

    frozen, template = _build_frozen_reward_model(
        spec, model_config, template_config, distributed_config=distributed_config,
        remote_group=remote_group)
    return FunctionWorker(RewardModel(frozen, template, display_name=spec.display_name), remote_group=worker_group)


def _build_frozen_reward_model(spec: 'RewardSpec', model_config: 'ModelConfig',
                               template_config: 'TemplateConfig', *,
                               distributed_config: Optional['DistributedConfig'] = None,
                               remote_group: Optional[str] = None):
    """One frozen forward model plus its own template -> ``(model, template)`` (the scalar-RM pair).

    The head is sized by the SPEC's ``task_type`` / ``num_labels`` (which ``resolve_reward_model`` forced
    for a plain checkpoint reused as a scalar RM), falling back to the checkpoint's metadata for a real
    seq_cls/reranker RM; ``template_name`` overrides the policy's template and ``max_length`` is cleared
    so the frozen model is not truncated to the policy's budget; an optional frozen LoRA is attached.
    """
    from swift.dev.builders import build_model, build_template, load_model_processor
    from swift.dev.builders.adapter import configure_frozen_adapter
    from swift.dev.builders.model import frozen_auxiliary_distributed_config
    from swift.dev.config import DistributedConfig
    from swift.dev.model.loader import resolve_model_info

    # Re-resolve with the spec's OWN task_type/num_labels: a plain checkpoint reused as a scalar RM was
    # forced to seq_cls at resolve time (RewardSpec.task_type/num_labels), and re-inferring from the
    # checkpoint metadata alone would rebuild it as causal_lm -- no classification head, no scalar score.
    model_info = resolve_model_info(
        spec.model_id, model_type=spec.model_type, revision=spec.revision,
        task_type=spec.task_type, num_labels=spec.num_labels)
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
    if dist.mode == 'remote':
        from swift.dev.builders.remote import _remote_task_head_kwargs, build_remote_model
        reward_model = build_remote_model(reward_model_config, distributed_config)
        if spec.adapter:
            if spec.adapter.startswith('twinkle://'):
                from peft import LoraConfig
                from swift.dev.builders.adapter import _lora_common_kwargs
                from swift.dev.config import TunerConfig

                adapter_config = LoraConfig(**_lora_common_kwargs(TunerConfig(), task_type=None))
            else:
                from peft import PeftConfig

                try:
                    adapter_config = PeftConfig.from_pretrained(spec.adapter)
                except Exception as e:
                    raise ValueError(
                        f'reward model {spec.model_id!r}: --mode remote cannot read the frozen '
                        f'adapter {spec.adapter!r} ({e}). The client must read the checkpoint to '
                        'build the matching LoRA config: use a hub id, a client-visible path, or '
                        'the server store\'s twinkle:// URI.') from e
            reward_model.add_adapter_to_model(
                'reward', adapter_config, **_remote_task_head_kwargs(model_info.task_type, reward_model_config))
            reward_model.load(spec.adapter, load_optimizer=False)
        return reward_model, reward_template
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
    from swift.dev.builders import build_parallel_mesh, build_sampler, build_template, load_model_processor

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
    from swift.dev.builders.sampler import build_engine_args

    shaped = dict(build_engine_args(backend, None, rollout_config))
    shaped.update(engine_args or {})
    return shaped
