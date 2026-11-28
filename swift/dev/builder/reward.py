# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
import copy
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from swift.dev.config import DistributedConfig, ModelConfig, QuantizeConfig, TemplateConfig
    from swift.dev.reward.scorer import RewardSpec

#: Upper bound on a judge's generation: the verdict is a short analysis plus a ``Reward: <score>`` line.
JUDGE_MAX_NEW_TOKENS = 512


def _attached_head_checkpoint(adapter: Optional[str]) -> bool:
    """True when a client-visible adapter dir carries a separate attached readout head.

    A remotely-trained seq_cls/reranker RM saves its trained head as ``task_head.safetensors`` (the
    host's attached-head slot), NOT as peft ``modules_to_save`` inside ``adapter_model.safetensors``.
    The legacy local frozen-adapter path (PeftModel.from_pretrained into an in-model head) never reads
    that file, so it scores with a random head. Detect the format so the local path can reproduce the
    host forward through the same attached-head channel the remote branch uses.
    """
    import os
    return bool(adapter) and not adapter.startswith('twinkle://') and os.path.isfile(
        os.path.join(adapter, 'task_head.safetensors'))


def _trained_lora_target_modules(adapter: str) -> list:
    """The module suffixes the adapter actually trained, read from its own keys.

    ``adapter_config.json`` stores the INPUT ``target_modules`` (often the bare string ``all-linear``);
    re-expanding that on a local model whose architecture differs from the training host wraps extra
    modules the adapter never trained (e.g. hybrid ``linear_attn`` projections), desynchronizing the
    hidden states. Reading the trained suffixes from ``adapter_model.safetensors`` makes the reload
    wrap exactly the trained modules, independent of the local architecture's all-linear expansion.
    """
    import os
    import re
    from safetensors import safe_open
    modules = set()
    with safe_open(os.path.join(adapter, 'adapter_model.safetensors'), framework='pt') as f:
        for key in f.keys():
            match = re.search(r'\.([A-Za-z_]+)\.lora_[AB]\.', key)
            if match:
                modules.add(match.group(1))
    return sorted(modules)


def _build_local_attached_head_reward_model(spec: 'RewardSpec', reward_model_config: 'ModelConfig',
                                            model_dir: str) -> Any:
    """Reproduce the serving host's frozen RM forward locally for an attached-head checkpoint.

    Builds the base through twinkle's own ``TransformersModel`` (the host backend, resolving
    ``architectures[0]`` exactly as the server does) rather than the dev family loader, attaches the
    readout head through the adapter channel and ``load``s the trained ``task_head.safetensors`` -- the
    SAME mechanism the remote branch uses -- so a remote-trained RM reused as a local ``--orm`` scores
    identically (bf16) instead of through a random in-model head. ``model_dir`` is the resolved local
    snapshot dir (the host loads from a local dir, not the hub id, so the architecture must match).

    Like the remote branch, this deliberately does NOT ``set_processor(InputProcessor)``/``set_template``:
    the dev InputProcessor injects explicit ``position_ids`` (arange, or the template's mRoPE positions),
    and Qwen3_5's hybrid linear-attention forward routes explicit positions differently from the
    ``position_ids=None`` model default the serving host uses -- scoring [1.232] vs the host's [1.968]
    on the same checkpoint/encoding (defect94). Leaving twinkle's default processor keeps position_ids
    unset so the local forward reproduces the host bit-for-bit (bf16). The reward template is still
    returned to the caller for encoding; only the model-side processor injection is dropped.
    """
    import torch
    from peft import PeftConfig
    from twinkle.model import TransformersModel

    dtype = getattr(reward_model_config, 'torch_dtype', None) or 'bfloat16'
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    model = TransformersModel(model_id=model_dir, torch_dtype=dtype)
    adapter_config = PeftConfig.from_pretrained(spec.adapter)
    adapter_config.target_modules = _trained_lora_target_modules(spec.adapter)
    model.add_adapter_to_model('reward', adapter_config, is_trainable=False,
                               task=spec.task_type, num_labels=spec.num_labels)
    model.load(spec.adapter, load_optimizer=False)
    return model


def build_reward_model(spec: 'RewardSpec', *, model_config: 'ModelConfig',
                       template_config: 'TemplateConfig', distributed_config: 'DistributedConfig',
                       rollout_config: Any, backend: str = 'vllm', sampler: Any = None,
                       engine_args: Optional[dict] = None, remote_group: Optional[str] = None,
                       worker_group: Optional[str] = None, judge_template: Optional[str] = None,
                       quantize_config: Optional['QuantizeConfig'] = None, instance_id: str = '') -> Any:
    """One resolved reward spec -> ONE scorer worker (``worker(trajectories) -> scores``), whatever its kind.

    ``scalar`` builds the frozen seq-cls model on ``remote_group`` and wraps it in a ``FunctionWorker``;
    the judge kinds become a ``GenerativeWorker`` over the sampler (``generative_reuse``) or over a
    judge engine of their own on ``remote_group`` (``generative_independent``), whose engine starts
    from the same SHAPED engine args the sampler got (rollout-derived knobs + the caller's overrides);
    ``api`` is a ``FunctionWorker`` around :class:`ApiJudgeReward`. The worker itself is placed on
    ``worker_group`` (a CPU group: it only calls the handles it holds). Rule scorers are NOT built
    here -- :func:`swift.dev.recipe.utils.resolve_reward_scorers` wraps them directly.

    ``instance_id`` distinguishes this worker from the other specs' workers: ray derives an actor
    name from the construction call site, which is the same line for every spec built in a loop.
    """
    from twinkle.reward import FunctionWorker

    if spec.kind == 'api':
        from swift.dev.reward.scorer import ApiJudgeReward

        return FunctionWorker(
            ApiJudgeReward(spec.model_id, model=(engine_args or {}).get('model'), judge_template=judge_template),
            remote_group=worker_group, instance_id=instance_id)

    if spec.kind != 'scalar':
        import functools

        from twinkle.reward import GenerativeWorker

        from swift.dev.builder import to_sampling_params
        from swift.dev.reward.scorer import DEFAULT_JUDGE_TEMPLATE, judge_trajectory, parse_judge_score

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
            remote_group=worker_group, instance_id=instance_id)

    from swift.dev.reward.scorer import RewardModel

    frozen, template = _build_frozen_reward_model(
        spec, model_config, template_config, distributed_config=distributed_config,
        remote_group=remote_group)
    return FunctionWorker(RewardModel(frozen, template, display_name=spec.display_name),
                          remote_group=worker_group, instance_id=instance_id)


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
    from swift.dev.builder import build_model, build_template, load_model_processor
    from swift.dev.builder.adapter import configure_frozen_adapter
    from swift.dev.builder.model import frozen_auxiliary_distributed_config
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
    # The run's config carries the POLICY's portable id in _model_id_or_path (process.py), and
    # load_model_processor prefers it over .model for family matching. Repointing .model alone
    # left the reward model resolved against the policy -- its tokenizer encoded reward inputs
    # into foreign token ids (a smaller-vocab judge engine rejects them outright).
    reward_model_config._model_id_or_path = spec.model_id
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
        from swift.dev.builder.remote import _remote_task_head_kwargs, build_remote_model
        reward_model = build_remote_model(reward_model_config, distributed_config)
        if spec.adapter:
            if spec.adapter.startswith('twinkle://'):
                from peft import LoraConfig
                from swift.dev.builder.adapter import _lora_common_kwargs
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
    from swift.dev.reward.scorer import SCALAR_TASK_TYPES
    if (dist.mode == 'local' and not spec.parallel_spec and spec.task_type in SCALAR_TASK_TYPES
            and _attached_head_checkpoint(spec.adapter)):
        # A remote-trained attached-head RM reused locally: reproduce the host forward (defect94),
        # not the legacy in-model-head PeftModel path that leaves the head random.
        reward_model = _build_local_attached_head_reward_model(
            spec, reward_model_config, model_info.model_dir)
    else:
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
    from swift.dev.builder import build_parallel_mesh, build_sampler, build_template, load_model_processor

    judge_backend = backend if backend in {'vllm', 'sglang', 'transformers'} else 'vllm'
    judge_model_config = copy.copy(model_config)
    judge_model_config.model = spec.model_id
    # Same stale-id contract as the scalar path: load_model_processor prefers _model_id_or_path,
    # so the copy must repoint it too or the judge resolves to the POLICY's tokenizer.
    judge_model_config._model_id_or_path = spec.model_id
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
        # Same ray contract as the infer samplers: a remote judge actor needs a real mesh even
        # at dp=1 (twinkle dispatch raises on device_mesh=None with remote_group set).
        judge_mesh = build_parallel_mesh(
            distributed_config,
            none_if_single_rank=(distributed_config is None or distributed_config.mode != 'ray'))
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
    from swift.dev.builder.sampler import build_engine_args

    shaped = dict(build_engine_args(backend, None, rollout_config))
    shaped.update(engine_args or {})
    return shaped
