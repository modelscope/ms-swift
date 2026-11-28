# Copyright (c) ModelScope Contributors. All rights reserved.
"""One policy assembly; value requirements, not algorithm names, select the critic."""

from __future__ import annotations
import copy
from typing import TYPE_CHECKING, Optional

from swift.dev.config.adapter_config import has_adapter
from swift.dev.config.process import normalize_advantage_estimator, normalize_rl_type
from swift.dev.recipe.configs import RecipeConfigs
from swift.dev.recipe.online import build_policy_stream
from swift.dev.recipe.preparation import prepare_training_model
from swift.dev.recipe.resources import PolicyResources
from swift.dev.recipe.utils import (
    REF_GROUP,
    TEACHER_GROUP,
    auxiliary_mesh,
    build_rl_sampler,
    effective_colocate,
    resolve_reward_scorers,
    single_teacher_id,
)

if TYPE_CHECKING:
    from twinkle import DeviceMesh
    from twinkle.advantage.base import Advantage
    from twinkle_agentic.async_rl.cluster_adapter import RLResourcePlan

    from swift.dev.config import RewardConfig, RLConfig
    from swift.dev.recipe.preparation import TrainingModel
    from swift.dev.template import Template


def needs_critic(rl: RLConfig) -> bool:
    normalize_advantage_estimator(rl)
    return rl.advantage_estimator == "gae"


def policy_group_requests(configs: RecipeConfigs) -> list[tuple[str, Optional[str]]]:
    from swift.dev.reward.scorer import split_channel

    rl, reward = configs.rl, configs.reward
    assert rl is not None and reward is not None
    normalize_rl_type(rl)
    requests = []
    disable = has_adapter(configs.tuner) and not rl.ref_adapters and not rl.sync_ref_model
    if needs_critic(rl) or (rl.beta and rl.calculate_KL is not False):
        if not disable and (rl.ref_model or needs_critic(rl)):
            spec = (
                (rl.reference_parallel_spec or configs.distributed.parallel_spec)
                if rl.sync_ref_model
                else rl.ref_parallel_spec
            )
            requests.append((REF_GROUP, spec))
    teacher = single_teacher_id(rl.teacher_model, algo="policy teacher signal")
    if teacher and not rl._teacher_use_disable_adapter:
        requests.append((TEACHER_GROUP, rl.teacher_parallel_spec))
    requests.extend((f"reward_{i}", reward.orm_parallel_spec) for i in range(len(split_channel(list(reward.orm))[1])))
    if split_channel(list(reward.prm))[1]:
        requests.append(("prm_0", reward.prm_parallel_spec))
    return requests


def build_advantage(
    rl: RLConfig, reward: RewardConfig, *, use_prm: bool, has_teacher: bool, remote_group=None
) -> Advantage:
    from twinkle.advantage import ESTIMATORS, GAEAdvantage

    normalize_rl_type(rl)
    common = dict(env_reward_weight=reward.env_reward_weight)
    if use_prm:
        common.update(use_prm=True, prm_orm_mix=reward.prm_orm_mix, prm_weights=reward.prm_weights)
    if remote_group is not None:
        common["remote_group"] = remote_group
    if needs_critic(rl):
        options = dict(
            kl_kind="k1",
            gamma=rl.gamma,
            gae_lambda=rl.lam,
            gae_lambda_alpha=rl.lam_alpha,
            normalize=rl.whiten_rewards,
            kl_coef=rl.kl_coef,
            **common,
        )
        if has_teacher or rl.rollout_importance_sampling_mode or rl.log_rollout_offpolicy_metrics:
            from .signals import PolicyGAE

            return PolicyGAE(rl, **options)
        return GAEAdvantage(**options)
    options = dict(
        num_generations=rl.num_generations,
        reward_weights=reward.orm_weights,
        scale_rewards=rl.scale_rewards or "group",
        beta=float(rl.beta or 0.0),
        kl_in_reward=bool(rl.kl_in_reward and rl.beta),
        sdar_loss_coef=rl.sdar_loss_coef,
        rollout_importance_sampling_mode=rl.rollout_importance_sampling_mode,
        log_rollout_offpolicy_metrics=rl.log_rollout_offpolicy_metrics,
        segment_discount=rl.segment_discount,
        **common,
    )
    if rl.advantage_estimator == "gpg":
        options["f_norm"] = rl.f_norm
    if rl.advantage_reweight == "rlsd":
        options.update(
            teacher_reweight="rlsd",
            rlsd_lambda=rl.rlsd_lambda,
            rlsd_lambda_warmup_steps=rl.rlsd_lambda_warmup_steps,
            rlsd_lambda_decay_steps=rl.rlsd_lambda_decay_steps,
            rlsd_reweight_clip_range=rl.rlsd_reweight_clip_range,
            rlsd_negative_only=rl.rlsd_negative_only,
        )
    elif has_teacher and not use_prm and rl.sdar_loss_coef <= 0:
        options.update(teacher_reweight="expand", teacher_kl_coef=rl.teacher_kl_coef)
    return ESTIMATORS[rl.advantage_estimator](**options)


def build_critic(configs: RecipeConfigs, template: Template, max_steps: int) -> TrainingModel:
    from swift.dev.builder import build_model
    from swift.dev.builder.optimizer import configure_optimizer
    from swift.dev.loss import configure_ppo_value_loss, configure_ppo_value_metric
    from swift.dev.processor import InputProcessor, input_processor_kwargs
    from swift.dev.reward.scorer import split_channel

    rl = configs.rl
    assert rl is not None and configs.reward is not None
    value_config = copy.deepcopy(configs.model)
    value_config.task_type, value_config.num_labels = "seq_cls", 1
    orm_models = split_channel(list(configs.reward.orm))[1]
    value_config.model = orm_models[0] if orm_models else configs.model.model
    if configs.distributed.mode == "remote":
        from peft import LoraConfig

        from swift.dev.builder.adapter import _lora_common_kwargs
        from swift.dev.builder.remote import build_remote_model, configure_remote_processor

        if not has_adapter(configs.tuner) or configs.tuner.tuner != "lora":
            raise ValueError("A remote critic requires tuner=lora to attach its value head.")
        value = build_remote_model(value_config, configs.distributed)
        value.add_adapter_to_model(
            "default",
            LoraConfig(**_lora_common_kwargs(configs.tuner, task_type=None)),
            task="seq_cls",
            num_labels=1,
            gradient_accumulation_steps=configs.train.gradient_accumulation_steps,
        )
        configure_remote_processor(value, configs.template, value_config, configs.distributed)
    else:
        value = build_model(
            value_config,
            configs.distributed,
            configs.train,
            megatron_config=configs.megatron,
            moe_config=configs.moe,
            instance_id="value",
            torch_dtype="float32",
        )
        value.set_processor(
            InputProcessor, **input_processor_kwargs(template, configs.template, configs.distributed, value_config)
        )
        value.set_template(template)
    configure_ppo_value_loss(value, rl)
    configure_optimizer(
        value,
        configs.train,
        num_training_steps=max_steps,
        distributed_config=configs.distributed,
        tuner_config=None,
        plugin_config=None,
    )
    configure_ppo_value_metric(value, rl)
    return value


def build_policy_objects(
    configs: RecipeConfigs,
    template: Template,
    *,
    max_steps: int,
    batch_size: int,
    resource_plan: Optional[RLResourcePlan] = None,
    auxiliary_meshes: Optional[dict[str, Optional[DeviceMesh]]] = None,
    teacher_prompt: bool = False,
) -> PolicyResources:
    from twinkle.processor import RoutingReplay, TrajectoryFilter, TrajectoryUpdate

    from swift.dev.builder import build_dp_mesh, build_ref_model
    from swift.dev.builder.streaming import CONTROL_GROUP, build_rollout, inflight_rows, rollout_sampling_params
    from swift.dev.loss import configure_rl_loss, configure_rl_metrics
    from swift.dev.reward.prm import build_prm_segment
    from swift.dev.utils.scoring import PrivilegedView
    from twinkle.sampler.vllm_sampler.quant_sync import check_quantized_weight_sync

    rl, rollout_config, reward = configs.rl, configs.rollout, configs.reward
    assert rl is not None and rollout_config is not None and reward is not None
    normalize_rl_type(rl)
    remote = configs.distributed.mode == "remote"
    sp = configs.template.sequence_parallel_size
    mesh = None if remote or (configs.distributed.mode == "ray" and sp <= 1) else build_dp_mesh(configs.distributed, sp)
    check_quantized_weight_sync(template.model_info.config, configs.model.model_type, configs.tuner, rollout_config)
    build_config = configs.model
    resume = configs.checkpoint.resume_from_checkpoint if configs.checkpoint is not None else None
    if resume and not has_adapter(configs.tuner) and not remote:
        build_config = copy.deepcopy(configs.model) if rl.rl_type == "ppo" else copy.copy(configs.model)
        build_config.model = resume

    def metric(model):
        temperature = configs.generation.temperature if configs.generation is not None else None
        configure_rl_metrics(model, rl, sampling_temperature=1.0 if temperature is None else float(temperature))

    model = prepare_training_model(
        configs,
        template,
        num_training_steps=max_steps,
        device_mesh=mesh,
        build_config=build_config,
        parameter_dtype="float32",
        configure_loss=lambda model: configure_rl_loss(model, rl),
        configure_metric=metric,
    )
    critic = build_critic(configs, template, max_steps) if needs_critic(rl) else None
    colocate = effective_colocate(rollout_config, resource_plan)
    sampler = build_rl_sampler(
        model_config=configs.model,
        rollout_config=rollout_config,
        template=template,
        tuner_config=configs.tuner,
        resource_plan=resource_plan,
        colocate=colocate,
        enable_sampling_replay=rl.enable_sampling_replay,
        distributed_config=configs.distributed,
        template_config=configs.template,
    )
    engine, adapter = build_policy_stream(
        model, sampler, rollout_config, configs.distributed, resource_plan, lend=critic is None
    )
    params = rollout_sampling_params(rl.max_completion_length, configs.generation)
    rollout = build_rollout(
        sampler,
        template,
        rollout_config,
        sampling_params=params,
        remote=not remote,
        inflight=inflight_rows(rollout_config, batch_size, max(1, rl.num_generations)),
        seed=configs.train.seed,
    )
    ref_spec = (
        (rl.reference_parallel_spec or configs.distributed.parallel_spec) if rl.sync_ref_model else rl.ref_parallel_spec
    )
    reference = None
    if critic is not None or (rl.beta and rl.calculate_KL is not False):
        reference = build_ref_model(
            name=REF_GROUP,
            model_id=rl.ref_model or (configs.model.model if critic is not None else None),
            model_config=configs.model,
            template=template,
            template_config=configs.template,
            distributed_config=configs.distributed,
            device_mesh=auxiliary_mesh(REF_GROUP, ref_spec, auxiliary_meshes),
            parallel_spec=ref_spec,
            adapters=tuple(rl.ref_adapters or ()),
            policy_model=model,
            disable_adapter=has_adapter(configs.tuner) and not rl.ref_adapters and not rl.sync_ref_model,
            model_type=rl.ref_model_type,
            model_revision=rl.ref_model_revision,
        )
    teacher = build_ref_model(
        name=TEACHER_GROUP,
        model_id=single_teacher_id(rl.teacher_model, algo="policy teacher signal"),
        model_config=configs.model,
        template=template,
        template_config=configs.template,
        distributed_config=configs.distributed,
        device_mesh=auxiliary_mesh(TEACHER_GROUP, rl.teacher_parallel_spec, auxiliary_meshes),
        parallel_spec=rl.teacher_parallel_spec,
        adapters=tuple(rl.teacher_adapters or ()),
        policy_model=model,
        disable_adapter=rl._teacher_use_disable_adapter,
        model_type=rl.teacher_model_type,
        model_revision=rl.teacher_model_revision,
        deepspeed=rl.teacher_deepspeed,
    )
    worker_group = None if remote else CONTROL_GROUP
    scorer_kwargs = dict(
        reward_config=reward,
        model_config=configs.model,
        template_config=configs.template,
        distributed_config=configs.distributed,
        rollout_config=rollout_config,
        backend=rollout_config.rollout_sampler,
        sampler=sampler,
        worker_group=worker_group,
        quantize_config=configs.quantize,
    )
    orm_scorers, orm_names = resolve_reward_scorers("orm", **scorer_kwargs)
    prm_scorers, prm_names = resolve_reward_scorers("prm", **scorer_kwargs)
    return PolicyResources(
        model=model,
        critic=critic,
        sampler=sampler,
        rollout=rollout,
        engine=engine,
        adapter=adapter,
        reference=reference,
        teacher=teacher,
        orm_scorers=orm_scorers,
        orm_names=orm_names,
        prm_scorers=prm_scorers,
        prm_names=prm_names,
        sampling_params=params,
        colocate=colocate,
        advantage=build_advantage(
            rl,
            reward,
            use_prm=bool(prm_scorers),
            has_teacher=teacher is not None or teacher_prompt,
            remote_group=worker_group,
        ),
        privileged=PrivilegedView(template=template, remote_group=worker_group) if teacher_prompt else None,
        routing=RoutingReplay(remote_group=worker_group) if rl.router_replay_mode != "disabled" else None,
        traj_filter=TrajectoryFilter(remote_group=worker_group) if critic is not None else None,
        prm_segment=build_prm_segment(reward, rollout_config, remote_group=worker_group),
        traj_update=TrajectoryUpdate(remote_group=worker_group),
    )


def group_variance(rewards_block, weights):
    import torch
    from twinkle.advantage import nanstd, reduce_rewards

    if len(rewards_block) <= 1:
        return 0.0
    matrix = torch.tensor(
        [
            [float("nan") if value is None or value != value else float(value) for value in row] or [float("nan")]
            for row in rewards_block
        ],
        dtype=torch.float32,
    )
    if weights is not None and matrix.shape[1] != len(weights):
        weights = None
    std = nanstd(reduce_rewards(matrix, weights))
    return 0.0 if std != std else float(std)


def on_group(rl: RLConfig, reward: RewardConfig, sampling_params: dict, num_orms: int):
    if not rl.dynamic_sample and rl.advantage_estimator != "remax":
        return None
    from twinkle.data_format import attach_user_data

    from swift.dev.utils.scoring import build_reward_weights

    weights = build_reward_weights(reward.orm_weights, num_orms) if num_orms else None

    def verdict(row, samples, attempt):
        rows = []
        if rl.dynamic_sample and group_variance([list(sample.orm_reward or ()) for sample in samples], weights) <= 0:
            rows = [copy.deepcopy(row) for _ in range(rl.num_generations)]
        if rl.advantage_estimator == "remax":
            baseline = attach_user_data(
                copy.deepcopy(row), sampling_params=dict(sampling_params, temperature=0.0, top_p=1.0, top_k=-1)
            )
            baseline["role"] = "baseline"
            rows.append(baseline)
        return rows

    return verdict


def build_policy_driver(
    resources: PolicyResources,
    configs: RecipeConfigs,
    *,
    prompt_dataloader,
    batch_size: int,
    num_train_epochs: float,
    max_steps: Optional[int],
):
    from swift.dev.builder.streaming import build_streaming_driver

    rl, rollout = configs.rl, configs.rollout
    assert rl is not None and rollout is not None and configs.reward is not None
    return build_streaming_driver(
        dataloader=prompt_dataloader,
        rollout=resources.rollout,
        rollout_config=rollout,
        batch_size=batch_size,
        num_generations=rl.num_generations,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        checkpoint_manager=resources.engine,
        orms=resources.orm_scorers,
        orm_names=resources.orm_names,
        prms=resources.prm_scorers,
        prm_names=resources.prm_names,
        prm_segment=resources.prm_segment,
        on_group=on_group(rl, configs.reward, resources.sampling_params, len(resources.orm_scorers)),
        max_group_regenerations=rl.max_resample_times if rl.dynamic_sample else 0,
        cluster_adapter=resources.adapter,
        remote=True,
    )
