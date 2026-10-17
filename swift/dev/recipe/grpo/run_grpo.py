from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from swift.dev.recipe.rl.auxiliary import (
    REF_GROUP,
    SAMPLER_GROUP,
    TEACHER_GROUP,
    AuxiliarySpec,
    build_auxiliary,
    plan_rl_device_groups,
    prm_group,
    reward_group,
    spec_world_size,
)
from swift.dev.recipe.rl.prompts import prompt_rows_from_dataset
from swift.dev.recipe.rl.ref_only import REF_ONLY_REMOTE_GROUP, build_remote_batch_assembler
from swift.dev.recipe.rl.rewards import build_reward_scorers, split_reward_channel
from swift.dev.recipe.rl.sampler import (
    RolloutSamplerSpec,
    SyncableRollout,
    initialize_twinkle_rl,
    rollout_sampling_params,
)
from swift.dev.recipe.rl.streaming import build_streaming_driver
from swift.dev.utils.logger import get_logger

if TYPE_CHECKING:
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        GenerationConfig,
        LoggingConfig,
        MegatronConfig,
        ModelConfig,
        MoEConfig,
        QuantizeConfig,
        RLConfig,
        RewardConfig,
        RolloutConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )

logger = get_logger()


def _split_prm_channel(prm_items: Optional[Sequence[Any]]) -> Tuple[List[Any], Optional[str]]:
    """Split ``--prm`` into ``(rule_specs, model_id)``: the rules scored on completions and the one frozen PRM.

    ``prm`` is a merged heterogeneous selector (rule names + at most one reward-model id), the same grammar
    ``orm`` uses, so it is split by :func:`swift.dev.rewards.reward.is_registered_reward` -- registered name / class /
    callable is a rule, anything else is a model id. At most one model id is allowed because a PRM channel has
    a single ``prm_parallel_spec`` and a single dedicated DeviceGroup; two model ids would have nowhere
    distinct to live. Both the group PLANNING (run body) and the BUILD (:func:`_build_prm_channel`) call this
    one splitter, so they cannot drift on which item is the model.
    """
    from swift.dev.rewards.reward import is_registered_reward
    items = list(prm_items or [])
    rule_specs = [item for item in items if is_registered_reward(item)]
    model_ids = [item for item in items if not is_registered_reward(item)]
    if len(model_ids) > 1:
        raise ValueError(
            f'--prm resolves at most one frozen PRM model alongside any number of rule names, but got '
            f'{len(model_ids)} model ids: {model_ids}. Keep one model id in the prm channel.')
    return rule_specs, (model_ids[0] if model_ids else None)


def run_grpo(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    dataset_config: DatasetConfig,
    train_config: TrainConfig,
    distributed_config: DistributedConfig,
    checkpoint_config: CheckpointConfig,
    rollout_config: RolloutConfig,
    rl_config: RLConfig,
    reward_config: Optional['RewardConfig'] = None,
    tuner_config: Optional[TunerConfig] = None,
    generation_config: Optional[GenerationConfig] = None,
    logging_config: Optional[LoggingConfig] = None,
    quantize_config: Optional[QuantizeConfig] = None,
    megatron_config: Optional[MegatronConfig] = None,
    moe_config: Optional[MoEConfig] = None,
    *,
    engine_args: Optional[Dict[str, Any]] = None,
    output_dir: str = 'output',
    _save_final: bool = True,
) -> List[dict]:
    """Assemble and run online GRPO with weight sync. Returns the loss/metric history.

    Placement is chosen from ``rollout_config.sampler_mode`` (see :func:`plan_rl_device_groups`):
    ``DistributedConfig.nproc_per_node`` is the TRAINER GPU count, and the sampler's GPUs
    (:func:`RolloutSamplerSpec.world_size`, per backend) are placed alongside (disaggregated) or shared (colocate).
    """
    from swift.dev.builders import build_sampler
    from swift.dev.loss import configure_rl_loss, configure_rl_metrics
    from swift.dev.builders.optimizer import configure_optimizer, resolve_max_grad_norm
    from swift.dev.recipe.common.assembly import TrainAssembly
    from swift.dev.recipe.common.checkpoint import CheckpointSaver
    from swift.dev.recipe.grpo.assembler import GRPOAssembler, GroupCompletionHook, _grpo_log_line
    from swift.dev.recipe.common.step_runner import StepRunner

    assembly = TrainAssembly(
        'run_grpo',
        model_config,
        template_config,
        dataset_config,
        train_config,
        distributed_config,
        checkpoint_config,
        tuner_config,
        rl_config=rl_config,
        reward_config=reward_config,
        rollout_config=rollout_config,
        output_dir=output_dir,
        logging_config=logging_config,
        quantize_config=quantize_config,
        megatron_config=megatron_config,
        moe_config=moe_config)
    # Also imports the run's plugin files -- the reward names handed to the GRPO assembler below are
    # resolved against the registry they write into.
    assembly.prepare()
    assembly.build_template()
    prompts, prompt_extras = prompt_rows_from_dataset(dataset_config)
    prompt_rows = [dict(messages=messages, **extras) for messages, extras in zip(prompts, prompt_extras)]

    backend = RolloutSamplerSpec.backend(rollout_config)
    # Plan the DeviceGroups for any FULL-PARAMETER frozen auxiliary (RL_PLAN §3.5): the KL reference, an
    # RLSD/SDAR teacher, and each reward model is a separate Ray actor on its own GPUs. A LoRA/disable_lora
    # auxiliary reuses the policy actor and plans NO group. Each frozen model is described once by an
    # :class:`~swift.dev.recipe.rl.auxiliary.AuxiliarySpec`: :meth:`AuxiliarySpec.group_request` feeds the
    # planner here and :func:`build_auxiliary` builds the SAME spec at the assembler construction below, so
    # the plan and the build read one object and cannot drift (the invariant ``_frozen_model_needs_group``
    # used to guard).
    from swift.dev.recipe.common.assembly import single_teacher_id
    teacher_model_id = single_teacher_id(rl_config.teacher_model, algo='RLSD/SDAR')
    ref_active = bool(rl_config.beta and rl_config.calculate_KL is not False)
    # A full-parameter frozen reference / RLSD teacher is sharded by its OWN parallel spec (world size == the
    # ranks its mesh needs), so a large frozen model is scale-safe instead of pinned to one GPU. Both are
    # built from a COPY of the policy ModelConfig (``base_model_config`` defaults True): the reference is
    # usually the same checkpoint as the policy and the RLSD/SDAR teacher shares its processor knobs, so both
    # inherit the policy's model_type. Neither overrides model_type/revision/deepspeed -- exactly as the old
    # ``_build_frozen_model`` did, which copied the policy config and set only ``.model``.
    ref_spec = AuxiliarySpec(
        role='ref',
        model_id=rl_config.ref_model,
        name=REF_GROUP,
        # Paired EMA requires both sides on identical sharding: when sync_ref_model is active the reference
        # inherits the policy's own parallel_spec (or an explicit matching one); otherwise the card-frugal
        # ref_parallel_spec (None = single rank) is used.
        parallel_spec=((rl_config.reference_parallel_spec or distributed_config.parallel_spec)
                       if rl_config.sync_ref_model else rl_config.ref_parallel_spec),
        adapters=tuple(rl_config.ref_adapters or ()),
        disable_adapter=(tuner_config is not None and not rl_config.ref_adapters and not rl_config.sync_ref_model))
    teacher_spec = AuxiliarySpec(
        role='teacher',
        model_id=teacher_model_id,
        name=TEACHER_GROUP,
        parallel_spec=rl_config.teacher_parallel_spec,
        adapters=tuple(rl_config.teacher_adapters or ()),
        disable_adapter=rl_config._teacher_use_disable_adapter)
    auxiliary_groups: List[Tuple[str, int]] = []
    if ref_active:
        ref_request = ref_spec.group_request()
        if ref_request is not None:
            auxiliary_groups.append(ref_request)
    teacher_request = teacher_spec.group_request()
    if teacher_request is not None:
        auxiliary_groups.append(teacher_request)
    # Each MODEL id in the --orm channel (rule items need no GPUs) is a full-parameter frozen auxiliary with
    # its OWN reward_<i> group, sized by --orm_parallel_spec. split_reward_channel is the one splitter both
    # this planning and build_reward_scorers use, so the planned groups and the built scorers cannot drift.
    reward_world_size = spec_world_size(reward_config.orm_parallel_spec)
    for reward_index in range(len(split_reward_channel(reward_config.orm)[1])):
        auxiliary_groups.append((reward_group(reward_index), reward_world_size))
    # A frozen PRM model (a non-rule --prm item) is a full-parameter auxiliary too, so it gets its OWN group
    # exactly like a reward model, sized by --prm_parallel_spec; _split_prm_channel is the one splitter both
    # this planning and _build_prm_channel use, so the planned name/world size and the build cannot drift.
    if _split_prm_channel(reward_config.prm)[1] is not None:
        auxiliary_groups.append((prm_group(0), spec_world_size(reward_config.prm_parallel_spec)))
    from swift.dev.recipe.rl.resource import RLResourceBootstrap
    resource_plan = RLResourceBootstrap.build(
        model_config=model_config,
        distributed_config=distributed_config,
        rollout_config=rollout_config,
        template=assembly.template,
        prompt_rows=prompt_rows,
        auxiliary_groups=auxiliary_groups,
        max_completion_length=rl_config.max_completion_length)
    sampler_world_size = RolloutSamplerSpec.world_size(rollout_config, backend)
    if resource_plan is None:
        groups, sampler_remote_group, colocate = plan_rl_device_groups(
            distributed_config.nproc_per_node,
            rollout_config.sampler_mode,
            sampler_world_size,
            auxiliary_groups=auxiliary_groups)
    else:
        groups = RLResourceBootstrap.device_groups(
            resource_plan, adaptive=rollout_config.resource_mode == 'adaptive')
        sampler_remote_group, colocate = SAMPLER_GROUP, False
        logger.info('RL resource plan:\n%s', resource_plan.describe())
    # RL initializes twinkle itself rather than through the assembly: it needs two device groups (trainer
    # + sampler), whose placement was just planned.
    initialize_twinkle_rl(
        distributed_config,
        groups,
        seed=train_config.seed,
        full_determinism=train_config.full_determinism,
        tf32=train_config.tf32,
        sequence_parallel_size=template_config.sequence_parallel_size,
        sampler_parallel_width=RolloutSamplerSpec.parallel_width(rollout_config, backend))

    # Ulysses SP mesh (None unless sequence_parallel_size>1): planned before build_model so the trainable
    # policy is placed on the dp x ulysses mesh, the same one initialize_twinkle_rl installed globally for
    # the colocated sampler and the one the dp_size batch-width formula below reads.
    assembly.plan_sp_mesh()
    # Trainer: a Ray-actor model in the 'model' group (build_model sets remote_group='model' under
    # mode='ray'), with the tuner applied before loss/optimizer so those target its group.
    assembly.build_model()
    configure_rl_loss(assembly.model, rl_config)
    # Prompts are rolled out, not iterated by a dataloader, so the step budget is derived from the prompt
    # set the way assembly derives it from a dataloader: one generation batch per rollout, exhausted after
    # num_train_epochs passes (B1). Load the prompts + design-B mini-batch width first, then size max_steps
    # (the LR-scheduler horizon) from them; an explicit --max_steps still overrides.
    # Design-B mini-batch width: per_device_train_batch_size * dp_size, the same global-batch formula SFT
    # uses (builders/dataset._twinkle_loader_layout). Online RL is Ray-only and pure data-parallel over
    # nproc_per_node, so dp_size is read off build_dp_mesh -- the exact mesh the trainer model was
    # placed on and slice_dp splits each forward_backward across, so every rank gets per_device rows.
    from swift.dev.builders import build_dp_mesh, build_prompt_dataloader
    from swift.dev.recipe.rl.prompts import resolve_rollout_max_steps, rollout_step_budget
    # data_world_size derives nproc/ulysses from the mesh, so under SP the dp_size (and thus the global
    # train_batch_size) counts SP peers as ONE data rank -- they share a sample and split its sequence.
    dp_size = build_dp_mesh(distributed_config, template_config.sequence_parallel_size).data_world_size
    train_batch_size = train_config.per_device_train_batch_size * dp_size
    chord_features = _load_chord_features(rl_config, dataset_config, assembly.template)
    chord_train_batch_size = (
        int(rl_config.chord_sft_per_device_train_batch_size or 0) * dp_size if chord_features else 0)
    max_steps = resolve_rollout_max_steps(
        train_config.max_steps,
        rollout_step_budget(
            num_prompts=len(prompts),
            num_generations=rl_config.num_generations,
            train_batch_size=train_batch_size,
            gradient_accumulation_steps=assembly.ga,
            num_train_epochs=train_config.num_train_epochs,
            generation_batch_size=rollout_config.generation_batch_size,
            num_iterations=rl_config.num_iterations),
        recipe='run_grpo')
    assembly.resolve_step_intervals(max_steps)
    configure_optimizer(
        assembly.model,
        train_config,
        num_training_steps=max_steps,
        distributed_config=distributed_config,
        tuner_config=assembly.tuner_config,
        plugin_config=assembly.plugin_config)
    # Register the GRPO-family policy metric (GRPOMetric/GSPOMetric/CISPOMetric by loss_type) AFTER the
    # optimizer group exists (add_metric appends onto it). twinkle then accumulates it inside forward_backward
    # from old_logps/advantages, so its ratio/clip/entropy stats ride calculate_metric into every step record.
    configure_rl_metrics(assembly.model, rl_config)

    # Sampler: a vLLM/SGLang sampler placed in its group (shared 'model' for colocate, separate 'sampler'
    # otherwise). The backend's memory-saver flag (forced on for colocate) is required for the device
    # hand-over.
    sampler_engine_args = RolloutSamplerSpec.engine_args(rollout_config, engine_args, colocate, backend, tuner_config=assembly.tuner_config)
    if rl_config.enable_sampling_replay:
        # Sampling replay trains against the sampler's per-token support set, so the engine must export each
        # token's sampling mask. This is VLLMEngine's own construction flag (it derives vLLM's
        # enable_return_sampling_mask + processed_logprobs); set it here rather than in the shared
        # RolloutSamplerSpec.engine_args because only the GRPO loop consumes sampling masks.
        sampler_engine_args['enable_sampling_replay'] = True
    if rollout_config.resource_mode == 'adaptive':
        from swift.dev.recipe.rl.resource import AdaptiveResourceController
        controller = AdaptiveResourceController.build(
            model=assembly.model,
            model_config=model_config,
            template=assembly.template,
            rollout_config=rollout_config,
            backend=backend,
            engine_args=sampler_engine_args,
            plan=resource_plan)
        sampler = controller.sampler
        rollout = SyncableRollout(
            assembly.model,
            sampler,
            assembly.template,
            colocate=False,
            sleep_level=rollout_config.sleep_level,
            handover=controller.handover)
        rollout.resource_controller = controller
    else:
        sampler = build_sampler(
            model_config,
            backend=backend,
            engine_args=sampler_engine_args,
            device_mesh=RolloutSamplerSpec.device_mesh(rollout_config, backend),
            template=assembly.template,
            remote_group=sampler_remote_group)
        rollout = SyncableRollout(
            assembly.model, sampler, assembly.template, colocate=colocate, sleep_level=rollout_config.sleep_level)
    # Multi-turn is requested by setting max_turns; the engine is twinkle-native (no scheduler, no
    # gym). Per-round length is sampling_params.max_tokens, whole-trajectory length is
    # max_trajectory_tokens. Tools are opt-in via RolloutConfig.tools: the engine builds its own sandbox
    # env pool in-actor (each episode leases its own env; see swift.dev.rollout.sandbox), so the pool never
    # crosses the Ray boundary; harness / followup_fn remain caller-supplied extension points. The env pool
    # is closed by rollout.shutdown().
    if rollout_config.max_turns is not None:
        rollout.configure_multi_turn(
            max_turns=rollout_config.max_turns,
            max_trajectory_tokens=rollout_config.max_trajectory_tokens,
            rollout_config=rollout_config,
            remote_group=REF_ONLY_REMOTE_GROUP)

    reward_scorers, reward_scorer_names = build_reward_scorers(
        model_config, template_config, reward_config, distributed_config, sampler_model=model_config.model,
        backend=backend)
    # Process-reward (PRM) channel: the segmenter plugin + the ordered --prm scorers, resolved exactly like
    # the outcome channel above (a frozen PRM model gets its OWN prm_0 group, planned in auxiliary_groups).
    prm_scorer, prm_scorers = _build_prm_channel(
        model_config, template_config, reward_config, rollout_config, distributed_config,
        sampler_model=model_config.model, backend=backend)
    # The flat recipe body: the twinkle StreamingDriver iterator (built below) yields assembled + scored
    # TrainingBatches on a background thread, and this body trains them -- no loop class, no inheritance.
    # Every fail-loudly validation below runs BEFORE the StepRunner builds its RunTracker, whose reporters
    # initialize as a construction side effect.
    _check_streaming_feature(colocate, rl_config.advantage_estimator)

    # Late-bound component cells: the per-step extra-metric closure is injected into the recorder BEFORE
    # the assembler/driver exist, so it reads them through one-element cells filled right after each build.
    _assembler_cell: List[Any] = []
    _driver_cell: List[Any] = []
    _idle_snapshot = {'train': 0.0, 'idle': 0.0}
    _last_idle_ratio = 0.0

    def _fold_idle_ratio() -> None:
        """Fold the trainer-bubble ratio ONCE per batch, at the batch boundary, before training it.

        Mirrors the retired StreamingRolloutDriver._fold_idle_ratio: the driver adds train_active_time only
        AFTER a pull returns, so the delta against the last batch boundary is the PREVIOUS batch's train
        time, and idle_time's delta is the bubble spent starved waiting for THIS batch. Reported identically
        on every step of this batch.
        """
        nonlocal _idle_snapshot, _last_idle_ratio
        driver = _driver_cell[0] if _driver_cell else None
        if driver is None:
            return
        stats = driver.stats
        d_train = stats.train_active_time - _idle_snapshot['train']
        d_idle = stats.idle_time - _idle_snapshot['idle']
        _idle_snapshot = {'train': stats.train_active_time, 'idle': stats.idle_time}
        denom = d_train + d_idle
        _last_idle_ratio = (d_idle / denom) if denom > 0 else 0.0

    def _extra_metrics() -> Dict[str, Any]:
        """Driver-side per-step channels the model's calculate_metric cannot surface."""
        extra: Dict[str, Any] = {}
        assembler = _assembler_cell[0] if _assembler_cell else None
        if assembler is not None:
            extra.update(assembler.step_telemetry())
        driver = _driver_cell[0] if _driver_cell else None
        if driver is not None:
            stats = driver.stats
            extra.update({
                'trainer_idle_ratio': _last_idle_ratio,
                'off_policy_consumed': stats.off_policy_consumed,
                'dropped_stale': stats.dropped_stale,
                'stream_publishes': stats.publishes,
            })
        return extra

    from swift.dev.callbacks import CallbackHandler, build_callbacks

    recorder = StepRunner(
        assembly.model,
        CallbackHandler(build_callbacks(train_config), output_dir=output_dir),
        gradient_accumulation_steps=assembly.ga,
        max_grad_norm=resolve_max_grad_norm(train_config),
        max_steps=max_steps,
        logging_config=logging_config,
        output_dir=output_dir,
        manual_gc=bool(megatron_config and megatron_config.manual_gc),
        manual_gc_steps=megatron_config.manual_gc_steps if megatron_config else 0,
        torch_empty_cache_steps=train_config.torch_empty_cache_steps,
        log_formatter=_grpo_log_line,
        extra_metrics_fn=_extra_metrics)
    saver = CheckpointSaver(
        assembly.model,
        recorder,
        None,
        output_dir=output_dir,
        save_steps=checkpoint_config.save_steps,
        no_save_optim=checkpoint_config.no_save_optim or checkpoint_config.save_only_model,
        no_save_rng=checkpoint_config.no_save_rng or checkpoint_config.save_only_model,
        safe_serialization=checkpoint_config.safe_serialization,
        max_shard_size=checkpoint_config.max_shard_size,
        save_total_limit=checkpoint_config.save_total_limit)
    reference = build_auxiliary(
        ref_spec,
        template=assembly.template,
        student=assembly.model,
        distributed_config=distributed_config,
        model_config=model_config) if ref_active else None
    # EMA manager: paired 2-rank NCCL groups (policy rank k <-> reference rank k) for streaming reference
    # sync. Only when sync_ref_model is active AND the reference is a separate FrozenAuxiliary (not a
    # PolicyAuxiliary sharing the policy actor, which has no independent weights to update).
    ema_manager = None
    if rl_config.sync_ref_model and reference is not None and hasattr(reference, 'model'):
        from twinkle.checkpoint_engine import CheckpointEngineManager
        ema_manager = CheckpointEngineManager(
            model=assembly.model, sampler=reference.model, platform='GPU', mode='standalone')
    teacher = build_auxiliary(
        teacher_spec,
        template=assembly.template,
        student=assembly.model,
        distributed_config=distributed_config,
        model_config=model_config)
    assembler = GRPOAssembler(
        assembly.model,
        rollout,
        template=assembly.template,
        reference=reference,
        teacher=teacher,
        reward_scorers=reward_scorers,
        reward_scorer_names=reward_scorer_names,
        reward_weights=reward_config.orm_weights,
        prm_scorer=prm_scorer,
        prm_scorers=prm_scorers,
        prm_weights=reward_config.prm_weights,
        num_generations=rl_config.num_generations,
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=assembly.ga,
        advantage_estimator=rl_config.advantage_estimator,
        scale_rewards=rl_config.scale_rewards or 'group',
        rl_config=rl_config,
        reward_config=reward_config,
        sampling_params=rollout_sampling_params(rl_config, generation_config),
        output_dir=output_dir,
        tracker=recorder.tracker,
        global_step_fn=lambda: recorder.global_step,
        chord_features=chord_features,
        chord_train_batch_size=chord_train_batch_size)
    group_complete = GroupCompletionHook(assembler, rl_config)
    payload_processor = build_remote_batch_assembler(assembler, group_complete=group_complete)
    _assembler_cell.append(payload_processor)
    # The driver: twinkle's per-sample StreamingDriver in ITERATOR mode (consume=None), injected with the
    # assembler's score/assemble seams and the group-completion hook. Returns (driver, handover): the
    # colocate hand-over bracket is the recipe's, not the driver's. The prompt dataloader is a twinkle
    # DataLoader over the prompt rows; the driver flattens its batches, draws rows, and stamps its own
    # ordinal as prompt_id.
    prompt_dataloader = build_prompt_dataloader(prompts, prompt_extras, seed=train_config.seed)
    driver, handover = build_grpo_streaming_driver(
        rollout_config,
        colocate=colocate,
        run_id='run_grpo',
        model=assembly.model,
        rollout=rollout,
        output_dir=output_dir,
        prompt_dataloader=prompt_dataloader,
        num_train_epochs=train_config.num_train_epochs,
        assembler=assembler,
        payload_processor=payload_processor,
        global_step=lambda: recorder.global_step,
        group_complete=group_complete,
        adapter_name=('default' if tuner_config is not None else None),
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=assembly.ga,
        num_generations=rl_config.num_generations,
        reached_max=recorder.reached_max,
        max_group_regenerations=(rl_config.max_resample_times if rl_config.dynamic_sample else 0))
    _driver_cell.append(driver)
    if assembly.resume_dir:
        saver.resume(assembly.resume_model())

    from twinkle.perf import Perf

    from swift.dev.recipe.common.loop_utils import filter_perf_actors

    if Perf.initialize(filter_perf_actors({
            'model': assembly.model,
            'sampler': rollout.sampler,
            'reference': reference,
            'teacher': teacher,
    })):
        logger.info('[perf] profiling enabled for the online GRPO loop.')
    try:
        if handover.serializes or getattr(handover, 'adaptive_shared', False):
            # Version-0 hand-over: put the pre-training policy in the sampler before the first admission.
            # Every later cycle's enter rides the in-place publish's sync_fn (see the builder).
            handover.enter()
        for batch in driver:
            _fold_idle_ratio()
            optimizer_steps = 0
            try:
                for _ in range(batch.replay_count):
                    for mini_batch_index in range(batch.mini_batch_count):
                        recorder.micro_step += 1
                        if assembler.chord_enabled:
                            mu = payload_processor.chord_mu(recorder.global_step)
                            batch.forward_backward(
                                mini_batch_index,
                                loss_key='grpo',
                                loss_weight=1.0 - mu,
                                finalize_gradients=False,
                            )
                            batch.forward_backward(
                                mini_batch_index,
                                kind='sft',
                                loss_key='sft',
                                loss_weight=mu,
                                finalize_gradients=True,
                            )
                        else:
                            batch.forward_backward(mini_batch_index)
                        is_boundary = recorder.is_grad_sync_boundary()
                        assembly.model.clip_grad_and_step(
                            max_grad_norm=recorder.max_grad_norm,
                            gradient_accumulation_steps=recorder.gradient_accumulation_steps)
                        if is_boundary:
                            recorder.record_step()
                            optimizer_steps += 1
                            if ema_manager is not None and recorder.global_step % rl_config.ref_model_sync_steps == 0:
                                ema_manager.sync_ema(rl_config.ref_model_mixup_alpha)
                            if saver.save_steps and recorder.global_step % saver.save_steps == 0:
                                saver.save(f'checkpoint-{recorder.global_step}')
                            if recorder.reached_max():
                                break
                    if recorder.reached_max():
                        break
            finally:
                batch.release()
            # commit releases the driver thread blocked in the iterator bridge and reports the real step
            # count -- the publish cadence unit (twinkle publishes only after consume returns).
            driver.commit(optimizer_steps)
            if recorder.reached_max():
                break
    finally:
        if ((handover.serializes or getattr(handover, 'adaptive_shared', False))
                and handover.in_generation):
            # Ended mid-window (e.g. a budget hit during generation): give the device back to the trainer
            # so the post-run final save finds the model resident.
            handover.exit()
        Perf.shutdown()
        rollout.shutdown()
        recorder.tracker.close()
    if _save_final:
        saver.save_final(write_args_json=assembly.args_json_writer())
    return recorder.history


def _check_streaming_feature(colocate: bool, advantage_estimator: str) -> None:
    """Refuse a GRPO feature the per-sample stream cannot serve under this placement (fail-loudly).

    Called by the run body BEFORE the recorder's RunTracker opens any reporter, and again inside
    :func:`build_grpo_streaming_driver` as the backstop for a direct build.
    """
    if colocate and advantage_estimator == 'remax':
        raise ValueError(
            "advantage_estimator='remax' runs a greedy baseline generation at each group completion, which "
            'needs the sampler WHILE the trainer holds the device -- a colocated sampler cannot do that '
            '(ExclusiveDeviceHandover time-shares one device). Set --sampler_mode disaggregated, or use '
            "advantage_estimator='grpo'.")


def build_grpo_streaming_driver(rollout_config: Any,
                                *,
                                colocate: bool,
                                run_id: str,
                                model: Any,
                                rollout: Any,
                                output_dir: str,
                                prompt_dataloader: Any,
                                num_train_epochs: float,
                                assembler: Any,
                                payload_processor: Any,
                                global_step: Any,
                                group_complete: Any,
                                adapter_name: Optional[str],
                                train_batch_size: int,
                                gradient_accumulation_steps: int,
                                num_generations: int,
                                reached_max: Any,
                                max_group_regenerations: int = 0) -> Tuple[Any, Any]:
    """Build the per-sample StreamingDriver for the flat GRPO recipe -> ``(driver, handover)``.

    A thin delegate over :func:`~swift.dev.recipe.rl.streaming.build_streaming_driver`: GRPO's assembly unit
    is one WHOLE prompt group (``unit_size=num_generations``, group-relative advantage is undefined on a
    partial group) and its algorithm seams are the assembler's ``score_sample`` / ``assemble`` /
    ``assembly_ready`` plus the injected ``group_complete`` hook. The GRPO-specific colocate+remax refusal
    runs here (and in the run body before the recorder's RunTracker opens any reporter) as the backstop for
    a direct build.
    """
    _check_streaming_feature(colocate, assembler.advantage_estimator)
    return build_streaming_driver(
        rollout_config,
        colocate=colocate,
        run_id=run_id,
        model=model,
        rollout=rollout,
        output_dir=output_dir,
        prompt_dataloader=prompt_dataloader,
        num_train_epochs=num_train_epochs,
        adapter_name=adapter_name,
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_generations=num_generations,
        sampling_params=assembler.sampling_params,
        unit_size=num_generations,
        score=assembler.score_sample,
        assemble=assembler.assemble,
        assembly_ready=assembler.assembly_ready,
        payload_processor=payload_processor,
        global_step=global_step,
        group_complete=group_complete,
        max_group_regenerations=max_group_regenerations,
        reached_max=reached_max)


def _one(values: Optional[Sequence[Any]], field: str) -> Any:
    """The prm channel's single companion entry: ``None``, or exactly one value for the one model id."""
    if values is None:
        return None
    values = list(values)
    if len(values) != 1:
        raise ValueError(
            f'--{field} must carry exactly one entry for the one model id in the prm channel, but got '
            f'{len(values)}: {values}.')
    return values[0]


def _build_prm_channel(model_config: ModelConfig, template_config: TemplateConfig, reward_config: RewardConfig,
                       rollout_config: RolloutConfig, distributed_config: DistributedConfig, *,
                       sampler_model: Optional[str], backend: str) -> Tuple[Any, List[Any]]:
    """Build GRPO's process-reward (PRM) channel -> ``(prm_scorer, prm_scorers)``.

    ``prm_scorers`` is the ordered ``--prm`` channel resolved exactly like the outcome channel (rule PRMs
    first, then the single frozen PRM model), so the assembler's per-sample score seam scores every step
    through the same :func:`~swift.dev.rewards.reward.score_completions`. ``prm_scorer`` is the
    :class:`~swift.dev.rewards.prm.PRMScorer` plugin that segments a response into steps and broadcasts each
    step's score, resolved from ``reward_config.prm_scorer``; when unset it defaults to ``'turn'`` for a
    multi-turn rollout (``max_turns`` set -- one assistant turn is one step) and ``'delimiter'`` otherwise
    (one delimiter-separated line is one step). It
    is returned only when a PRM channel is actually wired, so a scorer with no rule/model -- which would score
    nothing -- is left ``None`` for validate to reject.

    The frozen PRM model is a SCALAR reward model on its OWN ``prm_0`` DeviceGroup sized by
    ``--prm_parallel_spec`` (a full-parameter frozen seq_cls scorer, heterogeneous per RL_PLAN §3.5); the
    group name and world size MUST match what the run body planned through
    ``plan_rl_device_groups(auxiliary_groups=...)``. Its loading details come from the ``prm_adapters``/
    ``prm_model_type``/``prm_model_revision``/``prm_template`` family -- one entry each for the one model id
    (:func:`_one`). As in :func:`build_reward_scorers`, a generative/api
    process judge is not placeable in online RL yet and is rejected rather than mis-scored as a scalar head.

    Returns ``(None, [])`` when ``--prm`` is empty, so the loop's ``_prm_active`` stays False.
    """
    rule_specs, prm_model_id = _split_prm_channel(reward_config.prm)
    if not rule_specs and prm_model_id is None:
        return None, []

    from swift.dev.builders import frozen_auxiliary_distributed_config
    from swift.dev.utils.plugin import PluginRegistry
    from swift.dev.rewards.reward import get_reward_funcs
    from swift.dev.rewards.scorer import build_model_reward, resolve_reward_model
    from swift.dev.rewards import PRM_STEP_SCORER

    rule_scorers, _rule_names = get_reward_funcs(rule_specs, reward_config) if rule_specs else ([], [])
    scorers: List[Any] = list(rule_scorers)
    if prm_model_id is not None:
        spec = resolve_reward_model(prm_model_id, _one(reward_config.prm_adapters, 'prm_adapters'),
                                    sampler_model, backend)
        if spec is None or spec.kind != 'scalar':
            kind = 'none' if spec is None else spec.kind
            raise ValueError(
                f'--prm model {prm_model_id!r} resolves to a {kind!r} reward, but online RL places only a '
                'SCALAR (seq_cls/reranker) frozen PRM today; a generative/api process judge is not placeable '
                'in online RL yet (see the phase③ auxiliary-placement unification).')
        spec.model_type = _one(reward_config.prm_model_type, 'prm_model_type')
        spec.revision = _one(reward_config.prm_model_revision, 'prm_model_revision')
        spec.template_name = _one(reward_config.prm_template, 'prm_template')
        spec.parallel_spec = reward_config.prm_parallel_spec
        scorers.append(
            build_model_reward(
                spec,
                model_config=model_config,
                template_config=template_config,
                distributed_config=frozen_auxiliary_distributed_config(
                    distributed_config, spec_world_size(spec.parallel_spec), parallel_spec=spec.parallel_spec),
                remote_group=prm_group(0)))

    default_scorer = 'turn' if rollout_config.max_turns is not None else 'delimiter'
    prm_scorer = PluginRegistry.resolve(PRM_STEP_SCORER, reward_config.prm_scorer or default_scorer,
                                        config=reward_config)
    return prm_scorer, scorers


def _load_chord_features(rl_config: RLConfig, dataset_config: DatasetConfig, template: Any) -> List[dict]:
    """Encode CHORD's expert SFT rows once; the assembler cycles over defensive copies."""
    if not rl_config.chord_sft_dataset:
        return []
    from copy import copy

    from swift.dev.builders import load_prompt_only_rows

    chord_config = copy(dataset_config)
    chord_config.dataset = list(rl_config.chord_sft_dataset)
    chord_config.val_dataset = []
    rows = load_prompt_only_rows(chord_config, None, split_dataset_ratio=0.0)
    features = [template.encode(row) for row in rows]
    if not features:
        raise ValueError('chord_sft_dataset produced no encodable rows.')
    return features
