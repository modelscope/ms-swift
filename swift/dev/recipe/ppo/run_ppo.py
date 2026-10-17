"""PPO assembly: run_ppo orchestration (proximal policy optimization).

Peer of ``run_grpo`` for the one online RL type with a learned value function. PPO here is built from
the pieces twinkle already provides, with no fabricated infrastructure:

  - policy loss = the shared clipped surrogate (``GRPOLoss`` with epsilon=cliprange) -- PPO's policy
    objective is exactly that clip, so ``configure_rl_loss`` maps ppo -> GRPOLoss;
  - critic = a ``seq_cls`` ``num_labels=1`` model forwarded with ``task='value'`` (which keeps the
    head's per-token output instead of pooling to the last token), emitting ``V(s_t)`` at every token,
    trained by the clipped value loss (``configure_ppo_value_loss`` -> twinkle ``PPOValueLoss``). The
    ``task='value'`` behaviour is symmetric across backends (transformers ``TransformersValuePatch`` /
    Megatron ``forward_step``);
  - reward = the shared outcome channel (``--orm`` -- rules and scalar reward models alike, resolved by
    the same builder GRPO/RFT use and scored through the skeleton's reward half) combined as a weighted sum
    once per completion, plus the standard per-token KL-to-reference penalty;
  - rollout = the same weight-syncable ``SyncableRollout`` (colocate / disaggregated) run_grpo uses.

PER-TOKEN GAE: because the critic emits a value at every response token, PPO runs in its full
token-level form. Each response token gets a reward ``r_t = -kl_coef * (logp_t - ref_logp_t)`` with the
reward model's scalar added at the final token; twinkle's ``GAEAdvantage`` then walks backwards over
the response with ``gamma``/``lam`` to produce per-token advantages (for the clipped policy surrogate)
and returns (for the clipped value loss). advantages/old_logps/returns/old_values are all captured ONCE
per rollout and re-used across ``num_ppo_epochs``, so the epochs are genuine PPO batch re-uses.

The per-token value head is the DDP / AccelerateStrategy path; it reads the base model's last hidden
state and does not yet cover sequence-parallel / packed layouts, so the critic runs without those.

Placement (colocate vs heterogeneous) and weight-sync are identical to run_grpo.
"""
from __future__ import annotations
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

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


def run_ppo(
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
    """Assemble and run online PPO (policy + critic + reference + reward model + weight-syncable rollout).

    Reuses run_grpo's device planning and rollout, then trains the policy (clipped surrogate) and the
    critic (clipped value loss) over each rollout for ``num_ppo_epochs``. Returns the loss history.
    """
    from swift.dev.builders import build_sampler
    from swift.dev.loss import (configure_ppo_value_loss, configure_ppo_value_metric, configure_rl_loss,
                                configure_rl_metrics)
    from swift.dev.builders.optimizer import configure_optimizer, resolve_max_grad_norm
    from swift.dev.recipe.common.assembly import TrainAssembly
    from swift.dev.recipe.rl.auxiliary import REF_GROUP, plan_rl_device_groups, reward_group, spec_world_size
    from swift.dev.recipe.rl.prompts import prompts_from_dataset
    from swift.dev.recipe.rl.rewards import build_reward_scorers, split_reward_channel
    from swift.dev.recipe.rl.sampler import (
        RolloutSamplerSpec,
        SyncableRollout,
        initialize_twinkle_rl,
        rollout_sampling_params,
    )

    if rl_config.rl_type != 'ppo':
        raise ValueError(f'run_ppo requires rl_type="ppo", got {rl_config.rl_type!r}.')
    assembly = TrainAssembly(
        'run_ppo',
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
    assembly.prepare()
    assembly.build_template()
    prompts = prompts_from_dataset(dataset_config)

    backend = RolloutSamplerSpec.backend(rollout_config)
    sampler_world_size = RolloutSamplerSpec.world_size(rollout_config, backend)
    # Plan DeviceGroups for PPO's FULL-PARAMETER frozen auxiliaries (RL_PLAN §3.5): the KL reference (a
    # frozen policy-init copy under full fine-tuning) and each reward model. A LoRA reference is the
    # adapter-disabled policy base ('disable_lora'), which reuses the policy actor and plans no group. Each
    # reward model's group is sized by --orm_parallel_spec (a wide frozen scorer shards across as many ranks
    # as it asks for), the SAME world size build_reward_scorers builds it with below, so the plan and the
    # build cannot drift.
    ref_disable_lora = tuner_config is not None and not rl_config.ref_adapters and not rl_config.sync_ref_model
    ref_remote_group: Optional[str] = None if ref_disable_lora else REF_GROUP
    # A full-parameter PPO reference is sharded by its OWN --ref_parallel_spec (world size == the ranks its
    # mesh needs), so a large frozen reference is scale-safe. When sync_ref_model is active, paired EMA
    # requires the reference to use the SAME parallel_spec as the policy (reference_parallel_spec overrides).
    effective_ref_spec = ((rl_config.reference_parallel_spec or distributed_config.parallel_spec)
                          if rl_config.sync_ref_model else rl_config.ref_parallel_spec)
    ref_world_size = spec_world_size(effective_ref_spec)
    auxiliary_groups: List[Tuple[str, int]] = [] if ref_disable_lora else [(REF_GROUP, ref_world_size)]
    reward_world_size = spec_world_size(reward_config.orm_parallel_spec)
    auxiliary_groups += [(reward_group(i), reward_world_size)
                         for i in range(len(split_reward_channel(reward_config.orm)[1]))]
    from swift.dev.recipe.rl.resource import RLResourceBootstrap
    resource_plan = RLResourceBootstrap.build(
        model_config=model_config,
        distributed_config=distributed_config,
        rollout_config=rollout_config,
        template=assembly.template,
        prompt_rows=[{'messages': prompt} for prompt in prompts],
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
        sampler_remote_group, colocate = 'sampler', False
        logger.info('RL resource plan:\n%s', resource_plan.describe())
    # PPO runs SP=1: the per-token value-head critic does not cover sequence-parallel / packed layouts
    # (see the module docstring), and policy+critic MUST share one mesh -- a SP policy with an SP=1 critic
    # would slice_dp each forward_backward at a different data_world_size, starving the critic's ranks. So
    # PPO deliberately plans no SP mesh, and validate_configs rejects sequence_parallel_size>1 for it.
    initialize_twinkle_rl(
        distributed_config,
        groups,
        seed=train_config.seed,
        full_determinism=train_config.full_determinism,
        tf32=train_config.tf32,
        sampler_parallel_width=RolloutSamplerSpec.parallel_width(rollout_config, backend))

    # The critic, the reference and the reward models all encode with the SAME template as the policy,
    # so a batch lines up token-for-token across the four forwards.
    template = assembly.template

    # Design-B mini-batch width: per_device_train_batch_size * dp_size, the same global-batch formula GRPO
    # uses. Online RL is Ray-only and pure data-parallel over nproc_per_node, so dp_size comes off
    # build_ray_dp_mesh -- the exact mesh the policy/critic were placed on and slice_dp splits each
    # forward_backward across. Every mini-batch must carry >= dp_size rows or slice_dp leaves some rank
    # with no data ("Batch too small"), which is why PPO feeds train_batch_size-row mini-batches, not the
    # one-sample-per-step it used to. Computed before max_steps because the step budget is now expressed in
    # optimizer steps, a count this mini-batch width defines.
    from swift.dev.builders import build_prompt_dataloader, build_ray_dp_mesh
    dp_size = build_ray_dp_mesh(distributed_config).data_world_size
    train_batch_size = train_config.per_device_train_batch_size * dp_size

    # Prompts are rolled out, not iterated by a dataloader, so the step budget is derived from the prompt
    # set. Under the unified optimizer-step cadence every recipe's global_step counts optimizer steps, PPO
    # included: one rollout splits into train_batch_size mini-batches and replays them num_ppo_epochs times,
    # so the LR-scheduler horizon is rollout_step_budget(num_iterations=num_ppo_epochs), NOT the rollout
    # count it used to be. --max_steps still overrides.
    from swift.dev.recipe.rl.prompts import resolve_rollout_max_steps, rollout_step_budget
    max_steps = resolve_rollout_max_steps(
        train_config.max_steps,
        rollout_step_budget(
            num_prompts=len(prompts),
            num_generations=rl_config.num_generations,
            train_batch_size=train_batch_size,
            gradient_accumulation_steps=assembly.ga,
            num_train_epochs=train_config.num_train_epochs,
            generation_batch_size=rollout_config.generation_batch_size,
            num_iterations=max(1, rl_config.num_ppo_epochs)),
        recipe='run_ppo')
    assembly.resolve_step_intervals(max_steps)

    # Policy (trainer): the clipped surrogate is the PPO policy loss (configure_rl_loss maps ppo). PPO
    # plans no SP mesh (SP=1) -- the critic runs SP=1 too, so policy/critic/dp_size all agree on a pure-DP
    # mesh of data_world_size = nproc_per_node.
    assembly.build_model()
    model = assembly.model
    configure_rl_loss(model, rl_config)
    configure_optimizer(
        model,
        train_config,
        num_training_steps=max_steps,
        distributed_config=distributed_config,
        tuner_config=assembly.tuner_config,
        plugin_config=assembly.plugin_config)
    # Register PPOMetric on the policy AFTER its optimizer group exists: twinkle accumulates it inside
    # forward_backward from advantages/old_logps, so its ratio/clip stats ride calculate_metric per step.
    configure_rl_metrics(model, rl_config)

    # Critic: a trainable seq_cls (num_labels=1) value model, trained by the clipped value loss.
    value_model = _build_value_model(
        model_config,
        reward_config,
        distributed_config,
        train_config,
        template,
        model_path=(os.path.join(assembly.resume_dir, 'value_model') if assembly.resume_dir else None),
        megatron_config=megatron_config,
        moe_config=moe_config)
    configure_ppo_value_loss(value_model, rl_config)
    # The critic is full-parameter (_build_value_model applies no tuner), so tuner_config is None: passing
    # the policy's LoRA+ tuner_config would let its ``ndim == 1`` rule scale the critic's bias / norm lr by
    # lorap_lr_ratio. plugin_config is still threaded so a user param_groups plugin can group critic params.
    configure_optimizer(
        value_model,
        train_config,
        num_training_steps=max_steps,
        distributed_config=distributed_config,
        tuner_config=None,
        plugin_config=assembly.plugin_config)
    # Register PPOValueMetric on the critic AFTER its optimizer group exists: twinkle accumulates it from
    # returns/old_values, so value_mean / explained_variance ride the critic's calculate_metric per step.
    configure_ppo_value_metric(value_model, rl_config)

    # Rollout (weight-syncable, exactly as run_grpo), reward model(s) and reference for the KL penalty.
    sampler_engine_args = RolloutSamplerSpec.engine_args(rollout_config, engine_args, colocate, backend, tuner_config=assembly.tuner_config)
    if rollout_config.resource_mode == 'adaptive':
        from swift.dev.recipe.rl.resource import AdaptiveResourceController
        controller = AdaptiveResourceController.build(
            model=model,
            model_config=model_config,
            template=template,
            rollout_config=rollout_config,
            backend=backend,
            engine_args=sampler_engine_args,
            plan=resource_plan,
            additional_handover_models=(value_model, ))
        sampler = controller.sampler
        rollout = SyncableRollout(
            model, sampler, template, colocate=False, sleep_level=rollout_config.sleep_level,
            handover=controller.handover)
        rollout.resource_controller = controller
    else:
        sampler = build_sampler(
            model_config,
            backend=backend,
            engine_args=sampler_engine_args,
            device_mesh=RolloutSamplerSpec.device_mesh(rollout_config, backend),
            template=template,
            remote_group=sampler_remote_group)
        rollout = SyncableRollout(
            model, sampler, template, colocate=colocate, sleep_level=rollout_config.sleep_level)
    reference = _build_reference(
        model_config, tuner_config, template, rl_config, distributed_config, remote_group=ref_remote_group)
    # EMA manager: paired 2-rank NCCL groups (policy rank k <-> reference rank k) for streaming reference
    # sync. Only when sync_ref_model is active AND the reference is a real model (not the 'disable_lora'
    # string sentinel, which shares the policy actor and has no independent weights to update).
    ema_manager = None
    if rl_config.sync_ref_model and reference is not None and reference != 'disable_lora':
        from twinkle.checkpoint_engine import CheckpointEngineManager
        ema_manager = CheckpointEngineManager(
            model=assembly.model,
            sampler=reference,
            platform='GPU',
            mode='standalone',
            channel='ema:reference')
    # The outcome-reward channel is resolved by the SAME shared builder GRPO/RFT use: the ``--orm`` channel's
    # rules plus each reward MODEL id become one ordered ``RewardScorer`` list, so PPO scores rules and
    # reward models together through the skeleton's ``_score`` (a weighted sum over the channel, not the old
    # per-model mean). Each reward model was placed on its own ``reward_<i>`` group sized by
    # ``--orm_parallel_spec`` (planned above); a generative/api judge is rejected there until the phase③
    # auxiliary-placement unification, exactly as for GRPO.
    reward_scorers, reward_scorer_names = build_reward_scorers(
        model_config, template_config, reward_config, distributed_config, sampler_model=model_config.model,
        backend=backend)

    # The flat recipe body: the twinkle StreamingDriver iterator (built below) yields scored + assembled
    # TrainingBatches on a background thread, and this body trains each on BOTH models -- no loop class, no
    # inheritance. PPO has no sync-only feature (dynamic_sample/remax are GRPO advantage estimators; PPO's
    # per-token GAE has no group contract and no whole-batch prepass), so every PPO run rides the per-sample
    # stream: 'none' runs it at max_staleness=0 (admit one window, drain it, train, publish -- the same
    # per-sample control flow, no overlap), 'one_step_off' pins staleness to 1, and 'fully_async' uses the
    # configured max_staleness. The critic is never published to the sampler, so no regime touches the
    # value_model. The builder runs the streaming config guard before the recorder's RunTracker opens any
    # reporter, so an unsound stream fails loudly with no side effect. A LoRA run pins each policy version
    # by its 'default' adapter; a full-parameter run has none and is served as merged base weights.
    from swift.dev.callbacks import CallbackHandler, build_callbacks
    from swift.dev.recipe.rl.ref_only import build_remote_batch_assembler
    from swift.dev.recipe.rl.streaming import build_streaming_driver
    from swift.dev.recipe.common.checkpoint import CheckpointSaver
    from swift.dev.recipe.ppo.assembler import PPOAssembler, _ppo_log_line
    from swift.dev.recipe.common.step_runner import StepRunner

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
        """Fold the critic's metrics + the driver-side per-step channels the model cannot surface.

        The critic's loss/grad_norm are namespaced ``value_loss`` / ``value_grad_norm`` so they cannot
        clobber the policy's authoritative ``loss``; its PPOValueMetric fields (value_mean /
        explained_variance / ...) ride under their own names. ``value_model.calculate_metric`` is read here
        -- inside record_step, after the critic stepped for this boundary -- so the merge reflects the step
        just completed (the re-home of the retired PPOLoop._extra_step_metrics). The assembler adds the
        batch version-span aggregate + the reward metric; this closure adds the driver's idle-ratio and
        lifetime stream counters.
        """
        from swift.dev.recipe.common.tracking import coerce_metric_scalars

        extra: Dict[str, Any] = {}
        assembler = _assembler_cell[0] if _assembler_cell else None
        if assembler is not None:
            extra.update(assembler.step_telemetry())
        for key, value in coerce_metric_scalars(value_model.calculate_metric(is_training=True)).items():
            if key == 'loss':
                extra['value_loss'] = value
            elif key == 'grad_norm':
                extra['value_grad_norm'] = value
            else:
                extra[key] = value
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
        log_formatter=_ppo_log_line,
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
    assembler = PPOAssembler(
        assembly.model,
        rollout,
        value_model=value_model,
        reference=reference,
        reward_scorers=reward_scorers,
        reward_scorer_names=reward_scorer_names,
        reward_weights=reward_config.orm_weights,
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=assembly.ga,
        rl_config=rl_config)
    payload_processor = build_remote_batch_assembler(assembler)
    _assembler_cell.append(payload_processor)
    # The driver: twinkle's per-sample StreamingDriver in ITERATOR mode (consume=None), injected with the
    # assembler's score/assemble seams. PPO's assembly unit is ONE sample (per-token GAE scores each row on
    # its own, no group contract), so unit_size=1 and no group-completion hook. Returns (driver, handover):
    # the colocate hand-over bracket is the recipe's, not the driver's. The prompt dataloader is a twinkle
    # DataLoader over the prompt rows (PPO prompts carry no extra columns); the driver flattens its batches,
    # draws rows, and stamps its own ordinal as prompt_id.
    prompt_dataloader = build_prompt_dataloader(prompts, None, seed=train_config.seed)
    driver, handover = build_streaming_driver(
        rollout_config,
        colocate=colocate,
        run_id='run_ppo',
        model=assembly.model,
        rollout=rollout,
        output_dir=output_dir,
        prompt_dataloader=prompt_dataloader,
        num_train_epochs=train_config.num_train_epochs,
        adapter_name=('default' if tuner_config is not None else None),
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=assembly.ga,
        num_generations=rl_config.num_generations,
        sampling_params=rollout_sampling_params(rl_config, generation_config),
        unit_size=1,
        score=assembler.score_sample,
        assemble=assembler.assemble,
        assembly_ready=assembler.assembly_ready,
        payload_processor=payload_processor,
        global_step=lambda: recorder.global_step,
        reached_max=recorder.reached_max)
    _driver_cell.append(driver)
    if assembly.resume_dir:
        policy_state = assembly.resume_model()
        saver.resume(policy_state)
        value_state = assembly.resume_model(
            value_model, os.path.join(assembly.resume_dir, 'value_model'), adapter_name='')
        # The critic's weights/optimizer are already restored above; its trainer_state is only cross-checked
        # here -- a mismatch means the two checkpoints came from different steps (both models step in phase
        # on the one recorder counter).
        if value_state is not None and int(value_state.get('consumed_train_samples', 0)) != recorder.global_step:
            raise ValueError('PPO policy and value checkpoints contain different completed-step counts.')

    from twinkle.perf import Perf

    from swift.dev.recipe.common.loop_utils import filter_perf_actors

    if Perf.initialize(filter_perf_actors({
            'model': assembly.model,
            'sampler': rollout.sampler,
            'reference': reference,
            'critic': value_model,
    })):
        logger.info('[perf] profiling enabled for the online PPO loop.')
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
                        batch.forward_backward(mini_batch_index)
                        is_boundary = recorder.is_grad_sync_boundary()
                        assembly.model.clip_grad_and_step(
                            max_grad_norm=recorder.max_grad_norm,
                            gradient_accumulation_steps=recorder.gradient_accumulation_steps)
                        batch.forward_backward(mini_batch_index, kind='critic')
                        value_model.clip_grad_and_step(
                            max_grad_norm=recorder.max_grad_norm,
                            gradient_accumulation_steps=recorder.gradient_accumulation_steps)
                        if is_boundary:
                            recorder.record_step()
                            optimizer_steps += 1
                            if ema_manager is not None and recorder.global_step % rl_config.ref_model_sync_steps == 0:
                                ema_manager.sync_ema(rl_config.ref_model_mixup_alpha)
                            if saver.save_steps and recorder.global_step % saver.save_steps == 0:
                                saver.save(f'checkpoint-{recorder.global_step}')
                                _save_value_checkpoint(
                                    value_model, os.path.join(output_dir, f'checkpoint-{recorder.global_step}'),
                                    checkpoint_config=checkpoint_config,
                                    consumed_train_samples=recorder.global_step)
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
        _save_value_checkpoint(
            value_model, os.path.join(output_dir, 'checkpoint-final'),
            checkpoint_config=checkpoint_config, consumed_train_samples=recorder.global_step)
    return recorder.history


def _build_value_model(model_config: ModelConfig, reward_config: Optional['RewardConfig'],
                       distributed_config: DistributedConfig,
                       train_config: TrainConfig, template: Any, *, model_path: Optional[str] = None,
                       megatron_config: Optional[MegatronConfig] = None,
                       moe_config: Optional[MoEConfig] = None) -> Any:
    """Build the trainable critic: a ``seq_cls`` num_labels=1 model, forwarded with ``task='value'``.

    Initialised from the first reward MODEL in the ``--orm`` channel when one is given (closest to TRL,
    which inits the value function from the reward model), else from the policy's own base. The value
    head IS the seq_cls
    ``score`` linear (trainable, in the optimizer) -- ``task='value'`` keeps its PER-TOKEN output
    ``V(s_t)`` instead of pooling to the last token, symmetric across transformers and Megatron. Placed
    with the SAME DistributedConfig as the policy.
    """
    from copy import copy

    from swift.dev.builders import build_model
    from swift.dev.processor import InputProcessor

    from swift.dev.recipe.rl.rewards import split_reward_channel

    orm_models = split_reward_channel(reward_config.orm)[1] if reward_config is not None else []
    init_from = orm_models[0] if orm_models else model_config.model
    value_cfg = copy(model_config)
    value_cfg.model = model_path or init_from
    value_cfg.task_type = 'seq_cls'  # the per-token value rides the seq_cls score head
    value_cfg.num_labels = 1
    value_model = build_model(
        value_cfg,
        distributed_config,
        train_config,
        megatron_config=megatron_config,
        moe_config=moe_config,
        # The critic shares the policy's 'model' DeviceGroup (it MUST: policy+critic slice_dp each
        # forward_backward at the same data_world_size), so it needs a distinct instance_id or its Ray
        # actor name collides with the policy's -- both are built from the same builders/model.py line
        # into the same group, and twinkle names the actor by group+class+caller+rank.
        instance_id='value')
    value_model.set_processor(InputProcessor, cp_partition_mode=distributed_config.cp_partition_mode)
    value_model.set_template(template)
    return value_model


def _build_reference(model_config: ModelConfig, tuner_config: Optional[TunerConfig], template: Any,
                     rl_config: RLConfig, distributed_config: DistributedConfig, *,
                     remote_group: Optional[str]) -> Any:
    """PPO's frozen reference for the KL penalty: 'disable_lora' (LoRA) or a frozen policy-init model.

    LoRA reuses the adapter-disabled base (no second model, no DeviceGroup); full fine-tuning loads a
    frozen copy of the policy's initial weights (PPO anchors the KL to the starting policy) as a separate
    Ray actor on its OWN ``remote_group`` DeviceGroup, built with the run's backend via
    :func:`frozen_auxiliary_distributed_config` (RL_PLAN §3.5, basic principles 1/2). ``remote_group`` is
    the 'ref' group the run body planned (None on the LoRA path, which returns before using it).

    When ``sync_ref_model`` is active the reference MUST use the same parallel_spec as the policy (paired
    EMA constraint), so ``reference_parallel_spec`` (or the policy's own spec when None) overrides the
    general ``ref_parallel_spec``.
    """
    if tuner_config is not None and not rl_config.ref_adapters and not rl_config.sync_ref_model:
        return 'disable_lora'
    from copy import copy

    from swift.dev.builders import build_model, frozen_auxiliary_distributed_config
    from swift.dev.recipe.common.assembly import configure_frozen_adapter
    from swift.dev.recipe.rl.auxiliary import spec_world_size

    ref_config = copy(model_config)
    ref_config.model = rl_config.ref_model or model_config.model
    ref_config.model_type = rl_config.ref_model_type or model_config.model_type
    ref_config.model_revision = rl_config.ref_model_revision or model_config.model_revision
    # Paired EMA requires both sides on identical sharding; otherwise the card-frugal ref_parallel_spec.
    ref_parallel_spec = ((rl_config.reference_parallel_spec or distributed_config.parallel_spec)
                         if rl_config.sync_ref_model else rl_config.ref_parallel_spec)
    ref_world_size = spec_world_size(ref_parallel_spec)
    device_mesh = None
    if ref_parallel_spec:
        from twinkle import DeviceMesh
        device_mesh = DeviceMesh.from_spec(ref_parallel_spec)
    ref = build_model(
        ref_config,
        frozen_auxiliary_distributed_config(distributed_config, ref_world_size, parallel_spec=ref_parallel_spec),
        device_mesh=device_mesh,
        remote_group=remote_group)
    return configure_frozen_adapter(ref, template, rl_config.ref_adapters, role='ref')


def _save_value_checkpoint(value_model: Any, checkpoint_dir: str, *,
                           checkpoint_config: CheckpointConfig, consumed_train_samples: int) -> None:
    """Persist the critic into ``<policy checkpoint>/value_model`` (PPO's two-model save half).

    The policy's :class:`~swift.dev.recipe.common.checkpoint.CheckpointSaver` owns the cadence and the main write;
    this mirrors the retired ``PPOLoop.save`` second half so every policy checkpoint (periodic and final)
    carries a step-aligned critic -- both models step in phase on the one recorder counter, so a policy
    checkpoint without its value_model is un-resumable.
    """
    from swift.dev.recipe.common.loop_utils import save_training_checkpoint

    save_training_checkpoint(
        value_model,
        'value_model',
        output_dir=checkpoint_dir,
        consumed_train_samples=consumed_train_samples,
        no_save_optim=checkpoint_config.no_save_optim or checkpoint_config.save_only_model,
        no_save_rng=checkpoint_config.no_save_rng or checkpoint_config.save_only_model,
        safe_serialization=checkpoint_config.safe_serialization,
        max_shard_size=checkpoint_config.max_shard_size,
        save_total_limit=checkpoint_config.save_total_limit)
