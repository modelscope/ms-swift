from __future__ import annotations
import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from swift.dev.recipe._distill import (
    DistillAssembler,
    DistillBatch,
    DistillRow,
    build_frozen_teacher,
    distill_rows_from_dataset,
    distill_sampling_params,
)
from swift.dev.recipe._streaming import build_streaming_driver
from swift.dev.recipe.checkpoint import CheckpointSaver
from swift.dev.recipe.rl_auxiliary import PolicyAuxiliary
from swift.dev.recipe.run_grpo import (
    SyncableRollout,
    _TEACHER_GROUP,
    _initialize_twinkle_rl,
    _sampler_backend,
    _sampler_engine_args,
    _sampler_world_size,
    plan_rl_device_groups,
    teacher_group_world_size,
)
from swift.dev.recipe.step_runner import StepRunner
from swift.dev.rollout import blocking_generate
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
        RolloutConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )
    from swift.dev.model import TrainableModel

logger = get_logger()


def run_gkd(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    dataset_config: DatasetConfig,
    train_config: TrainConfig,
    distributed_config: DistributedConfig,
    checkpoint_config: CheckpointConfig,
    rollout_config: RolloutConfig,
    rl_config: RLConfig,
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
    """Assemble and run on-policy GKD with weight sync. Returns the loss history.

    Mirrors :func:`run_rft`'s assembly (train/sampler device groups, weight-syncable ``SyncableRollout``,
    design-B mini-batch width) but sets the GKD loss (``configure_rl_loss``) and drives
    :class:`~swift.dev.recipe._distill.DistillAssembler` (signal='logits', teacher-scored distillation)
    instead of a policy-gradient or SFT step. The student
    is the trained model; the teacher is a frozen separate model (``rl_config.teacher_model``) or, for a
    LoRA student whose teacher is its own base, the adapter-disabled student
    (``rl_config._teacher_use_disable_adapter``).
    """
    from swift.dev.builders import build_dp_mesh, build_sampler
    from swift.dev.loss import configure_rl_loss
    from swift.dev.builders.optimizer import configure_optimizer, resolve_max_grad_norm
    from swift.dev.recipe.assembly import TrainAssembly, single_teacher_id

    if rl_config.rl_type != 'gkd':
        raise ValueError(f'run_gkd requires rl_type="gkd", got {rl_config.rl_type!r}.')
    assembly = TrainAssembly(
        'run_gkd',
        model_config,
        template_config,
        dataset_config,
        train_config,
        distributed_config,
        checkpoint_config,
        tuner_config,
        rl_config=rl_config,
        output_dir=output_dir,
        logging_config=logging_config,
        quantize_config=quantize_config,
        megatron_config=megatron_config,
        moe_config=moe_config)
    assembly.prepare()

    # A separate frozen teacher is a Ray actor on its own DeviceGroup (plan §3.3), which
    # plan_rl_device_groups must allocate BEFORE twinkle.initialize; the adapter-disabled teacher
    # (a PolicyAuxiliary) reuses the student's own actor and needs no group (teacher_world_size=0).
    separate_teacher = (not rl_config._teacher_use_disable_adapter
                        and single_teacher_id(rl_config.teacher_model, algo='GKD') is not None)
    teacher_world_size = teacher_group_world_size(rl_config) if separate_teacher else 0

    backend = _sampler_backend(rollout_config)
    sampler_world_size = _sampler_world_size(rollout_config, backend)
    groups, sampler_remote_group, colocate = plan_rl_device_groups(
        distributed_config.nproc_per_node, rollout_config.sampler_mode, sampler_world_size,
        auxiliary_groups=[(_TEACHER_GROUP, teacher_world_size)] if teacher_world_size else [])
    _initialize_twinkle_rl(
        distributed_config,
        groups,
        seed=train_config.seed,
        full_determinism=train_config.full_determinism,
        tf32=train_config.tf32)

    assembly.build_template()
    assembly.build_model()
    configure_rl_loss(assembly.model, rl_config)
    # Prompts are rolled out, not iterated by a dataloader, so the step budget is derived from the prompt
    # set (B1): one generation batch per round, exhausted after num_train_epochs passes. Load the prompts +
    # design-B mini-batch width first, then size max_steps (the LR horizon); --max_steps still overrides.
    prompts, prompt_extras, dataset_rows = distill_rows_from_dataset(dataset_config, assembly.template)
    if rl_config.lmbda < 1.0 and not dataset_rows:
        raise ValueError('GKD with lmbda < 1 distils some rounds on the dataset\'s own completions, but no dataset '
                         'row carries an assistant completion. Add completions, or set lmbda=1 for purely on-policy '
                         'GKD.')
    # Design-B mini-batch width (per_device_train_batch_size * dp_size); see run_grpo for why dp_size comes
    # off build_dp_mesh (online RL is Ray-only, pure data-parallel over nproc_per_node).
    train_batch_size = train_config.per_device_train_batch_size * build_dp_mesh(
        distributed_config).data_world_size
    # GKD replays each round once (no num_iterations), so the budget uses num_iterations=1. Under lmbda<1 the
    # off-policy rounds size their mini-batches from dataset_rows instead of the rollout, so the LR horizon is
    # approximate there; the scheduler's exhaustion still bounds the actual round count.
    from swift.dev.recipe.train_loop import resolve_rollout_max_steps, rollout_step_budget
    max_steps = resolve_rollout_max_steps(
        train_config.max_steps,
        rollout_step_budget(
            num_prompts=len(prompts),
            num_generations=rl_config.num_generations,
            train_batch_size=train_batch_size,
            gradient_accumulation_steps=assembly.ga,
            num_train_epochs=train_config.num_train_epochs,
            generation_batch_size=rollout_config.generation_batch_size),
        recipe='run_gkd')
    assembly.resolve_step_intervals(max_steps)
    configure_optimizer(
        assembly.model,
        train_config,
        num_training_steps=max_steps,
        distributed_config=distributed_config,
        tuner_config=assembly.tuner_config,
        plugin_config=assembly.plugin_config)

    sampler_engine_args = _sampler_engine_args(rollout_config, engine_args, colocate, backend)
    sampler = build_sampler(
        model_config,
        backend=backend,
        engine_args=sampler_engine_args,
        template=assembly.template,
        remote_group=sampler_remote_group)
    rollout = SyncableRollout(
        assembly.model, sampler, assembly.template, colocate=colocate, sleep_level=rollout_config.sleep_level)

    # The flat recipe body: two regimes selected by lmbda (the SAME predicate the retired
    # build_rollout_driver(sync_only=...) used -- see validate._sync_only_feature). A purely on-policy run
    # (lmbda == 1.0) rides the per-sample stream; an off-policy-capable run (lmbda != 1.0) needs whole
    # dataset rounds, which a per-sample stream structurally cannot serve, so it runs the synchronous
    # round loop. The lmbda range check and the teacher build run BEFORE the StepRunner builds its
    # RunTracker, whose reporters initialize as a construction side effect.
    if not 0.0 <= rl_config.lmbda <= 1.0:
        raise ValueError(f'distillation lmbda must be in [0, 1], got {rl_config.lmbda}.')
    from swift.dev.callbacks import CallbackHandler, build_callbacks

    # Late-bound component cells: the per-step extra-metric closure is injected into the recorder BEFORE
    # the driver exists, so it reads it through a one-element cell filled right after the build. The
    # assembler has no reward metric (distillation scores no outcome channel), so the extra channels are
    # the driver's alone.
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

    assembler = DistillAssembler(
        assembly.model,
        rollout,
        teacher=_build_teacher(assembly.model, assembly.template, rl_config, tuner_config, distributed_config),
        template=assembly.template,
        prompts=prompts,
        signal='logits',
        gkd_logits_topk=rl_config.gkd_logits_topk,
        sft_alpha=rl_config.sft_alpha,
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=assembly.ga)
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
        save_total_limit=checkpoint_config.save_total_limit,
        consumed_train_samples_fn=lambda: recorder.global_step)
    if assembly.resume_dir:
        saver.resume(assembly.resume_model())

    from twinkle.perf import Perf

    from swift.dev.recipe._loop_utils import filter_perf_actors

    if Perf.initialize(filter_perf_actors({'model': assembly.model, 'sampler': rollout.sampler})):
        logger.info('[perf] profiling enabled for the online GKD loop.')
    handover = None
    try:
        if rl_config.lmbda == 1.0:
            # A purely on-policy run rides the per-sample stream at max_staleness=0 ('none',
            # drain-before-publish) or 1 ('one_step_off', the only overlapping regime distillation
            # supports; 'fully_async' is rejected by validate._check_async_mode). The streaming config
            # guard runs inside build_streaming_driver.
            driver, handover = build_streaming_driver(
                rollout_config,
                colocate=colocate,
                run_id='run_gkd',
                model=assembly.model,
                rollout=rollout,
                output_dir=output_dir,
                prompts=prompts,
                prompt_extras=prompt_extras,
                adapter_name=('default' if tuner_config is not None else None),
                train_batch_size=train_batch_size,
                gradient_accumulation_steps=assembly.ga,
                num_generations=rl_config.num_generations,
                num_train_epochs=train_config.num_train_epochs,
                seed=train_config.seed,
                sampling_params=distill_sampling_params(rl_config, generation_config),
                unit_size=1,
                assemble=assembler.assemble,
                assembly_ready=assembler.assembly_ready,
                reached_max=recorder.reached_max)
            _driver_cell.append(driver)
            if handover.serializes:
                # Version-0 hand-over: put the pre-training policy in the sampler before the first admission.
                # Every later cycle's enter rides the in-place publish's sync_fn (see the builder).
                handover.enter()
            for batch in driver:
                _fold_idle_ratio()
                optimizer_steps = 0
                for mini_batch in assembler.plan_mini_batches(batch):
                    recorder.micro_step += 1
                    assembly.model.forward_backward(**assembler.forward_kwargs(mini_batch))
                    is_boundary = recorder.is_grad_sync_boundary()
                    assembly.model.clip_grad_and_step(
                        max_grad_norm=recorder.max_grad_norm,
                        gradient_accumulation_steps=recorder.gradient_accumulation_steps)
                    if is_boundary:
                        recorder.record_step()
                        optimizer_steps += 1
                        if saver.save_steps and recorder.global_step % saver.save_steps == 0:
                            saver.save(f'checkpoint-{recorder.global_step}')
                        if recorder.reached_max():
                            break
                # commit releases the driver thread blocked in the iterator bridge and reports the real step
                # count -- the publish cadence unit (twinkle publishes only after consume returns).
                driver.commit(optimizer_steps)
                if recorder.reached_max():
                    break
        else:
            # The off-policy-capable regime: per-round on/off-policy coin flip over the prompt set's
            # generation batches (the retired FixedBatchDriver's cadence). BOTH round kinds train -- an
            # off-policy round distils the dataset completions (and adds the SFT term via
            # apply_sft_loss); it is not a skipped round. round_index counts rounds (NOT optimizer steps,
            # W27) so the coin flip and the dataset window stay aligned with the batch being processed.
            from swift.dev.recipe.train_loop import PromptBatchScheduler

            scheduler = PromptBatchScheduler(
                len(prompts),
                generation_batch_size=rollout_config.generation_batch_size,
                num_train_epochs=train_config.num_train_epochs,
                seed=train_config.seed)
            round_index = 0
            for prompt_indices in scheduler:
                if recorder.reached_max():
                    break
                use_student = random.Random(train_config.seed + round_index).random() <= rl_config.lmbda
                round_index += 1
                if use_student:
                    # blocking_generate syncs the trained policy into the sampler first, so an on-policy
                    # round samples from the current policy.
                    samples = blocking_generate(
                        rollout,
                        [prompts[i] for i in prompt_indices],
                        num_samples=rl_config.num_generations,
                        sampling_params=distill_sampling_params(rl_config, generation_config),
                        prompt_extras=[prompt_extras[i] for i in prompt_indices])
                    expected = len(prompt_indices) * rl_config.num_generations
                    if len(samples) != expected:
                        raise RuntimeError(f'rollout returned {len(samples)} samples, expected {expected} '
                                           f'({len(prompt_indices)} prompts * {rl_config.num_generations} generations).')
                    for local_idx, prompt_idx in enumerate(prompt_indices):
                        for sample in samples[local_idx * rl_config.num_generations:(local_idx + 1) * rl_config.num_generations]:
                            sample.prompt_id = str(prompt_idx)
                    rows = [
                        DistillRow(sample.input_feature,
                                   sample.messages if sample.messages is not None else prompts[int(sample.prompt_id)],
                                   sample.extra or {}) for sample in samples
                    ]
                else:
                    # A rolling window over the dataset completions, so successive off-policy rounds cover
                    # new rows (run_gkd already rejected lmbda<1 without dataset completions).
                    count = len(dataset_rows)
                    offset = ((round_index - 1) * train_batch_size) % count
                    rows = [dataset_rows[(offset + i) % count] for i in range(count)]
                for mini_batch in assembler.plan_mini_batches(DistillBatch(rows, use_student)):
                    if recorder.reached_max():
                        break
                    recorder.micro_step += 1
                    assembly.model.forward_backward(**assembler.forward_kwargs(mini_batch))
                    is_boundary = recorder.is_grad_sync_boundary()
                    assembly.model.clip_grad_and_step(
                        max_grad_norm=recorder.max_grad_norm,
                        gradient_accumulation_steps=recorder.gradient_accumulation_steps)
                    if is_boundary:
                        recorder.record_step()
                        if saver.save_steps and recorder.global_step % saver.save_steps == 0:
                            saver.save(f'checkpoint-{recorder.global_step}')
    finally:
        if handover is not None and handover.serializes and handover.in_generation:
            # Ended mid-window (e.g. a budget hit during generation): give the device back to the trainer
            # so the post-run final save finds the model resident.
            handover.exit()
        Perf.shutdown()
        rollout.shutdown()
        recorder.tracker.close()
    if _save_final:
        saver.save_final(write_args_json=assembly.args_json_writer())
    return recorder.history


def _build_teacher(model: TrainableModel, template: Any, rl_config: RLConfig, tuner_config: Optional[TunerConfig],
                   distributed_config: DistributedConfig) -> Any:
    """Resolve the GKD teacher: the LoRA student's own adapter-disabled base, or a frozen Ray-actor model.

    There is no HTTP teacher server (removed, RL_PLAN §2.H): the teacher is either a
    :class:`~swift.dev.recipe.rl_auxiliary.PolicyAuxiliary` with ``disable_adapter=True`` -- the paper's
    headline self-distillation setting, scoring with ``forward_only(disable_lora=True)`` on the student's own
    actor (no separate group) -- or a :class:`~swift.dev.recipe.rl_auxiliary.FrozenAuxiliary` wrapping a
    separate frozen twinkle model actor built from ``teacher_model`` on its own ``'teacher'`` DeviceGroup
    (sized by ``teacher_parallel_spec``, planned by ``plan_rl_device_groups``). ``offload_teacher_model`` is
    applied inside the frozen teacher (its full-vocab logits are the memory hook), so it is rejected for the
    adapter-disabled base, which has no separate weights to offload.
    """
    if rl_config._teacher_use_disable_adapter:
        if tuner_config is None:
            raise ValueError('rl_config._teacher_use_disable_adapter=True requires a LoRA student (tuner_config), '
                             'since it distils from the adapter-disabled base of that same model.')
        if rl_config.offload_teacher_model:
            raise ValueError('offload_teacher_model requires a distinct frozen teacher model, not the '
                             'adapter-disabled base of the student (which has no separate weights to offload).')
        return PolicyAuxiliary(model, disable_adapter=True)
    from swift.dev.recipe.assembly import single_teacher_id
    teacher_model = single_teacher_id(rl_config.teacher_model, algo='GKD')
    if teacher_model is None:
        raise ValueError('GKD needs a teacher: set RLConfig.teacher_model, or use a LoRA student with '
                         '_teacher_use_disable_adapter=True to distil from its own frozen base.')
    return build_frozen_teacher(
        rl_config, template, teacher_model, distributed_config=distributed_config, remote_group=_TEACHER_GROUP,
        adapters=rl_config.teacher_adapters, role='teacher', offload=rl_config.offload_teacher_model)

