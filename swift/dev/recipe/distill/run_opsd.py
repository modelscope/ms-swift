"""On-policy self-distillation (OPSD) assembly: run_opsd orchestration.

Peer of ``run_gkd`` for OPSD (On-Policy Self-Distillation, Zhao et al. arXiv:2601.18734). A single
model acts as BOTH teacher and student, differing only in CONTEXT: the student conditions on the
question only, the teacher conditions on a PRIVILEGED view (question + reference/rubric, supplied
per-sample by the dataset's ``teacher_prompt`` column). The student generates on-policy, the teacher
re-scores those SAME response tokens under the privileged prompt, and the student is pulled toward
the teacher's per-token distribution with a dense, logits-free sampled-token k3 surrogate
(:class:`twinkle.loss.opsd.OPSDLoss`).

Why this is a DistillAssembler sibling
--------------------------------------
OPSD shares the whole distillation skeleton with GKD/MOPD -- a weight-synced sampler rollout over Ray ->
a frozen teacher ``forward_only`` -> student ``forward_backward`` with the teacher signal, over design-B
mini-batches on Ray data parallelism, with ``lmbda`` on/off-policy round mixing -- so the flat recipe drives
:class:`~swift.dev.recipe.distill.assembler.DistillAssembler` with ``signal='logps'``, a SIBLING of GKD's
``signal='logits'`` and MOPD's ``signal='multi_logps'`` (not a subclass of either). Only the teacher SIGNAL
differs: OPSD passes response-only ``teacher_logps`` from a privileged-prompt view instead of GKD's
full-vocab ``teacher_logits`` on the shared prompt, so the assembler's ``signal`` knob is the entire
difference. The teacher forward reuses the shared chunked ``batched_response_scores`` (correct at dp_size>1);
a separate teacher is a frozen Ray actor on its own DeviceGroup (plan §3.3), never an HTTP server
(no-server premise, RL_PLAN §2.H).

Teacher selection (``_build_opsd_teacher``, all :class:`~swift.dev.recipe.rl.auxiliary.AuxiliaryModel` wrappers):
  * LoRA student with ``_teacher_use_disable_adapter`` -> ``PolicyAuxiliary(disable_adapter=True)``: the
    teacher is the student's own frozen base on the privileged view (scored on the student's actor, no extra
    group);
  * ``teacher_model`` set -> ``FrozenAuxiliary``: a separate frozen Ray-actor model on the ``'teacher'``
    DeviceGroup;
  * neither -> ``PolicyAuxiliary``: DYNAMIC self-distillation, the student's CURRENT weights on the
    privileged view.
"""
from __future__ import annotations
import math
import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from swift.dev.recipe.distill.assembler import (
    DistillAssembler,
    DistillBatch,
    DistillRow,
    build_frozen_teacher,
    build_privileged_teacher_feature,
    distill_rows_from_dataset,
    distill_sampling_params,
    response_ids_from_feature,
)
from swift.dev.recipe.rl.ref_only import build_remote_batch_assembler
from swift.dev.recipe.rl.streaming import build_streaming_driver
from swift.dev.recipe.common.checkpoint import CheckpointSaver
from swift.dev.recipe.rl.auxiliary import (
    TEACHER_GROUP,
    PolicyAuxiliary,
    plan_rl_device_groups,
    teacher_group_world_size,
)
from swift.dev.recipe.rl.sampler import RolloutSamplerSpec, SyncableRollout, initialize_twinkle_rl
from swift.dev.recipe.common.step_runner import StepRunner
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
        RewardConfig,
        RolloutConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )
    from swift.dev.model import TrainableModel

logger = get_logger()


def run_opsd(
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
    """Assemble and run on-policy self-distillation with weight sync. Returns the loss history.

    Mirrors :func:`run_gkd`'s assembly (train/sampler device groups, weight-syncable ``SyncableRollout``,
    design-B mini-batch width), swapping the GKD loop for
    :class:`~swift.dev.recipe.distill.assembler.DistillAssembler` with ``signal='logps'`` and the GKD teacher for the
    OPSD teacher (``_build_opsd_teacher``). ``lmbda`` keeps its GKD meaning (per-round probability of
    on-policy generation vs. the dataset's own completion); ``sft_alpha``/``gkd_logits_topk`` are GKD-loss
    knobs that OPSDLoss does not read, so they are not forwarded.
    """
    from swift.dev.builders import build_dp_mesh, build_prompt_dataloader, build_sampler
    from swift.dev.loss import configure_rl_loss
    from swift.dev.builders.optimizer import configure_optimizer, resolve_max_grad_norm
    from swift.dev.recipe.common.assembly import TrainAssembly, single_teacher_id

    if rl_config.rl_type != 'opsd':
        raise ValueError(f'run_opsd requires rl_type="opsd", got {rl_config.rl_type!r}.')
    if rl_config.offload_teacher_model:
        raise ValueError('offload_teacher_model is GKD-only: its full-vocab teacher forward wraps reload/offload, '
                         'while OPSD scores response-only teacher_logps via a chunked forward_only with no offload '
                         'hook. A separate OPSD teacher already gets its own DeviceGroup, so offload is pointless. '
                         'Leave it off.')
    assembly = TrainAssembly(
        'run_opsd',
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
    prompts, prompt_extras, dataset_rows = distill_rows_from_dataset(dataset_config, assembly.template)
    _require_teacher_prompt(prompt_extras)

    # A separate frozen teacher is a Ray actor on its own DeviceGroup (plan §3.3), which
    # plan_rl_device_groups must allocate BEFORE twinkle.initialize; the adapter-disabled teacher and
    # dynamic-self (teacher_model unset) both reuse the student's own actor and plan no group.
    separate_teacher = (not rl_config._teacher_use_disable_adapter
                        and single_teacher_id(rl_config.teacher_model, algo='OPSD') is not None)
    teacher_world_size = teacher_group_world_size(rl_config) if separate_teacher else 0

    backend = RolloutSamplerSpec.backend(rollout_config)
    sampler_world_size = RolloutSamplerSpec.world_size(rollout_config, backend)
    auxiliary_groups = [(TEACHER_GROUP, teacher_world_size)] if teacher_world_size else []
    from swift.dev.recipe.rl.resource import RLResourceBootstrap
    resource_plan = RLResourceBootstrap.build(
        model_config=model_config,
        distributed_config=distributed_config,
        rollout_config=rollout_config,
        template=assembly.template,
        prompt_rows=[dict(messages=messages, **(extras or {}))
                     for messages, extras in zip(prompts, prompt_extras)],
        auxiliary_groups=auxiliary_groups,
        max_completion_length=rl_config.max_completion_length)
    sampler_world_size = RolloutSamplerSpec.world_size(rollout_config, backend)
    if resource_plan is None:
        groups, sampler_remote_group, colocate = plan_rl_device_groups(
            distributed_config.nproc_per_node, rollout_config.sampler_mode, sampler_world_size,
            auxiliary_groups=auxiliary_groups)
    else:
        groups = RLResourceBootstrap.device_groups(
            resource_plan, adaptive=rollout_config.resource_mode == 'adaptive')
        sampler_remote_group, colocate = 'sampler', False
        logger.info('RL resource plan:\n%s', resource_plan.describe())
    initialize_twinkle_rl(
        distributed_config,
        groups,
        seed=train_config.seed,
        full_determinism=train_config.full_determinism,
        tf32=train_config.tf32,
        sampler_parallel_width=RolloutSamplerSpec.parallel_width(rollout_config, backend))

    assembly.build_model()
    configure_rl_loss(assembly.model, rl_config)
    # Prompts are rolled out, not iterated by a dataloader, so the step budget is derived from the prompt
    # set (B1): one generation batch per round, exhausted after num_train_epochs passes. OPSD replays each
    # round once (no num_iterations), so the budget uses num_iterations=1; --max_steps still overrides.
    if rl_config.lmbda < 1.0 and not dataset_rows:
        raise ValueError('OPSD with lmbda < 1 distils some rounds on the dataset\'s own completions, but no dataset '
                         'row carries an assistant completion. Add completions, or set lmbda=1 for purely on-policy '
                         'OPSD.')
    # Design-B mini-batch width (per_device_train_batch_size * dp_size); see run_grpo for why dp_size comes
    # off build_dp_mesh (online RL is Ray-only, pure data-parallel over nproc_per_node).
    train_batch_size = train_config.per_device_train_batch_size * build_dp_mesh(
        distributed_config).data_world_size
    from swift.dev.recipe.rl.prompts import resolve_rollout_max_steps, rollout_step_budget
    max_steps = resolve_rollout_max_steps(
        train_config.max_steps,
        rollout_step_budget(
            num_prompts=len(prompts),
            num_generations=rl_config.num_generations,
            train_batch_size=train_batch_size,
            gradient_accumulation_steps=assembly.ga,
            num_train_epochs=train_config.num_train_epochs,
            generation_batch_size=rollout_config.generation_batch_size),
        recipe='run_opsd')
    assembly.resolve_step_intervals(max_steps)
    configure_optimizer(
        assembly.model,
        train_config,
        num_training_steps=max_steps,
        distributed_config=distributed_config,
        tuner_config=assembly.tuner_config,
        plugin_config=assembly.plugin_config)

    sampler_engine_args = RolloutSamplerSpec.engine_args(rollout_config, engine_args, colocate, backend, tuner_config=assembly.tuner_config)
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
            assembly.model, sampler, assembly.template, colocate=False,
            sleep_level=rollout_config.sleep_level, handover=controller.handover)
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
        teacher=_build_opsd_teacher(assembly.model, assembly.template, rl_config, tuner_config, distributed_config),
        template=assembly.template,
        signal='logps',
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=assembly.ga)
    payload_processor = build_remote_batch_assembler(assembler)
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
        save_total_limit=checkpoint_config.save_total_limit)
    if assembly.resume_dir:
        saver.resume(assembly.resume_model())

    from twinkle.perf import Perf

    from swift.dev.recipe.common.loop_utils import filter_perf_actors

    if Perf.initialize(filter_perf_actors({'model': assembly.model, 'sampler': rollout.sampler})):
        logger.info('[perf] profiling enabled for the online OPSD loop.')
    handover = None
    try:
        if rl_config.lmbda == 1.0:
            # A purely on-policy run rides the per-sample stream at max_staleness=0 ('none',
            # drain-before-publish) or 1 ('one_step_off', the only overlapping regime distillation
            # supports; 'fully_async' is rejected by validate._check_async_mode). The streaming config
            # guard runs inside build_streaming_driver.
            # The prompt dataloader is a twinkle DataLoader over the prompt rows (messages + passthrough
            # extras); the driver flattens its batches, draws rows, and stamps its own ordinal as prompt_id
            # (the assemble seam reads sample.prompt).
            prompt_dataloader = build_prompt_dataloader(prompts, prompt_extras, seed=train_config.seed)
            driver, handover = build_streaming_driver(
                rollout_config,
                colocate=colocate,
                run_id='run_opsd',
                model=assembly.model,
                rollout=rollout,
                output_dir=output_dir,
                prompt_dataloader=prompt_dataloader,
                num_train_epochs=train_config.num_train_epochs,
                adapter_name=('default' if tuner_config is not None else None),
                train_batch_size=train_batch_size,
                gradient_accumulation_steps=assembly.ga,
                num_generations=rl_config.num_generations,
                sampling_params=distill_sampling_params(rl_config, generation_config),
                unit_size=1,
                score=None,
                assemble=assembler.assemble,
                assembly_ready=assembler.assembly_ready,
                payload_processor=payload_processor,
                global_step=lambda: recorder.global_step,
                reached_max=recorder.reached_max)
            _driver_cell.append(driver)
            if handover.serializes or getattr(handover, 'adaptive_shared', False):
                # Version-0 hand-over: put the pre-training policy in the sampler before the first admission.
                # Every later cycle's enter rides the in-place publish's sync_fn (see the builder).
                handover.enter()
            for batch in driver:
                _fold_idle_ratio()
                optimizer_steps = 0
                try:
                    for mini_batch_index in range(batch.mini_batch_count):
                        recorder.micro_step += 1
                        batch.forward_backward(mini_batch_index)
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
                finally:
                    batch.release()
                # commit releases the driver thread blocked in the iterator bridge and reports the real step
                # count -- the publish cadence unit (twinkle publishes only after consume returns).
                driver.commit(optimizer_steps)
                if recorder.reached_max():
                    break
        else:
            # The off-policy-capable regime: per-round on/off-policy coin flip over the prompt set's
            # generation batches (the retired FixedBatchDriver's cadence). BOTH round kinds train -- an
            # off-policy round distils the dataset completions; it is not a skipped round. round_index
            # counts rounds (NOT optimizer steps, W27) so the coin flip and the dataset window stay
            # aligned with the batch being processed.
            # The generation batches come from the SAME twinkle DataLoader bridge the streaming path uses,
            # sized to the round's prompt count (batch_size=generation_batch_size) and looped for
            # ceil(num_train_epochs) passes with set_epoch advancing the seeded reshuffle per pass -- the
            # retired PromptBatchScheduler's cadence, now delegated to the loader.
            gen_batch_size = min(rollout_config.generation_batch_size or len(prompts), len(prompts))
            prompt_dataloader = build_prompt_dataloader(
                prompts, prompt_extras, seed=train_config.seed, batch_size=gen_batch_size)
            round_index = 0
            for epoch in range(max(1, math.ceil(train_config.num_train_epochs))):
                prompt_dataloader.set_epoch(epoch)
                for batch in prompt_dataloader:
                    if recorder.reached_max():
                        break
                    use_student = random.Random(train_config.seed + round_index).random() <= rl_config.lmbda
                    round_index += 1
                    if use_student:
                        # blocking_generate syncs the trained policy into the sampler first, so an on-policy
                        # round samples from the current policy. Each row is one prompt's messages plus its
                        # passthrough extras; prompt_id is the batch-local index the fallback below reads.
                        batch_messages = [row['messages'] for row in batch]
                        batch_extras = [{k: v for k, v in row.items() if k != 'messages'} for row in batch]
                        samples = blocking_generate(
                            rollout,
                            batch_messages,
                            num_samples=rl_config.num_generations,
                            sampling_params=distill_sampling_params(rl_config, generation_config),
                            prompt_extras=batch_extras)
                        expected = len(batch) * rl_config.num_generations
                        if len(samples) != expected:
                            raise RuntimeError(f'rollout returned {len(samples)} samples, expected {expected} '
                                               f'({len(batch)} prompts * {rl_config.num_generations} generations).')
                        for local_idx in range(len(batch)):
                            for sample in samples[local_idx * rl_config.num_generations:(local_idx + 1)
                                                  * rl_config.num_generations]:
                                sample.prompt_id = str(local_idx)
                        rows = [
                            DistillRow(sample.input_feature,
                                       sample.messages if sample.messages is not None
                                       else batch_messages[int(sample.prompt_id)],
                                       sample.extra or {}) for sample in samples
                        ]
                    else:
                        # A rolling window over the dataset completions, so successive off-policy rounds cover
                        # new rows (run_opsd already rejected lmbda<1 without dataset completions).
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
                if recorder.reached_max():
                    break
    finally:
        if (handover is not None
                and (handover.serializes or getattr(handover, 'adaptive_shared', False))
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


def _require_teacher_prompt(prompt_extras: List[dict]) -> None:
    """Fail loudly up front if the dataset lacks the ``teacher_prompt`` column OPSD is defined by.

    Without a privileged teacher view, teacher and student would condition on the same prompt and the
    distillation signal would be identically zero -- that is plain GKD, not OPSD, so it is rejected here
    rather than silently training on a no-op loss. A row that individually lacks it is caught later by
    the assembler's privileged-feature build.
    """
    if not any((extra or {}).get('teacher_prompt') for extra in prompt_extras):
        raise ValueError('OPSD requires a per-sample `teacher_prompt` dataset column carrying the privileged '
                         'context (reference/rubric). None of the rows provide it. Add it via a dataset plugin '
                         '(see examples/rl/opsd), or use --rl_type gkd for plain distillation.')


def _build_opsd_teacher(model: TrainableModel, template: Any, rl_config: RLConfig, tuner_config: Optional[TunerConfig],
                        distributed_config: DistributedConfig) -> Any:
    """Resolve the OPSD teacher: the student's adapter-disabled base, a frozen Ray-actor model, or dynamic-self.

    All three are :class:`~swift.dev.recipe.rl.auxiliary.AuxiliaryModel` wrappers scoring with ``forward_only``.
    A ``PolicyAuxiliary`` with ``disable_adapter=False`` (no separate teacher configured, the OPSD default) is
    DYNAMIC self-distillation: it scores the student's CURRENT weights on the privileged view. There is no
    HTTP teacher server (removed, RL_PLAN §2.H): the ``teacher_model_server`` field is gone entirely, so there
    is nothing to re-check here.
    """
    if rl_config._teacher_use_disable_adapter:
        if tuner_config is None:
            raise ValueError('rl_config._teacher_use_disable_adapter=True requires a LoRA student (tuner_config), '
                             'since it distils from the adapter-disabled base of that same model.')
        return PolicyAuxiliary(model, disable_adapter=True)
    from swift.dev.recipe.common.assembly import single_teacher_id
    teacher_model = single_teacher_id(rl_config.teacher_model, algo='OPSD')
    if teacher_model is None:
        return PolicyAuxiliary(model)
    return build_frozen_teacher(
        rl_config, template, teacher_model, distributed_config=distributed_config, remote_group=TEACHER_GROUP,
        adapters=rl_config.teacher_adapters, role='teacher')

