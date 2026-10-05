"""On-policy self-distillation (OPSD) assembly: run_opsd orchestration.

Peer of ``run_gkd`` for OPSD (On-Policy Self-Distillation, Zhao et al. arXiv:2601.18734). A single
model acts as BOTH teacher and student, differing only in CONTEXT: the student conditions on the
question only, the teacher conditions on a PRIVILEGED view (question + reference/rubric, supplied
per-sample by the dataset's ``teacher_prompt`` column). The student generates on-policy, the teacher
re-scores those SAME response tokens under the privileged prompt, and the student is pulled toward
the teacher's per-token distribution with a dense, logits-free sampled-token k3 surrogate
(:class:`twinkle.loss.opsd.OPSDLoss`).

Why this reuses the GKD skeleton
---------------------------------
OPSD's loop is GKD's, which is GRPO's: a weight-synced sampler rollout over Ray -> a frozen teacher
``forward_only`` -> student ``forward_backward`` with the teacher signal, over design-B mini-batches on
Ray data parallelism. Only the teacher SIGNAL differs -- OPSD passes response-only ``teacher_logps`` from
a privileged-prompt view instead of GKD's full-vocab ``teacher_logits`` on the shared prompt -- so
:class:`OPSDLoop` subclasses :class:`GKDLoop` and overrides just the teacher-scoring and the
``forward_backward`` kwargs. The teacher forward reuses :meth:`GRPOLoop._response_logps` (chunked
``forward_only``, correct at dp_size>1); a separate teacher is a frozen Ray actor on its own DeviceGroup
(plan §3.3), never an HTTP server (no-server premise, RL_PLAN §2.H).

Teacher selection (``_build_opsd_teacher``, all :class:`~swift.dev.recipe._teacher.Teacher` wrappers):
  * LoRA student with ``_teacher_use_disable_adapter`` -> ``DisableAdapterTeacher``: the teacher is the
    student's own frozen base on the privileged view (scored on the student's actor, no extra group);
  * ``teacher_model`` set -> ``FrozenModelTeacher``: a separate frozen Ray-actor model on the ``'teacher'``
    DeviceGroup;
  * neither -> ``DynamicSelfTeacher``: DYNAMIC self-distillation, the student's CURRENT weights on the
    privileged view.
"""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from swift.dev.recipe._distill import (
    DistillRow,
    build_frozen_teacher,
    build_privileged_teacher_feature,
    distill_rows_from_dataset,
    distill_sampling_params,
    response_ids_from_feature,
)
from swift.dev.recipe._teacher import DisableAdapterTeacher, DynamicSelfTeacher
from swift.dev.recipe.run_gkd import GKDLoop
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
        RLHFConfig,
        RolloutConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )
    from swift.dev.model import TrainableModel

logger = logging.getLogger(__name__)


def run_opsd(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    dataset_config: DatasetConfig,
    train_config: TrainConfig,
    distributed_config: DistributedConfig,
    checkpoint_config: CheckpointConfig,
    rollout_config: RolloutConfig,
    rlhf_config: RLHFConfig,
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
    design-B mini-batch width), swapping the GKD loop for :class:`OPSDLoop` and the GKD teacher for the
    OPSD teacher (``_build_opsd_teacher``). ``lmbda`` keeps its GKD meaning (per-round probability of
    on-policy generation vs. the dataset's own completion); ``sft_alpha``/``gkd_logits_topk`` are GKD-loss
    knobs that OPSDLoss does not read, so they are not forwarded (see :meth:`OPSDLoop._forward_kwargs`).
    """
    from swift.dev.builders import build_ray_dp_mesh, build_sampler
    from swift.dev.loss import configure_rlhf_loss
    from swift.dev.optimizer import configure_optimizer, resolve_max_grad_norm
    from swift.dev.recipe.assembly import TrainAssembly, single_teacher_id

    if rlhf_config.rlhf_type != 'opsd':
        raise ValueError(f'run_opsd requires rlhf_type="opsd", got {rlhf_config.rlhf_type!r}.')
    if rlhf_config.offload_teacher_model:
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
        rlhf_config=rlhf_config,
        output_dir=output_dir,
        logging_config=logging_config,
        quantize_config=quantize_config,
        megatron_config=megatron_config,
        moe_config=moe_config)
    assembly.prepare()

    # A separate frozen teacher is a Ray actor on its own DeviceGroup (plan §3.3), which
    # plan_rl_device_groups must allocate BEFORE twinkle.initialize; the adapter-disabled teacher and
    # dynamic-self (teacher_model unset) both reuse the student's own actor and plan no group.
    separate_teacher = (not rlhf_config._teacher_use_disable_adapter
                        and single_teacher_id(rlhf_config.teacher_model, algo='OPSD') is not None)
    teacher_world_size = teacher_group_world_size(rlhf_config) if separate_teacher else 0

    backend = _sampler_backend(rollout_config)
    sampler_world_size = _sampler_world_size(rollout_config, backend)
    groups, sampler_remote_group, colocate = plan_rl_device_groups(distributed_config.nproc_per_node,
                                                                   rollout_config.vllm_mode, sampler_world_size,
                                                                   teacher_world_size)
    _initialize_twinkle_rl(
        distributed_config, groups, seed=train_config.seed, full_determinism=train_config.full_determinism)

    assembly.build_template()
    assembly.build_model()
    configure_rlhf_loss(assembly.model, rlhf_config)
    # Prompts are rolled out, not iterated by a dataloader, so the step budget is derived from the prompt
    # set (B1): one generation batch per round, exhausted after num_train_epochs passes. OPSD replays each
    # round once (no num_iterations), so the budget uses num_iterations=1; --max_steps still overrides.
    prompts, prompt_extras, dataset_rows = distill_rows_from_dataset(dataset_config, assembly.template)
    _require_teacher_prompt(prompt_extras)
    if rlhf_config.lmbda < 1.0 and not dataset_rows:
        raise ValueError('OPSD with lmbda < 1 distils some rounds on the dataset\'s own completions, but no dataset '
                         'row carries an assistant completion. Add completions, or set lmbda=1 for purely on-policy '
                         'OPSD.')
    # Design-B mini-batch width (per_device_train_batch_size * dp_size); see run_grpo for why dp_size comes
    # off build_ray_dp_mesh (online RL is Ray-only, pure data-parallel over nproc_per_node).
    train_batch_size = train_config.per_device_train_batch_size * build_ray_dp_mesh(
        distributed_config).data_world_size
    from swift.dev.recipe.train_loop import resolve_rollout_max_steps, rollout_step_budget
    max_steps = resolve_rollout_max_steps(
        train_config.max_steps,
        rollout_step_budget(
            num_prompts=len(prompts),
            num_generations=rlhf_config.num_generations,
            train_batch_size=train_batch_size,
            gradient_accumulation_steps=assembly.ga,
            num_train_epochs=train_config.num_train_epochs,
            generation_batch_size=rollout_config.generation_batch_size),
        recipe='run_opsd')
    assembly.resolve_step_intervals(max_steps)
    configure_optimizer(
        assembly.model, train_config, num_training_steps=max_steps, distributed_config=distributed_config)

    sampler_engine_args = _sampler_engine_args(rollout_config, engine_args, colocate, backend)
    sampler = build_sampler(
        model_config,
        backend=backend,
        engine_args=sampler_engine_args,
        template=assembly.template,
        remote_group=sampler_remote_group)
    rollout = SyncableRollout(
        assembly.model, sampler, assembly.template, colocate=colocate, sleep_level=rollout_config.sleep_level)

    # async_mode selects the driver (see run_gkd for the full rationale): the per-sample stream now serves
    # EVERY regime, so OPSD rides it whenever the round is purely on-policy -- 'none' runs StreamingOPSDLoop
    # at max_staleness=0 (drain-before-publish, no overlap) and 'one_step_off' (the only overlapping regime
    # distillation supports) pins staleness to 1. An off-policy round (lmbda != 1.0, enforced by validate)
    # generates nothing to admit ahead, so it stays on OPSDLoop's synchronous _run_sync. A colocated sampler
    # sets serialize_generation for the exclusive-device hand-over.
    loop_cls = OPSDLoop
    streaming_kwargs: Dict[str, Any] = {}
    sync_only = rlhf_config.lmbda != 1.0
    if rollout_config.async_mode != 'none' or not sync_only:
        from swift.dev.recipe.opsd_async import StreamingOPSDLoop
        loop_cls = StreamingOPSDLoop
        streaming_kwargs = {
            'max_staleness': 0 if rollout_config.async_mode == 'none' else 1,
            'weight_sync_strategy': rollout_config.weight_sync_strategy,
            'allow_partial_rollout': rollout_config.allow_partial_rollout,
            'parameter_sync_step': rollout_config.parameter_sync_step,
            'adapter_name': ('default' if tuner_config is not None else None),
            'serialize_generation': colocate,
        }
    assembly.loop = loop_cls(
        assembly.model,
        rollout,
        prompts,
        teacher=_build_opsd_teacher(assembly.model, rlhf_config, tuner_config, teacher_world_size, distributed_config),
        template=assembly.template,
        prompt_extras=prompt_extras,
        dataset_rows=dataset_rows,
        lmbda=rlhf_config.lmbda,
        num_generations=rlhf_config.num_generations,
        seed=train_config.seed,
        rlhf_config=rlhf_config,
        max_steps=max_steps,
        train_batch_size=train_batch_size,
        generation_batch_size=rollout_config.generation_batch_size,
        num_train_epochs=train_config.num_train_epochs,
        gradient_accumulation_steps=assembly.ga,
        max_grad_norm=resolve_max_grad_norm(train_config),
        sampling_params=distill_sampling_params(rlhf_config, generation_config),
        logging_config=logging_config,
        output_dir=output_dir,
        save_steps=checkpoint_config.save_steps,
        no_save_optim=checkpoint_config.no_save_optim or checkpoint_config.save_only_model,
        no_save_rng=checkpoint_config.no_save_rng or checkpoint_config.save_only_model,
        safe_serialization=checkpoint_config.safe_serialization,
        max_shard_size=checkpoint_config.max_shard_size,
        save_total_limit=checkpoint_config.save_total_limit,
        manual_gc=bool(megatron_config and megatron_config.manual_gc),
        manual_gc_steps=megatron_config.manual_gc_steps if megatron_config else 0,
        **streaming_kwargs)
    if assembly.resume_dir:
        assembly.loop.resume(assembly.resume_model())
    try:
        history = assembly.loop.fit()
        if _save_final:
            assembly.save_final()
        return history
    finally:
        rollout.shutdown()


def _require_teacher_prompt(prompt_extras: List[dict]) -> None:
    """Fail loudly up front if the dataset lacks the ``teacher_prompt`` column OPSD is defined by.

    Without a privileged teacher view, teacher and student would condition on the same prompt and the
    distillation signal would be identically zero -- that is plain GKD, not OPSD, so it is rejected here
    rather than silently training on a no-op loss. A row that individually lacks it is caught later by
    :meth:`OPSDLoop._privileged_feature`.
    """
    if not any((extra or {}).get('teacher_prompt') for extra in prompt_extras):
        raise ValueError('OPSD requires a per-sample `teacher_prompt` dataset column carrying the privileged '
                         'context (reference/rubric). None of the rows provide it. Add it via a dataset plugin '
                         '(see examples/v5/rl/opsd), or use --rlhf_type gkd for plain distillation.')


def _build_opsd_teacher(model: TrainableModel, rlhf_config: RLHFConfig, tuner_config: Optional[TunerConfig],
                        teacher_world_size: int, distributed_config: DistributedConfig) -> Any:
    """Resolve the OPSD teacher: the student's adapter-disabled base, a frozen Ray-actor model, or dynamic-self.

    All three are :class:`~swift.dev.recipe._teacher.Teacher` wrappers scoring with ``forward_only``. A
    ``DynamicSelfTeacher`` (no separate teacher configured, the OPSD default) scores the student's CURRENT
    weights on the privileged view. There is no HTTP teacher server (removed, RL_PLAN §2.H): the
    ``teacher_model_server`` field is gone entirely, so there is nothing to re-check here.
    """
    if rlhf_config._teacher_use_disable_adapter:
        if tuner_config is None:
            raise ValueError('rlhf_config._teacher_use_disable_adapter=True requires a LoRA student (tuner_config), '
                             'since it distils from the adapter-disabled base of that same model.')
        return DisableAdapterTeacher(model)
    from swift.dev.recipe.assembly import single_teacher_id
    teacher_model = single_teacher_id(rlhf_config.teacher_model, algo='OPSD')
    if teacher_model is None:
        return DynamicSelfTeacher(model)
    template = model.template if hasattr(model, 'template') else None
    return build_frozen_teacher(
        rlhf_config, template, teacher_model, distributed_config=distributed_config, remote_group=_TEACHER_GROUP,
        teacher_world_size=teacher_world_size, adapters=rlhf_config.teacher_adapters, role='teacher')


class OPSDLoop(GKDLoop):
    """On-policy self-distillation loop: the student generates, a privileged teacher re-scores, distil.

    Subclasses :class:`GKDLoop` (hence :class:`GRPOLoop`) and reuses its weight-synced rollout, design-B
    mini-batch planning, per-step cadence and ``lmbda`` on/off-policy mixing verbatim; only the teacher
    signal changes. ``_teacher_kwargs`` builds each row's PRIVILEGED teacher view (``teacher_prompt`` in the
    user turn + the student's shared response tokens) and scores it with the inherited chunked
    ``_response_logps`` (a ``forward_only`` correct at dp_size>1), returning RESPONSE-ONLY ``teacher_logps``
    -- the form :class:`OPSDLoss` requires (teacher and student prompts differ in length, so GKD's
    full-sequence ``teacher_logits`` must not be used). ``_forward_kwargs`` drops GKD's
    ``topk``/``apply_sft_loss``, which the sampled-token OPSD loss does not read.
    """

    def _privileged_feature(self, row: DistillRow) -> dict:
        """Encode this row's privileged teacher view (``teacher_prompt`` + the shared response tokens)."""
        teacher_prompt = (row.extra or {}).get('teacher_prompt')
        if not teacher_prompt:
            raise ValueError('OPSD requires a per-sample `teacher_prompt` carrying the privileged context; '
                             'a row in this batch has none.')
        response_ids = response_ids_from_feature(row.feature)
        return build_privileged_teacher_feature(self.template, row.messages, teacher_prompt, response_ids, row.extra)

    def _teacher_kwargs(self, rows: List[DistillRow], use_student: bool) -> Dict[str, Any]:
        """Response-only teacher log-probs, one list per row, under each row's privileged view.

        ``self.teacher`` is always a :class:`~swift.dev.recipe._teacher.Teacher` (never ``None``: the
        no-separate-teacher case is a ``DynamicSelfTeacher``), so scoring is a single uniform call.
        """
        return {'teacher_logps': self._response_logps(self.teacher, [self._privileged_feature(row) for row in rows])}

    def _forward_kwargs(self, teacher_kwargs: Dict[str, Any], use_student: bool) -> Dict[str, Any]:
        """OPSDLoss reads only ``teacher_logps`` -- GKD's ``topk``/``apply_sft_loss`` do not apply."""
        return dict(teacher_kwargs)
