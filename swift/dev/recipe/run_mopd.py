"""Multi-teacher on-policy distillation (MOPD) assembly: run_mopd orchestration.

Peer of ``run_opsd`` for MOPD (Multi-teacher On-Policy Distillation, arXiv:2606.30406 Open-MOPD). The
student generates one on-policy trajectory; K frozen DOMAIN teachers each score that same trajectory
token-by-token; their per-token distributions are fused into a single distillation target by weighted
probability mixture (``log sum_k w_k p_k``), and the student is pulled toward it with the same
sampled-token k3 surrogate OPSD uses (:class:`twinkle.loss.mopd.MOPDLoss`). Compared with single-teacher
OPSD/GKD, MOPD reduces the cross-domain degradation a student suffers when trained on one teacher's
distribution alone.

Relationship to OPSD
--------------------
MOPD reuses the OPSD/GKD (hence GRPO) skeleton: :class:`MOPDLoop` subclasses :class:`OPSDLoop` and
overrides only the teacher-scoring -- instead of ONE privileged-view teacher it loops K frozen teachers
over the student's OWN on-policy prompt+response (no privileged view; plan §3.3 stage-one has every
teacher share the student's prompt) and returns a list of per-teacher response-only ``teacher_logps``
channels plus ``teacher_weights``. ``MOPDLoss`` fuses them; with a single teacher it degenerates exactly
to OPSD.

Differences from OPSD that matter here:
  * teachers are K separate frozen Ray actors (``teacher_model`` is a LIST), each scoring the student's
    own response tokens directly -- no privileged ``teacher_prompt`` view (that is OPSD's single-teacher
    trick);
  * every teacher must share the student's tokenizer, since each re-scores the student's response ids;
  * all K teachers share ONE ``'teacher'`` DeviceGroup (plan §3.3 stage-one default: card-frugal, serial
    ``forward_only``); ``teacher_parallel_spec`` sizes that group;
  * teacher offloading is rejected (K resident teachers on their own group; offload is a GKD-only hook).
"""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from swift.dev.recipe._distill import (
    DistillRow,
    build_frozen_teacher,
    distill_rows_from_dataset,
    distill_sampling_params,
)
from swift.dev.recipe._teacher import MultiTeacher
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
from swift.dev.recipe.run_opsd import OPSDLoop

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


def run_mopd(
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
    """Assemble and run multi-teacher on-policy distillation with weight sync. Returns the loss history.

    Mirrors :func:`run_opsd`'s assembly, swapping in K frozen teachers (:func:`_build_mopd_teachers`) and
    :class:`MOPDLoop`. ``teacher_weights`` (optional) sets the mixture weights; unset means uniform.
    """
    from swift.dev.builders import build_ray_dp_mesh, build_sampler
    from swift.dev.loss import configure_rlhf_loss
    from swift.dev.optimizer import configure_optimizer, resolve_max_grad_norm
    from swift.dev.recipe.assembly import TrainAssembly

    if rlhf_config.rlhf_type != 'mopd':
        raise ValueError(f'run_mopd requires rlhf_type="mopd", got {rlhf_config.rlhf_type!r}.')
    if rlhf_config.offload_teacher_model:
        raise ValueError('offload_teacher_model is not supported for MOPD: it keeps K teachers resident on their own '
                         'DeviceGroup to score every trajectory, and offload is a GKD-only full-vocab hook. Leave it '
                         'off (multi-teacher offload is not a thing here).')
    assembly = TrainAssembly(
        'run_mopd',
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

    # MOPD always builds K separate frozen teachers, so it always plans the shared 'teacher' DeviceGroup
    # (plan §3.3); teacher_parallel_spec sizes it, else one card-frugal rank.
    teacher_world_size = teacher_group_world_size(rlhf_config)

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
    # set (B1): one generation batch per round, exhausted after num_train_epochs passes. MOPD replays each
    # round once (no num_iterations), so the budget uses num_iterations=1; --max_steps still overrides.
    prompts, prompt_extras, dataset_rows = distill_rows_from_dataset(dataset_config, assembly.template)
    if rlhf_config.lmbda < 1.0 and not dataset_rows:
        raise ValueError('MOPD with lmbda < 1 distils some rounds on the dataset\'s own completions, but no dataset '
                         'row carries an assistant completion. Add completions, or set lmbda=1 for purely on-policy '
                         'MOPD.')
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
        recipe='run_mopd')
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
    rollout = SyncableRollout(assembly.model, sampler, assembly.template, colocate=colocate)

    teacher = _build_mopd_teachers(assembly.model, rlhf_config, teacher_world_size, distributed_config)
    assembly.loop = MOPDLoop(
        assembly.model,
        rollout,
        prompts,
        teacher=teacher,
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
        manual_gc_steps=megatron_config.manual_gc_steps if megatron_config else 0)
    if assembly.resume_dir:
        assembly.loop.resume(assembly.resume_model())
    try:
        history = assembly.loop.fit()
        if _save_final:
            assembly.save_final()
        return history
    finally:
        rollout.shutdown()


def _resolve_teacher_weights(weights: Optional[List[float]], num_teachers: int) -> Optional[List[float]]:
    """Validate teacher_weights against the teacher count (None -> uniform, resolved inside MOPDLoss)."""
    if weights is None:
        return None
    if len(weights) != num_teachers:
        raise ValueError(f'teacher_weights has {len(weights)} entries but {num_teachers} teachers were built; they '
                         'must match, or leave teacher_weights unset for uniform weights.')
    return list(weights)


def _build_mopd_teachers(model: TrainableModel, rlhf_config: RLHFConfig, teacher_world_size: int,
                         distributed_config: DistributedConfig) -> MultiTeacher:
    """Build the K frozen domain teachers as Ray actors sharing the ``'teacher'`` DeviceGroup.

    Each teacher is a frozen twinkle model actor (plan §3.3) sharing the student's processor/template so it
    encodes and scores the student's response tokens identically. All K share ONE ``'teacher'`` group
    (stage-one default: card-frugal, serial ``forward_only``); ``teacher_parallel_spec`` sizes that group.
    Per-teacher adapters are not applied (each teacher is a distinct full model), and the single-teacher
    ``_teacher_use_disable_adapter`` self-distillation mode does not apply to K domain teachers. There is no
    HTTP teacher server (removed, RL_PLAN §2.H): the ``teacher_model_server`` field is gone entirely, so
    there is nothing to re-check here. The K teachers plus their mixture weights are returned as one
    :class:`~swift.dev.recipe._teacher.MultiTeacher`.
    """
    if rlhf_config._teacher_use_disable_adapter:
        raise ValueError('MOPD distils from K separate domain teachers; _teacher_use_disable_adapter (a single '
                         'adapter-disabled base) is an OPSD/GKD self-distillation mode and does not apply here.')
    if rlhf_config.teacher_adapters:
        raise ValueError('MOPD builds each teacher as a distinct full model and does not apply per-teacher adapters; '
                         'teacher_adapters would be ambiguous (which teacher?) and is ignored. Remove it, or use OPSD '
                         'for a single adapter-based teacher.')
    teacher_models = rlhf_config.teacher_model
    if not teacher_models:
        raise ValueError('MOPD needs at least one teacher: set RLHFConfig.teacher_model to a list of model ids '
                         '(one per domain expert).')
    template = model.template if hasattr(model, 'template') else None
    teachers = [
        build_frozen_teacher(
            rlhf_config, template, model_id, distributed_config=distributed_config, remote_group=_TEACHER_GROUP,
            teacher_world_size=teacher_world_size, role='teacher') for model_id in teacher_models
    ]
    return MultiTeacher(teachers, _resolve_teacher_weights(rlhf_config.teacher_weights, len(teachers)))


class MOPDLoop(OPSDLoop):
    """Multi-teacher on-policy distillation loop.

    Reuses :class:`OPSDLoop`'s rollout/mini-batch/step skeleton and its ``_forward_kwargs`` (which passes
    the teacher signal straight through, dropping GKD's top-k/SFT knobs). Only ``_teacher_kwargs`` differs:
    it loops the K frozen teachers over the student's OWN on-policy features (no privileged view), producing
    one response-only ``teacher_logps`` channel per teacher via the inherited chunked ``_response_logps`` (a
    ``forward_only`` correct at dp_size>1), and forwards ``teacher_weights`` so :class:`MOPDLoss` fuses them
    by weighted mixture. With a single teacher the fused target is that teacher's log-probs, i.e. exactly
    OPSD.
    """

    def _teacher_kwargs(self, rows: List[DistillRow], use_student: bool) -> Dict[str, Any]:
        """K response-only teacher-logp channels over the student's own features, plus the mixture weights.

        ``self.teacher`` is a :class:`~swift.dev.recipe._teacher.MultiTeacher`: its ``teachers`` are each
        scored into one channel and its ``weights`` are the mixture weights ``MOPDLoss`` fuses by.
        """
        features = [row.feature for row in rows]
        channels = [self._response_logps(teacher, features) for teacher in self.teacher.teachers]
        return {'teacher_logps': channels, 'teacher_weights': self.teacher.weights}
