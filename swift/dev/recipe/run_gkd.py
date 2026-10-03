"""On-policy GKD assembly: run_gkd orchestration (generalized knowledge distillation).

Peer of ``run_grpo`` / ``run_rft`` for GKD. GKD trains a student to match a frozen teacher on sequences
the STUDENT itself generates, so it learns to correct its own mistakes rather than only imitating a fixed
corpus. This recipe is built on the SAME online-RL primitives as GRPO/RFT -- a weight-syncing vLLM
sampler rollout over Ray, a data-parallel trainer, and design-B mini-batches -- because on-policy
generation must be backend-general (a separate sampler + weight sync works for both transformers and
megatron; in-process ``model.generate`` is transformers-only and megatron raises). The teacher is a
frozen twinkle model actor (or the LoRA student's own adapter-disabled base) scored with ``forward_only``,
never an HTTP server (no-server / all-Ray premise, RL_PLAN §2.H).

Per round (one rollout, split into ``train_batch_size`` mini-batches):
  1. with probability ``lmbda`` roll out on-policy completions from the student's live weights (pushed
     into the sampler first, so the behaviour policy tracks the trained one); otherwise take the
     dataset's own completions for this round (GKD's on/off-policy mixing);
  2. score each mini-batch with the teacher ``forward_only(return_logits=True)`` -> full-vocab
     ``teacher_logits``, lazily per mini-batch so peak memory is one mini-batch, not the whole round;
  3. student ``forward_backward(teacher_logits=...)`` -> GKDLoss (β-JSD, optional top-k); ``sft_alpha>0``
     adds supervised CE on the dataset-sourced rounds.

Placement, backend and Ray requirements are therefore identical to ``run_grpo`` (Ray-only, vLLM sampler,
heterogeneous or colocate device groups).
"""
from __future__ import annotations
import copy
import logging
import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from swift.dev.recipe._distill import (
    DistillRow,
    build_frozen_teacher,
    distill_rows_from_dataset,
    distill_sampling_params,
)
from swift.dev.recipe._teacher import DisableAdapterTeacher
from swift.dev.recipe.grpo import GRPOLoop
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


def run_gkd(
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
    """Assemble and run on-policy GKD with weight sync. Returns the loss history.

    Mirrors :func:`run_rft`'s assembly (train/sampler device groups, weight-syncable ``SyncableRollout``,
    design-B mini-batch width) but sets the GKD loss (``configure_rlhf_loss``) and drives
    :class:`GKDLoop` (teacher-scored distillation) instead of a policy-gradient or SFT step. The student
    is the trained model; the teacher is a frozen separate model (``rlhf_config.teacher_model``) or, for a
    LoRA student whose teacher is its own base, the adapter-disabled student
    (``rlhf_config._teacher_use_disable_adapter``).
    """
    from swift.dev.builders import build_ray_dp_mesh, build_sampler
    from swift.dev.loss import configure_rlhf_loss
    from swift.dev.optimizer import configure_optimizer, resolve_max_grad_norm
    from swift.dev.recipe.assembly import TrainAssembly, single_teacher_id

    if rlhf_config.rlhf_type != 'gkd':
        raise ValueError(f'run_gkd requires rlhf_type="gkd", got {rlhf_config.rlhf_type!r}.')
    assembly = TrainAssembly(
        'run_gkd',
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
    # plan_rl_device_groups must allocate BEFORE twinkle.initialize; the adapter-disabled teacher
    # (DisableAdapterTeacher) reuses the student's own actor and needs no group (teacher_world_size=0).
    separate_teacher = (not rlhf_config._teacher_use_disable_adapter
                        and single_teacher_id(rlhf_config.teacher_model, algo='GKD') is not None)
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
    # set (B1): one generation batch per round, exhausted after num_train_epochs passes. Load the prompts +
    # design-B mini-batch width first, then size max_steps (the LR horizon); --max_steps still overrides.
    prompts, prompt_extras, dataset_rows = distill_rows_from_dataset(dataset_config, assembly.template)
    if rlhf_config.lmbda < 1.0 and not dataset_rows:
        raise ValueError('GKD with lmbda < 1 distils some rounds on the dataset\'s own completions, but no dataset '
                         'row carries an assistant completion. Add completions, or set lmbda=1 for purely on-policy '
                         'GKD.')
    # Design-B mini-batch width (per_device_train_batch_size * dp_size); see run_grpo for why dp_size comes
    # off build_ray_dp_mesh (online RL is Ray-only, pure data-parallel over nproc_per_node).
    train_batch_size = train_config.per_device_train_batch_size * build_ray_dp_mesh(
        distributed_config).data_world_size
    # GKD replays each round once (no num_iterations), so the budget uses num_iterations=1. Under lmbda<1 the
    # off-policy rounds size their mini-batches from dataset_rows instead of the rollout, so the LR horizon is
    # approximate there; the scheduler's exhaustion still bounds the actual round count.
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
        recipe='run_gkd')
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

    assembly.loop = GKDLoop(
        assembly.model,
        rollout,
        prompts,
        teacher=_build_teacher(assembly.model, rlhf_config, tuner_config, teacher_world_size, distributed_config),
        template=assembly.template,
        prompt_extras=prompt_extras,
        dataset_rows=dataset_rows,
        lmbda=rlhf_config.lmbda,
        sft_alpha=rlhf_config.sft_alpha,
        gkd_logits_topk=rlhf_config.gkd_logits_topk,
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


def _build_teacher(model: TrainableModel, rlhf_config: RLHFConfig, tuner_config: Optional[TunerConfig],
                   teacher_world_size: int, distributed_config: DistributedConfig) -> Any:
    """Resolve the GKD teacher: the LoRA student's own adapter-disabled base, or a frozen Ray-actor model.

    There is no HTTP teacher server (removed, RL_PLAN §2.H): the teacher is either a
    :class:`DisableAdapterTeacher` -- the paper's headline self-distillation setting, scoring with
    ``forward_only(disable_lora=True)`` on the student's own actor (no separate group) -- or a
    :class:`FrozenModelTeacher` wrapping a separate frozen twinkle model actor built from ``teacher_model``
    on its own ``'teacher'`` DeviceGroup (``teacher_world_size`` ranks, planned by ``plan_rl_device_groups``).
    ``offload_teacher_model`` is applied inside the frozen teacher (its full-vocab logits are the memory
    hook), so it is rejected for the adapter-disabled base, which has no separate weights to offload.
    """
    if rlhf_config._teacher_use_disable_adapter:
        if tuner_config is None:
            raise ValueError('rlhf_config._teacher_use_disable_adapter=True requires a LoRA student (tuner_config), '
                             'since it distils from the adapter-disabled base of that same model.')
        if rlhf_config.offload_teacher_model:
            raise ValueError('offload_teacher_model requires a distinct frozen teacher model, not the '
                             'adapter-disabled base of the student (which has no separate weights to offload).')
        return DisableAdapterTeacher(model)
    from swift.dev.recipe.assembly import single_teacher_id
    teacher_model = single_teacher_id(rlhf_config.teacher_model, algo='GKD')
    if teacher_model is None:
        raise ValueError('GKD needs a teacher: set RLHFConfig.teacher_model, or use a LoRA student with '
                         '_teacher_use_disable_adapter=True to distil from its own frozen base.')
    template = model.template if hasattr(model, 'template') else None
    return build_frozen_teacher(
        rlhf_config, template, teacher_model, distributed_config=distributed_config, remote_group=_TEACHER_GROUP,
        teacher_world_size=teacher_world_size, adapters=rlhf_config.teacher_adapters, role='teacher',
        offload=rlhf_config.offload_teacher_model)


class GKDLoop(GRPOLoop):
    """On-policy GKD loop: student rolls out, a frozen teacher scores, the student distils toward it.

    Reuses :class:`GRPOLoop`'s weight-synced rollout (``_generate``), design-B mini-batch planning
    (``_plan_mini_batches``) and per-step cadence; overrides ``fit`` to replace the advantage/policy-gradient
    step with teacher-scored distillation. Each round is on-policy (a rollout) or off-policy (dataset
    completions) per ``lmbda``, and both reduce to a list of :class:`DistillRow` so the mini-batch split and
    teacher scoring never branch on the source. The teacher signal is full-vocab ``teacher_logits``, scored
    lazily one mini-batch at a time.
    """

    def __init__(
        self,
        model: TrainableModel,
        rollout_engine: Any,
        prompts: List[List[dict]],
        *,
        teacher: Any,
        template: Any = None,
        prompt_extras: Optional[List[dict]] = None,
        dataset_rows: Optional[List[DistillRow]] = None,
        lmbda: float = 0.5,
        sft_alpha: float = 0.0,
        gkd_logits_topk: Optional[int] = None,
        num_generations: int = 1,
        seed: int = 42,
        rlhf_config: Optional['RLHFConfig'] = None,
        max_steps: int = 1,
        train_batch_size: int = 1,
        generation_batch_size: Optional[int] = None,
        num_train_epochs: float = 1.0,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        sampling_params: Optional[dict] = None,
        logging_config: Optional['LoggingConfig'] = None,
        output_dir: str = 'output',
        save_steps: Optional[int] = None,
        no_save_optim: bool = False,
        no_save_rng: bool = False,
        safe_serialization: bool = True,
        max_shard_size: str = '5GB',
        save_total_limit: Optional[int] = None,
        manual_gc: bool = False,
        manual_gc_steps: int = 0,
    ):
        # Resolve the dataset rows BEFORE super().__init__, so a misconfiguration still happens before the
        # tracker initialises its reporters (a RunTracker side effect). The teacher offload now lives in
        # FrozenModelTeacher, built at recipe time (also before the tracker).
        if not 0.0 <= lmbda <= 1.0:
            raise ValueError(f'GKD lmbda must be in [0, 1], got {lmbda}.')
        resolved_dataset_rows = list(dataset_rows or [])

        super().__init__(
            model,
            rollout_engine,
            prompts,
            prompt_extras=prompt_extras,
            teacher=teacher,
            template=template,
            num_generations=num_generations,
            rlhf_config=rlhf_config,
            max_steps=max_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            train_batch_size=train_batch_size,
            generation_batch_size=generation_batch_size,
            num_train_epochs=num_train_epochs,
            seed=seed,
            max_grad_norm=max_grad_norm,
            sampling_params=sampling_params,
            logging_config=logging_config,
            output_dir=output_dir,
            save_steps=save_steps,
            no_save_optim=no_save_optim,
            no_save_rng=no_save_rng,
            safe_serialization=safe_serialization,
            max_shard_size=max_shard_size,
            save_total_limit=save_total_limit,
            manual_gc=manual_gc,
            manual_gc_steps=manual_gc_steps)
        self.dataset_rows = resolved_dataset_rows
        self.lmbda = lmbda
        self.sft_alpha = sft_alpha
        self.gkd_logits_topk = gkd_logits_topk
        self.seed = seed

    def _uses_student_generation(self, round_index: int) -> bool:
        """Per-round on/off-policy coin flip: with probability ``lmbda`` this round rolls out on-policy."""
        return random.Random(self.seed + round_index).random() <= self.lmbda

    def _round_rows(self, round_index: int, use_student: bool, prompt_indices: Sequence[int]) -> List[DistillRow]:
        """This round's training rows, uniform across the on-policy and off-policy sources.

        An on-policy round rolls out the scheduler's generation batch (``prompt_indices``); an off-policy
        round ignores it and takes the dataset's own completions window instead (GKD's on/off mixing).
        """
        if use_student:
            samples = self._generate(prompt_indices)
            return [
                DistillRow(sample.input_feature,
                           sample.messages if sample.messages is not None else self.prompts[int(sample.prompt_id)],
                           sample.extra or {}) for sample in samples
            ]
        return self._dataset_round_rows(round_index)

    def _dataset_round_rows(self, round_index: int) -> List[DistillRow]:
        """A rolling window over the dataset completions, so successive off-policy rounds cover new rows."""
        if not self.dataset_rows:
            raise RuntimeError('an off-policy GKD round needs dataset completions, but the dataset provided none; '
                               'set lmbda=1 for purely on-policy GKD, or add assistant completions to the dataset.')
        count = len(self.dataset_rows)
        offset = (round_index * self.train_batch_size) % count
        return [self.dataset_rows[(offset + i) % count] for i in range(count)]

    def _teacher_logits(self, features: List[dict]) -> Any:
        """Full-vocab teacher logits for ONE mini-batch, scored lazily to bound peak memory.

        The signal is materialised per mini-batch (``train_batch_size`` rows), never for a whole round: a
        full-vocab ``[rows, seq, vocab]`` tensor over an entire rollout would not fit. ``forward_only`` is a
        slice_dp method, so the mini-batch (>= dp_size rows) feeds every DP rank; the collected logits are
        then handed to ``forward_backward``, which slice_dp-splits them along dim0 to match each rank's
        ``inputs`` shard.
        """
        inputs = [copy.deepcopy(feature) for feature in features]
        outputs = self.teacher.forward_only(inputs=inputs, return_logits=True)
        return outputs['logits']

    def _teacher_kwargs(self, rows: List[DistillRow], use_student: bool) -> Dict[str, Any]:
        """The teacher signal for one mini-batch. GKD scores the SHARED prompt+response (full-vocab logits).

        A hook so the OPSD/MOPD subclasses -- which reuse this loop's rollout/mini-batch/step skeleton but a
        different teacher view and signal (response-only ``teacher_logps``) -- can override just this and
        :meth:`_forward_kwargs`.
        """
        return {'teacher_logits': self._teacher_logits([row.feature for row in rows])}

    def _forward_kwargs(self, teacher_kwargs: Dict[str, Any], use_student: bool) -> Dict[str, Any]:
        """The extra ``forward_backward`` kwargs beyond ``inputs``/GA: teacher signal + GKD loss knobs.

        GKD's JSD loss scores top-k teacher logits when ``gkd_logits_topk`` is set and adds supervised CE on
        the dataset-sourced rounds when ``sft_alpha > 0``.
        """
        return {
            **teacher_kwargs,
            'topk': self.gkd_logits_topk,
            'apply_sft_loss': bool(self.sft_alpha > 0 and not use_student),
        }

    def fit(self) -> list:
        """Train over the prompt set for ``num_train_epochs`` passes of teacher-scored distillation.

        Design B (see :meth:`GRPOLoop.fit`): the prompt-set scheduler yields one generation batch per round
        and is exhausted after ``num_train_epochs`` passes (B1); an explicit ``max_steps`` still caps the
        optimizer-step count early. Each round's rows are split into ``train_batch_size``-row mini-batches,
        one ``forward_backward`` runs per mini-batch and ``gradient_accumulation_steps`` mini-batches make
        one optimizer step. A round is on-policy (a weight-synced rollout of the batch) or off-policy
        (dataset completions) per ``lmbda``.
        """
        from swift.dev.recipe.train_loop import finish_manual_gc, start_manual_gc

        gc_was_enabled = start_manual_gc(self.manual_gc)
        try:
            # round_index is the scheduler's emitted-batch position, NOT global_step: it seeds the per-round
            # lmbda coin flip and offsets the off-policy dataset window, so it must stay aligned with the
            # prompt batch actually being processed. Deriving it from enumerate keeps the two in lockstep
            # within a run and across a resume (the scheduler restarts from batch 0, so the pairing restarts
            # with it); seeding it from global_step misaligns them whenever GA>1 or a round spans several
            # mini-batches, because global_step counts optimizer steps, not rounds (W27).
            for round_index, prompt_indices in enumerate(self._prompt_batches):
                if self._reached_max():
                    break
                use_student = self._uses_student_generation(round_index)
                rows = self._round_rows(round_index, use_student, prompt_indices)
                for mini_batch in self._plan_mini_batches(rows):
                    if self._reached_max():
                        break
                    teacher_kwargs = self._teacher_kwargs(mini_batch, use_student)
                    self._run_micro_step({
                        'inputs': [copy.deepcopy(row.feature) for row in mini_batch],
                        'gradient_accumulation_steps': self.gradient_accumulation_steps,
                        **self._forward_kwargs(teacher_kwargs, use_student),
                    })
            return self.history
        finally:
            finish_manual_gc(gc_was_enabled)
            self.tracker.close()

    # GKD has no reference model and no RL-loss channels, so it resets the two GRPOLoop hooks that would
    # otherwise inject a reference sync and policy-gradient metrics. The rest of the per-step cadence
    # (counter, GC tick, log-on-tracker, periodic save) is TrainLoop's.
    def _pre_metric_step(self) -> None:
        """No reference model to sync on a distillation step (overrides GRPOLoop's reference sync)."""

    def _extra_step_metrics(self, metrics: dict) -> dict:
        """GKD logs grad_norm beside the distillation loss; it has no entropy/rollout-ratio channels."""
        extra: dict = {}
        if metrics.get('grad_norm') is not None:
            extra['grad_norm'] = float(metrics['grad_norm'])
        return extra
