"""Rejection-sampling fine-tuning (RFT / RAFT / ReST) assembly: run_rft orchestration.

Peer of ``run_grpo`` for RFT. Where GRPO turns rewards into group-relative advantages and takes a
policy-gradient step, RFT uses the SAME rollout+reward machinery for a simpler purpose: sample N
completions per prompt from the current policy, score them, KEEP only the good ones (best-of-N, a reward
threshold, or top-K), and run plain cross-entropy SFT on the kept set. Repeating this over
``rft_iterations`` rounds -- each round re-sampling from the just-improved policy -- is the
rejection-sampling loop (RAFT/ReST): the policy bootstraps toward its own high-reward outputs with no
advantage estimator, no reference model, and no importance ratio.

What it reuses
--------------
Rollout, weight sync and reward are exactly GRPO's: :class:`RFTLoop` subclasses :class:`GRPOLoop` to
inherit ``_generate`` (which pushes the trained policy into the sampler before each rollout, so the
behaviour policy tracks the trained one), ``_score`` and ``_weighted_rewards``; it overrides ``fit`` to
replace the advantage/policy-gradient step with select-then-SFT. Device placement, sampler construction,
the dataset prompt loader and reward-model scorers are imported from ``run_grpo`` (these are the shared
online-RL rollout primitives; converging them into a common module is the class-hierarchy refactor,
RL_PLAN §3.8 / step 1b). The loss is SFT's ``cross_entropy`` (``configure_loss``), NOT an RL loss -- RFT
has no entry in ``_RLHF_LOSS_NAME`` on purpose.

Placement, backend and Ray requirements are therefore identical to ``run_grpo`` (Ray-only, vLLM sampler,
heterogeneous or colocate device groups).
"""
from __future__ import annotations
import copy
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from swift.dev.recipe.grpo import GRPOLoop, RolloutBatch
from swift.dev.recipe.run_grpo import (
    SyncableRollout,
    _build_reward_model_scorers,
    _grpo_sampling_params,
    _initialize_twinkle_rl,
    _prompt_rows_from_dataset,
    _reward_group,
    _sampler_backend,
    _sampler_engine_args,
    _sampler_world_size,
    plan_rl_device_groups,
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

logger = logging.getLogger(__name__)


def _rft_kept_per_prompt(rlhf_config: 'RLHFConfig', num_samples: int) -> int:
    """Upper bound on completions RFT keeps PER PROMPT -- the step budget's effective ``num_generations``.

    Mirrors :meth:`RFTLoop._select_keep`'s per-mode count from config alone (no rewards exist yet, since the
    budget is sized before any rollout): ``best_of_n`` keeps exactly 1, ``top_k`` keeps ``min(rft_top_k, N)``,
    and ``threshold`` can keep all N (the only reward-dependent mode, so its budget is the all-clear upper
    bound). ``rft_max_samples_per_prompt`` caps every mode. MUST stay in sync with ``_select_keep``.
    """
    select = rlhf_config.rft_select
    if select == 'best_of_n':
        kept = 1
    elif select == 'top_k':
        kept = min(max(1, rlhf_config.rft_top_k), num_samples)
    else:  # threshold
        kept = num_samples
    cap = rlhf_config.rft_max_samples_per_prompt
    if cap is not None:
        kept = min(kept, cap)
    return max(1, kept)


def run_rft(
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
    """Assemble and run rejection-sampling fine-tuning with weight sync. Returns the loss history.

    Mirrors :func:`run_grpo`'s assembly (train/sampler device groups, weight-syncable ``SyncableRollout``,
    reward scorers) but sets the SFT ``cross_entropy`` loss instead of an RL loss, and drives
    :class:`RFTLoop` (select-then-SFT) instead of :class:`GRPOLoop` (advantage-then-policy-gradient).
    """
    from swift.dev.builders import build_sampler
    from swift.dev.loss import configure_loss
    from swift.dev.optimizer import configure_optimizer, resolve_max_grad_norm
    from swift.dev.recipe.assembly import TrainAssembly

    if rlhf_config.rlhf_type != 'rft':
        raise ValueError(f'run_rft requires rlhf_type="rft", got {rlhf_config.rlhf_type!r}.')
    assembly = TrainAssembly(
        'run_rft',
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

    num_samples = rlhf_config.rft_num_samples or rlhf_config.num_generations
    if num_samples < 1:
        raise ValueError(f'RFT needs at least one sampled completion per prompt, got {num_samples}.')

    backend = _sampler_backend(rollout_config)
    sampler_world_size = _sampler_world_size(rollout_config, backend)
    # Each frozen reward model is a full-parameter seq_cls scorer on its own DeviceGroup (RL_PLAN §3.5); the
    # names/world sizes must match what _build_reward_model_scorers builds below. RFT has no reference or
    # teacher (it is cross-entropy SFT on filtered completions), so reward models are its only auxiliary.
    reward_models = list(rlhf_config.reward_model or [])
    auxiliary_groups: List[Tuple[str, int]] = [(_reward_group(i), 1) for i in range(len(reward_models))]
    groups, sampler_remote_group, colocate = plan_rl_device_groups(
        distributed_config.nproc_per_node,
        rollout_config.vllm_mode,
        sampler_world_size,
        auxiliary_groups=auxiliary_groups)
    _initialize_twinkle_rl(
        distributed_config,
        groups,
        seed=train_config.seed,
        full_determinism=train_config.full_determinism,
        sequence_parallel_size=template_config.sequence_parallel_size)

    assembly.build_template()
    # Ulysses SP mesh (None unless sequence_parallel_size>1): planned before build_model so the trainable
    # policy is placed on the dp x ulysses mesh, the same one _initialize_twinkle_rl installed globally for
    # the colocated sampler and the one the train_batch_size formula below reads. RFT is single-model
    # SFT-style (no critic), so its policy mesh and batch width agree exactly as GRPO's do.
    assembly.plan_sp_mesh()
    assembly.build_model()
    # RFT trains with plain cross-entropy SFT on the filtered completions -- NOT an RL loss.
    configure_loss(assembly.model, loss_type='cross_entropy', reduction='sum')
    # Prompts are rolled out, not iterated by a dataloader, so the step budget -- the LR-scheduler horizon AND
    # RFTLoop.fit's cap -- is DERIVED from the prompt set exactly as run_grpo/run_gkd derive it (an explicit
    # --max_steps still overrides). RFT runs rft_iterations rejection-sampling rounds, each one full generation
    # pass over the prompt set plus one SFT pass over the KEPT completions, so the budget sizes those rounds
    # with the per-prompt kept count as the effective num_generations and no replay (num_iterations=1). The old
    # `train_config.max_steps or 1` let the -1 "unset" sentinel through (-1 is truthy), so RFTLoop.fit's
    # `global_step >= max_steps` guard tripped at step 0 and the run trained nothing while misreporting it as
    # "kept too few completions".
    prompts, prompt_extras = _prompt_rows_from_dataset(dataset_config)
    from swift.dev.builders import build_ray_dp_mesh
    from swift.dev.recipe.train_loop import resolve_rollout_max_steps, rollout_step_budget
    # Design-B mini-batch width (per_device_train_batch_size * dp_size); see run_grpo for why dp_size comes
    # off build_ray_dp_mesh (online RL is Ray-only, pure data-parallel over nproc_per_node).
    train_batch_size = train_config.per_device_train_batch_size * build_ray_dp_mesh(
        distributed_config, template_config.sequence_parallel_size).data_world_size
    max_steps = resolve_rollout_max_steps(
        train_config.max_steps,
        rollout_step_budget(
            num_prompts=len(prompts),
            num_generations=_rft_kept_per_prompt(rlhf_config, num_samples),
            train_batch_size=train_batch_size,
            gradient_accumulation_steps=assembly.ga,
            num_train_epochs=rlhf_config.rft_iterations,
            num_iterations=1),
        recipe='run_rft')
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

    reward_model_plugins, reward_model_names = _build_reward_model_scorers(
        model_config, template_config, rlhf_config, distributed_config)
    loop = RFTLoop(
        assembly.model,
        rollout,
        prompts,
        prompt_extras=prompt_extras,
        template=assembly.template,
        num_generations=num_samples,
        reward_funcs=list(rlhf_config.orm) or None,
        reward_model_plugins=reward_model_plugins,
        reward_model_names=reward_model_names,
        reward_weights=rlhf_config.orm_weights,
        rlhf_config=rlhf_config,
        rft_select=rlhf_config.rft_select,
        rft_threshold=rlhf_config.rft_threshold,
        rft_top_k=rlhf_config.rft_top_k,
        rft_iterations=rlhf_config.rft_iterations,
        rft_max_samples_per_prompt=rlhf_config.rft_max_samples_per_prompt,
        max_steps=max_steps,
        gradient_accumulation_steps=assembly.ga,
        train_batch_size=train_batch_size,
        max_grad_norm=resolve_max_grad_norm(train_config),
        sampling_params=_grpo_sampling_params(rlhf_config, generation_config),
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
    assembly.loop = loop
    if assembly.resume_dir:
        loop.resume(assembly.resume_model())
    try:
        history = loop.fit()
        if _save_final:
            assembly.save_final()
        return history
    finally:
        rollout.shutdown()


class RFTLoop(GRPOLoop):
    """Rejection-sampling fine-tuning loop: sample N, score, keep the good, cross-entropy SFT on them.

    Reuses :class:`GRPOLoop`'s weight-synced rollout (``_generate``) and reward (``_score`` /
    ``_weighted_rewards``); overrides ``fit`` so each round selects the kept completions and trains them
    with plain SFT instead of a policy-gradient step. ``num_generations`` is the per-prompt sample count N.
    """

    def __init__(self,
                 *args: Any,
                 rft_select: str = 'best_of_n',
                 rft_threshold: float = 0.0,
                 rft_top_k: int = 1,
                 rft_iterations: int = 1,
                 rft_max_samples_per_prompt: Optional[int] = None,
                 **kwargs: Any):
        super().__init__(*args, **kwargs)
        if rft_select not in ('best_of_n', 'threshold', 'top_k'):
            raise ValueError(f"rft_select must be one of 'best_of_n'/'threshold'/'top_k', got {rft_select!r}.")
        if rft_iterations < 1:
            raise ValueError(f'rft_iterations must be >= 1, got {rft_iterations}.')
        if rft_max_samples_per_prompt is not None and rft_max_samples_per_prompt < 1:
            raise ValueError(f'rft_max_samples_per_prompt must be >= 1 or None, got {rft_max_samples_per_prompt}.')
        self.rft_select = rft_select
        self.rft_threshold = rft_threshold
        self.rft_top_k = max(1, rft_top_k)
        self.rft_iterations = rft_iterations
        self.rft_max_samples_per_prompt = rft_max_samples_per_prompt

    def _select_keep(self, group_rewards: List[float]) -> List[int]:
        """Local indices (into one prompt's N samples) to keep, ordered best-first."""
        order = sorted(range(len(group_rewards)), key=lambda i: group_rewards[i], reverse=True)
        if self.rft_select == 'best_of_n':
            keep = order[:1]
        elif self.rft_select == 'top_k':
            keep = order[:self.rft_top_k]
        else:  # threshold: keep every sample at/above the threshold, best-first.
            keep = [i for i in order if group_rewards[i] >= self.rft_threshold]
        if self.rft_max_samples_per_prompt is not None:
            keep = keep[:self.rft_max_samples_per_prompt]
        return keep

    def _select_samples(self, samples: List[Any], rewards: List[float]) -> RolloutBatch:
        """Apply the rejection filter across every prompt group, returning the kept samples.

        Returns a samples-only ``RolloutBatch``: RFT trains the kept completions with plain cross-entropy, so
        it carries no advantage / log-prob columns (they stay ``None``) and only ``.samples`` is consumed
        downstream.
        """
        n = self.num_generations
        if len(samples) % n != 0:
            raise RuntimeError(f'{len(samples)} samples is not a multiple of num_generations={n}; '
                               'group-wise rejection sampling is impossible.')
        selected: List[Any] = []
        for start in range(0, len(samples), n):
            group = samples[start:start + n]
            group_rewards = rewards[start:start + n]
            for local in self._select_keep(group_rewards):
                selected.append(group[local])
        return RolloutBatch(samples=selected)

    def fit(self) -> list:
        """Run up to ``rft_iterations`` rejection-sampling rounds, capped at ``max_steps`` optimizer steps.

        Design B (see :meth:`GRPOLoop.fit`): each round's kept completions are split into
        ``train_batch_size``-row mini-batches, one ``forward_backward`` per mini-batch and
        ``gradient_accumulation_steps`` mini-batches per optimizer step. RFT trains the selected completions
        with plain SFT, so a mini-batch carries only ``inputs`` -- no advantages/old_logps. A round that
        keeps fewer than ``train_batch_size`` completions cannot fill one mini-batch and is skipped (the kept
        count is reward-dependent, unlike GRPO's deterministic rollout size, so this warns rather than fails).
        """
        from swift.dev.recipe.train_loop import finish_manual_gc, start_manual_gc

        prompt_indices = list(range(len(self.prompts)))
        gc_was_enabled = start_manual_gc(self.manual_gc)
        try:
            for _ in range(self.rft_iterations):
                if self.global_step >= self.max_steps:
                    break
                # _generate syncs the trained policy into the sampler first, so this round samples from
                # the policy the previous round produced (the rejection-sampling bootstrap).
                samples = self._generate(prompt_indices)
                rewards_per_func = self._score(samples)
                # Feed the whole generated rollout (before rejection) into the driver-side reward component,
                # so the logged train/total_reward tracks the policy's improving sample quality -- the signal
                # that shows RFT working, since the SFT loss on the kept subset alone hides it.
                self._accumulate_reward_metric(samples, rewards_per_func)
                rewards = self._weighted_rewards(rewards_per_func).detach().float().cpu().tolist()
                selected = self._select_samples(samples, rewards)
                if len(selected) == 0:
                    logger.warning('RFT round kept no completions (rewards below the filter); skipping the SFT pass.')
                    continue
                if len(selected) < self.train_batch_size:
                    logger.warning('RFT round kept %d completions, fewer than train_batch_size=%d; skipping.',
                                   len(selected), self.train_batch_size)
                    continue
                for mini_batch in self._plan_mini_batches(selected):
                    if self.global_step >= self.max_steps:
                        break
                    self._run_micro_step({
                        'inputs': [copy.deepcopy(sample.input_feature) for sample in mini_batch.samples],
                        'gradient_accumulation_steps': self.gradient_accumulation_steps,
                    })
            if self.global_step == 0:
                # No round produced a full mini-batch, so the run trained nothing. Saving now would hand
                # back an untrained checkpoint that looks like a completed run -- fail loudly instead
                # (contract: an empty selected set is never a silent 0-step run).
                raise RuntimeError(
                    f'RFT took 0 optimizer steps across {self.rft_iterations} rejection-sampling round(s) '
                    f'(max_steps={self.max_steps}); no round kept enough completions to fill a train_batch_size='
                    f'{self.train_batch_size} mini-batch, so nothing was trained and no checkpoint is written. '
                    'Check max_steps > 0, then lower rft_threshold / rft_select, raise rft_num_samples, or enlarge '
                    'the prompt dataset so at least one round fills a mini-batch.')
            return self.history
        finally:
            finish_manual_gc(gc_was_enabled)
            self.tracker.close()


    # RFT trains the kept completions with plain cross-entropy SFT: no reference model, so it resets the one
    # GRPOLoop hook that would inject a reference sync. It KEEPS GRPOLoop's _extra_step_metrics (the
    # driver-side reward merge) and _log_step (loss + reward head signal), which its fit feeds via
    # _accumulate_reward_metric. The rest of the per-step cadence (counter, GC tick, log-on-tracker,
    # periodic save) is TrainLoop's.
    def _pre_metric_step(self) -> None:
        """No reference model to sync on the SFT-style RFT step (overrides GRPOLoop's reference sync)."""
