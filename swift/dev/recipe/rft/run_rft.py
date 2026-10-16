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
Rollout, weight sync and reward are the shared online-RL machinery: the flat run body blocks on one
whole-prompt-set rollout per round through :func:`~swift.dev.rollout.blocking_generate` (which pushes the
trained policy into the sampler before each rollout, so the behaviour policy tracks the trained one),
scores it through the shared :mod:`~swift.dev.recipe.common.scoring` outcome-reward chokepoint
(``score_samples`` / ``weighted_rewards`` / ``accumulate_reward_metric``), and hands the scored samples to
:class:`~swift.dev.recipe.rft.selector.RFTSelector` (select-then-SFT) instead of an advantage/policy-gradient step.
Device placement, sampler construction, the dataset prompt loader and reward-model scorers are imported
from ``run_grpo`` (the shared online-RL rollout primitives). The loss is SFT's ``cross_entropy``
(``configure_loss``), NOT an RL loss -- RFT has no entry in ``_RL_LOSS_NAME`` on purpose.

Placement, backend and Ray requirements are therefore identical to ``run_grpo`` (Ray-only, vLLM sampler,
heterogeneous or colocate device groups).
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from swift.dev.recipe.common.scoring import copy_inputs, score_samples, split_mini_batches, weighted_rewards
from swift.dev.recipe.common.checkpoint import CheckpointSaver
from swift.dev.recipe.rft.selector import RFTSelector, _rft_log_line
from swift.dev.recipe.rl.auxiliary import plan_rl_device_groups, reward_group, spec_world_size
from swift.dev.recipe.rl.prompts import prompt_rows_from_dataset
from swift.dev.recipe.rl.rewards import build_reward_scorers
from swift.dev.recipe.rl.sampler import (
    RolloutSamplerSpec,
    SyncableRollout,
    initialize_twinkle_rl,
    rollout_sampling_params,
)
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
        RolloutConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )

logger = get_logger()


class RFTState:
    """The next rejection-sampling round that has not been committed to a checkpoint."""

    def __init__(self):
        self.next_round = 0

    def get_state(self) -> dict:
        return {'version': 1, 'next_round': self.next_round}

    def load_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            raise TypeError(f'RFT state must be a dict, got {type(state).__name__}.')
        version = int(state.get('version', 1))
        if version != 1:
            raise ValueError(f'Unsupported RFT state version: {version}.')
        next_round = int(state.get('next_round', 0))
        if next_round < 0:
            raise ValueError(f'RFT next_round must be non-negative, got {next_round}.')
        self.next_round = next_round


def _rft_kept_per_prompt(rl_config: 'RLConfig', num_samples: int) -> int:
    """Upper bound on completions RFT keeps PER PROMPT -- the step budget's effective ``num_generations``.

    Mirrors :meth:`RFTSelector._select_keep`'s per-mode count from config alone (no rewards exist yet, since the
    budget is sized before any rollout): ``best_of_n`` keeps exactly 1, ``top_k`` keeps ``min(rft_top_k, N)``,
    and ``threshold`` can keep all N (the only reward-dependent mode, so its budget is the all-clear upper
    bound). ``rft_max_samples_per_prompt`` caps every mode. MUST stay in sync with ``_select_keep``.
    """
    select = rl_config.rft_select
    if select == 'best_of_n':
        kept = 1
    elif select == 'top_k':
        kept = min(max(1, rl_config.rft_top_k), num_samples)
    else:  # threshold
        kept = num_samples
    cap = rl_config.rft_max_samples_per_prompt
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
    """Assemble and run rejection-sampling fine-tuning with weight sync. Returns the loss history.

    Mirrors :func:`run_grpo`'s assembly (train/sampler device groups, weight-syncable ``SyncableRollout``,
    reward scorers) but sets the SFT ``cross_entropy`` loss instead of an RL loss, and drives
    :class:`~swift.dev.recipe.rft.selector.RFTSelector` (select-then-SFT) in a blocking round loop instead of a
    streaming driver.
    """
    from swift.dev.builders import build_sampler
    from swift.dev.loss import configure_loss
    from swift.dev.builders.optimizer import configure_optimizer, resolve_max_grad_norm
    from swift.dev.recipe.common.assembly import TrainAssembly

    if rl_config.rl_type != 'rft':
        raise ValueError(f'run_rft requires rl_type="rft", got {rl_config.rl_type!r}.')
    assembly = TrainAssembly(
        'run_rft',
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

    num_samples = rl_config.rft_num_samples or rl_config.num_generations
    if num_samples < 1:
        raise ValueError(f'RFT needs at least one sampled completion per prompt, got {num_samples}.')

    backend = RolloutSamplerSpec.backend(rollout_config)
    sampler_world_size = RolloutSamplerSpec.world_size(rollout_config, backend)
    # Each frozen reward model is a full-parameter seq_cls scorer on its own DeviceGroup (RL_PLAN §3.5); the
    # names/world sizes must match what build_reward_scorers builds below. RFT has no reference or
    # teacher (it is cross-entropy SFT on filtered completions), so reward models are its only auxiliary.
    reward_models = list(rl_config.reward_model or [])
    reward_world_size = spec_world_size(rl_config.orm_parallel_spec)
    auxiliary_groups: List[Tuple[str, int]] = [
        (reward_group(i), reward_world_size) for i in range(len(reward_models))]
    groups, sampler_remote_group, colocate = plan_rl_device_groups(
        distributed_config.nproc_per_node,
        rollout_config.sampler_mode,
        sampler_world_size,
        auxiliary_groups=auxiliary_groups)
    initialize_twinkle_rl(
        distributed_config,
        groups,
        seed=train_config.seed,
        full_determinism=train_config.full_determinism,
        tf32=train_config.tf32,
        sequence_parallel_size=template_config.sequence_parallel_size)

    assembly.build_template()
    # Ulysses SP mesh (None unless sequence_parallel_size>1): planned before build_model so the trainable
    # policy is placed on the dp x ulysses mesh, the same one initialize_twinkle_rl installed globally for
    # the colocated sampler and the one the train_batch_size formula below reads. RFT is single-model
    # SFT-style (no critic), so its policy mesh and batch width agree exactly as GRPO's do.
    assembly.plan_sp_mesh()
    assembly.build_model()
    # RFT trains with plain cross-entropy SFT on the filtered completions -- NOT an RL loss.
    configure_loss(assembly.model, loss_type='cross_entropy', reduction='sum')
    # Prompts are rolled out, not iterated by a dataloader, so the step budget -- the LR-scheduler horizon AND
    # the run body's cap -- is DERIVED from the prompt set exactly as run_grpo/run_gkd derive it (an explicit
    # --max_steps still overrides). RFT runs rft_iterations rejection-sampling rounds, each one full generation
    # pass over the prompt set plus one SFT pass over the KEPT completions, so the budget sizes those rounds
    # with the per-prompt kept count as the effective num_generations and no replay (num_iterations=1). The old
    # `train_config.max_steps or 1` let the -1 "unset" sentinel through (-1 is truthy), so the run body's
    # `global_step >= max_steps` guard tripped at step 0 and the run trained nothing while misreporting it as
    # "kept too few completions".
    prompts, prompt_extras = prompt_rows_from_dataset(dataset_config)
    from swift.dev.builders import build_dp_mesh
    from swift.dev.recipe.rl.prompts import resolve_rollout_max_steps, rollout_step_budget
    # Design-B mini-batch width (per_device_train_batch_size * dp_size); see run_grpo for why dp_size comes
    # off build_dp_mesh (online RL is Ray-only, pure data-parallel over nproc_per_node).
    train_batch_size = train_config.per_device_train_batch_size * build_dp_mesh(
        distributed_config, template_config.sequence_parallel_size).data_world_size
    max_steps = resolve_rollout_max_steps(
        train_config.max_steps,
        rollout_step_budget(
            num_prompts=len(prompts),
            num_generations=_rft_kept_per_prompt(rl_config, num_samples),
            train_batch_size=train_batch_size,
            gradient_accumulation_steps=assembly.ga,
            num_train_epochs=rl_config.rft_iterations,
            num_iterations=1),
        recipe='run_rft')
    assembly.resolve_step_intervals(max_steps)
    configure_optimizer(
        assembly.model,
        train_config,
        num_training_steps=max_steps,
        distributed_config=distributed_config,
        tuner_config=assembly.tuner_config,
        plugin_config=assembly.plugin_config)

    sampler_engine_args = RolloutSamplerSpec.engine_args(rollout_config, engine_args, colocate, backend)
    sampler = build_sampler(
        model_config,
        backend=backend,
        engine_args=sampler_engine_args,
        template=assembly.template,
        remote_group=sampler_remote_group)
    rollout = SyncableRollout(
        assembly.model, sampler, assembly.template, colocate=colocate, sleep_level=rollout_config.sleep_level)

    reward_scorers, reward_scorer_names = build_reward_scorers(
        model_config, template_config, rl_config, distributed_config, sampler_model=model_config.model,
        backend=backend)
    if rl_config.rft_iterations < 1:
        raise ValueError(f'rft_iterations must be >= 1, got {rl_config.rft_iterations}.')
    # RFT's control flow is a synchronous multi-round rejection-sampling loop (no driver): each of
    # rft_iterations rounds blocking-generates over the WHOLE prompt set through blocking_generate (which
    # pushes the just-trained policy into the sampler first), scores, selects the kept completions and
    # SFTs them. The old FixedBatchDriver (build_rollout_driver(sync_only=True)) was a dead shell -- fit
    # never called driver.run -- so the flat body drops it entirely.
    from swift.dev.callbacks import CallbackHandler, build_callbacks

    selector = RFTSelector(
        num_generations=num_samples,
        rft_select=rl_config.rft_select,
        rft_threshold=rl_config.rft_threshold,
        rft_top_k=rl_config.rft_top_k,
        rft_max_samples_per_prompt=rl_config.rft_max_samples_per_prompt,
        reward_scorer_names=reward_scorer_names,
        reward_weights=rl_config.orm_weights)
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
        log_formatter=_rft_log_line,
        extra_metrics_fn=selector.step_telemetry)
    rft_state = RFTState()
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
        components={'rft': rft_state})
    if assembly.resume_dir:
        saver.resume(assembly.resume_model())
    start_round = rft_state.next_round

    from twinkle.perf import Perf

    from swift.dev.recipe.common.loop_utils import filter_perf_actors

    if Perf.initialize(filter_perf_actors({'model': assembly.model, 'sampler': rollout.sampler})):
        logger.info('[perf] profiling enabled for the online RFT loop.')
    prompt_indices = list(range(len(prompts)))
    try:
        for round_index in range(start_round, rl_config.rft_iterations):
            if recorder.global_step >= max_steps:
                break
            # blocking_generate syncs the trained policy into the sampler first, so this round samples
            # from the policy the previous round produced (the rejection-sampling bootstrap).
            samples = blocking_generate(
                rollout,
                [prompts[i] for i in prompt_indices],
                num_samples=num_samples,
                sampling_params=rollout_sampling_params(rl_config, generation_config),
                prompt_extras=[prompt_extras[i] for i in prompt_indices])
            expected = len(prompt_indices) * num_samples
            if len(samples) != expected:
                raise RuntimeError(f'rollout returned {len(samples)} samples, expected {expected} '
                                   f'({len(prompt_indices)} prompts * {num_samples} generations).')
            for local_idx, prompt_idx in enumerate(prompt_indices):
                for sample in samples[local_idx * num_samples:(local_idx + 1) * num_samples]:
                    sample.prompt_id = str(prompt_idx)
            rewards_per_func = score_samples(samples, reward_scorers=reward_scorers)
            # Feed the whole generated rollout (before rejection) into the reward component, so the logged
            # train/total_reward tracks the policy's improving sample quality -- the signal that shows RFT
            # working, since the SFT loss on the kept subset alone hides it.
            selector.accumulate_metric(samples, rewards_per_func)
            rewards = weighted_rewards(rewards_per_func, rl_config.orm_weights).detach().float().cpu().tolist()
            selected = selector.select(samples, rewards)
            save_at_round_boundary = False
            if len(selected) == 0:
                logger.warning('RFT round kept no completions (rewards below the filter); skipping the SFT pass.')
            elif len(selected) < train_batch_size:
                logger.warning('RFT round kept %d completions, fewer than train_batch_size=%d; skipping.',
                               len(selected), train_batch_size)
            else:
                dropped = len(selected.samples) % train_batch_size
                if dropped:
                    logger.warning('Dropping %d of %d kept completions that do not fill a train_batch_size=%d '
                                   'mini-batch.', dropped, len(selected.samples), train_batch_size)
                for mini_batch in split_mini_batches(selected.samples, train_batch_size):
                    if recorder.global_step >= max_steps:
                        break
                    recorder.micro_step += 1
                    assembly.model.forward_backward(
                        inputs=copy_inputs([sample.input_feature for sample in mini_batch]),
                        gradient_accumulation_steps=assembly.ga)
                    is_boundary = recorder.is_grad_sync_boundary()
                    assembly.model.clip_grad_and_step(
                        max_grad_norm=recorder.max_grad_norm,
                        gradient_accumulation_steps=recorder.gradient_accumulation_steps)
                    if is_boundary:
                        recorder.record_step()
                        if saver.save_steps and recorder.global_step % saver.save_steps == 0:
                            save_at_round_boundary = True
                    if recorder.reached_max():
                        break
            # Generated/selected samples are not checkpointed. Commit the round only after its SFT pass has
            # completed, or after max_steps has deliberately truncated it, then persist at this safe boundary.
            rft_state.next_round = round_index + 1
            if save_at_round_boundary:
                saver.save(f'checkpoint-{recorder.global_step}')
            if recorder.reached_max():
                break
        if recorder.global_step == 0:
            # No round produced a full mini-batch, so the run trained nothing. Saving now would hand back
            # an untrained checkpoint that looks like a completed run -- fail loudly instead (contract: an
            # empty selected set is never a silent 0-step run).
            raise RuntimeError(
                f'RFT took 0 optimizer steps across {rl_config.rft_iterations} rejection-sampling round(s) '
                f'(max_steps={max_steps}); no round kept enough completions to fill a train_batch_size='
                f'{train_batch_size} mini-batch, so nothing was trained and no checkpoint is written. '
                'Check max_steps > 0, then lower rft_threshold / rft_select, raise rft_num_samples, or enlarge '
                'the prompt dataset so at least one round fills a mini-batch.')
    finally:
        Perf.shutdown()
        rollout.shutdown()
        recorder.tracker.close()
    if _save_final:
        saver.save_final(write_args_json=assembly.args_json_writer())
    return recorder.history

