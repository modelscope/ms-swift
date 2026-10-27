# Copyright (c) ModelScope Contributors. All rights reserved.
# RFT's visible training main: five segments, then the blocking rejection-sampling round loop.
from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, List, Optional

from swift.dev.utils.logger import get_logger

from swift.dev.recipe.rft.assembly import initialize_rft_cluster, rft_group_requests
from swift.dev.recipe.utils import (
    StepState, build_save_knobs, initialize_rl_devices, prompt_rows_from_dataset,
    resolve_rollout_max_steps, resume_training, rollout_step_budget, save_training_step,
    snapshot_initial_adapter)

if TYPE_CHECKING:
    from swift.dev.config import (
        CheckpointConfig, DatasetConfig, DistributedConfig, GenerationConfig, LoggingConfig, MegatronConfig,
        ModelConfig, MoEConfig, QuantizeConfig, RLConfig, RewardConfig, RolloutConfig, TemplateConfig,
        TrainConfig, TunerConfig)

logger = get_logger()


def _rft_log_line(record: dict, loss: float) -> str:
    """Console head signal for RFT: loss plus this round's mean total reward when one was computed.

    The reward is what actually shows rejection sampling working -- the SFT loss on the kept
    completions starts near its floor and stays flat -- so the step line leads with ``reward`` when
    the round produced one.
    """
    reward = record.get('train/total_reward')
    reward_str = f'  reward={reward:.4f}' if reward is not None else ''
    return f'step {record.get("step")}  loss={loss:.4f}{reward_str}'


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
    """Upper bound on completions RFT keeps PER PROMPT -- the budget's effective ``num_generations``.

    Mirrors :meth:`RFTSelector._select_keep` from config alone (no rewards exist yet, since the
    budget is sized before any rollout): ``best_of_n`` keeps 1, ``top_k`` keeps ``min(rft_top_k, N)``,
    ``threshold`` can keep all N (the reward-dependent mode, so its budget is the all-clear bound).
    ``rft_max_samples_per_prompt`` caps every mode; MUST stay in sync with ``_select_keep``.
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


class RFTSelector:
    """The rejection-sampling selector + reward-metric holder RFT's round loop drives.

    ``num_generations`` is the per-prompt sample count N (``rft_num_samples`` or
    ``num_generations``): every block of N samples is one prompt's generation set, and the filter
    runs per block. The selector owns the ``CompletionRewardMetric`` the loop feeds once per round;
    ``step_telemetry`` folds it into every step record (``StepState``'s ``extra_metrics_fn``).
    """

    def __init__(self, *, num_generations: int, rft_select: str = 'best_of_n',
                 rft_threshold: float = 0.0, rft_top_k: int = 1,
                 rft_max_samples_per_prompt: Optional[int] = None,
                 reward_scorer_names: Optional[List[str]] = None,
                 reward_weights: Optional[List[float]] = None):
        if rft_select not in ('best_of_n', 'threshold', 'top_k'):
            raise ValueError(f"rft_select must be one of 'best_of_n'/'threshold'/'top_k', got {rft_select!r}.")
        if rft_max_samples_per_prompt is not None and rft_max_samples_per_prompt < 1:
            raise ValueError(f'rft_max_samples_per_prompt must be >= 1 or None, got {rft_max_samples_per_prompt}.')
        self.num_generations = num_generations
        self.rft_select = rft_select
        self.rft_threshold = rft_threshold
        self.rft_top_k = max(1, rft_top_k)
        self.rft_max_samples_per_prompt = rft_max_samples_per_prompt
        #: The outcome-channel roster (the ``--orm`` channel's rule + model labels), for the metric's
        #: per-scorer columns; an empty roster reports only the weighted ``total``.
        self.reward_scorer_names = list(reward_scorer_names or [])
        self.reward_weights = reward_weights
        #: The round reward metric, fed once per round and read back by :meth:`step_telemetry`.
        self._reward_metric = None

    def accumulate_metric(self, samples: List[Any], rewards_per_func: Any) -> None:
        """Feed one round's WHOLE generated rollout into the reward metric (reset-then-accumulate).

        Called before rejection so the logged reward tracks the policy's improving sample quality --
        filtering first would make the metric read only the kept winners and hide the trend.
        """
        from swift.dev.utils.scoring import accumulate_reward_metric
        from twinkle.metric import CompletionRewardMetric

        if self._reward_metric is None:
            self._reward_metric = CompletionRewardMetric()
        accumulate_reward_metric(self._reward_metric, samples, rewards_per_func,
                                 scorer_names=self.reward_scorer_names, reward_weights=self.reward_weights)

    def step_telemetry(self) -> dict:
        """The per-optimizer-step metric fold read by the loop's ``StepState``.

        Every step of one round reports the same round-level reward (RFT re-samples only between
        rounds, so the reward is naturally one value per round); ``calculate`` is non-resetting and
        returns an empty mapping when nothing was accumulated.
        """
        if self._reward_metric is None:
            return {}
        return dict(self._reward_metric.calculate())

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

    def select(self, samples: List[Any], rewards: List[float]) -> List[Any]:
        """Apply the rejection filter across every prompt group, returning the kept samples.

        The kept completions train with plain cross-entropy, so only the sample list is consumed
        downstream -- no advantage / log-prob columns exist in RFT.
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
        return selected


def _resume_round(checkpoint_config: 'CheckpointConfig', rft_state: RFTState) -> int:
    """Seed the round counter from the checkpoint's ``component_states['rft']`` (0 when absent).

    The model-side restore already ran (:func:`resume_training`); a weights-only checkpoint carries
    no trainer_state.json, which means a fresh round 0 -- the same fresh-run reading the model side
    took. ``rft_state`` is updated in place; the return is the seeded ``next_round``.
    """
    path = checkpoint_config.resume_from_checkpoint
    trainer_state_path = os.path.join(path, 'trainer_state.json')
    if not os.path.isfile(trainer_state_path):
        return rft_state.next_round
    with open(trainer_state_path) as f:
        component_states = json.load(f).get('component_states', {})
    if 'rft' in component_states:
        rft_state.load_state(component_states['rft'])
    return rft_state.next_round


def run_rft(model_config: 'ModelConfig', template_config: 'TemplateConfig',
            dataset_config: 'DatasetConfig', train_config: 'TrainConfig',
            distributed_config: 'DistributedConfig', checkpoint_config: 'CheckpointConfig',
            rollout_config: 'RolloutConfig', rl_config: 'RLConfig',
            reward_config: Optional['RewardConfig'] = None, tuner_config: Optional['TunerConfig'] = None,
            generation_config: Optional['GenerationConfig'] = None,
            logging_config: Optional['LoggingConfig'] = None,
            quantize_config: Optional['QuantizeConfig'] = None,
            megatron_config: Optional['MegatronConfig'] = None,
            moe_config: Optional['MoEConfig'] = None, *, output_dir: str = 'output',
            _save_final: bool = True) -> List[dict]:
    """RFT's five-segment main: prompts -> devices -> cluster -> loop scaffolding -> the round loop.

    A synchronous multi-round rejection-sampling loop, no driver: each of ``rft_iterations`` rounds
    pushes the just-trained policy into the sampler, blocking-generates over the WHOLE prompt set,
    scores, selects the kept completions (:class:`RFTSelector`) and SFTs them. Checkpoints commit at
    round boundaries only (the round component rides along), so a resume replays whole rounds.
    """
    # segment 1: prompts (the resource estimator reads the prompt lengths off the raw rows).
    prompt_rows, prompts, prompt_extras = prompt_rows_from_dataset(dataset_config)
    # segment 2: devices (configs + template + twinkle + perf; an estimate rewrites the parallel sizes).
    template, resource_plan, _ = initialize_rl_devices(
        model_config=model_config, template_config=template_config, dataset_config=dataset_config,
        train_config=train_config, distributed_config=distributed_config,
        checkpoint_config=checkpoint_config, rollout_config=rollout_config, rl_config=rl_config,
        reward_config=reward_config, tuner_config=tuner_config, logging_config=logging_config,
        quantize_config=quantize_config, megatron_config=megatron_config, moe_config=moe_config,
        output_dir=output_dir, prompt_rows=prompt_rows,
        group_requests=(lambda: rft_group_requests(reward_config)))
    num_samples = rl_config.rft_num_samples or rl_config.num_generations
    if num_samples < 1:
        raise ValueError(f'RFT needs at least one sampled completion per prompt, got {num_samples}.')
    if rl_config.rft_iterations < 1:
        raise ValueError(f'rft_iterations must be >= 1, got {rl_config.rft_iterations}.')
    # The optimizer's LR-scheduler horizon, derived BEFORE the cluster builds the optimizer: each of
    # the rft_iterations rounds generates the whole prompt set and SFTs the kept completions, so the
    # budget sizes those rounds with the per-prompt kept count as the effective num_generations and
    # no replay (num_iterations=1); an explicit --max_steps still overrides.
    from swift.dev.builders import build_dp_mesh

    dp_size = build_dp_mesh(distributed_config, template_config.sequence_parallel_size).data_world_size
    train_batch_size = train_config.per_device_train_batch_size * dp_size
    max_steps = resolve_rollout_max_steps(
        train_config.max_steps,
        rollout_step_budget(
            num_prompts=len(prompts), num_generations=_rft_kept_per_prompt(rl_config, num_samples),
            train_batch_size=train_batch_size,
            gradient_accumulation_steps=train_config.gradient_accumulation_steps,
            num_train_epochs=rl_config.rft_iterations, num_iterations=1),
        recipe='run_rft')
    # segment 3: the cluster -- the policy, the sampler, the orm reward channel, nothing else.
    cluster = initialize_rft_cluster(
        model_config=model_config, template_config=template_config, template=template,
        train_config=train_config, distributed_config=distributed_config,
        checkpoint_config=checkpoint_config, rollout_config=rollout_config, rl_config=rl_config,
        reward_config=reward_config, tuner_config=tuner_config, max_steps=max_steps,
        quantize_config=quantize_config, megatron_config=megatron_config, moe_config=moe_config,
        generation_config=generation_config, resource_plan=resource_plan, output_dir=output_dir)
    # segment 4: the dissolved step cadence + the save closure (round-atomic, see save below).
    from swift.dev.builders.optimizer import resolve_max_grad_norm

    selector = RFTSelector(
        num_generations=num_samples, rft_select=rl_config.rft_select,
        rft_threshold=rl_config.rft_threshold, rft_top_k=rl_config.rft_top_k,
        rft_max_samples_per_prompt=rl_config.rft_max_samples_per_prompt,
        reward_scorer_names=cluster.orm_scorer_names, reward_weights=reward_config.orm_weights)
    state = StepState(
        cluster.model, train_config,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        max_steps=max_steps, max_grad_norm=resolve_max_grad_norm(train_config),
        logging_config=logging_config, output_dir=output_dir,
        save_steps=checkpoint_config.save_steps,
        manual_gc=bool(megatron_config and megatron_config.manual_gc),
        manual_gc_steps=megatron_config.manual_gc_steps if megatron_config else 0,
        empty_cache_steps=train_config.torch_empty_cache_steps,
        extra_metrics_fn=selector.step_telemetry, log_formatter=_rft_log_line)
    save_knobs = build_save_knobs(
        checkpoint_config, dataset_config, distributed_config, tuner_config, output_dir=output_dir)
    # Before any training step: a base-modifying LoRA init (pissa ...) needs its initial snapshot.
    snapshot_initial_adapter(cluster.model, save_knobs.initial_adapter_dir)
    rft_state = RFTState()

    def save(name: str, is_final: bool = False) -> str:
        """One training checkpoint with the round component, so a resume replays whole rounds only.

        Generated / selected samples are never checkpointed, so a checkpoint saved mid-round would
        have the round's optimizer steps but not its completions; committing ``next_round`` and the
        save together at the round boundary keeps every checkpoint round-atomic.
        """
        state.publish_state()
        return save_training_step(
            cluster.model, state.handler, name, save_knobs,
            consumed_train_samples=state.global_step,
            component_states={'rft': rft_state.get_state()}, is_final=is_final)

    start_round = 0
    if checkpoint_config.resume_from_checkpoint:
        state.global_step, state.micro_step, _ = resume_training(
            cluster.model, checkpoint_config=checkpoint_config, distributed_config=distributed_config,
            megatron_config=megatron_config, tuner_config=tuner_config,
            gradient_accumulation_steps=train_config.gradient_accumulation_steps)
        start_round = _resume_round(checkpoint_config, rft_state)
    from twinkle.perf import Perf

    from swift.dev.recipe.utils import filter_perf_actors

    if Perf.initialize(filter_perf_actors({'model': cluster.model, 'sampler': cluster.sampler})):
        logger.info('[perf] profiling enabled for the online RFT loop.')
    # segment 5: the visible round loop.
    save_steps = checkpoint_config.save_steps
    ga = train_config.gradient_accumulation_steps
    from swift.dev.utils.scoring import copy_inputs

    state.handler.on_train_begin()
    try:
        for round_index in range(start_round, rl_config.rft_iterations):
            if state.reached_max():
                break
            entered = False
            try:
                # borrow the device for generation (colocate) / push the trained policy into the
                # sampler, so this round samples from the policy the previous round produced -- the
                # rejection-sampling bootstrap.
                cluster.sampler.wake_up(tags=['weights'])
                cluster.engine.sync_weights(merge_and_sync=True)
                from swift.dev.recipe.offload import offload_for_rollout
                offload_for_rollout(cluster.model, rollout_config)
                cluster.sampler.wake_up(tags=['kv_cache'])
                entered = True
                samples = cluster.rollout.generate(
                    prompts, num_samples=num_samples, sampling_params=cluster.sampling_params,
                    prompt_extras=prompt_extras or None)
                expected = len(prompts) * num_samples
                if len(samples) != expected:
                    raise RuntimeError(f'rollout returned {len(samples)} samples, expected {expected} '
                                       f'({len(prompts)} prompts * {num_samples} generations).')
                import torch

                from swift.dev.utils.scoring import weighted_rewards

                rewards_per_func = cluster.worker.score_group(samples)
                # None (an unparseable judge verdict) rides as nan: the metric fold and the weighted
                # sum below consume the same nan-normalized sample-major matrix.
                rewards_matrix = torch.tensor(
                    [[float('nan') if value is None or value != value else float(value) for value in row]
                     for row in rewards_per_func], dtype=torch.float32)
                # Feed the whole generated rollout (before rejection) into the reward metric, so the
                # logged train/total_reward tracks the policy's improving sample quality -- the SFT
                # loss on the kept subset alone hides it.
                selector.accumulate_metric(samples, rewards_matrix)
                rewards = weighted_rewards(rewards_matrix, reward_config.orm_weights).tolist()
                selected = selector.select(samples, rewards)
                # return the device to the trainer: sleep the sampler, then reload the weights.
                from swift.dev.recipe.offload import sleep_for_rollout
                sleep_for_rollout(cluster.sampler, rollout_config.sleep_level)
                cluster.model.reload_to_gpu()
                entered = False
            finally:
                if entered:
                    # ended mid-window (a count-contract failure during generation): give the device
                    # back so the final save finds the model resident.
                    from swift.dev.recipe.offload import sleep_for_rollout
                    sleep_for_rollout(cluster.sampler, rollout_config.sleep_level)
                    cluster.model.reload_to_gpu()
            round_save_due = False
            if len(selected) == 0:
                logger.warning('RFT round kept no completions (rewards below the filter); skipping the SFT pass.')
            elif len(selected) < train_batch_size:
                logger.warning('RFT round kept %d completions, fewer than train_batch_size=%d; skipping.',
                               len(selected), train_batch_size)
            else:
                dropped = len(selected) % train_batch_size
                if dropped:
                    logger.warning('Dropping %d of %d kept completions that do not fill a train_batch_size=%d '
                                   'mini-batch.', dropped, len(selected), train_batch_size)
                for start in range(0, len(selected) - train_batch_size + 1, train_batch_size):
                    if state.reached_max():
                        break
                    mini = selected[start:start + train_batch_size]
                    state.micro_step += 1
                    cluster.model.forward_backward(
                        inputs=copy_inputs([sample.encoded for sample in mini]),
                        gradient_accumulation_steps=ga)
                    boundary = state.is_grad_sync_boundary()
                    cluster.model.clip_grad_and_step(max_grad_norm=state.max_grad_norm,
                                                     gradient_accumulation_steps=ga)
                    if boundary:
                        state.record()
                        # The periodic save is DEFERRED to the round boundary (the round component
                        # commits with it, keeping every checkpoint round-atomic); snapshot the
                        # callback's one-shot request before the tail clears it.
                        wanted = state.handler.control.should_save
                        state.step_tail()
                        if wanted or (save_steps and state.global_step % save_steps == 0):
                            round_save_due = True
                    if state.reached_max():
                        break
            # Commit the round only after its SFT pass has completed (or max_steps deliberately
            # truncated it), then persist at this safe boundary.
            rft_state.next_round = round_index + 1
            if round_save_due:
                save(f'checkpoint-{state.global_step}')
            if state.reached_max() or state.should_stop():
                break
        if state.global_step == 0:
            # No round produced a full mini-batch, so the run trained nothing. Saving now would hand
            # back an untrained checkpoint that looks like a completed run -- fail loudly instead
            # (contract: an empty selected set is never a silent 0-step run).
            raise RuntimeError(
                f'RFT took 0 optimizer steps across {rl_config.rft_iterations} rejection-sampling round(s) '
                f'(max_steps={max_steps}); no round kept enough completions to fill a train_batch_size='
                f'{train_batch_size} mini-batch, so nothing was trained and no checkpoint is written. '
                'Check max_steps > 0, then lower rft_threshold / rft_select, raise rft_num_samples, or enlarge '
                'the prompt dataset so at least one round fills a mini-batch.')
    finally:
        Perf.shutdown()
        # MultiTurnRollout owns only the sandbox env pool; the sampler's GPU memory is the Cluster's
        # to release (a Ray-deployed rollout is a proxy whose sampler lives worker-side).
        cluster.rollout.close()
        cluster.sampler.shutdown()
        state.tracker.close()
    state.handler.on_train_end()
    if _save_final:
        # the final checkpoint is synchronous; its args.json was written at initialize_devices.
        cluster.model.wait_async_saves()
        save('checkpoint-final', is_final=True)
    return state.history
