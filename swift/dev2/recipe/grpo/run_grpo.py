# Copyright (c) ModelScope Contributors. All rights reserved.
# GRPO's visible training main: five segments, then ONE loop body picked by placement.
from __future__ import annotations

import math
import os
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from swift.dev.utils.logger import get_logger  # TODO(dev2): bridge until the utils batch
from swift.dev2.recipe.grpo.assembly import (
    greedy_params, group_variance, initialize_cluster, initialize_devices, prompt_rows_from_dataset,
    resolve_rollout_max_steps, rollout_step_budget, weighted_reward)

if TYPE_CHECKING:
    from swift.dev2.config import (
        CheckpointConfig, DatasetConfig, DistributedConfig, GenerationConfig, LoggingConfig, MegatronConfig,
        ModelConfig, MoEConfig, QuantizeConfig, RLConfig, RewardConfig, RolloutConfig, TemplateConfig,
        TrainConfig, TunerConfig)

logger = get_logger()


class _StepState:
    """The loop's step cadence (the retired recorder, dissolved into this recipe): counters, metrics.

    Owns the global/micro step counters, the metric record (``model.calculate_metric`` + the
    megatron MTP drain), the ``RunTracker`` write, the console line, and the callback fan-out --
    nothing else; the loops, the save cadence, and the publish cadence stay visible in the branches.
    """

    def __init__(self, model: Any, train_config: 'TrainConfig', *, gradient_accumulation_steps: int,
                 max_steps: int, max_grad_norm: float, logging_config: Optional['LoggingConfig'],
                 output_dir: str, save_steps: Optional[int] = None, manual_gc: bool = False,
                 manual_gc_steps: int = 0, empty_cache_steps: Optional[int] = None,
                 extra_metrics_fn: Optional[Callable[[], Dict[str, Any]]] = None):
        # The tracker initializes its reporters as a construction side effect (validation ran above).
        # TODO(dev2): bridge until the recipe/common batch migrates RunTracker
        from swift.dev.recipe.common.tracking import RunTracker

        from swift.dev2.callbacks import build_callbacks
        from swift.dev2.callbacks.handler import CallbackHandler

        self.model = model
        self.ga = gradient_accumulation_steps
        self.max_steps = max_steps
        self.max_grad_norm = max_grad_norm
        self.save_steps = save_steps
        self.manual_gc = manual_gc
        self.manual_gc_steps = manual_gc_steps
        self.empty_cache_steps = empty_cache_steps
        self._extra_metrics_fn = extra_metrics_fn
        self.global_step = 0
        self.micro_step = 0
        self.history: List[dict] = []
        self.tracker = RunTracker(logging_config, output_dir)
        self.handler = CallbackHandler(build_callbacks(train_config), output_dir=output_dir)

    def is_grad_sync_boundary(self) -> bool:
        """Whether the current ``micro_step`` closes an optimizer window."""
        # TODO(dev2): bridge until the recipe/common batch migrates the cadence helpers
        from swift.dev.recipe.common.loop_utils import is_grad_sync_boundary

        return is_grad_sync_boundary(self.micro_step, self.ga)

    def reached_max(self) -> bool:
        return self.max_steps > 0 and self.global_step >= self.max_steps

    def record(self) -> dict:
        """Count one optimizer step: metrics, tracker write, console line, callback fan-out."""
        # TODO(dev2): bridge until the recipe/common batch migrates the cadence helpers
        from swift.dev.recipe.common.loop_utils import collect_empty_cache, collect_manual_gc

        from twinkle.perf import Perf

        self.global_step += 1
        collect_manual_gc(self.manual_gc, self.manual_gc_steps, self.global_step)
        collect_empty_cache(self.model, self.empty_cache_steps, self.global_step)
        Perf.advance_step()
        metrics = self.model.calculate_metric(is_training=True)
        pop = getattr(self.model, 'pop_mtp_metrics', None)
        if pop is not None:
            # megatron tracks the MTP channel outside twinkle's metrics; drain it or it grows unbounded
            metrics.update({key: float(value) for key, value in (pop() or {}).items()})
        loss = float(metrics['loss']) if metrics.get('loss') is not None else float('nan')
        record = {'step': self.global_step, **metrics, 'loss': loss}
        if self._extra_metrics_fn is not None:
            record.update(self._extra_metrics_fn())
        record = self.tracker.log(record, self.global_step)
        self.history.append(record)
        self.handler.state.global_step = self.global_step
        self.handler.state.micro_step = self.micro_step
        self.handler.state.loss = loss
        self.handler.state.metrics = record
        if self.tracker.should_log(self.global_step) or self.handler.control.should_log:
            self.handler.control.should_log = False
            gn = record.get('grad_norm')
            line = (f"step {self.global_step}  loss={loss:.4f}"
                    + (f"  grad_norm={gn:.4f}" if isinstance(gn, (int, float)) else '')
                    + ''.join(f'  {key}={value:.4f}' for key, value in record.items()
                              if key.startswith('loss_') and isinstance(value, (int, float))))
            logger.info(line)
            self.handler.on_log(logs=record)
        return record


def _score_group(cluster: Any, samples: List[Any], with_prm: bool = False) -> List[List[Any]]:
    """The orm matrix as sample-major rows (``[N][n_scorers]``), None for an unparseable verdict.

    ``worker.score_group`` already transposes and normalizes (the same protocol the driver's
    built-in group-barrier scoring uses); ``with_prm`` routes through ``score_group_prm`` so both
    channels score in ONE pass and each sample's process rewards land on it. This thin wrapper
    keeps the colocate loop reading uniformly whether scorers are wired or not.
    """
    if with_prm and cluster.prm_scorer is not None:
        rewards, prm_refs = cluster.worker.score_group_prm(samples)
        for sample, ref in zip(samples, prm_refs):
            sample.prm_process = ref
        return rewards
    return cluster.worker.score_group(samples) if cluster.orm_scorers else [[] for _ in samples]


def _response_logps(scorer_model: Any, features: List[Any], *, train_batch_size: int,
                    disable_lora: bool = False) -> List[List[float]]:
    """One response-token logp list per feature (chunked forward_only, DP>1-correct)."""
    # TODO(dev2): bridge until the recipe/common batch migrates the chunked scoring helpers
    from swift.dev.recipe.common.scoring import batched_response_scores

    forward_kwargs = {'disable_lora': True} if disable_lora else None
    return batched_response_scores(
        scorer_model, features, train_batch_size=train_batch_size, output_key='logps',
        forward_kwargs=forward_kwargs)


def _advantage_pipeline(cluster: Any, rl_config: 'RLConfig', reward_config: 'RewardConfig',
                        samples: List[Any], train_batch_size: int, global_step: int) -> Tuple[Any, ...]:
    """The shared advantage pipeline: logps forwards, KL, group-relative advantage, PRM / teacher fusion.

    Reads the scoring results off the samples (``orm_reward`` / ``baseline`` / ``prm_process``, written
    by the group-completion hook or the colocate loop). Returns ``(advantages, old_logps, ref_logps,
    teacher_logps)``. The reference / teacher / policy-old forwards run on their own DeviceGroups
    (disjoint ones overlap, same-group ones chain); PRM fusion and teacher-signal reweighting are
    mutually exclusive per-token sources, and PRM wins (validate rejects enabling both).
    """
    import torch

    # TODO(dev2): bridge until the recipe/common batch migrates the overlap scheduler
    from swift.dev.recipe.common.scoring import overlap_by_placement

    from swift.dev2.rewards.advantage import compute_advantages

    rewards = [sample.orm_reward for sample in samples]
    baselines = ([sample.baseline for sample in samples]
                 if any(sample.baseline is not None for sample in samples) else None)
    prm_process = ([sample.prm_process for sample in samples]
                   if any(sample.prm_process is not None for sample in samples) else None)
    old_logps: List[Any] = [list(sample.rollout_logps) for sample in samples]
    preserve = bool(rl_config.rollout_importance_sampling_mode or rl_config.log_rollout_offpolicy_metrics)
    # ref / teacher / policy-old each forward on their own group; a privileged teacher_prompt column
    # with no explicit teacher scores on the policy actor itself (placement_key None).
    tasks: List[Tuple[Any, Callable[[], Any]]] = []
    slots: List[str] = []
    if cluster.reference is not None:
        reference = cluster.reference
        tasks.append((reference.placement_key, lambda: _response_logps(
            reference.model, [sample.encoded for sample in samples], train_batch_size=train_batch_size,
            disable_lora=reference.disable_adapter)))
        slots.append('ref')
    if cluster.teacher is not None or any((s.extra or {}).get('teacher_prompt') for s in samples):
        if cluster.teacher is not None:
            teacher = cluster.teacher
            t_key, t_model, t_disable = teacher.placement_key, teacher.model, teacher.disable_adapter
        else:
            t_key, t_model, t_disable = None, cluster.model, False
        privileged = cluster.worker.privileged_features(samples)
        tasks.append((t_key, lambda: _response_logps(
            t_model, privileged, train_batch_size=train_batch_size, disable_lora=t_disable)))
        slots.append('teacher')
    if preserve:
        tasks.append((None, lambda: _response_logps(
            cluster.model, [sample.encoded for sample in samples], train_batch_size=train_batch_size)))
        slots.append('old')
    scored = dict(zip(slots, overlap_by_placement(tasks)))
    ref_logps = scored.get('ref')
    teacher_logps = scored.get('teacher')
    if preserve:
        old_logps = scored['old']
    kl_in_reward = bool(rl_config.kl_in_reward and rl_config.beta)
    if kl_in_reward and ref_logps is None:
        raise ValueError('kl_in_reward=True requires an active reference model.')
    kl_values = (torch.tensor(cluster.worker.sequence_kl(old_logps, ref_logps), dtype=torch.float32)
                 if kl_in_reward else None)
    # None (an unparseable judge verdict) rides as nan, exactly as dev recorded it; a no-scorer row
    # (no --orm wired) stays [nan] -- a zero-column tensor cannot carry a group mean.
    reward_matrix = torch.tensor(
        [[float('nan') if value is None or value != value else float(value) for value in row]
         or [float('nan')] for row in rewards], dtype=torch.float32)
    estimator = rl_config.advantage_estimator
    advantages = compute_advantages(
        reward_matrix,
        rl_config.num_generations,
        reward_weights=reward_config.orm_weights,
        advantage_estimator=estimator,
        scale_rewards=rl_config.scale_rewards or 'group',
        kl_in_reward=kl_in_reward,
        beta=float(rl_config.beta or 0.0),
        kl_values=kl_values,
        response_lengths=([len(sample.response_token_ids) for sample in samples]
                          if estimator == 'opo' else None),
        f_norm=(float(rl_config.f_norm) if estimator == 'gpg' else 1.0),
        reward_baselines=(torch.tensor(baselines, dtype=torch.float32) if baselines is not None else None))
    if prm_process is not None:
        advantages = cluster.worker.fuse_prm(advantages, prm_process, float(cluster.prm_orm_mix))
    elif teacher_logps is not None:
        # teacher-signal per-token reweighting; sdar keeps the scalar advantage (the loss reads
        # teacher_logps itself), rlsd / expand turn it per-token on the worker.
        if rl_config.advantage_reweight == 'rlsd':
            if rl_config.rlsd_lambda_warmup_steps > 0 and global_step < rl_config.rlsd_lambda_warmup_steps:
                lam = rl_config.rlsd_lambda * global_step / rl_config.rlsd_lambda_warmup_steps
            elif rl_config.rlsd_lambda_decay_steps > 0 and global_step >= rl_config.rlsd_lambda_warmup_steps:
                progress = ((global_step - rl_config.rlsd_lambda_warmup_steps)
                            / rl_config.rlsd_lambda_decay_steps)
                lam = rl_config.rlsd_lambda * max(1.0 - progress, 0.0)
            else:
                lam = rl_config.rlsd_lambda
            advantages = cluster.worker.reweight_advantages(
                advantages, old_logps, teacher_logps, mode='rlsd', lam=lam,
                clip_range=rl_config.rlsd_reweight_clip_range, negative_only=rl_config.rlsd_negative_only)
        elif rl_config.sdar_loss_coef <= 0:
            advantages = cluster.worker.reweight_advantages(
                advantages, old_logps, teacher_logps, mode='expand',
                teacher_kl_coef=rl_config.teacher_kl_coef)
    return advantages, old_logps, ref_logps, teacher_logps


def _prepare_routing_replay(model: Any, rl_config: 'RLConfig', samples: List[Any], train_batch_size: int) -> None:
    """Establish each sample's MoE expert routing BEFORE the training forward replays it, in place.

    ``R3`` requires the sampler-exported routing (fail loudly when the engine exported none);
    ``R2`` RECORDS the training model's own routing under the just-synced weights, chunked the same
    way a scoring pass is (slice_dp needs >= dp_size rows per forward).
    """
    # TODO(dev2): bridge until the recipe/common batch migrates the chunking / copy helpers
    from swift.dev.recipe.common.scoring import copy_inputs, score_chunks

    mode = rl_config.router_replay_mode
    if mode == 'disabled':
        return
    if mode == 'R3':
        missing = sum(1 for sample in samples if sample.encoded.get('routed_experts') is None)
        if missing:
            raise RuntimeError(
                f'router_replay_mode="R3" replays the sampler-reported MoE routing, but {missing} of '
                f'{len(samples)} rollout samples carry none: the sampling engine did not export '
                'routed_experts. Use router_replay_mode="R2" to RECORD the routing on the training '
                'model instead, or "disabled" to recompute it.')
        return
    if mode != 'R2':
        raise ValueError(f'Unknown router_replay_mode {mode!r}; expected one of disabled/R2/R3.')
    features = [sample.encoded for sample in samples]
    rows: List[Any] = []
    for chunk in score_chunks(features, train_batch_size):
        out = model.forward_only(inputs=copy_inputs(chunk), router_replay_mode='R2')
        recorded = out.get('routed_experts') if isinstance(out, dict) else None
        if recorded is None:
            raise RuntimeError(
                'router_replay_mode="R2" asked the training model to RECORD its MoE routing, but the '
                'forward returned none. The policy must be an MoE model built with routing replay '
                'enabled; a dense policy has no expert routing to record (use "disabled").')
        for tensor in (recorded if isinstance(recorded, (list, tuple)) else [recorded]):
            for index in range(tensor.shape[0]):
                rows.append(tensor[index])
    if len(rows) != len(samples):
        raise RuntimeError(f'R2 routing RECORD returned {len(rows)} rows for {len(samples)} samples; '
                           'per-row routing alignment is impossible.')
    for sample, routed in zip(samples, rows):
        length = len(sample.encoded['input_ids'])
        if routed.shape[0] < length:
            raise RuntimeError(f'R2 recorded routing covers {routed.shape[0]} tokens for a {length}-token '
                               'sequence; the RECORD forward must route the whole sequence.')
        sample.encoded['routed_experts'] = routed[:length]
        sample.encoded['replay_mask'] = sample.encoded['completion_mask']


def _forward_kwargs(rl_config: 'RLConfig', ga: int, mini: Any, advantages: Any, old_logps: Any,
                    ref_logps: Any, teacher_logps: Any) -> Dict[str, Any]:
    """One mini-batch's parallel-list ``forward_backward`` kwargs (CHORD's explicit loss controls excluded)."""
    # TODO(dev2): bridge until the recipe/common batch migrates the copy-inputs helper
    from swift.dev.recipe.common.scoring import copy_inputs

    kwargs: Dict[str, Any] = {
        'inputs': copy_inputs([sample.encoded for sample in mini.samples]),
        'gradient_accumulation_steps': ga,
        'advantages': advantages,
        'old_logps': old_logps,
        'rollout_logps': [list(sample.rollout_logps) for sample in mini.samples],
        'truncated': [bool(rl_config.overlong_filter and getattr(sample, 'truncated', False))
                      for sample in mini.samples],
    }
    if ref_logps is not None:
        kwargs['ref_logps'] = ref_logps
    if teacher_logps is not None and rl_config.sdar_loss_coef > 0:
        kwargs['teacher_logps'] = teacher_logps
    if rl_config.router_replay_mode != 'disabled':
        kwargs['router_replay_mode'] = rl_config.router_replay_mode
    if rl_config.enable_sampling_replay:
        kwargs['sampling_masks'] = [sample.sampling_mask for sample in mini.samples]
    return kwargs


def _train_mini_batches(state: _StepState, rl_config: 'RLConfig', samples: List[Any], *,
                        train_batch_size: int, advantages: Any, old_logps: Any, ref_logps: Any,
                        teacher_logps: Any, chord_fn: Optional[Callable[[], List[Any]]] = None,
                        ema_manager: Any = None, save: Optional[Callable[[str], Any]] = None) -> int:
    """Train one pull's mini-batches (replays x chunks); returns the optimizer steps actually run."""
    model = state.model
    ga = state.ga
    if len(samples) < train_batch_size:
        raise RuntimeError(
            f'rollout produced {len(samples)} samples but train_batch_size={train_batch_size} '
            '(per_device_train_batch_size * dp_size), so no full mini-batch fits and no optimizer '
            'step is possible. Enlarge the prompt dataset or num_generations, or lower '
            'per_device_train_batch_size / the DP world size.')
    minis = [samples[start:start + train_batch_size]
             for start in range(0, len(samples) - train_batch_size + 1, train_batch_size)]
    dropped = len(samples) - len(minis) * train_batch_size
    if dropped:
        logger.warning('Dropping %d of %d rollout samples that do not fill a train_batch_size=%d mini-batch.',
                       dropped, len(samples), train_batch_size)
    optimizer_steps = 0
    for _ in range(max(1, rl_config.num_iterations)):
        for index, mini in enumerate(minis):
            state.micro_step += 1
            lo, hi = index * train_batch_size, (index + 1) * train_batch_size
            kwargs = _forward_kwargs(
                rl_config, ga, mini, advantages[lo:hi], old_logps[lo:hi],
                ref_logps[lo:hi] if ref_logps is not None else None,
                teacher_logps[lo:hi] if teacher_logps is not None else None)
            if chord_fn is not None:
                # CHORD's dual RPC: the two losses accumulate on ONE gradient (weights sum to 1),
                # so only the second RPC finalizes; the overlap guard below is the collision backstop.
                overlap = {'loss_key', 'loss_weight', 'finalize_gradients', 'kind'} & set(kwargs)
                if overlap:
                    raise ValueError(f'forward kwargs duplicate explicit loss controls: {sorted(overlap)}')
                warmup = int(rl_config.chord_mu_warmup_steps or 0)
                decay = int(rl_config.chord_mu_decay_steps or 0)
                peak = float(rl_config.chord_mu_peak or 0.0)
                valley = float(rl_config.chord_mu_valley or 0.0)
                step = state.global_step
                if warmup > 0 and step < warmup:
                    mu = peak * step / warmup
                elif decay == 0 or step >= warmup + decay:
                    mu = valley
                else:
                    mu = valley + (peak - valley) * 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / decay))
                model.forward_backward(**kwargs, loss_key='grpo', loss_weight=1.0 - mu,
                                       finalize_gradients=False)
                model.forward_backward(inputs=chord_fn(), gradient_accumulation_steps=ga, kind='sft',
                                       loss_key='sft', loss_weight=mu, finalize_gradients=True)
            else:
                model.forward_backward(**kwargs)
            boundary = state.is_grad_sync_boundary()
            model.clip_grad_and_step(max_grad_norm=state.max_grad_norm,
                                     gradient_accumulation_steps=ga)
            if boundary:
                state.record()
                optimizer_steps += 1
                if ema_manager is not None and state.global_step % rl_config.ref_model_sync_steps == 0:
                    ema_manager.sync_ema(rl_config.ref_model_mixup_alpha)
                if save is not None and state.save_steps and state.global_step % state.save_steps == 0:
                    save(f'checkpoint-{state.global_step}')
                if state.reached_max():
                    break
        if state.reached_max():
            break
    return optimizer_steps


def _colocate_loop(cluster: Any, rl_config: 'RLConfig', rollout_config: 'RolloutConfig',
                   reward_config: 'RewardConfig', dataloader: Any, state: _StepState, *,
                   train_batch_size: int, chord_fn: Optional[Callable[[], List[Any]]] = None,
                   ema_manager: Any = None, save: Optional[Callable[[str], Any]] = None) -> None:
    """The colocate branch (deep_refactor.md 71-81): a plain loop, no driver, no version face.

    The sampler time-shares one device with the trainer, so each round brackets generation with the
    four-line borrow and the two-line return below; remax and dynamic_sample regenerate HERE (the
    sampler still holds the device at group-completion time), which the driver form could not allow.
    Scoring results are written back onto the samples, the same fixed-format contract the driver
    path yields, so both branches feed the same advantage / training pipeline.
    """
    from swift.dev2.rewards.advantage import build_reward_weights

    num_generations = rl_config.num_generations
    # the orm channel's weight vector (None when no scorer is wired)
    weights = (build_reward_weights(reward_config.orm_weights, len(cluster.orm_scorers))
               if cluster.orm_scorers else None)
    entered = False
    try:
        for batch_rows in dataloader:
            # borrow the device for generation: wake the sampler's weights so the sync has somewhere
            # to write, sync the trained policy in, step the trainer aside, wake ONLY the kv cache.
            cluster.sampler.wake_up(tags=['weights'])
            cluster.engine.sync_weights(merge_and_sync=True)
            cluster.model.offload_to_cpu()
            cluster.sampler.wake_up(tags=['kv_cache'])
            entered = True
            prompts = [row['messages'] for row in batch_rows]
            extras = [{key: value for key, value in row.items() if key != 'messages'} for row in batch_rows]
            samples = cluster.rollout.generate(
                prompts, num_samples=num_generations, sampling_params=cluster.sampling_params,
                prompt_extras=extras or None)
            rewards = _score_group(cluster, samples, with_prm=True)
            for sample, reward in zip(samples, rewards):
                sample.orm_reward = reward
            if rl_config.dynamic_sample:
                # dynamic_sample: regenerate a zero-variance group (its advantage is degenerate) on
                # the same prompt, bounded by max_resample_times; the cap accept keeps the last try.
                for start in range(0, len(samples), num_generations):
                    attempts = 0
                    extra = extras[start // num_generations] or None
                    group = samples[start:start + num_generations]
                    while (attempts < rl_config.max_resample_times
                           and group_variance([s.orm_reward for s in group], weights) <= 0.0):
                        regen = cluster.rollout.generate(
                            [prompts[start // num_generations]], num_samples=num_generations,
                            sampling_params=cluster.sampling_params,
                            prompt_extras=[extra] if extra else None)
                        samples[start:start + num_generations] = regen
                        group = regen
                        for sample, reward in zip(group, _score_group(cluster, regen, with_prm=True)):
                            sample.orm_reward = reward
                        attempts += 1
            if rl_config.advantage_estimator == 'remax':
                # remax baseline per group: the greedy completion of the same prompt, scored the same
                # way, AFTER any dynamic_sample regen so the baseline lands on the final siblings.
                for start in range(0, len(samples), num_generations):
                    prompt = samples[start].prompt or prompts[start // num_generations]
                    extra = samples[start].extra or None
                    greedy = cluster.rollout.generate(
                        [prompt], num_samples=1, sampling_params=greedy_params(cluster.sampling_params),
                        prompt_extras=[extra] if extra else None)
                    if len(greedy) != 1:
                        raise RuntimeError(
                            f'remax greedy baseline returned {len(greedy)} samples, expected 1.')
                    greedy[0].prompt = list(prompt)
                    row = _score_group(cluster, [greedy[0]])
                    baseline = weighted_reward(row[0] if row else [], weights)
                    for sample in samples[start:start + num_generations]:
                        sample.baseline = baseline
            advantages, old_logps, ref_logps, teacher_logps = _advantage_pipeline(
                cluster, rl_config, reward_config, samples, train_batch_size, state.global_step)
            _prepare_routing_replay(cluster.model, rl_config, samples, train_batch_size)
            # return the device to the trainer: sleep the sampler, then reload the trainer's weights.
            cluster.sampler.sleep(level=rollout_config.sleep_level or 1)
            cluster.model.reload_to_gpu()
            entered = False
            _train_mini_batches(state, rl_config, samples, train_batch_size=train_batch_size,
                                advantages=advantages, old_logps=old_logps, ref_logps=ref_logps,
                                teacher_logps=teacher_logps, chord_fn=chord_fn, ema_manager=ema_manager,
                                save=save)
            if state.reached_max():
                break
    finally:
        if entered:
            # ended mid-window (a budget hit during generation): give the device back so the final
            # save finds the model resident.
            cluster.sampler.sleep(level=rollout_config.sleep_level or 1)
            cluster.model.reload_to_gpu()


def _disaggregated_loop(cluster: Any, rl_config: 'RLConfig', rollout_config: 'RolloutConfig',
                        reward_config: 'RewardConfig', tuner_config: Optional['TunerConfig'],
                        dataloader: Any, state: _StepState, *, train_batch_size: int,
                        num_train_epochs: float, output_dir: str,
                        chord_fn: Optional[Callable[[], List[Any]]] = None, ema_manager: Any = None,
                        save: Optional[Callable[[str], Any]] = None,
                        metrics_cells: Optional[Tuple[List[Any], List[List[int]]]] = None) -> None:
    """The disaggregated branch (deep_refactor.md 63-70): the StreamingDriver streams, the body trains.

    Per pull: the visible advantage pipeline, routing replay, the mini-batch training rounds, then
    the visible publish -- the cadence counter is inline, ``in_place`` overwrites the sampler's live
    copy (abort in-flight generations first, then sync), ``adapter_snapshot`` hands the driver the
    next version's save.
    """
    from swift.dev2.recipe.grpo.assembly import build_driver

    driver = build_driver(
        cluster=cluster, rl_config=rl_config, rollout_config=rollout_config,
        reward_config=reward_config, tuner_config=tuner_config, train_batch_size=train_batch_size,
        gradient_accumulation_steps=state.ga, prompt_dataloader=dataloader,
        num_train_epochs=num_train_epochs, max_steps=state.max_steps, output_dir=output_dir)
    if metrics_cells is not None:
        metrics_cells[0].append(driver)
    strategy = rollout_config.weight_sync_strategy
    steps_since_publish = 0
    total_steps = 0
    for samples in driver:
        if rl_config.log_completions:
            cluster.worker.log_completions(
                samples, [sample.orm_reward for sample in samples], state.global_step + 1,
                os.path.join(output_dir, 'completions.jsonl'))
        state.tracker.log_prompts([sample.prompt for sample in samples], samples,
                                  state.global_step + 1, 1)
        if metrics_cells is not None:
            # This batch's version-span aggregate, reported on every step it feeds (see the cells).
            versions = [sample.policy_version for sample in samples]
            if metrics_cells[1]:
                metrics_cells[1][0] = versions
            else:
                metrics_cells[1].append(versions)
        advantages, old_logps, ref_logps, teacher_logps = _advantage_pipeline(
            cluster, rl_config, reward_config, samples, train_batch_size, state.global_step)
        _prepare_routing_replay(cluster.model, rl_config, samples, train_batch_size)
        optimizer_steps = _train_mini_batches(
            state, rl_config, samples, train_batch_size=train_batch_size, advantages=advantages,
            old_logps=old_logps, ref_logps=ref_logps, teacher_logps=teacher_logps, chord_fn=chord_fn,
            ema_manager=ema_manager, save=save)
        # the visible publish: cadence inline, then one request; the driver brackets the push with
        # its version accounting and the adapter's decide/register (commit releases its thread).
        steps_since_publish += optimizer_steps
        publish_due = steps_since_publish >= rollout_config.parameter_sync_step
        if publish_due:
            if strategy == 'in_place':
                # in_place overwrites the sampler's single live weight copy, so the visible half of
                # the publish runs here: abort every in-flight generation, then push the weights.
                def _visible_sync(window: Any) -> None:
                    cluster.rollout.abort_all_inflight()
                    if cluster.adapter is not None:
                        cluster.adapter.publish()
                    else:
                        cluster.engine.sync_weights()

                driver.request_publish(_visible_sync)
            else:
                # adapter_snapshot: the default push (snapshot_model) saves the next version's adapter.
                driver.request_publish()
            steps_since_publish = 0
        driver.commit(optimizer_steps)
        if state.reached_max():
            break


def run_grpo(model_config: 'ModelConfig', template_config: 'TemplateConfig',
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
    """GRPO's five-segment main: prompts -> devices -> cluster -> loop scaffolding -> ONE visible loop."""
    # segment 1: prompts (the resource estimator reads the prompt lengths off the raw rows).
    prompt_rows, prompts, prompt_extras = prompt_rows_from_dataset(dataset_config)
    # segment 2: devices (configs + template + twinkle + perf; an estimate rewrites the parallel sizes).
    template, resource_plan, auxiliary_meshes = initialize_devices(
        model_config=model_config, template_config=template_config, dataset_config=dataset_config,
        train_config=train_config, distributed_config=distributed_config,
        checkpoint_config=checkpoint_config, rollout_config=rollout_config, rl_config=rl_config,
        reward_config=reward_config, tuner_config=tuner_config, logging_config=logging_config,
        quantize_config=quantize_config, megatron_config=megatron_config, moe_config=moe_config,
        output_dir=output_dir, prompt_rows=prompt_rows)
    # the optimizer's LR-scheduler horizon, derived BEFORE the cluster builds the optimizer.
    # TODO(dev2): bridge until the builders batch migrates build_dp_mesh
    from swift.dev.builders import build_dp_mesh

    dp_size = build_dp_mesh(distributed_config, template_config.sequence_parallel_size).data_world_size
    train_batch_size = train_config.per_device_train_batch_size * dp_size
    budget = rollout_step_budget(
        num_prompts=len(prompts), num_generations=rl_config.num_generations,
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        num_train_epochs=train_config.num_train_epochs,
        generation_batch_size=rollout_config.generation_batch_size,
        num_iterations=rl_config.num_iterations)
    max_steps = resolve_rollout_max_steps(train_config.max_steps, budget, recipe='run_grpo')
    # segment 3: the cluster -- every actor GRPO trains with, built explicitly, nothing hidden.
    cluster = initialize_cluster(
        model_config=model_config, template_config=template_config, template=template,
        train_config=train_config, distributed_config=distributed_config,
        checkpoint_config=checkpoint_config, rollout_config=rollout_config, rl_config=rl_config,
        reward_config=reward_config, tuner_config=tuner_config, max_steps=max_steps,
        quantize_config=quantize_config, megatron_config=megatron_config, moe_config=moe_config,
        generation_config=generation_config, resource_plan=resource_plan,
        auxiliary_meshes=auxiliary_meshes, output_dir=output_dir)
    # segment 4: dataloader + the dissolved step cadence + the save closure.
    from swift.dev2.builders import build_prompt_dataloader
    from swift.dev2.builders.optimizer import resolve_max_grad_norm

    dataloader = build_prompt_dataloader(
        prompts, prompt_extras, seed=train_config.seed,
        batch_size=(rollout_config.generation_batch_size or len(prompts)))
    # Late-bound cells: _StepState builds before the disaggregated loop creates the driver, so
    # _extra_metrics reads the driver handle and the batch's version spans through these cells.
    _driver_cell: List[Any] = []
    _last_batch_versions: List[List[int]] = []

    def _extra_metrics() -> Dict[str, Any]:
        extra: Dict[str, Any] = {}
        if _driver_cell:
            # Straight reads of the driver's own counters; the idle ratio is folded BY the driver
            # at each pull boundary (stats.last_pull_idle_ratio) -- the recipe does no timing math.
            stats = _driver_cell[0].stats
            extra.update({
                'trainer_idle_ratio': stats.last_pull_idle_ratio,
                'off_policy_consumed': stats.off_policy_consumed,
                'dropped_stale': stats.dropped_stale,
                'stream_publishes': stats.publishes,
            })
            versions = _last_batch_versions[0] if _last_batch_versions else []
            if versions:
                extra.update({
                    'partial_ratio': sum(1 for span in versions if span > 0) / len(versions),
                    'max_partial_span': max(versions),
                    'version_span_mean': sum(versions) / len(versions),
                })
        return extra

    state = _StepState(
        cluster.model, train_config,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        max_steps=max_steps, max_grad_norm=resolve_max_grad_norm(train_config),
        logging_config=logging_config, output_dir=output_dir,
        save_steps=checkpoint_config.save_steps,
        manual_gc=bool(megatron_config and megatron_config.manual_gc),
        manual_gc_steps=megatron_config.manual_gc_steps if megatron_config else 0,
        empty_cache_steps=train_config.torch_empty_cache_steps,
        extra_metrics_fn=_extra_metrics)

    def save(name: str) -> str:
        """One training checkpoint (weights + trainer state + rotation), then the save callbacks."""
        # TODO(dev2): bridge until the recipe/common batch migrates the checkpoint writer
        from swift.dev.recipe.common.loop_utils import save_training_checkpoint

        save_training_checkpoint(
            cluster.model, name, output_dir=output_dir, consumed_train_samples=state.global_step,
            component_states={},
            no_save_optim=(checkpoint_config.no_save_optim or checkpoint_config.save_only_model),
            no_save_rng=(checkpoint_config.no_save_rng or checkpoint_config.save_only_model),
            safe_serialization=checkpoint_config.safe_serialization,
            max_shard_size=checkpoint_config.max_shard_size,
            save_total_limit=checkpoint_config.save_total_limit)
        ckpt_dir = os.path.join(output_dir, name)
        state.handler.on_save(checkpoint_dir=ckpt_dir)
        return ckpt_dir

    ema_manager = None
    if rl_config.sync_ref_model and cluster.reference is not None and hasattr(cluster.reference, 'model'):
        # EMA pair: policy rank k <-> reference rank k, synced every ref_model_sync_steps.
        from twinkle.checkpoint_engine import CheckpointEngineManager

        ema_manager = CheckpointEngineManager(
            model=cluster.model, sampler=cluster.reference.model, platform='GPU', mode='standalone')
    chord_fn = None
    if rl_config.chord_sft_dataset:
        # TODO(dev2): bridge until the recipe batch migrates the CHORD feature loader
        from swift.dev.recipe.grpo.run_grpo import _load_chord_features

        chord_features = _load_chord_features(rl_config, dataset_config, template)
        chord_train_batch_size = int(rl_config.chord_sft_per_device_train_batch_size or 0) * (
            train_batch_size // max(1, train_config.per_device_train_batch_size))
        chord_offset = [0]

        def chord_fn() -> List[Any]:  # noqa: F811
            block = [deepcopy(chord_features[(chord_offset[0] + i) % len(chord_features)])
                     for i in range(chord_train_batch_size)]
            chord_offset[0] += chord_train_batch_size
            return block

    resume_dir = checkpoint_config.resume_from_checkpoint
    if resume_dir:
        # Full-param resume weights already loaded; the state seeds micro_step = cur_step (grad-sync).
        # TODO(dev2): bridge until the builders batch migrates the resume guards
        from swift.dev.builders import is_megatron_backend

        from swift.dev.recipe.common.loop_utils import num_optimizer_steps

        resume_kwargs = {'resume_only_model': checkpoint_config.resume_only_model}
        if tuner_config is not None:
            resume_kwargs['adapter_name'] = 'default'
        if is_megatron_backend(distributed_config):
            resume_kwargs.update(
                no_load_optim=checkpoint_config.no_load_optim, no_load_rng=checkpoint_config.no_load_rng,
                data_parallel_random_init=bool(megatron_config
                                               and megatron_config.data_parallel_random_init))
        trainer_state_path = os.path.join(resume_dir, 'trainer_state.json')
        if not os.path.isfile(trainer_state_path):
            if not checkpoint_config.resume_only_model:
                raise FileNotFoundError(
                    f'{trainer_state_path} is missing, so this model-only checkpoint cannot resume '
                    'optimizer, scheduler, RNG, or progress state. Set --resume_only_model true to '
                    'start a new run from its weights, or resume from a checkpoint saved without '
                    '--no_save_optim/--save_only_model.')
            if resume_kwargs.get('adapter_name'):
                cluster.model.load(resume_dir, adapter_name=resume_kwargs['adapter_name'])
            state.global_step = 0
        else:
            trainer_state = cluster.model.resume_from_checkpoint(resume_dir, **resume_kwargs)
            state.micro_step = int(trainer_state['cur_step'])
            state.global_step = num_optimizer_steps(
                int(trainer_state['cur_step']), train_config.gradient_accumulation_steps)
    # TODO(dev2): bridge until the recipe/common batch migrates filter_perf_actors
    from twinkle.perf import Perf

    from swift.dev.recipe.common.loop_utils import filter_perf_actors

    if Perf.initialize(filter_perf_actors({
            'model': cluster.model,
            'sampler': cluster.sampler,
            'reference': cluster.reference,
            'teacher': cluster.teacher,
    })):
        logger.info('[perf] profiling enabled for the online GRPO loop.')
    # segment 5: ONE visible loop, picked by placement.
    state.handler.on_train_begin()
    try:
        if cluster.colocate:
            _colocate_loop(cluster, rl_config, rollout_config, reward_config, dataloader, state,
                           train_batch_size=train_batch_size, chord_fn=chord_fn,
                           ema_manager=ema_manager, save=save)
        else:
            _disaggregated_loop(cluster, rl_config, rollout_config, reward_config, tuner_config,
                                dataloader, state, train_batch_size=train_batch_size,
                                num_train_epochs=train_config.num_train_epochs, output_dir=output_dir,
                                chord_fn=chord_fn, ema_manager=ema_manager, save=save,
                                metrics_cells=(_driver_cell, _last_batch_versions))
    finally:
        Perf.shutdown()
        # MultiTurnRollout owns only the sandbox env pool; the sampler's GPU memory is the
        # Cluster's to release (a Ray-deployed rollout is a proxy whose sampler lives worker-side).
        cluster.rollout.close()
        cluster.sampler.shutdown()
        state.tracker.close()
    state.handler.on_train_end()
    if _save_final:
        # the final checkpoint is synchronous; its args.json is written by initialize_devices' copy
        cluster.model.wait_async_saves()
        save('checkpoint-final')
    return state.history
