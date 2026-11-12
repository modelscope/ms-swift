# Copyright (c) ModelScope Contributors. All rights reserved.
# GRPO's visible training main: five segments, then ONE loop over the streaming driver.
from __future__ import annotations
import math
import os
import sys
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import torch

from swift.dev.recipe.batching import plan_rollout_batch, record_rollout_step
from swift.dev.recipe.grpo.assembly import build_driver, grpo_group_requests, initialize_cluster
from twinkle.perf import span as perf_span

from swift.dev.recipe.utils import (
    RolloutStepState,
    _log_line,
    build_save_knobs,
    collect,
    initialize_rl_devices,
    optimizer_batch_slices,
    prompt_rows_from_dataset,
    resolve_generation_batch_size,
    resolve_rollout_max_steps,
    resume_training,
    rollout_step_budget,
    save_final_checkpoint,
    save_training_step,
    snapshot_initial_adapter,
    stream_batch_size,
)
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
        RewardConfig,
        RLConfig,
        RolloutConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )

logger = get_logger()


def _rlsd_lambda(rl_config: 'RLConfig', global_step: int) -> float:
    """RLSD's mixing weight at ``global_step``: linear warmup, then linear decay to 0, else constant."""
    warmup, decay = rl_config.rlsd_lambda_warmup_steps, rl_config.rlsd_lambda_decay_steps
    if warmup > 0 and global_step < warmup:
        return rl_config.rlsd_lambda * global_step / warmup
    if decay > 0 and global_step >= warmup:
        return rl_config.rlsd_lambda * max(1.0 - (global_step - warmup) / decay, 0.0)
    return rl_config.rlsd_lambda


def _advantage_pipeline(cluster: Any, rl_config: 'RLConfig', reward_config: 'RewardConfig',
                        trajs: List[Any], samples: List[Any], train_batch_size: int, global_step: int,
                        *, teacher_prompt: bool = False,
                        correction_out: Optional[Dict[str, float]] = None,
                        batching_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[Any, ...]:
    """The shared advantage pipeline: forwards and signal components called HERE, in order.

    ``trajs`` are the driver's trajectory references and ``samples`` their scored ``RolloutSample``
    (``orm_reward`` / ``baseline`` / ``prm_rewards``). Each lane (reference / teacher / policy-old)
    is ONE whole-batch ``forward_only`` call -- the batch splits are the dispatch's (``slice_dp`` per
    rank) and the kernel's (``micro_batch_size``). Every per-token step runs on its signal component
    over references (``cluster.scores`` aligns a forward to the response tokens, ``cluster.kl`` /
    ``process_reward`` / ``combine`` / ``teacher_reweight`` fold them), so no per-token lane
    materialises here until the final collect; the scalar advantage itself is computed here from the
    light samples. PRM fusion and teacher-signal reweighting are mutually exclusive per-token sources
    and PRM wins (validate rejects enabling both). ``teacher_prompt`` says the dataset carries a
    privileged ``teacher_prompt`` column. Returns ``(advantages, old_logps, ref_logps,
    teacher_logps)``.
    """
    from twinkle.advantage import compute_advantages
    from twinkle_agentic.async_rl import weighted_score

    # Every logps-bearing forward must divide logits by the SAMPLING temperature: the sampler's
    # logps are taken at that temperature, and a T=1 training forward against T-scaled rollout
    # logps skews every importance ratio by (T-1)·logp (legacy aligned this with ``logits / T``).
    temperature = float(cluster.sampling_params.get('temperature', 1.0))
    # one per-device block per rank: train_batch_size (per_device x dp) spread over the dp ranks
    micro_batch_size = max(1, train_batch_size // cluster.model.device_mesh.data_world_size)
    batching = {"micro_batch_size": micro_batch_size, **(batching_kwargs or {})}
    scores = cluster.scores
    ref_logps = teacher_logps = None
    if cluster.reference is not None:
        # a disable_adapter auxiliary IS the policy object itself (identity derives disable_lora AND
        # the policy-group placement).
        ref_out = cluster.reference.forward_only(
            inputs=trajs, **batching, temperature=temperature,
            **({'disable_lora': True} if cluster.reference is cluster.model else {}))
        ref_logps = scores(trajs, ref_out)
    if cluster.teacher is not None or teacher_prompt:
        # teacher_prompt without a dedicated teacher: the policy grades the privileged view itself.
        t_model = cluster.teacher if cluster.teacher is not None else cluster.model
        t_disable = cluster.teacher is not None and cluster.teacher is cluster.model
        view = cluster.privileged(trajs) if cluster.privileged is not None else trajs
        teacher_out = t_model.forward_only(
            inputs=view, **batching, temperature=temperature, **({'disable_lora': True} if t_disable else {}))
        teacher_logps = scores(view, teacher_out)
    rollout_logps = scores(trajs, output_key='rollout_logprobs')
    old_logps = rollout_logps
    if rl_config.rollout_importance_sampling_mode or rl_config.log_rollout_offpolicy_metrics:
        old_logps = scores(trajs, cluster.model.forward_only(inputs=trajs, **batching, temperature=temperature))
        if correction_out is not None:
            # The legacy-compatible off-policy diagnostics, from the training-side forward and the
            # sampler's own log-probs BEFORE any optimizer step this batch.
            correction_out.update(cluster.correction(old_logps, rollout_logps))
    # Fail-loudly on the dead-parameter trap: RLSD/SDAR consume the teacher signal per token, so
    # enabling either without a teacher lane (no --teacher_model, no teacher_prompt column, no
    # adapter-disabled self-teacher) would silently degrade the run to plain GRPO.
    if teacher_logps is None and (rl_config.advantage_reweight == 'rlsd' or rl_config.sdar_loss_coef > 0):
        raise ValueError(
            'advantage_reweight=rlsd / sdar_loss_coef>0 requires a teacher signal (a --teacher_model, a '
            'teacher_prompt column, or the adapter-disabled self-teacher); without one the reweighting '
            'would silently degrade to plain GRPO. Add a teacher, or drop the flag.')
    kl_values = None
    if cluster.kl is not None:
        if ref_logps is None:
            raise ValueError('kl_in_reward=True requires an active reference model.')
        kl_values = collect(cluster.kl(old_logps, ref_logps, kind='k3'))
    # None (an unparseable judge verdict) rides as nan, exactly as dev recorded it; a no-scorer
    # row (no --orm wired) stays [nan] -- a zero-column tensor cannot carry a group mean.
    reward_matrix = torch.tensor(
        [[float('nan') if value is None or value != value else float(value) for value in (sample.orm_reward or ())]
         or [float('nan')] for sample in samples], dtype=torch.float32)
    estimator = rl_config.advantage_estimator
    # remax: the baseline episode's ORM row, folded with the same weights as the group's rewards
    baselines = ([weighted_score(sample.baseline, reward_config.orm_weights) for sample in samples]
                 if any(sample.baseline is not None for sample in samples) else None)
    advantages: Any = compute_advantages(
        reward_matrix,
        rl_config.num_generations,
        reward_weights=reward_config.orm_weights,
        advantage_estimator=estimator,
        scale_rewards=rl_config.scale_rewards or 'group',
        kl_in_reward=kl_values is not None,
        beta=float(rl_config.beta or 0.0),
        kl_values=kl_values,
        response_lengths=([sample.token_lengths[1] if sample.token_lengths else 0 for sample in samples]
                          if estimator == 'opo' else None),
        f_norm=(float(rl_config.f_norm) if estimator == 'gpg' else 1.0),
        reward_baselines=baselines)
    if cluster.process_reward is not None:
        # advantage = mix * process + (1 - mix) * orm_advantage on every response token: mix=0 is
        # standard GRPO, mix=1 pure PRM.
        mix = float(reward_config.prm_orm_mix)
        process = cluster.process_reward(trajs, samples, weights=reward_config.prm_weights)
        advantages = cluster.combine([process, advantages], weights=[mix, 1.0 - mix])
    elif cluster.teacher_reweight is not None and teacher_logps is not None:
        # teacher-signal per-token reweighting; sdar keeps the scalar advantage (the loss reads
        # teacher_logps itself), rlsd / expand turn it per-token.
        if rl_config.advantage_reweight == 'rlsd':
            advantages = cluster.teacher_reweight(
                advantages, old_logps, teacher_logps, mode='rlsd', lam=_rlsd_lambda(rl_config, global_step),
                clip_range=rl_config.rlsd_reweight_clip_range, negative_only=rl_config.rlsd_negative_only)
        else:
            advantages = cluster.teacher_reweight(
                advantages, old_logps, teacher_logps, mode='expand', teacher_kl_coef=rl_config.teacher_kl_coef)
    # collect once at the training consumption point; the lanes crossed between components as references
    return tuple(collect(lane) for lane in (advantages, old_logps, ref_logps, teacher_logps))


def _prepare_routing_replay(
    model: Any,
    routing: Any,
    rl_config: 'RLConfig',
    trajs: List[Any],
    samples: List[Any],
    train_batch_size: int,
    batching_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    """Establish each trajectory's MoE expert routing BEFORE the training forward replays it.

    ``R3`` requires the sampler-exported routing (fail loudly when the engine exported none);
    ``R2`` RECORDS the training model's own routing under the just-synced weights -- ONE whole-batch
    ``forward_only`` (the dispatch splits it per rank, ``micro_batch_size`` keeps each rank's forward
    at one per-device block). ``routing`` (a ``RoutingReplay``) stitches the recorded rows into new
    trajectories where the forward handle and each trajectory resolve and returns them as references,
    so the heavy columns never pass through the driver. Returns the trajectories to train on.
    """
    from swift.dev.utils.scoring import copy_inputs

    mode = rl_config.router_replay_mode
    if mode == 'disabled':
        return trajs
    if mode == 'R3':
        missing = sum(1 for sample in samples if not sample.routing_exported)
        if missing:
            raise RuntimeError(
                f'router_replay_mode="R3" replays the sampler-reported MoE routing, but {missing} of '
                f'{len(samples)} rollout samples carry none: the sampling engine did not export '
                'routed_experts. Use router_replay_mode="R2" to RECORD the routing on the training '
                'model instead, or "disabled" to recompute it.')
        return trajs
    if mode != 'R2':
        raise ValueError(f'Unknown router_replay_mode {mode!r}; expected one of disabled/R2/R3.')
    micro_batch_size = max(1, train_batch_size // model.device_mesh.data_world_size)
    out = model.forward_only(
        inputs=copy_inputs(trajs), router_replay_mode='R2',
        **{"micro_batch_size": micro_batch_size, **(batching_kwargs or {})},
    )
    return list(collect(routing(trajs, out))[0])


def _forward_kwargs(
    rl_config: 'RLConfig',
    update_trajs: Any,
    update_samples: Any,
    advantages: Any,
    old_logps: Any,
    ref_logps: Any,
    teacher_logps: Any,
    temperature: float,
    *,
    micro_batch_size: int,
) -> Dict[str, Any]:
    """One optimizer batch's aligned kwargs; Twinkle splits the per-rank micro-batches.

    The rollout log-probs and sampling masks ride the trajectories themselves (the model reads
    ``rollout_logprobs`` / ``sampling_mask`` off its inputs).
    """
    from swift.dev.utils.scoring import copy_inputs

    kwargs: Dict[str, Any] = {
        'inputs': copy_inputs(update_trajs),
        'gradient_accumulation_steps': 1,
        "micro_batch_size": micro_batch_size,
        'advantages': advantages,
        'old_logps': old_logps,
        'truncated': [bool(rl_config.overlong_filter and sample.truncated) for sample in update_samples],
        'temperature': temperature,
    }
    if ref_logps is not None:
        kwargs['ref_logps'] = ref_logps
    if teacher_logps is not None and rl_config.sdar_loss_coef > 0:
        kwargs['teacher_logps'] = teacher_logps
    if rl_config.router_replay_mode != 'disabled':
        kwargs['router_replay_mode'] = rl_config.router_replay_mode
    return kwargs


def _chord_forward_backward(
    model: Any,
    rl_config: 'RLConfig',
    forward_kwargs: Dict[str, Any],
    chord_fn: Callable[[], List[Any]],
    *,
    step: int,
    num_micro_batches: int,
) -> None:
    """Accumulate the weighted RL and SFT losses before finalizing one CHORD update."""
    # CHORD's dual RPC: the two losses accumulate on ONE gradient (weights sum to 1),
    # so only the second RPC finalizes; the overlap guard below is the collision backstop.
    overlap = {'loss_key', 'loss_weight', 'finalize_gradients', 'kind'} & set(forward_kwargs)
    if overlap:
        raise ValueError(f'forward kwargs duplicate explicit loss controls: {sorted(overlap)}')
    warmup = int(rl_config.chord_mu_warmup_steps or 0)
    decay = int(rl_config.chord_mu_decay_steps or 0)
    peak = float(rl_config.chord_mu_peak or 0.0)
    valley = float(rl_config.chord_mu_valley or 0.0)
    if warmup > 0 and step < warmup:
        mu = peak * step / warmup
    elif decay == 0 or step >= warmup + decay:
        mu = valley
    else:
        mu = valley + (peak - valley) * 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / decay))
    # collect: a worker-side failure inside forward_backward must fail THIS step
    collect(model.forward_backward(**forward_kwargs, loss_key='grpo', loss_weight=1.0 - mu,
                                   finalize_gradients=False))
    sft_inputs = chord_fn()
    sft_micro_batch_size = len(sft_inputs) // model.device_mesh.data_world_size
    for _ in range(num_micro_batches - 1):
        sft_inputs.extend(chord_fn())
    collect(model.forward_backward(
        inputs=sft_inputs,
        gradient_accumulation_steps=1,
        kind='sft',
        micro_batch_size=sft_micro_batch_size,
        loss_key='sft',
        loss_weight=mu,
        finalize_gradients=True,
        **{
            key: forward_kwargs[key]
            for key in ("dynamic_batching", "max_tokens_per_micro_batch", "packing_algorithm")
            if key in forward_kwargs
        },
    ))


def _train_mini_batches(
    state: 'RolloutStepState',
    rl_config: "RLConfig",
    trajs: List[Any],
    samples: List[Any],
    *,
    train_batch_size: int,
    advantages: Any,
    old_logps: Any,
    ref_logps: Any,
    teacher_logps: Any,
    temperature: float,
    batching_kwargs: Optional[Dict[str, Any]] = None,
    chord_fn: Optional[Callable[[], List[Any]]] = None,
    ema_manager: Any = None,
    save: Optional[Callable[[str], Any]] = None,
) -> int:
    """Batch consecutive replays into full GA updates, dropping only the final replay tail."""
    model = state.model
    dp_size = model.device_mesh.data_world_size
    columns = (trajs, samples, advantages, old_logps, ref_logps, teacher_logps)
    if any(column is not None and len(column) != len(trajs) for column in columns):
        raise ValueError("GRPO trajectories and per-sample training signals must have the same length.")
    update_batch_size = train_batch_size * state.ga
    total_samples = len(trajs) * max(1, rl_config.num_iterations)
    dropped = total_samples % update_batch_size
    if dropped:
        logger.warning(
            f"Dropping {dropped} GRPO sample occurrences from the final replay tail "
            f"that cannot fill an optimizer batch of {update_batch_size}."
        )
    optimizer_steps = 0
    for lo, hi in optimizer_batch_slices(total_samples, train_batch_size, state.ga):
        if state.reached_max() or state.should_stop():
            return optimizer_steps
        # Repeat the traversal, including across update boundaries, without copying all replays.
        indices = [index % len(trajs) for index in range(lo, hi)]
        plan = plan_rollout_batch(
            state, [trajs[index] for index in indices],
            samples=[samples[index] for index in indices],
            micro_batch_kwargs={'micro_batch_size': train_batch_size // dp_size, **(batching_kwargs or {})})
        if plan is not None:
            indices = plan.apply(indices)
        batch = ([column[index] for index in indices] if column is not None else None for column in columns)
        kwargs = _forward_kwargs(rl_config, *batch, temperature, micro_batch_size=train_batch_size // dp_size)
        kwargs.update(batching_kwargs or {})
        micro_batches = (hi - lo) // train_batch_size  # Nominal count; DBS schedules physical batches internally.
        if chord_fn is not None:
            _chord_forward_backward(
                model, rl_config, kwargs, chord_fn, step=state.global_step, num_micro_batches=micro_batches
            )
        else:
            # collect: a worker-side failure inside forward_backward must fail THIS step
            collect(model.forward_backward(**kwargs))
        # clip_grad_and_step is declared sync=True on the model: worker exceptions surface here.
        model.clip_grad_and_step(max_grad_norm=state.max_grad_norm, gradient_accumulation_steps=1)
        state.micro_step += micro_batches
        record_rollout_step(state, plan)
        state.record_rollout_training(indices)
        # Sample worker memory for THIS iteration before step_tail can run a periodic
        # checkpoint (rows are cached on the state; later reads reuse them).
        state.rollout_memory_metrics()
        optimizer_steps += 1
        if ema_manager is not None and state.global_step % rl_config.ref_model_sync_steps == 0:
            ema_manager.sync_ema(rl_config.ref_model_mixup_alpha)
        state.step_tail(save=save)
    return optimizer_steps


def _record_reward_metric(samples: List[Any], *, orm_names: List[str], orm_weights: Any,
                          reward_metric: Any) -> None:
    """Feed one consumed batch's per-sample ORM rewards into the driver-side reward aggregate.

    The samples arrive scored (``orm_reward`` written by the driver, one column per ORM worker).
    The metric resets per call, so the ``train/<col>_reward`` keys describe exactly the batch the
    upcoming optimizer steps train on (the same CompletionRewardMetric contract PPO's loop uses). A
    no-scorer run (teacher-only OPD-RL) accumulates nothing and the keys simply stay absent.
    """
    if not orm_names or not samples or reward_metric is None:
        return
    from swift.dev.utils.scoring import build_reward_weights

    column_weights = build_reward_weights(orm_weights, len(orm_names))
    matrix = torch.tensor(
        [[float('nan') if r is None or r != r else float(r) for r in sample.orm_reward]
         for sample in samples], dtype=torch.float32).T
    rewards = {
        name: [float(v) for v in column if v == v]
        for name, column in zip(orm_names, matrix)
    }
    total = (matrix * column_weights.view(-1, 1)).nansum(dim=0)
    rewards['total'] = [float(v) for v in total if v == v]
    reward_metric.reset()
    reward_metric.accumulate(
        rewards=rewards,
        completion_lengths=[sample.token_lengths[1] if sample.token_lengths else 0 for sample in samples])


def _train_loop(
    cluster: Any,
    driver: Any,
    rl_config: 'RLConfig',
    rollout_config: 'RolloutConfig',
    reward_config: 'RewardConfig',
    state: 'RolloutStepState',
    *,
    train_batch_size: int,
    batching_kwargs: Dict[str, Any],
    output_dir: str,
    teacher_prompt: bool = False,
    chord_fn: Optional[Callable[[], List[Any]]] = None,
    ema_manager: Any = None,
    save: Optional[Callable[[str], Any]] = None,
    last_batch_versions: Optional[List[List[int]]] = None,
    correction_out: Optional[Dict[str, float]] = None,
    reward_metric: Any = None,
    completion_log: Any = None,
) -> None:
    """One loop for every placement: take a batch from the driver, train on it, sync every K batches.

    Colocate time-shares the devices: the sync (``engine.sync_weights``) wakes the sampler, pushes
    the weights, offloads the trainer and wakes the KV cache; once a batch is handed out (nothing
    is generating at staleness 0 with ``parameter_sync_step`` 1) the loop sleeps the sampler and
    reloads the trainer before the advantage forwards.
    """
    from swift.dev.recipe.offload import sleep_for_rollout

    sync_every = rollout_config.parameter_sync_step
    temperature = float(cluster.sampling_params.get('temperature', 1.0))
    # Initial publication stays outside the measured iterations: a resumed policy differs from the
    # weights the sampler loaded, and colocate hands the devices to the sampler here.
    cluster.engine.sync_weights()
    lent = cluster.colocate
    batches = 0
    try:
        state.begin_rollout_iteration()
        for trajs, samples in driver:
            if cluster.colocate:
                sleep_for_rollout(cluster.sampler, rollout_config.sleep_level)
                cluster.model.reload_to_gpu()
                lent = False
            state.observe_rollout_samples(samples)
            _record_reward_metric(
                samples, orm_names=cluster.orm_names, orm_weights=reward_config.orm_weights,
                reward_metric=reward_metric)
            # '_logging' is the excluded band in fold_iteration_metrics' phase reduction.
            with perf_span('_logging'):
                if completion_log is not None:
                    rows = completion_log(
                        trajs, samples, step=state.global_step + 1,
                        path=os.path.join(output_dir, 'completions.jsonl') if rl_config.log_completions else None)
                    if state.tracker.logs_prompts:
                        state.tracker.log_prompts(rows, state.global_step + 1)
            if last_batch_versions is not None:
                # This batch's version-span aggregate, reported on every step it feeds: version_span is
                # the per-sample lag the driver stamps at hand-out (current - admission version).
                last_batch_versions[:1] = [[sample.version_span for sample in samples]]
            advantages, old_logps, ref_logps, teacher_logps = _advantage_pipeline(
                cluster, rl_config, reward_config, trajs, samples, train_batch_size, state.global_step,
                teacher_prompt=teacher_prompt, correction_out=correction_out, batching_kwargs=batching_kwargs,
            )
            trajs = _prepare_routing_replay(cluster.model, cluster.routing, rl_config, trajs, samples,
                                            train_batch_size, batching_kwargs)
            _train_mini_batches(
                state,
                rl_config,
                trajs,
                samples,
                train_batch_size=train_batch_size,
                batching_kwargs=batching_kwargs,
                advantages=advantages,
                old_logps=old_logps,
                ref_logps=ref_logps,
                teacher_logps=teacher_logps,
                temperature=temperature,
                chord_fn=chord_fn,
                ema_manager=ema_manager,
                save=save,
            )
            batches += 1
            finished = state.reached_max() or state.should_stop()
            publish = not finished and batches % sync_every == 0
            if publish:
                with perf_span('update_weights'):
                    cluster.engine.sync_weights()
                lent = cluster.colocate
            state.fold_iteration_metrics(publications=int(publish), terminal=finished)
            if finished:
                break
            state.begin_rollout_iteration()
    finally:
        # Release the driver thread and its episodes before the caller's teardown shuts the rollout.
        driver.close()
        if lent:
            sleep_for_rollout(cluster.sampler, rollout_config.sleep_level)
            cluster.model.reload_to_gpu()


def _load_chord_features(rl_config: 'RLConfig', dataset_config: 'DatasetConfig', template: Any) -> List[dict]:
    """Encode CHORD's expert SFT rows once; the assembler cycles over defensive copies."""
    if not rl_config.chord_sft_dataset:
        return []
    from copy import copy

    from swift.dev.builders import load_prompt_only_rows

    chord_config = copy(dataset_config)
    chord_config.dataset = list(rl_config.chord_sft_dataset)
    chord_config.val_dataset = []
    rows = load_prompt_only_rows(chord_config, None, split_dataset_ratio=0.0)
    features = [template.encode(row) for row in rows]
    if not features:
        raise ValueError('chord_sft_dataset produced no encodable rows.')
    return features


def _grpo_extra_metrics(
    driver_cell: List[Any],
    last_batch_versions: List[List[int]],
    rollout_correction: Dict[str, float],
    reward_metric: Any,
) -> Dict[str, Any]:
    """Read the current rollout, staleness, and reward aggregates for step logging."""
    extra = {f"rollout_correction/{key}": value for key, value in rollout_correction.items()}
    if driver_cell:
        # Straight reads of the driver's own counters; the idle ratio is folded BY the driver at each
        # hand-out -- the recipe does no timing math.
        stats = driver_cell[0].stats()
        extra.update(
            {
                "trainer_idle_ratio": stats['train_idle_ratio'],
                "dropped_stale": stats['dropped_stale'],
                "policy_version": stats['version'],
            }
        )
        versions = last_batch_versions[0] if last_batch_versions else []
        if versions:
            extra.update(
                {
                    "partial_ratio": sum(1 for span in versions if span > 0) / len(versions),
                    "max_partial_span": max(versions),
                    "version_span_mean": sum(versions) / len(versions),
                }
            )
    extra.update(reward_metric.calculate())
    return extra


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
    template, resource_plan, auxiliary_meshes = initialize_rl_devices(
        model_config=model_config, template_config=template_config, dataset_config=dataset_config,
        train_config=train_config, distributed_config=distributed_config,
        checkpoint_config=checkpoint_config, rollout_config=rollout_config, rl_config=rl_config,
        reward_config=reward_config, tuner_config=tuner_config, logging_config=logging_config,
        quantize_config=quantize_config, megatron_config=megatron_config, moe_config=moe_config,
        output_dir=output_dir, prompt_rows=prompt_rows,
        group_requests=(lambda: grpo_group_requests(
            rl_config, reward_config, distributed_config, tuner_config)))
    # the optimizer's LR-scheduler horizon, derived BEFORE the cluster builds the optimizer.
    from swift.dev.builders import build_dp_mesh

    dp_size = build_dp_mesh(distributed_config, template_config.sequence_parallel_size).data_world_size
    train_batch_size = train_config.per_device_train_batch_size * dp_size
    ga = train_config.gradient_accumulation_steps
    # One resolved value feeds the driver batch, the step budget and the dataloader: the derived
    # default fills whole optimizer updates (never the whole dataset -- see resolve_generation_batch_size).
    generation_batch_size = resolve_generation_batch_size(
        rollout_config.generation_batch_size, num_prompts=len(prompts),
        num_generations=rl_config.num_generations, num_iterations=rl_config.num_iterations,
        train_batch_size=train_batch_size, gradient_accumulation_steps=ga)
    colocate = rollout_config.sampler_mode == "colocate"
    batch_size = stream_batch_size(
        colocate=colocate, generation_batch_size=generation_batch_size,
        num_generations=rl_config.num_generations, train_batch_size=train_batch_size,
        gradient_accumulation_steps=ga)
    num_train_epochs = math.ceil(train_config.num_train_epochs)
    budget = rollout_step_budget(
        num_prompts=len(prompts), num_generations=rl_config.num_generations,
        train_batch_size=train_batch_size, gradient_accumulation_steps=ga,
        num_train_epochs=num_train_epochs, generation_batch_size=generation_batch_size,
        num_iterations=rl_config.num_iterations, stream_batch_rows=batch_size)
    max_steps = resolve_rollout_max_steps(train_config.max_steps, budget, recipe='run_grpo')
    # segment 3: the cluster -- every actor GRPO trains with, built explicitly, nothing hidden.
    teacher_prompt = any(extra.get('teacher_prompt') for extra in prompt_extras)
    cluster = initialize_cluster(
        model_config=model_config, template_config=template_config, template=template,
        train_config=train_config, distributed_config=distributed_config,
        checkpoint_config=checkpoint_config, rollout_config=rollout_config, rl_config=rl_config,
        reward_config=reward_config, tuner_config=tuner_config, max_steps=max_steps,
        batch_size=batch_size, quantize_config=quantize_config, megatron_config=megatron_config,
        moe_config=moe_config, generation_config=generation_config, resource_plan=resource_plan,
        auxiliary_meshes=auxiliary_meshes, teacher_prompt=teacher_prompt)
    # segment 4: dataloader + driver + the step cadence + the save closure.
    from swift.dev.builders import build_prompt_dataloader
    from swift.dev.builders.optimizer import resolve_max_grad_norm

    dataloader = build_prompt_dataloader(
        prompts, prompt_extras, seed=train_config.seed,
        shuffle=dataset_config.train_dataloader_shuffle,
        batch_size=generation_batch_size,
    )
    # A batch feeds batch_size * num_iterations / (train_batch_size * GA) optimizer steps.
    steps_per_batch = max(1, batch_size * max(1, rl_config.num_iterations) // (train_batch_size * ga))
    driver = build_driver(
        cluster=cluster, rl_config=rl_config, rollout_config=rollout_config, reward_config=reward_config,
        prompt_dataloader=dataloader, batch_size=batch_size, num_train_epochs=num_train_epochs,
        max_steps=math.ceil(max_steps / steps_per_batch))
    _driver_cell: List[Any] = [driver]
    _last_batch_versions: List[List[int]] = []
    # The last batch's legacy rollout_correction/* diagnostics (signed KL, ppl, chi2).
    rollout_correction: Dict[str, float] = {}

    # Fed from the per-sample orm_reward the driver's scoring writes -- the same
    # CompletionRewardMetric contract PPO's loop uses.
    from twinkle.metric import CompletionRewardMetric

    _reward_metric = CompletionRewardMetric()

    def _extra_metrics() -> Dict[str, Any]:
        return _grpo_extra_metrics(_driver_cell, _last_batch_versions, rollout_correction, _reward_metric)

    def _grpo_log_line(record: dict, loss: float) -> str:
        """GRPO's console head: the default line plus this round's mean total reward.

        For RL the reward, not the policy loss, is the convergence signal (a healthy run's loss
        wanders while the reward climbs), so the head line leads with it when one was computed.
        """
        line = _log_line(record, loss)
        reward = record.get('train/total_reward')
        if reward is not None:
            # The tracker may have formatted the scalar into a display string; coerce with a
            # guard -- a console head must never crash the training loop.
            try:
                reward = f'{float(reward):.4f}'
            except (TypeError, ValueError):
                reward = str(reward)
            line = f'{line}  reward={reward}'
        return line

    state = RolloutStepState(
        cluster.model, train_config,
        padding_free=template_config.padding_free,
        balance_batch=rl_config.balance_batch,
        gradient_accumulation_steps=ga,
        max_steps=max_steps, max_grad_norm=resolve_max_grad_norm(train_config),
        logging_config=logging_config, output_dir=output_dir,
        save_steps=checkpoint_config.save_steps,
        manual_gc=bool(megatron_config and megatron_config.manual_gc),
        manual_gc_steps=megatron_config.manual_gc_steps if megatron_config else 0,
        empty_cache_steps=train_config.torch_empty_cache_steps,
        extra_metrics_fn=_extra_metrics,
        log_formatter=_grpo_log_line)
    completion_log = None
    if rl_config.log_completions or state.tracker.logs_prompts:
        from swift.dev.builders.streaming import CONTROL_GROUP
        from swift.dev.metrics import CompletionLog

        completion_log = CompletionLog(remote_group=CONTROL_GROUP)

    state.setup_rollout_metrics(
        template.performance_device_groups,
        mode='colocate' if cluster.colocate else rollout_config.resource_mode,
        parameter_sync_step=rollout_config.parameter_sync_step,
        num_generations=rl_config.num_generations,
        attributes={
            'adv_pipeline_scope': 'auxiliary_forwards_and_advantage_resolution',
            'unavailable_metrics': ['perf/mfu/actor'],
        })

    save_knobs = build_save_knobs(
        checkpoint_config, dataset_config, distributed_config, tuner_config, output_dir=output_dir)
    # Before any training step: a base-modifying LoRA init (pissa ...) needs its initial snapshot.
    snapshot_initial_adapter(cluster.model, save_knobs.initial_adapter_dir)

    batching_kwargs = {"micro_batch_size": train_config.per_device_train_batch_size}
    if train_config.dynamic_batching:
        batching_kwargs.update(
            dynamic_batching=True,
            max_tokens_per_micro_batch=train_config.max_tokens_per_micro_batch,
            packing_algorithm=train_config.packing_algorithm,
        )

    def save(name: str, is_final: bool = False) -> str:
        """One training checkpoint: the twinkle-native write, the converted adapter, the hub push."""
        # Full-parameter training on a packed source exports BF16 weights. Re-quantize the
        # export back to the source's packed format so the artifact is loadable by the
        # quantized rollout engine; LoRA saves deltas and has nothing to re-quantize.
        requantize_source = (model_config.model
                             if (tuner_config is None or tuner_config.tuner == 'full') else None)
        component_states = {"training": {"micro_step": state.micro_step}}
        return save_training_step(cluster.model, state.handler, name, save_knobs,
            consumed_train_samples=state.global_step,
            component_states=component_states,
            is_final=is_final,
            requantize_source=requantize_source,
        )

    ema_manager = None
    if rl_config.sync_ref_model and cluster.reference is not None:
        # EMA pair: policy rank k <-> reference rank k, synced every ref_model_sync_steps.
        from twinkle.checkpoint_engine import CheckpointEngineManager

        ema_manager = CheckpointEngineManager(
            model=cluster.model, sampler=cluster.reference, mode='standalone')
    chord_fn = None
    chord_offset = [0]
    chord_train_batch_size = 0
    if rl_config.chord_sft_dataset:
        chord_features = _load_chord_features(rl_config, dataset_config, template)
        chord_train_batch_size = int(rl_config.chord_sft_per_device_train_batch_size or 0) * (
            train_batch_size // max(1, train_config.per_device_train_batch_size))

        def chord_fn() -> List[Any]:  # noqa: F811
            block = [deepcopy(chord_features[(chord_offset[0] + i) % len(chord_features)])
                     for i in range(chord_train_batch_size)]
            chord_offset[0] += chord_train_batch_size
            return block

    if checkpoint_config.resume_from_checkpoint:
        state.global_step, state.micro_step, _ = resume_training(
            cluster.model, checkpoint_config=checkpoint_config, distributed_config=distributed_config,
            megatron_config=megatron_config, tuner_config=tuner_config,
        )
        chord_offset[0] = state.micro_step * chord_train_batch_size
    from twinkle.perf import Perf

    from swift.dev.recipe.utils import filter_perf_actors

    if Perf.initialize(filter_perf_actors({
            'model': cluster.model,
            'sampler': cluster.sampler,
            'reference': cluster.reference,
            'teacher': cluster.teacher,
    })):
        logger.info('[perf] profiling enabled for the online GRPO loop.')
    # segment 5: ONE visible loop.
    state.handler.on_train_begin()
    try:
        if not (state.reached_max() or state.should_stop()):
            _train_loop(cluster, driver, rl_config, rollout_config, reward_config, state,
                        train_batch_size=train_batch_size,
                        batching_kwargs=batching_kwargs,
                        output_dir=output_dir,
                        teacher_prompt=teacher_prompt,
                        chord_fn=chord_fn,
                        ema_manager=ema_manager,
                        save=save,
                        last_batch_versions=_last_batch_versions,
                        correction_out=rollout_correction,
                        reward_metric=_reward_metric,
                        completion_log=completion_log,
                        )
    finally:
        try:
            state.finish_rollout_metrics(
                stream_stats=driver.stats(),
                status='failed' if sys.exc_info()[0] is not None else 'loop_completed')
        finally:
            try:
                Perf.shutdown()
            finally:
                try:
                    cluster.rollout.close()
                finally:
                    try:
                        cluster.sampler.shutdown()
                    finally:
                        state.tracker.close()
    state.handler.on_train_end()
    # The final checkpoint is synchronous; its args.json is written by initialize_rl_devices' copy.
    save_final_checkpoint(cluster.model, save, enabled=_save_final)
    return state.history
