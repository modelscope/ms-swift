# Copyright (c) ModelScope Contributors. All rights reserved.
# GRPO's visible training main: five segments, then ONE loop body picked by placement.
from __future__ import annotations
import math
import os
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import torch

from swift.dev.recipe.grpo.assembly import (
    greedy_params,
    group_variance,
    grpo_group_requests,
    initialize_cluster,
    weighted_reward,
)
from swift.dev.recipe.utils import (
    StepState,
    build_save_knobs,
    collect,
    initialize_rl_devices,
    optimizer_batch_slices,
    prompt_rows_from_dataset,
    resolve_rollout_max_steps,
    resume_training,
    rollout_step_budget,
    save_final_checkpoint,
    save_training_step,
    snapshot_initial_adapter,
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


def _advantage_pipeline(
    cluster: Any,
    rl_config: "RLConfig",
    reward_config: "RewardConfig",
    samples: List[Any],
    train_batch_size: int,
    global_step: int,
    correction_out: Optional[Dict[str, float]] = None,
    batching_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, ...]:
    """The shared advantage pipeline: forwards called HERE, one whole-batch handle per lane.

    Reads the scoring results off the samples (``orm_reward`` / ``baseline`` / ``prm_process``, written
    by the group-completion hook or the colocate loop). Each lane (reference / teacher / policy-old)
    is ONE whole-batch ``forward_only`` call made in this loop -- the batch splits are the dispatch's
    (``slice_dp`` per rank) and the kernel's (``micro_batch_size``, one per-device block per rank),
    never a caller-side chunk loop. The teacher's privileged view rides its call's ``inputs`` as a
    reference, resolved worker-side. The handles are collected into a dict and handed to
    ``grpo_advantages`` -- key-read, row alignment, the KL fold, the group-relative advantage and
    the PRM / teacher fusion all consume them on the worker, so no per-token signal materialises in
    the driver. PRM fusion and teacher-signal reweighting are mutually exclusive per-token sources
    and PRM wins (validate rejects enabling both). Returns ``(advantages, old_logps, ref_logps,
    teacher_logps)``.
    """
    features = [sample.encoded for sample in samples]
    # Every logps-bearing forward must divide logits by the SAMPLING temperature: the sampler's
    # logps are taken at that temperature, and a T=1 training forward against T-scaled rollout
    # logps skews every importance ratio by (T-1)·logp (legacy aligned this with ``logits / T``).
    temperature = float(cluster.sampling_params.get("temperature", 1.0))
    # one per-device block per rank: train_batch_size (per_device x dp) spread over the dp ranks
    micro_batch_size = max(1, train_batch_size // cluster.model.device_mesh.data_world_size)
    batching = {"micro_batch_size": micro_batch_size, **(batching_kwargs or {})}
    scored: Dict[str, Any] = {}
    if cluster.reference is not None:
        ref_shared = cluster.reference is cluster.model
        # a disable_adapter auxiliary IS the policy object itself (identity derives disable_lora AND
        # the policy-group placement).
        scored["ref"] = cluster.reference.forward_only(
            inputs=features,
            **batching,
            temperature=temperature,
            **({"disable_lora": True} if ref_shared else {}),
        )
    if cluster.teacher is not None or any((s.extra or {}).get("teacher_prompt") for s in samples):
        if cluster.teacher is not None:
            t_shared = cluster.teacher is cluster.model
            t_model, t_disable = cluster.teacher, t_shared
        else:
            # teacher_prompt without a dedicated teacher: the policy grades the privileged view itself.
            t_model, t_disable = cluster.model, False
        privileged = cluster.worker.privileged_features(samples)
        scored["teacher"] = t_model.forward_only(
            inputs=privileged,
            **batching,
            temperature=temperature,
            **({"disable_lora": True} if t_disable else {}),
        )
    preserve = bool(rl_config.rollout_importance_sampling_mode or rl_config.log_rollout_offpolicy_metrics)
    if preserve:
        scored["old"] = cluster.model.forward_only(inputs=features, **batching, temperature=temperature)
    if preserve and correction_out is not None:
        # The legacy-compatible off-policy diagnostics, computed from the training-side forward and
        # the sampler's own log-probs BEFORE any optimizer step this pull.
        correction_out.update(cluster.worker.correction_metrics(samples, scored))
    # collect once at the training consumption point; the lanes crossed driver->worker as references
    return collect(cluster.worker.grpo_advantages(samples, scored, rl_config, reward_config, global_step))


def _prepare_routing_replay(
    model: Any,
    rl_config: "RLConfig",
    samples: List[Any],
    train_batch_size: int,
    batching_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """Establish each sample's MoE expert routing BEFORE the training forward replays it, in place.

    ``R3`` requires the sampler-exported routing (fail loudly when the engine exported none);
    ``R2`` RECORDS the training model's own routing under the just-synced weights -- ONE whole-batch
    ``forward_only`` (the dispatch splits it per rank, ``micro_batch_size`` keeps each rank's forward
    at one per-device block), collected here because the recorded rows write back onto the samples.
    """
    from swift.dev.utils.scoring import copy_inputs

    mode = rl_config.router_replay_mode
    if mode == "disabled":
        return
    if mode == "R3":
        missing = sum(1 for sample in samples if sample.encoded.get("routed_experts") is None)
        if missing:
            raise RuntimeError(
                f'router_replay_mode="R3" replays the sampler-reported MoE routing, but {missing} of '
                f"{len(samples)} rollout samples carry none: the sampling engine did not export "
                'routed_experts. Use router_replay_mode="R2" to RECORD the routing on the training '
                'model instead, or "disabled" to recompute it.'
            )
        return
    if mode != "R2":
        raise ValueError(f"Unknown router_replay_mode {mode!r}; expected one of disabled/R2/R3.")
    features = [sample.encoded for sample in samples]
    micro_batch_size = max(1, train_batch_size // model.device_mesh.data_world_size)
    out = model.forward_only(
        inputs=copy_inputs(features),
        router_replay_mode="R2",
        **{"micro_batch_size": micro_batch_size, **(batching_kwargs or {})},
    )
    recorded = out.get("routed_experts")
    if recorded is None:
        raise RuntimeError(
            'router_replay_mode="R2" asked the training model to RECORD its MoE routing, but the '
            "forward returned none. The policy must be an MoE model built with routing replay "
            'enabled; a dense policy has no expert routing to record (use "disabled").'
        )
    tensors = recorded if isinstance(recorded, (list, tuple)) else [recorded]
    rows = [tensor[index] for tensor in tensors for index in range(tensor.shape[0])]
    if len(rows) != len(samples):
        raise RuntimeError(
            f"R2 routing RECORD returned {len(rows)} rows for {len(samples)} samples; "
            "per-row routing alignment is impossible."
        )
    for sample, routed in zip(samples, rows):
        length = len(sample.encoded["input_ids"])
        if routed.shape[0] < length:
            raise RuntimeError(
                f"R2 recorded routing covers {routed.shape[0]} tokens for a {length}-token "
                "sequence; the RECORD forward must route the whole sequence."
            )
        sample.encoded["routed_experts"] = routed[:length]
        sample.encoded["replay_mask"] = sample.encoded["completion_mask"]


def _forward_kwargs(
    rl_config: "RLConfig",
    update_samples: Any,
    advantages: Any,
    old_logps: Any,
    ref_logps: Any,
    teacher_logps: Any,
    temperature: float,
    *,
    micro_batch_size: int,
) -> Dict[str, Any]:
    """One optimizer batch's aligned kwargs; Twinkle splits the per-rank micro-batches."""
    from swift.dev.utils.scoring import copy_inputs

    kwargs: Dict[str, Any] = {
        "inputs": copy_inputs([sample.encoded for sample in update_samples]),
        "gradient_accumulation_steps": 1,
        "micro_batch_size": micro_batch_size,
        "advantages": advantages,
        "old_logps": old_logps,
        "rollout_logps": [list(sample.rollout_logps) for sample in update_samples],
        "truncated": [
            bool(rl_config.overlong_filter and getattr(sample, "truncated", False)) for sample in update_samples
        ],
        "temperature": temperature,
    }
    if ref_logps is not None:
        kwargs["ref_logps"] = ref_logps
    if teacher_logps is not None and rl_config.sdar_loss_coef > 0:
        kwargs["teacher_logps"] = teacher_logps
    if rl_config.router_replay_mode != "disabled":
        kwargs["router_replay_mode"] = rl_config.router_replay_mode
    if rl_config.enable_sampling_replay:
        kwargs["sampling_masks"] = [sample.sampling_mask for sample in update_samples]
    return kwargs


def _chord_forward_backward(
    model: Any,
    rl_config: "RLConfig",
    forward_kwargs: Dict[str, Any],
    chord_fn: Callable[[], List[Any]],
    *,
    step: int,
    num_micro_batches: int,
) -> None:
    """Accumulate the weighted RL and SFT losses before finalizing one CHORD update."""
    overlap = {"loss_key", "loss_weight", "finalize_gradients", "kind"} & set(forward_kwargs)
    if overlap:
        raise ValueError(f"forward kwargs duplicate explicit loss controls: {sorted(overlap)}")
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
    model.forward_backward(**forward_kwargs, loss_key="grpo", loss_weight=1.0 - mu, finalize_gradients=False)
    sft_inputs = chord_fn()
    sft_micro_batch_size = len(sft_inputs) // model.device_mesh.data_world_size
    for _ in range(num_micro_batches - 1):
        sft_inputs.extend(chord_fn())
    model.forward_backward(
        inputs=sft_inputs,
        gradient_accumulation_steps=1,
        kind="sft",
        micro_batch_size=sft_micro_batch_size,
        loss_key="sft",
        loss_weight=mu,
        finalize_gradients=True,
        **{
            key: forward_kwargs[key]
            for key in ("dynamic_batching", "max_tokens_per_micro_batch", "packing_algorithm")
            if key in forward_kwargs
        },
    )


def _train_mini_batches(
    state: StepState,
    rl_config: "RLConfig",
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
    columns = (samples, advantages, old_logps, ref_logps, teacher_logps)
    if any(column is not None and len(column) != len(samples) for column in columns):
        raise ValueError("GRPO samples and per-sample training signals must have the same length.")
    update_batch_size = train_batch_size * state.ga
    total_samples = len(samples) * max(1, rl_config.num_iterations)
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
        indices = [index % len(samples) for index in range(lo, hi)]
        batch = ([column[index] for index in indices] if column is not None else None for column in columns)
        kwargs = _forward_kwargs(rl_config, *batch, temperature, micro_batch_size=train_batch_size // dp_size)
        kwargs.update(batching_kwargs or {})
        micro_batches = (hi - lo) // train_batch_size  # Nominal count; DBS schedules physical batches internally.
        if chord_fn is not None:
            _chord_forward_backward(
                model, rl_config, kwargs, chord_fn, step=state.global_step, num_micro_batches=micro_batches
            )
        else:
            model.forward_backward(**kwargs)
        model.clip_grad_and_step(max_grad_norm=state.max_grad_norm, gradient_accumulation_steps=1)
        state.micro_step += micro_batches
        state.record()
        optimizer_steps += 1
        if ema_manager is not None and state.global_step % rl_config.ref_model_sync_steps == 0:
            ema_manager.sync_ema(rl_config.ref_model_mixup_alpha)
        state.step_tail(save=save)
    return optimizer_steps


def _record_reward_metric(samples: List[Any], *, orm_scorers: List[Any], orm_weights: Any, reward_metric: Any) -> None:
    """Feed one consumed batch's per-sample ORM rewards into the driver-side reward aggregate.

    Both online-RL loops (the colocate plain loop and the StreamingDriver pull loop) call this with
    the batch they are about to train on -- samples arrive pre-scored (``orm_reward`` written by the
    colocate loop's ``_score_group`` or the driver's built-in group-barrier scoring, the same
    fixed-format contract). The metric resets per call, so the ``train/<col>_reward`` keys describe
    exactly the batch the upcoming optimizer steps train on (the same CompletionRewardMetric
    contract PPO's loop uses). A no-scorer run (teacher-only OPD-RL) accumulates nothing and the
    keys simply stay absent.
    """
    if not orm_scorers or not samples or reward_metric is None:
        return
    from swift.dev.rewards.advantage import build_reward_weights
    from swift.dev.utils.plugin import PluginRegistry

    column_names = [PluginRegistry.display_name(scorer) for scorer in orm_scorers]
    column_weights = build_reward_weights(orm_weights, len(orm_scorers))
    matrix = torch.tensor(
        [[float("nan") if r is None or r != r else float(r) for r in sample.orm_reward] for sample in samples],
        dtype=torch.float32,
    ).T
    rewards = {name: [float(v) for v in column if v == v] for name, column in zip(column_names, matrix)}
    total = (matrix * column_weights.view(-1, 1)).nansum(dim=0)
    rewards["total"] = [float(v) for v in total if v == v]
    reward_metric.reset()
    reward_metric.accumulate(rewards=rewards, completion_lengths=[len(sample.response_token_ids) for sample in samples])


def _resample_zero_variance_groups(
    cluster: Any,
    rl_config: "RLConfig",
    samples: List[Any],
    prompts: List[Any],
    extras: List[dict],
    weights: Any,
) -> None:
    """Regenerate zero-variance groups on their original prompts up to the configured cap."""
    num_generations = rl_config.num_generations
    for start in range(0, len(samples), num_generations):
        attempts = 0
        extra = extras[start // num_generations] or None
        group = samples[start : start + num_generations]
        while attempts < rl_config.max_resample_times and group_variance([s.orm_reward for s in group], weights) <= 0.0:
            regen = cluster.rollout.generate(
                [prompts[start // num_generations]],
                num_samples=num_generations,
                sampling_params=cluster.sampling_params,
                prompt_extras=[extra] if extra else None,
            )
            samples[start : start + num_generations] = regen
            group = regen
            for sample, reward in zip(group, _score_group(cluster, regen, with_prm=True)):
                sample.orm_reward = reward
            attempts += 1


def _set_remax_baselines(cluster, samples, prompts, extras, num_generations, weights):
    """Score one greedy baseline per group after any dynamic resampling."""
    for start in range(0, len(samples), num_generations):
        prompt = samples[start].prompt or prompts[start // num_generations]
        extra = samples[start].extra or None
        greedy = cluster.rollout.generate(
            [prompt],
            num_samples=1,
            sampling_params=greedy_params(cluster.sampling_params),
            prompt_extras=[extra] if extra else None,
        )
        if len(greedy) != 1:
            raise RuntimeError(f"remax greedy baseline returned {len(greedy)} samples, expected 1.")
        greedy[0].prompt = list(prompt)
        row = _score_group(cluster, [greedy[0]])
        baseline = weighted_reward(row[0] if row else [], weights)
        for sample in samples[start : start + num_generations]:
            sample.baseline = baseline


def _colocate_loop(
    cluster: Any,
    rl_config: "RLConfig",
    rollout_config: "RolloutConfig",
    reward_config: "RewardConfig",
    dataloader: Any,
    state: StepState,
    *,
    train_batch_size: int,
    batching_kwargs: Dict[str, Any],
    chord_fn: Optional[Callable[[], List[Any]]] = None,
    ema_manager: Any = None,
    save: Optional[Callable[[str], Any]] = None,
    correction_out: Optional[Dict[str, float]] = None,
    reward_metric: Any = None,
    output_dir: str = "output",
) -> None:
    """The colocate branch (deep_refactor.md 71-81): a plain loop, no driver, no version face.

    The sampler time-shares one device with the trainer, so each round brackets generation with the
    four-line borrow and the two-line return below; remax and dynamic_sample regenerate HERE (the
    sampler still holds the device at group-completion time), which the driver form could not allow.
    Scoring results are written back onto the samples, the same fixed-format contract the driver
    path yields, so both branches feed the same advantage / training pipeline.
    """
    from swift.dev.rewards.advantage import build_reward_weights

    num_generations = rl_config.num_generations
    # the orm channel's weight vector (None when no scorer is wired)
    weights = build_reward_weights(reward_config.orm_weights, len(cluster.orm_scorers)) if cluster.orm_scorers else None

    entered = False
    try:
        for batch_rows in dataloader:
            if state.reached_max() or state.should_stop():
                break
            # borrow the device for generation: wake the sampler's weights so the sync has somewhere
            # to write, sync the trained policy in, step the trainer aside, wake ONLY the kv cache.
            cluster.sampler.wake_up(tags=["weights"])
            cluster.engine.sync_weights(merge_and_sync=True)
            from swift.dev.recipe.offload import offload_for_rollout

            offload_for_rollout(cluster.model, rollout_config)
            cluster.sampler.wake_up(tags=["kv_cache"])
            entered = True
            prompts = [row["messages"] for row in batch_rows]
            extras = [{key: value for key, value in row.items() if key != "messages"} for row in batch_rows]
            samples = cluster.rollout.generate(
                prompts,
                num_samples=num_generations,
                sampling_params=cluster.sampling_params,
                prompt_extras=extras or None,
            )
            rewards = _score_group(cluster, samples, with_prm=True)
            for sample, reward in zip(samples, rewards):
                sample.orm_reward = reward
            _record_reward_metric(
                samples,
                orm_scorers=cluster.orm_scorers,
                orm_weights=reward_config.orm_weights,
                reward_metric=reward_metric,
            )
            if rl_config.dynamic_sample:
                _resample_zero_variance_groups(cluster, rl_config, samples, prompts, extras, weights)
            if rl_config.advantage_estimator == "remax":
                _set_remax_baselines(cluster, samples, prompts, extras, num_generations, weights)
            if rl_config.log_completions:
                cluster.worker.log_completions(
                    samples,
                    [sample.orm_reward for sample in samples],
                    state.global_step + 1,
                    os.path.join(output_dir, "completions.jsonl"),
                )
            # return the device to the trainer NOW -- before the advantage/replay forwards: the reference,
            # old-lane and routing-replay forward_only calls below run on the TRAINER's weights, so they
            # must not fire while those weights are still offloaded to CPU (the borrow window covers only
            # the sampler's generate + scoring, which megatron's cuda-side inputs cannot survive against
            # CPU-resident weights).
            from swift.dev.recipe.offload import sleep_for_rollout

            sleep_for_rollout(cluster.sampler, rollout_config.sleep_level)
            cluster.model.reload_to_gpu()
            entered = False
            advantages, old_logps, ref_logps, teacher_logps = _advantage_pipeline(
                cluster,
                rl_config,
                reward_config,
                samples,
                train_batch_size,
                state.global_step,
                correction_out=correction_out,
                batching_kwargs=batching_kwargs,
            )
            _prepare_routing_replay(cluster.model, rl_config, samples, train_batch_size, batching_kwargs)
            _train_mini_batches(
                state,
                rl_config,
                samples,
                train_batch_size=train_batch_size,
                batching_kwargs=batching_kwargs,
                advantages=advantages,
                old_logps=old_logps,
                ref_logps=ref_logps,
                teacher_logps=teacher_logps,
                temperature=float(cluster.sampling_params.get("temperature", 1.0)),
                chord_fn=chord_fn,
                ema_manager=ema_manager,
                save=save,
            )
            if state.reached_max() or state.should_stop():
                break
    finally:
        if entered:
            # ended mid-window (a budget hit during generation): give the device back so the final
            # save finds the model resident.
            from swift.dev.recipe.offload import sleep_for_rollout

            sleep_for_rollout(cluster.sampler, rollout_config.sleep_level)
            cluster.model.reload_to_gpu()


def _log_completions(cluster, rl_config, samples, state, output_dir):
    if rl_config.log_completions:
        cluster.worker.log_completions(
            samples,
            [sample.orm_reward for sample in samples],
            state.global_step + 1,
            os.path.join(output_dir, "completions.jsonl"),
        )


def _disaggregated_loop(
    cluster: Any,
    rl_config: "RLConfig",
    rollout_config: "RolloutConfig",
    reward_config: "RewardConfig",
    tuner_config: Optional["TunerConfig"],
    dataloader: Any,
    state: StepState,
    *,
    train_batch_size: int,
    batching_kwargs: Dict[str, Any],
    num_train_epochs: float,
    output_dir: str,
    chord_fn: Optional[Callable[[], List[Any]]] = None,
    ema_manager: Any = None,
    save: Optional[Callable[[str], Any]] = None,
    metrics_cells: Optional[Tuple[List[Any], List[List[int]]]] = None,
    correction_out: Optional[Dict[str, float]] = None,
    reward_metric: Any = None,
) -> None:
    """The disaggregated branch (deep_refactor.md 63-70): the StreamingDriver streams, the body trains.

    Per pull: the visible advantage pipeline, routing replay, the mini-batch training rounds, then
    the visible publish -- the cadence counter is inline, ``in_place`` overwrites the sampler's live
    copy (abort in-flight generations first, then sync), ``adapter_snapshot`` hands the driver the
    next version's save.
    """
    from swift.dev.recipe.grpo.assembly import build_driver

    if state.reached_max() or state.should_stop():
        return
    driver = build_driver(
        cluster=cluster,
        rl_config=rl_config,
        rollout_config=rollout_config,
        reward_config=reward_config,
        tuner_config=tuner_config,
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=state.ga,
        prompt_dataloader=dataloader,
        num_train_epochs=num_train_epochs,
        max_steps=state.max_steps,
        output_dir=output_dir,
    )
    if metrics_cells is not None:
        metrics_cells[0].append(driver)
    strategy = rollout_config.weight_sync_strategy
    steps_since_publish = 0
    try:
        for samples in driver:
            # The driver's built-in group-barrier scoring has already written orm_reward onto every
            # sample (the same contract the colocate loop's _score_group yields), so the reward
            # aggregate can be recorded from the pull itself -- same metric, same reset-per-batch
            # semantics as the colocate branch.
            _record_reward_metric(
                samples,
                orm_scorers=cluster.orm_scorers,
                orm_weights=reward_config.orm_weights,
                reward_metric=reward_metric,
            )
            _log_completions(cluster, rl_config, samples, state, output_dir)
            state.tracker.log_prompts([sample.prompt for sample in samples], samples, state.global_step + 1, 1)
            if metrics_cells is not None:
                # This batch's version-span aggregate, reported on every step it feeds (see the cells).
                # version_span is the per-sample lag the driver stamps at collect (current - policy_version,
                # 0 for adapter_snapshot which spans nothing) -- NOT policy_version, the absolute admission
                # version, which climbs every publish and would report a synchronous run as off-policy.
                versions = [sample.version_span for sample in samples]
                metrics_cells[1][:1] = [versions]
            advantages, old_logps, ref_logps, teacher_logps = _advantage_pipeline(
                cluster,
                rl_config,
                reward_config,
                samples,
                train_batch_size,
                state.global_step,
                correction_out=correction_out,
                batching_kwargs=batching_kwargs,
            )
            _prepare_routing_replay(cluster.model, rl_config, samples, train_batch_size, batching_kwargs)
            optimizer_steps = _train_mini_batches(
                state,
                rl_config,
                samples,
                train_batch_size=train_batch_size,
                batching_kwargs=batching_kwargs,
                advantages=advantages,
                old_logps=old_logps,
                ref_logps=ref_logps,
                teacher_logps=teacher_logps,
                temperature=float(cluster.sampling_params.get("temperature", 1.0)),
                chord_fn=chord_fn,
                ema_manager=ema_manager,
                save=save,
            )
            # the visible publish: cadence inline, then one request; the driver brackets the push with
            # its version accounting and the adapter's decide/register (commit releases its thread).
            steps_since_publish += optimizer_steps
            publish_due = steps_since_publish >= rollout_config.parameter_sync_step
            if publish_due:
                if strategy == "in_place":
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
            if state.reached_max() or state.should_stop():
                break

    finally:
        # Release the driver thread before the caller's teardown shuts the rollout/sampler:
        # an exception on the recipe thread otherwise leaves it blocked on the bridge ack.
        driver.abort()


def _load_chord_features(rl_config: "RLConfig", dataset_config: "DatasetConfig", template: Any) -> List[dict]:
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
        raise ValueError("chord_sft_dataset produced no encodable rows.")
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
        stats = driver_cell[0].stats
        extra.update(
            {
                "trainer_idle_ratio": stats.last_pull_idle_ratio,
                "off_policy_consumed": stats.off_policy_consumed,
                "dropped_stale": stats.dropped_stale,
                "stream_publishes": stats.publishes,
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


def run_grpo(
    model_config: "ModelConfig",
    template_config: "TemplateConfig",
    dataset_config: "DatasetConfig",
    train_config: "TrainConfig",
    distributed_config: "DistributedConfig",
    checkpoint_config: "CheckpointConfig",
    rollout_config: "RolloutConfig",
    rl_config: "RLConfig",
    reward_config: Optional["RewardConfig"] = None,
    tuner_config: Optional["TunerConfig"] = None,
    generation_config: Optional["GenerationConfig"] = None,
    logging_config: Optional["LoggingConfig"] = None,
    quantize_config: Optional["QuantizeConfig"] = None,
    megatron_config: Optional["MegatronConfig"] = None,
    moe_config: Optional["MoEConfig"] = None,
    *,
    output_dir: str = "output",
    _save_final: bool = True,
) -> List[dict]:
    """GRPO's five-segment main: prompts -> devices -> cluster -> loop scaffolding -> ONE visible loop."""
    # segment 1: prompts (the resource estimator reads the prompt lengths off the raw rows).
    prompt_rows, prompts, prompt_extras = prompt_rows_from_dataset(dataset_config)
    # segment 2: devices (configs + template + twinkle + perf; an estimate rewrites the parallel sizes).
    template, resource_plan, auxiliary_meshes = initialize_rl_devices(
        model_config=model_config,
        template_config=template_config,
        dataset_config=dataset_config,
        train_config=train_config,
        distributed_config=distributed_config,
        checkpoint_config=checkpoint_config,
        rollout_config=rollout_config,
        rl_config=rl_config,
        reward_config=reward_config,
        tuner_config=tuner_config,
        logging_config=logging_config,
        quantize_config=quantize_config,
        megatron_config=megatron_config,
        moe_config=moe_config,
        output_dir=output_dir,
        prompt_rows=prompt_rows,
        group_requests=(lambda: grpo_group_requests(rl_config, reward_config, distributed_config, tuner_config)),
    )
    # the optimizer's LR-scheduler horizon, derived BEFORE the cluster builds the optimizer.
    from swift.dev.builders import build_dp_mesh

    dp_size = build_dp_mesh(distributed_config, template_config.sequence_parallel_size).data_world_size
    train_batch_size = train_config.per_device_train_batch_size * dp_size
    generation_batch_size = rollout_config.generation_batch_size
    streaming = rollout_config.sampler_mode != "colocate"
    budget = rollout_step_budget(
        num_prompts=len(prompts),
        num_generations=rl_config.num_generations,
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        num_train_epochs=math.ceil(train_config.num_train_epochs) if streaming else 1,
        generation_batch_size=generation_batch_size,
        num_iterations=rl_config.num_iterations,
        streaming=streaming,
        group_pull=streaming,
    )
    max_steps = resolve_rollout_max_steps(train_config.max_steps, budget, recipe="run_grpo")
    # segment 3: the cluster -- every actor GRPO trains with, built explicitly, nothing hidden.
    cluster = initialize_cluster(
        model_config=model_config,
        template_config=template_config,
        template=template,
        train_config=train_config,
        distributed_config=distributed_config,
        checkpoint_config=checkpoint_config,
        rollout_config=rollout_config,
        rl_config=rl_config,
        reward_config=reward_config,
        tuner_config=tuner_config,
        max_steps=max_steps,
        quantize_config=quantize_config,
        megatron_config=megatron_config,
        moe_config=moe_config,
        generation_config=generation_config,
        resource_plan=resource_plan,
        auxiliary_meshes=auxiliary_meshes,
        output_dir=output_dir,
    )
    # segment 4: dataloader + the dissolved step cadence + the save closure.
    from swift.dev.builders import build_prompt_dataloader
    from swift.dev.builders.optimizer import resolve_max_grad_norm

    dataloader = build_prompt_dataloader(
        prompts,
        prompt_extras,
        seed=train_config.seed,
        shuffle=dataset_config.train_dataloader_shuffle,
        batch_size=(generation_batch_size or len(prompts)),
    )
    # Late-bound cells: StepState builds before the disaggregated loop creates the driver, so
    # _extra_metrics reads the driver handle and the batch's version spans through these cells.
    _driver_cell: List[Any] = []
    _last_batch_versions: List[List[int]] = []
    # The last pull's legacy rollout_correction/* diagnostics (signed KL, ppl, chi2).
    rollout_correction: Dict[str, float] = {}

    # BOTH online-RL branches feed this from the per-sample orm_reward their scoring writes
    # (colocate: _score_group; disaggregated: the driver's built-in group-barrier scoring) -- the same
    # CompletionRewardMetric contract PPO's loop uses.
    from twinkle.metric import CompletionRewardMetric

    _reward_metric = CompletionRewardMetric()

    def _extra_metrics() -> Dict[str, Any]:
        return _grpo_extra_metrics(_driver_cell, _last_batch_versions, rollout_correction, _reward_metric)

    state = StepState(
        cluster.model,
        train_config,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        max_steps=max_steps,
        max_grad_norm=resolve_max_grad_norm(train_config),
        logging_config=logging_config,
        output_dir=output_dir,
        save_steps=checkpoint_config.save_steps,
        manual_gc=bool(megatron_config and megatron_config.manual_gc),
        manual_gc_steps=megatron_config.manual_gc_steps if megatron_config else 0,
        empty_cache_steps=train_config.torch_empty_cache_steps,
        extra_metrics_fn=_extra_metrics,
    )

    save_knobs = build_save_knobs(
        checkpoint_config, dataset_config, distributed_config, tuner_config, output_dir=output_dir
    )
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
        requantize_source = model_config.model if (tuner_config is None or tuner_config.tuner == "full") else None
        component_states = {"training": {"micro_step": state.micro_step}}
        return save_training_step(
            cluster.model,
            state.handler,
            name,
            save_knobs,
            consumed_train_samples=state.global_step,
            component_states=component_states,
            is_final=is_final,
            requantize_source=requantize_source,
        )

    ema_manager = None
    if rl_config.sync_ref_model and cluster.reference is not None:
        # EMA pair: policy rank k <-> reference rank k, synced every ref_model_sync_steps.
        from twinkle.checkpoint_engine import CheckpointEngineManager

        ema_manager = CheckpointEngineManager(model=cluster.model, sampler=cluster.reference, mode="standalone")
    chord_fn = None
    chord_offset = [0]
    chord_train_batch_size = 0
    if rl_config.chord_sft_dataset:
        chord_features = _load_chord_features(rl_config, dataset_config, template)
        chord_train_batch_size = int(rl_config.chord_sft_per_device_train_batch_size or 0) * (
            train_batch_size // max(1, train_config.per_device_train_batch_size)
        )

        def chord_fn() -> List[Any]:  # noqa: F811
            block = [
                deepcopy(chord_features[(chord_offset[0] + i) % len(chord_features)])
                for i in range(chord_train_batch_size)
            ]
            chord_offset[0] += chord_train_batch_size
            return block

    if checkpoint_config.resume_from_checkpoint:
        state.global_step, state.micro_step, _ = resume_training(
            cluster.model,
            checkpoint_config=checkpoint_config,
            distributed_config=distributed_config,
            megatron_config=megatron_config,
            tuner_config=tuner_config,
        )
        chord_offset[0] = state.micro_step * chord_train_batch_size
    from twinkle.perf import Perf

    from swift.dev.recipe.utils import filter_perf_actors

    if Perf.initialize(
        filter_perf_actors(
            {
                "model": cluster.model,
                "sampler": cluster.sampler,
                "reference": cluster.reference,
                "teacher": cluster.teacher,
            }
        )
    ):
        logger.info("[perf] profiling enabled for the online GRPO loop.")
    # segment 5: ONE visible loop, picked by placement.
    state.handler.on_train_begin()
    try:
        if cluster.colocate:
            _colocate_loop(
                cluster,
                rl_config,
                rollout_config,
                reward_config,
                dataloader,
                state,
                train_batch_size=train_batch_size,
                batching_kwargs=batching_kwargs,
                chord_fn=chord_fn,
                ema_manager=ema_manager,
                save=save,
                correction_out=rollout_correction,
                reward_metric=_reward_metric,
                output_dir=output_dir,
            )
        else:
            _disaggregated_loop(
                cluster,
                rl_config,
                rollout_config,
                reward_config,
                tuner_config,
                dataloader,
                state,
                train_batch_size=train_batch_size,
                batching_kwargs=batching_kwargs,
                num_train_epochs=train_config.num_train_epochs,
                output_dir=output_dir,
                chord_fn=chord_fn,
                ema_manager=ema_manager,
                save=save,
                metrics_cells=(_driver_cell, _last_batch_versions),
                correction_out=rollout_correction,
                reward_metric=_reward_metric,
            )
    finally:
        Perf.shutdown()
        # MultiTurnRollout owns only the sandbox env pool; the sampler's GPU memory is the
        # Cluster's to release (a Ray-deployed rollout is a proxy whose sampler lives worker-side).
        cluster.rollout.close()
        cluster.sampler.shutdown()
        state.tracker.close()
    state.handler.on_train_end()
    save_final_checkpoint(cluster.model, save, enabled=_save_final)
    return state.history
