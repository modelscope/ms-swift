# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared policy training with optional value estimation and critic updates."""

from __future__ import annotations
import math
import os
import sys
from typing import TYPE_CHECKING, Callable, Optional

from twinkle.perf import span as perf_span

from swift.dev.config.adapter_config import has_adapter
from swift.dev.config.process import normalize_rl_type, policy_replay_count
from swift.dev.recipe.batching import balance_lanes, balance_update
from swift.dev.recipe.chord import chord_forward_backward, expert_batches, load_chord_features
from swift.dev.recipe.configs import RecipeConfigs
from swift.dev.recipe.online import fit_rollout
from swift.dev.recipe.policy.assembly import build_policy_driver, build_policy_objects, policy_group_requests
from swift.dev.recipe.resources import PolicyResources
from swift.dev.recipe.utils import (
    RolloutStepState,
    _log_line,
    build_save_knobs,
    collect,
    effective_colocate,
    filter_perf_actors,
    finalize_rl_resources,
    initialize_rl_devices,
    optimizer_batch_slices,
    prompt_rows_from_dataset,
    resolve_generation_batch_size,
    resolve_rollout_max_steps,
    resume_training,
    rollout_step_budget,
    save_training_checkpoint,
    save_training_step,
    snapshot_initial_adapter,
    stream_batch_size,
)
from swift.dev.utils.logger import get_logger
from swift.dev.utils.storage import local_run_dir

if TYPE_CHECKING:
    from twinkle.metric import CompletionRewardMetric
    from twinkle.processor import RoutingReplay
    from twinkle_agentic.async_rl import StreamingDriver

    from swift.dev.config import CheckpointConfig, RLConfig
    from swift.dev.recipe.online import TrainingRows
    from swift.dev.recipe.preparation import TrainingModel

logger = get_logger()


def _check_policy_value_step_alignment(global_step: int, value_state: Optional[dict]) -> None:
    if value_state is not None and (
        int(value_state.get("consumed_train_samples", 0)) != global_step
        or ("cur_step" in value_state and int(value_state["cur_step"]) != global_step)
    ):
        raise ValueError("PPO policy and value checkpoints contain different completed-step counts.")


def _save_value_checkpoint(
    model: TrainingModel,
    checkpoint_dir: str,
    checkpoint: CheckpointConfig,
    global_step: int,
    *,
    async_save: bool = False,
) -> None:
    save_training_checkpoint(
        model,
        "value_model",
        output_dir=checkpoint_dir,
        consumed_train_samples=global_step,
        component_states={},
        no_save_optim=checkpoint.no_save_optim or checkpoint.save_only_model,
        no_save_rng=checkpoint.no_save_rng or checkpoint.save_only_model,
        safe_serialization=checkpoint.safe_serialization,
        max_shard_size=checkpoint.max_shard_size,
        save_total_limit=checkpoint.save_total_limit,
        async_save=async_save,
    )


def _forward_kwargs(rl_config: RLConfig, update_trajs, temperature: float, *, micro_batch_size: int) -> dict:
    from swift.dev.utils.scoring import copy_inputs

    kwargs = dict(
        inputs=copy_inputs(update_trajs),
        gradient_accumulation_steps=1,
        micro_batch_size=micro_batch_size,
        temperature=temperature,
    )
    if rl_config.router_replay_mode != "disabled":
        kwargs["router_replay_mode"] = rl_config.router_replay_mode
    return kwargs


def _prepare_routing_replay(
    model: TrainingModel,
    routing: Optional[RoutingReplay],
    rl_config: RLConfig,
    trajs: TrainingRows,
    train_batch_size: int,
    batching_kwargs: Optional[dict] = None,
) -> TrainingRows:
    from swift.dev.utils.scoring import copy_inputs

    mode = rl_config.router_replay_mode
    if mode == "disabled":
        return trajs
    if routing is None:
        raise ValueError("Routing replay requires its RoutingReplay worker.")
    if mode == "R3":
        routing.verify(trajs)
        return trajs
    if mode != "R2":
        raise ValueError(f"Unknown router_replay_mode {mode!r}.")
    out = model.forward_only(
        inputs=copy_inputs(trajs),
        router_replay_mode="R2",
        **{
            "micro_batch_size": max(1, train_batch_size // model.device_mesh.data_world_size),
            **(batching_kwargs or {}),
        },
    )
    return list(collect(routing(trajs, out))[0])


def _train_mini_batches(
    state: RolloutStepState,
    rl_config: RLConfig,
    trajs: list,
    lengths: list[int],
    *,
    train_batch_size: int,
    temperature: float,
    repeats: int,
    batching_kwargs: Optional[dict] = None,
    critic: Optional[TrainingModel] = None,
    chord_fn: Optional[Callable] = None,
    ema_manager=None,
    save=None,
    group_size: int = 1,
) -> int:
    """Replay continuously; record an update only after every participating optimizer succeeds."""
    model, dp = state.model, state.model.device_mesh.data_world_size
    update_size = train_batch_size * state.ga
    if group_size > 1 and (len(trajs) % group_size or train_batch_size % (dp * group_size)):
        raise ValueError("REAL update and per-rank micro-batches must contain complete generation groups.")
    total = len(trajs) * max(1, repeats)
    if total % update_size:
        logger.warning("Dropping %d policy sample occurrences from the final replay tail.", total % update_size)
    steps = 0
    for lo, hi in optimizer_batch_slices(total, train_batch_size, state.ga):
        if state.reached_max() or state.should_stop():
            break
        indices = [i % len(trajs) for i in range(lo, hi)]
        original = [trajs[i] for i in indices]
        update, metrics = balance_update(
            state,
            original,
            [lengths[i] for i in indices],
            group_size=group_size,
            micro_batch_kwargs={"micro_batch_size": train_batch_size // dp, **(batching_kwargs or {})},
        )
        kwargs = _forward_kwargs(rl_config, update, temperature, micro_batch_size=train_batch_size // dp)
        kwargs.update(batching_kwargs or {})
        micro_batches = (hi - lo) // train_batch_size
        state.handler.on_step_begin()
        if chord_fn is not None:
            chord_forward_backward(
                model, rl_config, kwargs, chord_fn, step=state.global_step, num_micro_batches=micro_batches
            )
        else:
            model.forward_backward(**kwargs)
        state.handler.on_pre_optimizer_step()
        model.clip_grad_and_step(max_grad_norm=state.max_grad_norm, gradient_accumulation_steps=1)
        if critic is not None:
            critic.forward_backward(
                inputs=original, gradient_accumulation_steps=1, task="value", micro_batch_size=train_batch_size // dp
            )
            critic.clip_grad_and_step(max_grad_norm=state.max_grad_norm, gradient_accumulation_steps=1)
        state.handler.on_optimizer_step()
        state.micro_step += micro_batches
        state.record(extra_metrics=metrics)
        if critic is None:
            state.record_rollout_training(indices)
            state.rollout_memory_metrics()
        steps += 1
        if ema_manager is not None and state.global_step % rl_config.ref_model_sync_steps == 0:
            ema_manager.sync_ema(rl_config.ref_model_mixup_alpha)
        state.step_tail(save=save)
    return steps


def _prepare_policy_batch(
    resources: PolicyResources,
    configs: RecipeConfigs,
    state: RolloutStepState,
    trajs: list,
    facts: dict,
    *,
    train_batch_size: int,
    batching: dict,
):
    """Resolve all selected signal lanes once, restoring input order before optimizer replay."""
    rl = configs.rl
    assert rl is not None
    scores = facts["scores"]
    if resources.traj_filter is not None:
        trajs, kept = resources.traj_filter.select(trajs, where=lambda row: bool(row.get("response_token_ids")))
        if not trajs:
            raise RuntimeError("the pulled batch has no response tokens to train on.")
        state.record_rollout_training(kept["indices"])
        scores = [scores[i] for i in kept["indices"]]
        facts = dict(facts, **kept)
    lengths = [p + r for p, r in zip(facts.get("prompt_lengths") or (), facts.get("response_lengths") or ())]
    temperature = float(resources.sampling_params.get("temperature", 1.0))
    lane, order, restore = balance_lanes(
        state,
        trajs,
        lengths,
        group_size=1 if resources.critic is not None else rl.num_generations,
        micro_batch_kwargs=batching,
        episode_ids=facts.get("episode_ids"),
    )
    scores = scores if order is None else [scores[i] for i in order]
    pad = (-len(lane)) % resources.model.device_mesh.data_world_size if resources.critic is None else 0
    if pad:
        dummies = resources.traj_update.pad_rows(lane[0], pad_count=pad)
        lane = list(lane) + dummies
        scores = list(scores) + [{} for _ in dummies]
    values = None
    if resources.critic is not None:
        values = resources.critic.forward_only(
            inputs=lane,
            output_frame="response",
            output_fields=("values",),
            task="value",
            micro_batch_size=batching["micro_batch_size"],
            compute_loss=False,
        )
    ref = None
    if resources.reference is not None:
        ref = resources.reference.forward_only(
            inputs=lane,
            output_frame="response",
            output_fields=("logps",),
            temperature=temperature,
            **batching,
            **({"disable_lora": True} if resources.reference is resources.model else {}),
        )
    teacher = None
    if resources.teacher is not None or resources.privileged is not None:
        model = resources.teacher if resources.teacher is not None else resources.model
        view = collect(resources.privileged(lane)) if resources.privileged is not None else lane
        teacher = model.forward_only(
            inputs=view,
            output_frame="response",
            output_fields=("logps",),
            temperature=temperature,
            **batching,
            **({"disable_lora": True} if resources.teacher is resources.model else {}),
        )
    old = None
    if rl.rollout_importance_sampling_mode or rl.log_rollout_offpolicy_metrics:
        old = resources.model.forward_only(
            inputs=lane, output_frame="response", output_fields=("logps",), temperature=temperature, **batching
        )
    targets, summary = resources.advantage(
        lane,
        ref_out=ref,
        value_out=values,
        teacher_out=teacher,
        old_out=old,
        scores=scores,
        global_step=state.global_step,
    )
    if pad:
        if len(targets) != len(lane):
            raise ValueError("DP pad strip requires the estimator to preserve row count.")
        targets = targets[:-pad]
    if restore is not None:
        targets = [targets[i] for i in restore]
        for key in ("reward_totals", "unscored_flags"):
            if summary.get(key):
                summary[key] = [summary[key][i] for i in restore]
    targets = _prepare_routing_replay(resources.model, resources.routing, rl, targets, train_batch_size, batching)
    return targets, lengths, summary, facts


def _record_reward_metric(batch_facts, *, orm_weights, reward_metric):
    import torch

    from swift.dev.utils.scoring import build_reward_weights

    rows = [((row.get("rewards") or {}).get("orm") or {}) for row in (batch_facts.get("scores") or ())]
    if not rows or not rows[0] or reward_metric is None:
        return
    shapes = {tuple(row) for row in rows}
    if len(shapes) != 1:
        raise ValueError(f"the batch carries different ORM scorer sets {sorted(shapes)}.")
    names = list(rows[0])
    weights = build_reward_weights(orm_weights, len(names))
    matrix = torch.tensor(
        [[float("nan") if value is None or value != value else float(value) for value in row.values()] for row in rows],
        dtype=torch.float32,
    ).T
    rewards = {name: [float(v) for v in column if v == v] for name, column in zip(names, matrix)}
    total = (matrix * weights.view(-1, 1)).nansum(dim=0)
    rewards["total"] = [float(v) for v in total if v == v]
    reward_metric.reset()
    reward_metric.accumulate(rewards=rewards, completion_lengths=list(batch_facts.get("response_lengths") or ()))


def _policy_extra_metrics(
    critic: Optional[TrainingModel],
    driver: Optional[StreamingDriver],
    reward_metric: CompletionRewardMetric,
    versions: list[int],
    advantage_summary: dict,
    correction: dict,
) -> dict:
    extra = reward_metric.calculate()
    extra.update({f"advantage/{key}": value for key, value in advantage_summary.items()})
    extra.update({f"rollout_correction/{key}": value for key, value in correction.items()})
    if versions:
        extra.update(
            partial_ratio=sum(v > 0 for v in versions) / len(versions),
            max_partial_span=max(versions),
            version_span_mean=sum(versions) / len(versions),
        )
    if critic is not None:
        for key, value in (critic.calculate_metric(is_training=True) or {}).items():
            extra[{"loss": "value_loss", "grad_norm": "value_grad_norm"}.get(key, key)] = value
    elif driver is not None:
        stats = driver.stats()
        extra.update(
            trainer_idle_ratio=stats["train_idle_ratio"],
            dropped_stale=stats["dropped_stale"],
            policy_version=stats["version"],
            no_slot=stats["no_slot"],
            no_slot_dropped=stats["no_slot_dropped"],
        )
    return extra


def _policy_log_line(record: dict, loss: float) -> str:
    line = _log_line(record, loss)
    for key, label, spec in (
        ("advantage/absmean", "adv", ".4g"),
        ("advantage/zero_rows", "zero_adv", ".0%"),
        ("value_loss", "value_loss", ".4f"),
        ("train/total_reward", "reward", ".4f"),
        ("train/explained_variance", "explained_variance", ".4f"),
    ):
        value = record.get(key)
        if value is not None:
            try:
                value = format(float(value), spec)
            except (TypeError, ValueError):
                value = str(value)
            line += f"  {label}={value}"
    return line


def run_policy(configs: RecipeConfigs, *, output_dir: str = "output", _save_final: bool = True) -> list[dict]:
    from twinkle.metric import CompletionRewardMetric
    from twinkle.perf import Perf

    from swift.dev.builder import build_prompt_dataloader, is_megatron_backend, resolve_rl_dp_size
    from swift.dev.builder.optimizer import resolve_max_grad_norm

    if configs.reward is None:
        from swift.dev.config import RewardConfig

        configs.reward = RewardConfig()
    rl, rollout, train = configs.rl, configs.rollout, configs.train
    checkpoint, reward = configs.checkpoint, configs.reward
    assert rl is not None and rollout is not None and checkpoint is not None and reward is not None
    normalize_rl_type(rl)
    repeats = policy_replay_count(rl)
    prompt_rows, prompts, extras = prompt_rows_from_dataset(configs.dataset, system=configs.template.system)
    template, resource_plan, meshes = initialize_rl_devices(
        configs,
        output_dir=output_dir,
        prompt_rows=prompt_rows,
        group_requests=lambda: policy_group_requests(configs),
    )
    dp = resolve_rl_dp_size(configs.distributed, configs.template.sequence_parallel_size)
    train_batch_size, ga = train.per_device_train_batch_size * dp, train.gradient_accumulation_steps
    generation_batch_size = resolve_generation_batch_size(
        rollout.generation_batch_size,
        num_prompts=len(prompts),
        num_generations=rl.num_generations,
        num_iterations=repeats,
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=ga,
    )
    batch_size = stream_batch_size(
        colocate=effective_colocate(rollout, resource_plan),
        generation_batch_size=generation_batch_size,
        num_generations=rl.num_generations,
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=ga,
    )
    epochs = math.ceil(train.num_train_epochs)
    budget = rollout_step_budget(
        num_prompts=len(prompts),
        num_generations=rl.num_generations,
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=ga,
        num_train_epochs=epochs,
        generation_batch_size=generation_batch_size,
        num_iterations=repeats,
        stream_batch_rows=batch_size,
    )
    max_steps = resolve_rollout_max_steps(train.max_steps, budget, recipe="run_policy")
    resources = build_policy_objects(
        configs,
        template,
        max_steps=max_steps,
        batch_size=batch_size,
        resource_plan=resource_plan,
        auxiliary_meshes=meshes,
        teacher_prompt=any(extra.get("teacher_prompt") for extra in extras),
    )
    remote = configs.distributed.mode == "remote"
    dataloader = build_prompt_dataloader(
        prompts,
        extras,
        seed=configs.dataset.data_seed,
        shuffle=configs.dataset.train_dataloader_shuffle,
        batch_size=generation_batch_size,
    )
    steps_per_batch = max(1, batch_size * max(1, repeats) // (train_batch_size * ga))
    driver = build_policy_driver(
        resources,
        configs,
        prompt_dataloader=dataloader,
        batch_size=batch_size,
        num_train_epochs=epochs,
        max_steps=math.ceil(max_steps / steps_per_batch),
    )
    reward_metric = CompletionRewardMetric()
    versions, advantage_summary, correction = [], {}, {}

    def extra_metrics():
        return _policy_extra_metrics(resources.critic, driver, reward_metric, versions, advantage_summary, correction)

    state = RolloutStepState(
        resources.model,
        train,
        padding_free=configs.template.padding_free,
        dp_balance=train.dp_balance,
        gradient_accumulation_steps=ga,
        max_steps=max_steps,
        max_grad_norm=resolve_max_grad_norm(train),
        logging_config=configs.logging,
        output_dir=output_dir,
        save_steps=checkpoint.save_steps,
        manual_gc=bool(configs.megatron and configs.megatron.manual_gc),
        manual_gc_steps=configs.megatron.manual_gc_steps if configs.megatron else 0,
        empty_cache_steps=train.torch_empty_cache_steps,
        extra_metrics_fn=extra_metrics,
        extra_metrics_granularity="iteration" if resources.critic is None else "step",
        log_formatter=_policy_log_line,
    )
    if not remote:
        state.setup_rollout_metrics(
            template.performance_device_groups,
            mode="colocate" if resources.colocate else rollout.resource_mode,
            parameter_sync_step=rollout.parameter_sync_step,
            num_generations=rl.num_generations,
            attributes={"adv_pipeline_scope": "policy_signal_resolution", "unavailable_metrics": ["perf/mfu/actor"]},
        )
    save_knobs = build_save_knobs(
        checkpoint, configs.dataset, configs.distributed, configs.tuner, output_dir=output_dir
    )
    snapshot_initial_adapter(resources.model, save_knobs.initial_adapter_dir)

    def save(name, is_final=False):
        path = save_training_step(
            resources.model,
            state.handler,
            name,
            save_knobs,
            consumed_train_samples=state.global_step,
            component_states={"training": {"micro_step": state.micro_step}},
            is_final=is_final,
            requantize_source=configs.model.model
            if resources.critic is None and not has_adapter(configs.tuner)
            else None,
        )
        if resources.critic is not None:
            if remote:
                resources.critic.save(
                    name,
                    save_optimizer=not (checkpoint.no_save_optim or checkpoint.save_only_model),
                    consumed_train_samples=state.global_step,
                )
            else:
                _save_value_checkpoint(
                    resources.critic,
                    path,
                    checkpoint,
                    state.global_step,
                    async_save=checkpoint.async_save and not is_final,
                )
        return path

    ema = None
    if rl.sync_ref_model and resources.reference is not None:
        from twinkle.checkpoint_engine import CheckpointEngineManager

        ema = CheckpointEngineManager(
            model=resources.model,
            sampler=resources.reference,
            mode="standalone",
            channel="ema:reference" if resources.critic is not None else "ema",
        )
    features = load_chord_features(rl, configs.dataset, template) if rl.chord_sft_dataset else []
    if checkpoint.resume_from_checkpoint:
        state.global_step, state.micro_step, _ = resume_training(
            resources.model,
            checkpoint_config=checkpoint,
            distributed_config=configs.distributed,
            megatron_config=configs.megatron,
            tuner_config=configs.tuner,
        )
        if resources.critic is not None:
            options = dict(resume_only_model=checkpoint.resume_only_model)
            if is_megatron_backend(configs.distributed):
                options.update(
                    no_load_optim=checkpoint.no_load_optim,
                    no_load_rng=checkpoint.no_load_rng,
                    data_parallel_random_init=bool(configs.megatron and configs.megatron.data_parallel_random_init),
                )
            value_state = resources.critic.resume_from_checkpoint(
                os.path.join(checkpoint.resume_from_checkpoint, "value_model"), **options
            )
            _check_policy_value_step_alignment(state.global_step, value_state)
    chord_fn = None
    if features:
        size = rl.chord_sft_per_device_train_batch_size * dp
        chord_fn = expert_batches(features, size, offset=state.micro_step * size).__next__
    batching = {"micro_batch_size": train.per_device_train_batch_size, "micro_batch_order": train.micro_batch_order}
    if train.dynamic_batching:
        batching.update(
            dynamic_batching=True,
            max_tokens_per_micro_batch=train.max_tokens_per_micro_batch,
            packing_algorithm=train.packing_algorithm,
        )
    completion_log = None
    if rl.log_completions or state.tracker.logs_prompts:
        from swift.dev.builder.streaming import CONTROL_GROUP
        from swift.dev.metric import CompletionLog

        completion_log = CompletionLog(remote_group=None if remote else CONTROL_GROUP)
    if not remote:
        Perf.initialize(
            filter_perf_actors(
                dict(
                    model=resources.model,
                    value_model=resources.critic,
                    sampler=resources.sampler,
                    reference=resources.reference,
                    teacher=resources.teacher,
                )
            )
        )

    def train_batch(rows, facts):
        versions[:] = facts.get("version_spans") or []
        with perf_span("_logging"):
            if completion_log is not None:
                logged = completion_log(
                    rows,
                    step=state.global_step + 1,
                    scores=facts.get("scores"),
                    path=os.path.join(local_run_dir(output_dir), "completions.jsonl") if rl.log_completions else None,
                )
                if state.tracker.logs_prompts:
                    state.tracker.log_prompts(logged, state.global_step + 1)
        _record_reward_metric(facts, orm_weights=reward.orm_weights, reward_metric=reward_metric)
        targets, lengths, summary, kept = _prepare_policy_batch(
            resources, configs, state, rows, facts, train_batch_size=train_batch_size, batching=batching
        )
        advantage_summary.clear()
        for source, key in (("absmean_advantage", "absmean"), ("zero_advantage_rows", "zero_rows")):
            if source in summary:
                advantage_summary[key] = summary[source]
        if resources.critic is not None:
            advantage_summary["unscored_orm_rows"] = sum(summary.get("unscored_flags") or ())
            reward_metric.reset()
            reward_metric.accumulate(
                rewards={"total": list(summary.get("reward_totals") or ())},
                completion_lengths=list(kept.get("response_lengths") or ()),
            )
        else:
            for key in ("unscored_orm_rows", "unscored_baseline_rows", "unscored_prm_rows"):
                if key in summary:
                    advantage_summary[key] = summary[key]
        correction.update(summary.get("rollout_correction") or {})
        _train_mini_batches(
            state,
            rl,
            targets,
            lengths,
            train_batch_size=train_batch_size,
            temperature=float(resources.sampling_params.get("temperature", 1.0)),
            repeats=repeats,
            batching_kwargs=batching,
            critic=resources.critic,
            chord_fn=chord_fn,
            ema_manager=ema,
            save=save,
            group_size=rl.num_generations if rl.rl_type == "real" else 1,
        )

    state.handler.on_train_begin()
    try:
        if not (state.reached_max() or state.should_stop()):
            fit_rollout(
                state,
                driver,
                resources.sampler,
                resources.engine,
                rollout,
                colocate=resources.colocate,
                companion_models=() if resources.critic is None else (resources.critic,),
                train_batch=train_batch,
                sync_every=rollout.parameter_sync_step if resources.critic is None else None,
            )
        state.handler.on_train_end()
        if _save_final:
            resources.model.wait_async_saves()
            if resources.critic is not None:
                resources.critic.wait_async_saves()
            save("checkpoint-final", is_final=True)
    finally:
        auxiliary = [
            ("adapter", resources.adapter),
            ("reference", resources.reference),
            ("teacher", resources.teacher),
            ("advantage", resources.advantage),
            ("routing", resources.routing),
            ("privileged", resources.privileged),
            ("traj_filter", resources.traj_filter),
            ("prm_segment", resources.prm_segment),
            ("traj_update", resources.traj_update),
            *((f"orm_scorers[{i}]", scorer) for i, scorer in enumerate(resources.orm_scorers)),
            *((f"prm_scorers[{i}]", scorer) for i, scorer in enumerate(resources.prm_scorers)),
        ]
        finalize_rl_resources(
            state,
            model=resources.model,
            sampler=resources.sampler,
            engine=resources.engine,
            rollout=resources.rollout,
            value_model=resources.critic,
            remote_run=remote,
            driver=driver,
            companions=(ema,),
            recipe="policy",
            auxiliary_closers=[
                (label, id(resource), close)
                for label, resource in auxiliary
                if resource is not None and callable(close := getattr(resource, "close", None))
            ],
            finish=lambda: (
                state.finish_rollout_metrics(
                    stream_stats=state.stream_stats,
                    status="failed" if sys.exc_info()[0] is not None else "loop_completed",
                )
                if not remote
                else None
            ),
        )
    return state.history
