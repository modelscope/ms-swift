# Copyright (c) ModelScope Contributors. All rights reserved.
# GRPO's visible training main: five segments, then ONE loop body picked by placement.
from __future__ import annotations
import math
import os
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from swift.dev.recipe.grpo.assembly import (
    greedy_params,
    group_variance,
    grpo_group_requests,
    initialize_cluster,
    weighted_reward,
)
from swift.dev.recipe.utils import (
    REF_GROUP,
    TEACHER_GROUP,
    StepState,
    build_save_knobs,
    initialize_rl_devices,
    prompt_rows_from_dataset,
    resolve_rollout_max_steps,
    resume_training,
    rollout_step_budget,
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


def _advantage_pipeline(cluster: Any, rl_config: 'RLConfig', reward_config: 'RewardConfig',
                        samples: List[Any], train_batch_size: int, global_step: int,
                        correction_out: Optional[Dict[str, float]] = None) -> Tuple[Any, ...]:
    """recipe 编排多路信号前向，优势计算消费分数引用。"""
    features = [sample.encoded for sample in samples]
    ref_shared = cluster.reference is cluster.model
    teacher_shared = cluster.teacher is cluster.model
    preserve = bool(rl_config.rollout_importance_sampling_mode or rl_config.log_rollout_offpolicy_metrics)
    tasks, slots = [], []
    if cluster.reference is not None:
        tasks.append((None if ref_shared else REF_GROUP, cluster.reference, features, dict(
            train_batch_size=train_batch_size,
            forward_kwargs={'disable_lora': True} if ref_shared else None)))
        slots.append('ref')
    if cluster.teacher is not None or cluster.worker.has_teacher_prompt(samples):
        privileged = cluster.worker.privileged_features(samples)
        teacher = cluster.teacher if cluster.teacher is not None else cluster.model
        tasks.append((None if cluster.teacher is None or teacher_shared else TEACHER_GROUP,
                      teacher, privileged, dict(
                          train_batch_size=train_batch_size,
                          forward_kwargs={'disable_lora': True} if teacher_shared else None)))
        slots.append('teacher')
    if preserve:
        tasks.append((None, cluster.model, features, dict(train_batch_size=train_batch_size)))
        slots.append('old')
    scored = cluster.worker.response_scores(tasks)
    signals = dict(zip(slots, (scored[index] for index in range(len(tasks)))))
    result = cluster.worker.grpo_advantages(
        samples, rl_config, reward_config, global_step, cluster.prm_orm_mix,
        old_logps=signals.get('old'), ref_logps=signals.get('ref'),
        teacher_logps=signals.get('teacher'))
    if correction_out is not None and preserve:
        correction_out.update(cluster.worker.correction_metrics(signals.get('old'), samples))
    return (result['advantages'], result['old_logps'], signals.get('ref'), signals.get('teacher'))


def _prepare_routing_replay(cluster: Any, rl_config: 'RLConfig', samples: List[Any], train_batch_size: int) -> None:
    if rl_config.router_replay_mode == 'disabled':
        return
    features = cluster.worker.prepare_routing_replay(
        cluster.model, samples, rl_config.router_replay_mode, train_batch_size)
    for index, sample in enumerate(samples):
        sample.encoded = features[index]


def _forward_kwargs(rl_config: 'RLConfig', ga: int, mini: Any, advantages: Any, old_logps: Any,
                    ref_logps: Any, teacher_logps: Any) -> Dict[str, Any]:
    """One mini-batch's parallel-list ``forward_backward`` kwargs (CHORD's explicit loss controls excluded)."""
    from swift.dev.utils.scoring import copy_inputs

    kwargs: Dict[str, Any] = {
        'inputs': copy_inputs([sample.encoded for sample in mini]),
        'gradient_accumulation_steps': ga,
        'advantages': advantages,
        'old_logps': old_logps,
        'rollout_logps': [sample.rollout_logprobs for sample in mini],
        'truncated': [bool(rl_config.overlong_filter and getattr(sample, 'truncated', False))
                      for sample in mini],
    }
    if ref_logps is not None:
        kwargs['ref_logps'] = ref_logps
    if teacher_logps is not None and rl_config.sdar_loss_coef > 0:
        kwargs['teacher_logps'] = teacher_logps
    if rl_config.router_replay_mode != 'disabled':
        kwargs['router_replay_mode'] = rl_config.router_replay_mode
    if rl_config.enable_sampling_replay:
        kwargs['sampling_masks'] = [sample.sampling_mask for sample in mini]
    return kwargs


def _train_mini_batches(state: StepState, rl_config: 'RLConfig', samples: List[Any], *,
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
                state.step_tail(save=save)
                if state.reached_max() or state.should_stop():
                    break
        if state.reached_max() or state.should_stop():
            break
    return optimizer_steps


def _colocate_loop(cluster: Any, rl_config: 'RLConfig', rollout_config: 'RolloutConfig',
                   reward_config: 'RewardConfig', dataloader: Any, state: StepState, *,
                   train_batch_size: int, chord_fn: Optional[Callable[[], List[Any]]] = None,
                   ema_manager: Any = None, save: Optional[Callable[[str], Any]] = None,
                   correction_out: Optional[Dict[str, float]] = None) -> None:
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
                cluster, rl_config, reward_config, samples, train_batch_size, state.global_step,
                correction_out=correction_out)
            _prepare_routing_replay(cluster, rl_config, samples, train_batch_size)
            # return the device to the trainer: sleep the sampler, then reload the trainer's weights.
            cluster.sampler.sleep(level=rollout_config.sleep_level or 1)
            cluster.model.reload_to_gpu()
            entered = False
            _train_mini_batches(state, rl_config, samples, train_batch_size=train_batch_size,
                                advantages=advantages, old_logps=old_logps, ref_logps=ref_logps,
                                teacher_logps=teacher_logps, chord_fn=chord_fn, ema_manager=ema_manager,
                                save=save)
            if state.reached_max() or state.should_stop():
                break
    finally:
        if entered:
            # ended mid-window (a budget hit during generation): give the device back so the final
            # save finds the model resident.
            cluster.sampler.sleep(level=rollout_config.sleep_level or 1)
            cluster.model.reload_to_gpu()


def _disaggregated_loop(cluster: Any, rl_config: 'RLConfig', rollout_config: 'RolloutConfig',
                        reward_config: 'RewardConfig', tuner_config: Optional['TunerConfig'],
                        dataloader: Any, state: StepState, *, train_batch_size: int,
                        num_train_epochs: float, output_dir: str,
                        chord_fn: Optional[Callable[[], List[Any]]] = None, ema_manager: Any = None,
                        save: Optional[Callable[[str], Any]] = None,
                        metrics_cells: Optional[Tuple[List[Any], List[List[int]]]] = None,
                        correction_out: Optional[Dict[str, float]] = None) -> None:
    """The disaggregated branch (deep_refactor.md 63-70): the StreamingDriver streams, the body trains.

    Per pull: the visible advantage pipeline, routing replay, the mini-batch training rounds, then
    the visible publish -- the cadence counter is inline, ``in_place`` overwrites the sampler's live
    copy after driver pause/drain, ``adapter_snapshot`` hands the driver the
    next version's save.
    """
    from swift.dev.recipe.grpo.assembly import build_driver

    driver = build_driver(
        cluster=cluster, rl_config=rl_config, rollout_config=rollout_config,
        reward_config=reward_config, tuner_config=tuner_config, train_batch_size=train_batch_size,
        gradient_accumulation_steps=state.ga, prompt_dataloader=dataloader,
        num_train_epochs=num_train_epochs, max_steps=state.max_steps, output_dir=output_dir)
    if metrics_cells is not None:
        metrics_cells[0].append(driver)
    strategy = rollout_config.weight_sync_strategy
    steps_since_publish = 0
    try:
        for samples in driver:
            if rl_config.log_completions:
                cluster.worker.log_completions(
                    samples, [sample.orm_reward for sample in samples], state.global_step + 1,
                    os.path.join(output_dir, 'completions.jsonl'))
            state.tracker.log_prompts([sample.prompt for sample in samples], samples,
                                      state.global_step + 1, 1)
            if metrics_cells is not None:
                versions = [sample.policy_version for sample in samples]
                if metrics_cells[1]:
                    metrics_cells[1][0] = versions
                else:
                    metrics_cells[1].append(versions)
            advantages, old_logps, ref_logps, teacher_logps = _advantage_pipeline(
                cluster, rl_config, reward_config, samples, train_batch_size, state.global_step,
                correction_out=correction_out)
            _prepare_routing_replay(cluster, rl_config, samples, train_batch_size)
            optimizer_steps = _train_mini_batches(
                state, rl_config, samples, train_batch_size=train_batch_size, advantages=advantages,
                old_logps=old_logps, ref_logps=ref_logps, teacher_logps=teacher_logps, chord_fn=chord_fn,
                ema_manager=ema_manager, save=save)
            steps_since_publish += optimizer_steps
            if steps_since_publish >= rollout_config.parameter_sync_step:
                if strategy == 'in_place':
                    def _visible_sync(window: Any) -> None:
                        if cluster.adapter is not None:
                            cluster.adapter.publish()
                        else:
                            cluster.engine.sync_weights()
                    driver.request_publish(_visible_sync)
                else:
                    driver.request_publish()
                steps_since_publish = 0
            driver.commit(optimizer_steps)
            if state.reached_max() or state.should_stop():
                break
    finally:
        driver.close()


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
    from swift.dev.builders import build_prompt_dataloader
    from swift.dev.builders.optimizer import resolve_max_grad_norm

    dataloader = build_prompt_dataloader(
        prompts, prompt_extras, seed=train_config.seed,
        batch_size=(rollout_config.generation_batch_size or len(prompts)))
    # Late-bound cells: StepState builds before the disaggregated loop creates the driver, so
    # _extra_metrics reads the driver handle and the batch's version spans through these cells.
    _driver_cell: List[Any] = []
    _last_batch_versions: List[List[int]] = []
    # The last pull's legacy rollout_correction/* diagnostics (signed KL, ppl, chi2).
    rollout_correction: Dict[str, float] = {}

    def _extra_metrics() -> Dict[str, Any]:
        extra: Dict[str, Any] = {}
        extra.update({f'rollout_correction/{key}': value for key, value in rollout_correction.items()})
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

    state = StepState(
        cluster.model, train_config,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        max_steps=max_steps, max_grad_norm=resolve_max_grad_norm(train_config),
        logging_config=logging_config, output_dir=output_dir,
        save_steps=checkpoint_config.save_steps,
        manual_gc=bool(megatron_config and megatron_config.manual_gc),
        manual_gc_steps=megatron_config.manual_gc_steps if megatron_config else 0,
        empty_cache_steps=train_config.torch_empty_cache_steps,
        extra_metrics_fn=_extra_metrics)

    save_knobs = build_save_knobs(
        checkpoint_config, dataset_config, distributed_config, tuner_config, output_dir=output_dir)
    # Before any training step: a base-modifying LoRA init (pissa ...) needs its initial snapshot.
    snapshot_initial_adapter(cluster.model, save_knobs.initial_adapter_dir)

    def save(name: str, is_final: bool = False) -> str:
        """One training checkpoint: the twinkle-native write, the converted adapter, the hub push."""
        # Full-parameter training on a packed source exports BF16 weights. Re-quantize the
        # export back to the source's packed format so the artifact is loadable by the
        # quantized rollout engine; LoRA saves deltas and has nothing to re-quantize.
        requantize_source = (model_config.model
                             if (tuner_config is None or tuner_config.tuner == 'full') else None)
        return save_training_step(cluster.model, state.handler, name, save_knobs,
                                  consumed_train_samples=state.global_step, is_final=is_final,
                                  requantize_source=requantize_source)

    ema_manager = None
    if rl_config.sync_ref_model and cluster.reference is not None:
        # EMA pair: policy rank k <-> reference rank k, synced every ref_model_sync_steps.
        from twinkle.checkpoint_engine import CheckpointEngineManager

        ema_manager = CheckpointEngineManager(
            model=cluster.model, sampler=cluster.reference, platform='GPU', mode='standalone')
    chord_fn = None
    if rl_config.chord_sft_dataset:
        chord_features = _load_chord_features(rl_config, dataset_config, template)
        chord_train_batch_size = int(rl_config.chord_sft_per_device_train_batch_size or 0) * (
            train_batch_size // max(1, train_config.per_device_train_batch_size))
        chord_offset = [0]

        def chord_fn() -> List[Any]:  # noqa: F811
            block = [deepcopy(chord_features[(chord_offset[0] + i) % len(chord_features)])
                     for i in range(chord_train_batch_size)]
            chord_offset[0] += chord_train_batch_size
            return block

    if checkpoint_config.resume_from_checkpoint:
        state.global_step, state.micro_step, _ = resume_training(
            cluster.model, checkpoint_config=checkpoint_config, distributed_config=distributed_config,
            megatron_config=megatron_config, tuner_config=tuner_config,
            gradient_accumulation_steps=train_config.gradient_accumulation_steps)
    from twinkle.perf import Perf

    from swift.dev.recipe.utils import filter_perf_actors

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
                           ema_manager=ema_manager, save=save, correction_out=rollout_correction)
        else:
            _disaggregated_loop(cluster, rl_config, rollout_config, reward_config, tuner_config,
                                dataloader, state, train_batch_size=train_batch_size,
                                num_train_epochs=train_config.num_train_epochs, output_dir=output_dir,
                                chord_fn=chord_fn, ema_manager=ema_manager, save=save,
                                metrics_cells=(_driver_cell, _last_batch_versions),
                                correction_out=rollout_correction)
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
        save('checkpoint-final', is_final=True)
    return state.history
