# Copyright (c) ModelScope Contributors. All rights reserved.
# PPO's visible training main: five segments, then ONE loop body picked by placement.
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import torch

from swift.dev.utils.logger import get_logger

from swift.dev.recipe.ppo.assembly import (
    Cluster, build_ppo_driver, initialize_ppo_cluster, ppo_group_requests)
from swift.dev.recipe.utils import (
    StepState, build_save_knobs, collect, initialize_rl_devices, prompt_rows_from_dataset,
    resolve_rollout_max_steps, resume_training, rollout_step_budget, save_training_step,
    snapshot_initial_adapter)

if TYPE_CHECKING:
    from swift.dev.config import (
        CheckpointConfig, DatasetConfig, DistributedConfig, GenerationConfig, LoggingConfig, MegatronConfig,
        ModelConfig, MoEConfig, QuantizeConfig, RLConfig, RewardConfig, RolloutConfig, TemplateConfig,
        TrainConfig, TunerConfig)

logger = get_logger()


def _check_policy_value_step_alignment(global_step: int, value_state: Optional[Dict]) -> None:
    """Refuse a PPO resume whose policy and critic checkpoints completed DIFFERENT step counts.

    PPO saves the policy and the value model as two components under one checkpoint dir; a crash between
    the two writes leaves them a step apart, and resuming such a pair would train a policy whose rollout
    history no longer matches its critic's value baseline. ``global_step`` is the count restored from the
    policy state, ``value_state`` the critic's reloaded state -- None when there is no value component to
    cross-check (a policy-only resume), so the guard is skipped then. Pure and dependency-free so the
    invariant stays testable without a GPU / Ray build.
    """
    if value_state is not None and int(value_state.get('consumed_train_samples', 0)) != global_step:
        raise ValueError('PPO policy and value checkpoints contain different completed-step counts.')


def _gae_pipeline(cluster: Cluster, reward_config: 'RewardConfig', rl_config: 'RLConfig',
                  samples: List[Any], train_batch_size: int, reward_metric: Any
                  ) -> Tuple[List[Any], List[List[float]], List[List[float]], List[List[float]], List[List[float]]]:
    """The shared GAE anchors: forwards called HERE, handles consumed on the worker.

    The critic's values and the reference's logps are two whole-batch ``forward_only`` calls made
    in this loop (the dispatch splits each per rank; ``micro_batch_size`` keeps each rank's forward
    at one per-device block); the handles are handed to ``gae`` / ``reward_metrics`` -- key-read,
    row-aligned and folded on the worker, so the per-token rewards and the GAE anchors never touch
    the driver. Returns ``(kept, advantages, returns, values, old_logps)``.
    """
    features = [sample.encoded for sample in samples]
    if not features:
        raise RuntimeError('the pulled batch has no response tokens to train on.')
    # one per-device block per rank: train_batch_size (per_device x dp) spread over the dp ranks
    micro_batch_size = max(1, train_batch_size // cluster.model.device_mesh.data_world_size)
    # The reference's logps must be taken at the SAMPLING temperature, like every logps the ratio
    # math compares against (the sampler's own logps are T-scaled; a T=1 forward skews them).
    temperature = float(cluster.sampling_params.get('temperature', 1.0))
    ref_shared = cluster.reference is cluster.model
    # the critic's values (the value head via task='value') and the reference's logps; the policy
    # actor chains its own disable_lora pass when the reference IS the policy object.
    scored = {
        'values': cluster.value_model.forward_only(
            inputs=features, micro_batch_size=micro_batch_size,
            task='value', return_logits=True),
        'ref': cluster.reference.forward_only(
            inputs=features, micro_batch_size=micro_batch_size, temperature=temperature,
            **({'disable_lora': True} if ref_shared else {})),
    }
    # collect once at the training consumption point; the lanes crossed driver->worker as references
    indices, advantages, returns, values, old_logps = collect(
        cluster.worker.gae(samples, scored, reward_config, rl_config))
    kept = [samples[i] for i in indices]
    # the reward telemetry reflects exactly the batch the upcoming optimizer steps train on
    metrics = cluster.worker.reward_metrics(samples, scored, reward_config, rl_config)
    reward_metric.reset()
    reward_metric.accumulate(
        rewards={'total': metrics['total']}, completion_lengths=metrics['completion_lengths'])
    return kept, advantages, returns, values, old_logps


def _train_ppo_batch(cluster: Cluster, state: StepState, rl_config: 'RLConfig', samples: List[Any],
                     advantages: List[List[float]], returns: List[List[float]],
                     values: List[List[float]], old_logps: List[List[float]], *,
                     train_batch_size: int, temperature: float, ema_manager: Any,
                     save: Callable[[str], Any]) -> int:
    """Train one pull on BOTH models over the mini-batches x ``num_ppo_epochs`` replays.

    Each mini-batch steps the policy (the clipped surrogate: advantages + old_logps) and then the
    critic (the clipped value loss: ``task='value'`` + returns + old_values) within the SAME
    micro-step; the GAE anchors stay fixed across every replay (PPO's batch re-use). Returns the
    optimizer steps actually run.
    """
    model = cluster.model
    value_model = cluster.value_model
    optimizer_steps = 0
    for _ in range(max(1, rl_config.num_ppo_epochs)):
        for lo in range(0, len(samples) - train_batch_size + 1, train_batch_size):
            hi = lo + train_batch_size
            mini = [sample.encoded for sample in samples[lo:hi]]
            state.micro_step += 1
            model.forward_backward(
                inputs=mini, gradient_accumulation_steps=state.ga, temperature=temperature,
                advantages=advantages[lo:hi], old_logps=old_logps[lo:hi])
            boundary = state.is_grad_sync_boundary()
            model.clip_grad_and_step(
                max_grad_norm=state.max_grad_norm, gradient_accumulation_steps=state.ga)
            value_model.forward_backward(
                inputs=mini, gradient_accumulation_steps=state.ga,
                task='value', returns=returns[lo:hi], old_values=values[lo:hi])
            value_model.clip_grad_and_step(
                max_grad_norm=state.max_grad_norm, gradient_accumulation_steps=state.ga)
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


def _colocate_loop(cluster: Cluster, rl_config: 'RLConfig', reward_config: 'RewardConfig',
                   rollout_config: 'RolloutConfig', dataloader: Any, state: StepState,
                   reward_metric: Any, *, train_batch_size: int, ema_manager: Any,
                   save: Callable[[str], Any]) -> None:
    """The colocate branch: a plain loop over generation batches, the device borrowed per batch.

    The sampler time-shares one device with BOTH training models, so the borrow offloads the
    policy and the critic before generation and reloads both after it. The ORM scoring is ONE
    batched worker pass per batch -- the same field the driver's prefix path writes.
    """
    entered = False
    try:
        for batch_rows in dataloader:
            # borrow the device: wake the sampler's weights so the sync has somewhere to write,
            # sync the trained policy in, step BOTH trainers aside, wake ONLY the kv cache.
            cluster.sampler.wake_up(tags=['weights'])
            cluster.engine.sync_weights(merge_and_sync=True)
            cluster.model.offload_to_cpu()
            cluster.value_model.offload_to_cpu()
            cluster.sampler.wake_up(tags=['kv_cache'])
            entered = True
            prompts = [row['messages'] for row in batch_rows]
            extras = [{key: value for key, value in row.items() if key != 'messages'} for row in batch_rows]
            samples = cluster.rollout.generate(
                prompts, num_samples=rl_config.num_generations, sampling_params=cluster.sampling_params,
                prompt_extras=extras or None)
            rows = (cluster.worker.score_group(samples) if cluster.worker.orm_scorers
                    else [[] for _ in samples])
            for sample, row in zip(samples, rows):
                sample.orm_reward = row
            # return the device: sleep the sampler, then reload both trainers' weights.
            cluster.sampler.sleep(level=rollout_config.sleep_level or 1)
            cluster.model.reload_to_gpu()
            cluster.value_model.reload_to_gpu()
            entered = False
            kept, advantages, returns, values, old_logps = _gae_pipeline(
                cluster, reward_config, rl_config, samples, train_batch_size, reward_metric)
            _train_ppo_batch(cluster, state, rl_config, kept, advantages, returns, values, old_logps,
                             train_batch_size=train_batch_size,
                             temperature=float(cluster.sampling_params.get('temperature', 1.0)),
                             ema_manager=ema_manager, save=save)
            if state.reached_max() or state.should_stop():
                break
    finally:
        if entered:
            # ended mid-window (a budget hit during generation): give the device back so the
            # final save finds both models resident.
            cluster.sampler.sleep(level=rollout_config.sleep_level or 1)
            cluster.model.reload_to_gpu()
            cluster.value_model.reload_to_gpu()


def _streaming_loop(cluster: Cluster, rl_config: 'RLConfig', reward_config: 'RewardConfig',
                    rollout_config: 'RolloutConfig', tuner_config: Optional['TunerConfig'],
                    dataloader: Any, state: StepState, reward_metric: Any, *,
                    train_batch_size: int, num_train_epochs: float, output_dir: str,
                    ema_manager: Any, save: Callable[[str], Any],
                    versions_cell: List[List[int]]) -> None:
    """The disaggregated branch: the StreamingDriver streams scored samples, the body trains.

    Per pull (the driver has ALREADY batched the ORM scoring at the pull boundary): lift the
    samples through the GAE pipeline, train both models, then the visible publish --
    ``in_place`` aborts in-flight generations and pushes the live weights, ``adapter_snapshot``
    hands the driver the next version's save.
    """
    driver = build_ppo_driver(
        cluster=cluster, rl_config=rl_config, rollout_config=rollout_config, tuner_config=tuner_config,
        train_batch_size=train_batch_size, gradient_accumulation_steps=state.ga,
        prompt_dataloader=dataloader, num_train_epochs=num_train_epochs, max_steps=state.max_steps,
        output_dir=output_dir)
    strategy = rollout_config.weight_sync_strategy
    steps_since_publish = 0
    try:
        for samples in driver:
            versions_cell[:] = [getattr(sample, 'version_span', 0) for sample in samples]
            kept, advantages, returns, values, old_logps = _gae_pipeline(
                cluster, reward_config, rl_config, samples, train_batch_size, reward_metric)
            optimizer_steps = _train_ppo_batch(
                cluster, state, rl_config, kept, advantages, returns, values, old_logps,
                train_batch_size=train_batch_size,
                temperature=float(cluster.sampling_params.get('temperature', 1.0)),
                ema_manager=ema_manager, save=save)
            # the visible publish: cadence inline, then one request; the driver brackets the push with
            # its version accounting and the adapter's decide/register (commit releases its thread).
            steps_since_publish += optimizer_steps
            if steps_since_publish >= rollout_config.parameter_sync_step:
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
            if state.reached_max() or state.should_stop():
                break
    finally:
        # Release the driver thread before the caller's teardown shuts the rollout/sampler:
        # an exception on the recipe thread otherwise leaves it blocked on the bridge ack.
        driver.abort()


def _save_value_checkpoint(value_model: Any, checkpoint_dir: str, checkpoint_config: Any,
                           global_step: int, *, async_save: bool = False) -> None:
    """Persist the critic into ``<policy checkpoint>/value_model`` (PPO's two-model save half).

    Both models step in phase on the one counter, so every policy checkpoint (periodic and final)
    carries a step-aligned critic -- a policy checkpoint without its value_model is un-resumable.
    """
    from swift.dev.recipe.utils import save_training_checkpoint

    save_training_checkpoint(
        value_model, 'value_model', output_dir=checkpoint_dir, consumed_train_samples=global_step,
        component_states={},
        no_save_optim=(checkpoint_config.no_save_optim or checkpoint_config.save_only_model),
        no_save_rng=(checkpoint_config.no_save_rng or checkpoint_config.save_only_model),
        safe_serialization=checkpoint_config.safe_serialization,
        max_shard_size=checkpoint_config.max_shard_size,
        save_total_limit=checkpoint_config.save_total_limit,
        async_save=async_save)


def _ppo_log_line(record: Dict[str, Any], loss: float) -> str:
    """PPO's console head: the reward plus the critic's explained_variance (the convergence pair)."""
    parts = [f"step {record.get('step')}", f'loss={loss:.4f}']
    for key, label in (('value_loss', 'value_loss'), ('train/total_reward', 'reward'),
                       ('train/explained_variance', 'explained_variance')):
        value = record.get(key)
        if value is not None:
            # The tracker.log pass hands ``record`` back with its scalars already display-formatted into
            # numeric strings (e.g. '5.000000e-05'), so coerce before the :.4f; a raw float passes too.
            # A console head must never crash the training loop, so fall back to the value as-is.
            try:
                parts.append(f'{label}={float(value):.4f}')
            except (TypeError, ValueError):
                parts.append(f'{label}={value}')
    return '  '.join(parts)


def run_ppo(model_config: 'ModelConfig', template_config: 'TemplateConfig',
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
    """PPO's five-segment main: prompts -> devices -> cluster -> loop scaffolding -> ONE visible loop."""
    if rl_config.rl_type != 'ppo':
        raise ValueError(f'run_ppo expects rl_type="ppo", got {rl_config.rl_type!r}.')
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
        group_requests=(lambda: ppo_group_requests(
            rl_config, reward_config, distributed_config, tuner_config)))
    # the optimizer's LR-scheduler horizon (num_ppo_epochs replays per rollout), derived BEFORE the build.
    from swift.dev.builders import build_dp_mesh

    dp_size = build_dp_mesh(distributed_config, template_config.sequence_parallel_size).data_world_size
    train_batch_size = train_config.per_device_train_batch_size * dp_size
    budget = rollout_step_budget(
        num_prompts=len(prompts), num_generations=rl_config.num_generations,
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        num_train_epochs=train_config.num_train_epochs,
        generation_batch_size=rollout_config.generation_batch_size,
        num_iterations=max(1, rl_config.num_ppo_epochs))
    max_steps = resolve_rollout_max_steps(train_config.max_steps, budget, recipe='run_ppo')
    # segment 3: the cluster -- policy, critic, frozen reference, reward channel, built explicitly.
    cluster = initialize_ppo_cluster(
        model_config=model_config, template_config=template_config, template=template,
        train_config=train_config, distributed_config=distributed_config,
        checkpoint_config=checkpoint_config, rollout_config=rollout_config, rl_config=rl_config,
        reward_config=reward_config, tuner_config=tuner_config, max_steps=max_steps,
        quantize_config=quantize_config, megatron_config=megatron_config, moe_config=moe_config,
        generation_config=generation_config, resource_plan=resource_plan,
        auxiliary_meshes=auxiliary_meshes, output_dir=output_dir)
    # segment 4: dataloader + the dissolved step cadence + the save closure.
    from swift.dev.builders import build_prompt_dataloader, is_megatron_backend
    from swift.dev.builders.optimizer import resolve_max_grad_norm
    from twinkle.metric import CompletionRewardMetric

    dataloader = build_prompt_dataloader(
        prompts, prompt_extras, seed=train_config.seed,
        batch_size=(rollout_config.generation_batch_size or len(prompts)))
    reward_metric = CompletionRewardMetric()
    _versions_cell: List[List[int]] = []

    def _extra_metrics() -> Dict[str, Any]:
        """The critic's namespaced metrics + the batch's reward aggregate and version spans."""
        extra: Dict[str, Any] = {}
        for key, value in (cluster.value_model.calculate_metric(is_training=True) or {}).items():
            if key == 'loss':
                extra['value_loss'] = value
            elif key == 'grad_norm':
                extra['value_grad_norm'] = value
            else:
                extra[key] = value
        extra.update(reward_metric.calculate())
        if _versions_cell:
            extra.update({
                'partial_ratio': sum(1 for span in _versions_cell if span > 0) / len(_versions_cell),
                'max_partial_span': max(_versions_cell),
                'version_span_mean': sum(_versions_cell) / len(_versions_cell),
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
        extra_metrics_fn=_extra_metrics, log_formatter=_ppo_log_line)

    save_knobs = build_save_knobs(
        checkpoint_config, dataset_config, distributed_config, tuner_config, output_dir=output_dir)
    # Before any training step: a base-modifying LoRA init (pissa ...) needs its initial snapshot.
    snapshot_initial_adapter(cluster.model, save_knobs.initial_adapter_dir)

    def save(name: str, is_final: bool = False) -> str:
        """One training checkpoint: the policy, then the step-aligned critic inside it.

        The final save is synchronous on BOTH models, every periodic one may ride the background
        writer under ``async_save``.
        """
        ckpt_dir = save_training_step(cluster.model, state.handler, name, save_knobs,
                                      consumed_train_samples=state.global_step, is_final=is_final)
        _save_value_checkpoint(
            cluster.value_model, ckpt_dir, checkpoint_config, state.global_step,
            async_save=(checkpoint_config.async_save and not is_final))
        return ckpt_dir

    ema_manager = None
    if rl_config.sync_ref_model and cluster.reference is not None:
        # EMA pair: policy rank k <-> reference rank k, synced every ref_model_sync_steps. The
        # critic never enters a publish (the sampler only ever generates with the policy).
        from twinkle.checkpoint_engine import CheckpointEngineManager

        ema_manager = CheckpointEngineManager(
            model=cluster.model, sampler=cluster.reference, mode='standalone',
            channel='ema:reference')

    if checkpoint_config.resume_from_checkpoint:
        state.global_step, state.micro_step, _ = resume_training(
            cluster.model, checkpoint_config=checkpoint_config, distributed_config=distributed_config,
            megatron_config=megatron_config, tuner_config=tuner_config,
            gradient_accumulation_steps=train_config.gradient_accumulation_steps)
        # the critic reloads its own weights/optimizer from <resume>/value_model; its completed-step
        # count is cross-checked -- a mismatch means the two checkpoints came from different steps.
        value_kwargs = {'resume_only_model': checkpoint_config.resume_only_model}
        if is_megatron_backend(distributed_config):
            value_kwargs.update(
                no_load_optim=checkpoint_config.no_load_optim, no_load_rng=checkpoint_config.no_load_rng,
                data_parallel_random_init=bool(megatron_config and megatron_config.data_parallel_random_init))
        value_state = cluster.value_model.resume_from_checkpoint(
            os.path.join(checkpoint_config.resume_from_checkpoint, 'value_model'), **value_kwargs)
        _check_policy_value_step_alignment(state.global_step, value_state)
    from twinkle.perf import Perf

    from swift.dev.recipe.utils import filter_perf_actors

    if Perf.initialize(filter_perf_actors({
            'model': cluster.model,
            'value_model': cluster.value_model,
            'sampler': cluster.sampler,
            'reference': cluster.reference,
    })):
        logger.info('[perf] profiling enabled for the online PPO loop.')
    # segment 5: ONE visible loop, picked by placement (every async_mode rides the stream).
    state.handler.on_train_begin()
    try:
        if cluster.colocate:
            _colocate_loop(cluster, rl_config, reward_config, rollout_config, dataloader, state,
                           reward_metric, train_batch_size=train_batch_size, ema_manager=ema_manager,
                           save=save)
        else:
            _streaming_loop(cluster, rl_config, reward_config, rollout_config, tuner_config,
                            dataloader, state, reward_metric, train_batch_size=train_batch_size,
                            num_train_epochs=train_config.num_train_epochs, output_dir=output_dir,
                            ema_manager=ema_manager, save=save, versions_cell=_versions_cell)
    finally:
        Perf.shutdown()
        cluster.rollout.close()
        cluster.sampler.shutdown()
        state.tracker.close()
    state.handler.on_train_end()
    if _save_final:
        # the final checkpoint is synchronous on BOTH models; its args.json is written by
        # initialize_rl_devices' copy. The wait must pierce to the workers (decorated method).
        cluster.model.wait_async_saves()
        cluster.value_model.wait_async_saves()
        save('checkpoint-final', is_final=True)
    return state.history
