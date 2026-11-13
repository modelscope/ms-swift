# Copyright (c) ModelScope Contributors. All rights reserved.
# PPO's visible training main: five segments, then ONE loop over the streaming driver.
from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from swift.dev.recipe.batching import balance_update
from swift.dev.utils.logger import get_logger

from swift.dev.recipe.ppo.assembly import (
    Cluster, build_ppo_driver, initialize_ppo_cluster, ppo_group_requests)
from swift.dev.recipe.utils import (
    StepState, build_save_knobs, initialize_rl_devices, optimizer_batch_slices, prompt_rows_from_dataset,
    resolve_rollout_max_steps, resume_training, rollout_step_budget, resolve_generation_batch_size,
    save_training_step, snapshot_initial_adapter, stream_batch_size)

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


def _train_ppo_batch(cluster: Cluster, state: StepState, rl_config: 'RLConfig', trajs: List[Any],
                     lengths: List[int], *, train_batch_size: int, temperature: float, ema_manager: Any,
                     save: Callable[[str], Any]) -> int:
    """Repeat one batch continuously, then drop only the final incomplete optimizer batch.

    The trajectories carry every GAE anchor as top-level response-framed fields, so each update
    steps the policy (the clipped surrogate: ``advantages`` / ``old_logps`` off its inputs) and
    then the critic (the clipped value loss: ``task='value'`` over ``returns`` / ``old_values``)
    within the SAME update; the anchors stay fixed across every replay (PPO's batch re-use).
    Returns the optimizer steps actually run.
    """
    model = cluster.model
    value_model = cluster.value_model
    update_batch_size = train_batch_size * state.ga
    total_samples = len(trajs) * max(1, rl_config.num_ppo_epochs)
    dropped = total_samples % update_batch_size
    if dropped:
        logger.warning(
            f"Dropping {dropped} PPO sample occurrences from the final replay tail "
            f"that cannot fill an optimizer batch of {update_batch_size}."
        )
    optimizer_steps = 0
    micro_batch_size = train_batch_size // model.device_mesh.data_world_size
    for lo, hi in optimizer_batch_slices(total_samples, train_batch_size, state.ga):
        if state.reached_max() or state.should_stop():
            break
        indices = [index % len(trajs) for index in range(lo, hi)]
        # The policy's seq-mean-token-mean loss is permutation-invariant, so its batch is
        # reordered for even DP workloads; the value model's rank-local token mean is NOT,
        # so it always trains the batch in the original order.
        update, balance_metrics = balance_update(
            state, [trajs[index] for index in indices], [lengths[index] for index in indices],
            micro_batch_kwargs={'micro_batch_size': micro_batch_size})
        state.micro_step += (hi - lo) // train_batch_size
        # forward_backward is declared sync=True on the model: a worker-side failure raises
        # at this call, failing THIS step instead of riding an unread handle.
        model.forward_backward(
            inputs=update,
            gradient_accumulation_steps=1,
            micro_batch_size=micro_batch_size,
            temperature=temperature,
        )
        model.clip_grad_and_step(max_grad_norm=state.max_grad_norm, gradient_accumulation_steps=1)
        value_model.forward_backward(
            inputs=[trajs[index] for index in indices],
            gradient_accumulation_steps=1,
            micro_batch_size=micro_batch_size,
            task='value',
        )
        value_model.clip_grad_and_step(max_grad_norm=state.max_grad_norm, gradient_accumulation_steps=1)
        state.record(extra_metrics=balance_metrics)
        optimizer_steps += 1
        if ema_manager is not None and state.global_step % rl_config.ref_model_sync_steps == 0:
            ema_manager.sync_ema(rl_config.ref_model_mixup_alpha)
        state.step_tail(save=save)
    return optimizer_steps


def _train_loop(cluster: Cluster, driver: Any, rl_config: 'RLConfig',
                rollout_config: 'RolloutConfig', state: StepState, reward_metric: Any, *,
                train_batch_size: int, ema_manager: Any, save: Callable[[str], Any],
                versions_cell: List[List[int]]) -> None:
    """One loop for every placement: take a scored batch, lift it through GAE, train both models.

    Colocate time-shares the devices with BOTH training models: the critic is offloaded before the
    sync (which wakes the sampler, pushes the policy, offloads it and wakes the KV cache), and both
    are reloaded once a batch is handed out.
    """
    from swift.dev.builders.streaming import offload_kwargs
    from swift.dev.recipe.offload import sleep_for_rollout

    temperature = float(cluster.sampling_params.get('temperature', 1.0))
    # one per-device block per rank: train_batch_size (per_device x dp) spread over the dp ranks
    micro_batch_size = max(1, train_batch_size // cluster.model.device_mesh.data_world_size)
    # The reference's logps must be taken at the SAMPLING temperature, like every logps the ratio
    # math compares against (the sampler's own logps are T-scaled; a T=1 forward skews them).
    ref_shared = cluster.reference is cluster.model
    lent = False

    def sync() -> None:
        nonlocal lent
        if cluster.colocate:
            cluster.value_model.offload_to_cpu(**offload_kwargs(rollout_config))
            lent = True
        cluster.engine.sync_weights()

    def reclaim() -> None:
        nonlocal lent
        sleep_for_rollout(cluster.sampler, rollout_config.sleep_level)
        cluster.model.reload_to_gpu()
        cluster.value_model.reload_to_gpu()
        lent = False

    batches = 0
    try:
        # the initial publication: a resumed policy differs from the weights the sampler loaded
        sync()
        for trajs, batch_facts in driver:
            if cluster.colocate:
                reclaim()
            versions_cell[:] = batch_facts.get('version_spans') or []
            # the nonempty-response rule runs in the worker on the real rows; the kept rows'
            # summary replaces the batch facts (its metrics describe exactly what trains), and its
            # kept ``indices`` slice the row-aligned score column to the rows that train.
            trajs, kept_facts = cluster.traj_filter.select(
                trajs, where=lambda traj: bool(traj.get('response_token_ids')))
            if not trajs:
                raise RuntimeError('the pulled batch has no response tokens to train on.')
            scores = [batch_facts['scores'][index] for index in kept_facts['indices']]
            lengths = [p + r for p, r in zip(kept_facts.get('prompt_lengths') or (),
                                            kept_facts.get('response_lengths') or ())]
            # the critic's values (``task='value'``) and the reference's logps: two whole-batch
            # response-framed forwards over the kept rows; the policy actor chains its own
            # disable_lora pass when the reference IS the policy object.
            values_out = cluster.value_model.forward_only(
                inputs=trajs, output_frame='response', output_fields=('values',),
                micro_batch_size=micro_batch_size, task='value')
            ref_out = cluster.reference.forward_only(
                inputs=trajs, output_frame='response', output_fields=('logps',), temperature=temperature,
                micro_batch_size=micro_batch_size, **({'disable_lora': True} if ref_shared else {}))
            # ONE advantage call (estimator=gae, kl=k1): token rewards ``r - kl_coef*KL`` with the
            # weighted ORM score on the last token, GAE over the pre-update critic values; the
            # trajectories come back carrying advantages / returns / old_logps / old_values.
            trajs, gae_summary = cluster.advantage(
                trajs, ref_out=ref_out, value_out=values_out, scores=scores)
            # the reward telemetry reflects exactly the batch the upcoming optimizer steps train on
            reward_metric.reset()
            reward_metric.accumulate(
                rewards={'total': list(gae_summary.get('reward_totals') or ())},
                completion_lengths=list(kept_facts.get('response_lengths') or ()))
            _train_ppo_batch(
                cluster, state, rl_config, trajs, lengths,
                train_batch_size=train_batch_size, temperature=temperature, ema_manager=ema_manager, save=save)
            batches += 1
            if state.reached_max() or state.should_stop():
                break
            if batches % rollout_config.parameter_sync_step == 0:
                sync()
    finally:
        # Release the driver thread and its episodes before the caller's teardown shuts the rollout.
        driver.close()
        if lent:
            # ended with the devices lent: give them back so the final save finds both models resident
            reclaim()


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


def _ppo_extra_metrics(cluster, reward_metric, _versions_cell) -> Dict[str, Any]:
    """The critic's namespaced metrics + the batch's reward aggregate and version spans."""
    extra: Dict[str, Any] = {}
    for key, value in (cluster.value_model.calculate_metric(is_training=True) or {}).items():
        if key == "loss":
            extra["value_loss"] = value
        elif key == "grad_norm":
            extra["value_grad_norm"] = value
        else:
            extra[key] = value
    extra.update(reward_metric.calculate())
    if _versions_cell:
        extra.update(
            {
                "partial_ratio": sum(1 for span in _versions_cell if span > 0) / len(_versions_cell),
                "max_partial_span": max(_versions_cell),
                "version_span_mean": sum(_versions_cell) / len(_versions_cell),
            }
        )
    return extra


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
    # One resolved value feeds BOTH the step budget and the dataloader: the derived default fills
    # whole optimizer updates (never the whole dataset -- see resolve_generation_batch_size).
    generation_batch_size = resolve_generation_batch_size(
        rollout_config.generation_batch_size, num_prompts=len(prompts),
        num_generations=rl_config.num_generations, num_iterations=rl_config.num_ppo_epochs,
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps)
    num_generations = max(1, rl_config.num_generations)
    batch_size = stream_batch_size(
        colocate=rollout_config.sampler_mode == "colocate", generation_batch_size=generation_batch_size,
        num_generations=num_generations, train_batch_size=train_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps)
    num_train_epochs = math.ceil(train_config.num_train_epochs)
    budget = rollout_step_budget(
        num_prompts=len(prompts), num_generations=rl_config.num_generations,
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        generation_batch_size=generation_batch_size,
        num_iterations=rl_config.num_ppo_epochs,
        stream_batch_rows=batch_size,
    )
    max_steps = resolve_rollout_max_steps(train_config.max_steps, budget, recipe='run_ppo')
    # segment 3: the cluster -- policy, critic, frozen reference, reward channel, built explicitly.
    cluster = initialize_ppo_cluster(
        model_config=model_config, template_config=template_config, template=template,
        train_config=train_config, distributed_config=distributed_config,
        checkpoint_config=checkpoint_config, rollout_config=rollout_config, rl_config=rl_config,
        reward_config=reward_config, tuner_config=tuner_config, max_steps=max_steps,
        batch_size=batch_size, quantize_config=quantize_config, megatron_config=megatron_config,
        moe_config=moe_config, generation_config=generation_config, resource_plan=resource_plan,
        auxiliary_meshes=auxiliary_meshes)
    # segment 4: dataloader + driver + the step cadence + the save closure.
    from swift.dev.builders import build_prompt_dataloader, is_megatron_backend
    from swift.dev.builders.optimizer import resolve_max_grad_norm
    from twinkle.metric import CompletionRewardMetric

    dataloader = build_prompt_dataloader(
        prompts, prompt_extras, seed=train_config.seed,
        shuffle=dataset_config.train_dataloader_shuffle,
        batch_size=generation_batch_size,
    )
    # A batch feeds batch_size * num_ppo_epochs / (train_batch_size * GA) optimizer steps.
    steps_per_batch = max(1, batch_size * max(1, rl_config.num_ppo_epochs)
                          // (train_batch_size * train_config.gradient_accumulation_steps))
    driver = build_ppo_driver(
        cluster=cluster, rl_config=rl_config, rollout_config=rollout_config, prompt_dataloader=dataloader,
        batch_size=batch_size, num_train_epochs=num_train_epochs, max_steps=math.ceil(max_steps / steps_per_batch))
    reward_metric = CompletionRewardMetric()
    _versions_cell: List[List[int]] = []

    def _extra_metrics() -> Dict[str, Any]:
        """The critic's namespaced metrics + the batch's reward aggregate and version spans."""
        return _ppo_extra_metrics(cluster, reward_metric, _versions_cell)

    state = StepState(
        cluster.model, train_config,
        padding_free=template_config.padding_free,
        dp_balance=train_config.dp_balance,
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
        )
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
    # segment 5: ONE visible loop (every placement and async_mode rides the stream).
    state.handler.on_train_begin()
    try:
        if not (state.reached_max() or state.should_stop()):
            _train_loop(cluster, driver, rl_config, rollout_config, state, reward_metric,
                        train_batch_size=train_batch_size, ema_manager=ema_manager, save=save,
                        versions_cell=_versions_cell)
    finally:
        Perf.shutdown()
        cluster.rollout.close()
        cluster.sampler.shutdown()
        state.tracker.close()
    state.handler.on_train_end()
    if _save_final:
        # The final checkpoint is synchronous on BOTH models; its args.json is written by
        # initialize_rl_devices' copy. The wait must pierce to the workers (decorated method).
        cluster.model.wait_async_saves()
        cluster.value_model.wait_async_saves()
        save('checkpoint-final', is_final=True)
    return state.history
