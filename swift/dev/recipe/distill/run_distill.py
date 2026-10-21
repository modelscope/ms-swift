# Copyright (c) ModelScope Contributors. All rights reserved.
# The distillation recipes' visible training main: five segments, then ONE loop body by placement.
from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, Any, Callable, List, Optional

from swift.dev.utils.logger import get_logger

from swift.dev.recipe.distill.assembly import (
    DISTILL_SIGNALS, Cluster, build_distill_driver, distill_dataset_rows, distill_group_requests,
    initialize_distill_cluster)
from swift.dev.recipe.distill.worker import DistillRow
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


def _plan_mini_batches(rows: List[DistillRow], train_batch_size: int) -> List[List[DistillRow]]:
    """Split one round's rows into full ``train_batch_size`` mini-batches, warning on a dropped tail.

    Raises when the batch cannot fill even one mini-batch, so a caller never silently runs a no-op
    training round.
    """
    minis = [rows[start:start + train_batch_size]
             for start in range(0, len(rows) - train_batch_size + 1, train_batch_size)]
    if not minis:
        raise RuntimeError(
            f'the distillation round produced {len(rows)} rows but train_batch_size={train_batch_size} '
            '(per_device_train_batch_size * dp_size), so no full mini-batch fits and no optimizer '
            'step is possible. Enlarge the prompt dataset or num_generations, or lower '
            'per_device_train_batch_size / the DP world size.')
    dropped = len(rows) - len(minis) * train_batch_size
    if dropped:
        logger.warning('Dropping %d of %d distill rows that do not fill a train_batch_size=%d mini-batch.',
                       dropped, len(rows), train_batch_size)
    return minis


def _rows_from_samples(samples: List[Any]) -> List[DistillRow]:
    """Lift rollout samples into training rows: the encoded feature + the conversation + extras.

    A multi-turn record never occurs here (distillation rolls out single-turn), so a sample's own
    conversation defaults to the prompt it was admitted for.
    """
    return [
        DistillRow(sample.encoded,
                   sample.messages if sample.messages is not None else list(sample.prompt or []),
                   sample.extra or {}) for sample in samples
    ]


def _train_rows(cluster: Cluster, state: StepState, rows: List[DistillRow],
                use_student: bool, *, train_batch_size: int,
                save: Optional[Callable[[str], Any]]) -> int:
    """Train one round's mini-batches (teacher signal scored remotely per mini-batch). Returns steps."""
    optimizer_steps = 0
    for mini in _plan_mini_batches(rows, train_batch_size):
        if state.reached_max() or state.should_stop():
            break
        state.micro_step += 1
        # The worker builds the kwargs where the teacher forward runs -- the driver never sees the
        # full-vocab logits or the logps, only the light kwargs carrying them as refs.
        cluster.model.forward_backward(**cluster.worker.forward_kwargs(mini, use_student))
        boundary = state.is_grad_sync_boundary()
        cluster.model.clip_grad_and_step(
            max_grad_norm=state.max_grad_norm, gradient_accumulation_steps=state.ga)
        if boundary:
            state.record()
            optimizer_steps += 1
            state.step_tail(save=save)
    return optimizer_steps


def _distill_round_loop(cluster: Cluster, rl_config: 'RLConfig', rollout_config: 'RolloutConfig',
                        dataset_rows: List[DistillRow], dataloader: Any, state: StepState, *,
                        train_batch_size: int, num_train_epochs: float, seed: int,
                        save: Optional[Callable[[str], Any]]) -> None:
    """The per-round loop: an on/off-policy coin flip per generation batch, BOTH round kinds train.

    An on-policy round rolls the student (borrowing the device when colocated, or one weight sync
    when disaggregated); an off-policy round distils the dataset's own completions and adds GKD's
    supervised CE term -- it is not a skipped round. ``round_index`` counts ROUNDS (not optimizer
    steps) so the coin flip and the dataset window stay aligned with the batch being processed.
    """
    colocate = cluster.colocate
    round_index = 0
    entered = False
    try:
        for epoch in range(max(1, math.ceil(num_train_epochs))):
            dataloader.set_epoch(epoch)
            for batch in dataloader:
                if state.reached_max() or state.should_stop():
                    break
                use_student = random.Random(seed + round_index).random() <= rl_config.lmbda
                round_index += 1
                if use_student:
                    if colocate:
                        # borrow the device: wake the sampler's weights so the sync has somewhere to
                        # write, sync the trained policy in, step the trainer aside, wake the kv cache.
                        cluster.sampler.wake_up(tags=['weights'])
                        cluster.engine.sync_weights(merge_and_sync=True)
                        cluster.model.offload_to_cpu()
                        cluster.sampler.wake_up(tags=['kv_cache'])
                        entered = True
                    else:
                        cluster.engine.sync_weights()
                    samples = cluster.rollout.generate(
                        [row['messages'] for row in batch], num_samples=rl_config.num_generations,
                        sampling_params=cluster.sampling_params,
                        prompt_extras=[{k: v for k, v in row.items() if k != 'messages'} for row in batch] or None)
                    _train_rows(cluster, state, _rows_from_samples(samples), True,
                                train_batch_size=train_batch_size, save=save)
                    if colocate:
                        # return the device: sleep the sampler, then reload the trainer's weights.
                        cluster.sampler.sleep(level=rollout_config.sleep_level or 1)
                        cluster.model.reload_to_gpu()
                        entered = False
                else:
                    # A rolling window over the dataset completions, so successive off-policy rounds
                    # cover new rows (run_distill already rejected lmbda<1 without completions).
                    count = len(dataset_rows)
                    offset = ((round_index - 1) * train_batch_size) % count
                    rows = [dataset_rows[(offset + i) % count] for i in range(count)]
                    _train_rows(cluster, state, rows, False,
                                train_batch_size=train_batch_size, save=save)
    finally:
        if entered:
            # ended mid-window (e.g. reached_max inside a generation round): give the device back so
            # the post-run final save finds the model resident.
            cluster.sampler.sleep(level=rollout_config.sleep_level or 1)
            cluster.model.reload_to_gpu()


def _distill_streaming_loop(cluster: Cluster, rl_config: 'RLConfig', rollout_config: 'RolloutConfig',
                            tuner_config: Optional['TunerConfig'], dataloader: Any, state: StepState, *,
                            train_batch_size: int, num_train_epochs: float, output_dir: str,
                            save: Optional[Callable[[str], Any]]) -> None:
    """The purely on-policy disaggregated loop (lmbda == 1): the StreamingDriver streams, body trains.

    Per pull: lift the samples into rows, train the mini-batches through the remote worker, then the
    visible publish -- ``in_place`` aborts in-flight generations and pushes the live weights,
    ``adapter_snapshot`` hands the driver the next version's save.
    """
    driver = build_distill_driver(
        cluster=cluster, rl_config=rl_config, rollout_config=rollout_config, tuner_config=tuner_config,
        train_batch_size=train_batch_size, gradient_accumulation_steps=state.ga,
        prompt_dataloader=dataloader, num_train_epochs=num_train_epochs, max_steps=state.max_steps,
        output_dir=output_dir)
    strategy = rollout_config.weight_sync_strategy
    steps_since_publish = 0
    for samples in driver:
        optimizer_steps = _train_rows(
            cluster, state, _rows_from_samples(samples), True,
            train_batch_size=train_batch_size, save=save)
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


def run_distill(model_config: 'ModelConfig', template_config: 'TemplateConfig',
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
    """The distillation family's five-segment main (gkd / opsd / mopd via ``rl_config.rl_type``)."""
    if rl_config.rl_type not in DISTILL_SIGNALS:
        raise ValueError(f'run_distill expects rl_type gkd/opsd/mopd, got {rl_config.rl_type!r}.')
    signal = DISTILL_SIGNALS[rl_config.rl_type]
    if not 0.0 <= rl_config.lmbda <= 1.0:
        raise ValueError(f'distillation lmbda must be in [0, 1], got {rl_config.lmbda}.')
    # segment 1: prompts (the resource estimator reads the prompt lengths off the raw rows).
    raw_rows, prompts, prompt_extras = prompt_rows_from_dataset(dataset_config)
    # segment 2: devices (configs + template + twinkle + perf; an estimate rewrites the parallel sizes).
    template, resource_plan, auxiliary_meshes = initialize_rl_devices(
        model_config=model_config, template_config=template_config, dataset_config=dataset_config,
        train_config=train_config, distributed_config=distributed_config,
        checkpoint_config=checkpoint_config, rollout_config=rollout_config, rl_config=rl_config,
        reward_config=reward_config, tuner_config=tuner_config, logging_config=logging_config,
        quantize_config=quantize_config, megatron_config=megatron_config, moe_config=moe_config,
        output_dir=output_dir, prompt_rows=raw_rows,
        group_requests=(lambda: distill_group_requests(rl_config)))
    dataset_rows = distill_dataset_rows(template, raw_rows)
    if rl_config.lmbda < 1.0 and not dataset_rows:
        raise ValueError(f'{rl_config.rl_type} with lmbda < 1 distils some rounds on the dataset\'s own '
                         'completions, but no dataset row carries an assistant completion. Add '
                         'completions, or set lmbda=1 for purely on-policy distillation.')
    if rl_config.rl_type == 'opsd' and not any(
            (extra or {}).get('teacher_prompt') for extra in prompt_extras):
        # Without a privileged teacher view the distillation signal would be identically zero.
        raise ValueError('OPSD requires a per-sample `teacher_prompt` dataset column carrying the '
                         'privileged context (reference/rubric). None of the rows provide it. Add it '
                         'via a dataset plugin (see examples/rl/opsd), or use --rl_type gkd.')
    # the optimizer's LR-scheduler horizon, derived BEFORE the cluster builds the optimizer.
    from swift.dev.builders import build_dp_mesh

    dp_size = build_dp_mesh(distributed_config, template_config.sequence_parallel_size).data_world_size
    train_batch_size = train_config.per_device_train_batch_size * dp_size
    budget = rollout_step_budget(
        num_prompts=len(prompts), num_generations=rl_config.num_generations,
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        num_train_epochs=train_config.num_train_epochs,
        generation_batch_size=rollout_config.generation_batch_size)
    max_steps = resolve_rollout_max_steps(train_config.max_steps, budget, recipe='run_distill')
    # segment 3: the cluster -- every actor distillation trains with, built explicitly, nothing hidden.
    cluster = initialize_distill_cluster(
        model_config=model_config, template_config=template_config, template=template,
        train_config=train_config, distributed_config=distributed_config,
        checkpoint_config=checkpoint_config, rollout_config=rollout_config, rl_config=rl_config,
        tuner_config=tuner_config, max_steps=max_steps, signal=signal,
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        quantize_config=quantize_config, megatron_config=megatron_config, moe_config=moe_config,
        generation_config=generation_config, resource_plan=resource_plan,
        auxiliary_meshes=auxiliary_meshes, output_dir=output_dir)
    # segment 4: dataloader + the dissolved step cadence + the save closure.
    from swift.dev.builders import build_prompt_dataloader
    from swift.dev.builders.optimizer import resolve_max_grad_norm

    dataloader = build_prompt_dataloader(
        prompts, prompt_extras, seed=train_config.seed,
        batch_size=(rollout_config.generation_batch_size or len(prompts)))

    state = StepState(
        cluster.model, train_config,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        max_steps=max_steps, max_grad_norm=resolve_max_grad_norm(train_config),
        logging_config=logging_config, output_dir=output_dir,
        save_steps=checkpoint_config.save_steps,
        manual_gc=bool(megatron_config and megatron_config.manual_gc),
        manual_gc_steps=megatron_config.manual_gc_steps if megatron_config else 0,
        empty_cache_steps=train_config.torch_empty_cache_steps)

    save_knobs = build_save_knobs(
        checkpoint_config, dataset_config, distributed_config, tuner_config, output_dir=output_dir)
    # Before any training step: a base-modifying LoRA init (pissa ...) needs its initial snapshot.
    snapshot_initial_adapter(cluster.model, save_knobs.initial_adapter_dir)

    def save(name: str, is_final: bool = False) -> str:
        """One training checkpoint: the twinkle-native write, the converted adapter, the hub push."""
        return save_training_step(cluster.model, state.handler, name, save_knobs,
                                  consumed_train_samples=state.global_step, is_final=is_final)

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
            **{f'teacher_{index}': teacher for index, teacher in enumerate(cluster.teachers)},
    })):
        logger.info('[perf] profiling enabled for the online %s loop.', rl_config.rl_type.upper())
    # segment 5: ONE visible loop -- colocate trains a plain round loop; disaggregated picks by lmbda.
    state.handler.on_train_begin()
    try:
        if cluster.colocate:
            _distill_round_loop(cluster, rl_config, rollout_config, dataset_rows, dataloader,
                                state, train_batch_size=train_batch_size,
                                num_train_epochs=train_config.num_train_epochs, seed=train_config.seed,
                                save=save)
        elif rl_config.lmbda == 1.0:
            _distill_streaming_loop(cluster, rl_config, rollout_config, tuner_config, dataloader,
                                    state, train_batch_size=train_batch_size,
                                    num_train_epochs=train_config.num_train_epochs,
                                    output_dir=output_dir, save=save)
        else:
            _distill_round_loop(cluster, rl_config, rollout_config, dataset_rows, dataloader,
                                state, train_batch_size=train_batch_size,
                                num_train_epochs=train_config.num_train_epochs, seed=train_config.seed,
                                save=save)
    finally:
        Perf.shutdown()
        cluster.rollout.close()
        cluster.sampler.shutdown()
        state.tracker.close()
    state.handler.on_train_end()
    if _save_final:
        # the final checkpoint is synchronous; its args.json is written by initialize_rl_devices' copy
        cluster.model.wait_async_saves()
        save('checkpoint-final', is_final=True)
    return state.history


def run_gkd(model_config: 'ModelConfig', template_config: 'TemplateConfig', dataset_config: 'DatasetConfig',
            train_config: 'TrainConfig', distributed_config: 'DistributedConfig',
            checkpoint_config: 'CheckpointConfig', rollout_config: 'RolloutConfig', rl_config: 'RLConfig',
            **kwargs: Any) -> List[dict]:
    """GKD: on-policy distillation against a teacher's full-vocab logits (signal='logits')."""
    return run_distill(model_config, template_config, dataset_config, train_config,
                       distributed_config, checkpoint_config, rollout_config, rl_config, **kwargs)


def run_opsd(model_config: 'ModelConfig', template_config: 'TemplateConfig', dataset_config: 'DatasetConfig',
             train_config: 'TrainConfig', distributed_config: 'DistributedConfig',
             checkpoint_config: 'CheckpointConfig', rollout_config: 'RolloutConfig', rl_config: 'RLConfig',
             **kwargs: Any) -> List[dict]:
    """OPSD: on-policy self-distillation under a privileged teacher view (signal='logps')."""
    return run_distill(model_config, template_config, dataset_config, train_config,
                       distributed_config, checkpoint_config, rollout_config, rl_config, **kwargs)


def run_mopd(model_config: 'ModelConfig', template_config: 'TemplateConfig', dataset_config: 'DatasetConfig',
             train_config: 'TrainConfig', distributed_config: 'DistributedConfig',
             checkpoint_config: 'CheckpointConfig', rollout_config: 'RolloutConfig', rl_config: 'RLConfig',
             **kwargs: Any) -> List[dict]:
    """MOPD: multi-teacher on-policy distillation, K weighted teacher channels (signal='multi_logps')."""
    return run_distill(model_config, template_config, dataset_config, train_config,
                       distributed_config, checkpoint_config, rollout_config, rl_config, **kwargs)
