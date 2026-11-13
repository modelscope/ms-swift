# Copyright (c) ModelScope Contributors. All rights reserved.
# The distillation recipes' visible training main: five segments, then ONE loop over the streaming driver.
from __future__ import annotations

import math
import random
from itertools import count
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from swift.dev.recipe.batching import balance_update
from swift.dev.utils.logger import get_logger
from swift.dev.utils.scoring import copy_inputs

from swift.dev.recipe.distill.assembly import (
    DISTILL_SIGNALS, Cluster, build_distill_driver, distill_dataset_rows, distill_group_requests,
    initialize_distill_cluster)
from swift.dev.recipe.utils import (
    StepState, build_save_knobs, initialize_rl_devices, optimizer_batch_slices, prompt_rows_from_dataset,
    resolve_generation_batch_size, resolve_rollout_max_steps, resume_training, rollout_step_budget,
    save_final_checkpoint, save_training_step, snapshot_initial_adapter, stream_batch_size)

if TYPE_CHECKING:
    from swift.dev.config import (
        CheckpointConfig, DatasetConfig, DistributedConfig, GenerationConfig, LoggingConfig, MegatronConfig,
        ModelConfig, MoEConfig, QuantizeConfig, RLConfig, RewardConfig, RolloutConfig, TemplateConfig,
        TrainConfig, TunerConfig)

logger = get_logger()


def _train_rows(cluster: Cluster, state: StepState, trajs: List[Any], lengths: Optional[List[int]],
                use_student: bool, *, train_batch_size: int, save: Optional[Callable[[str], Any]]) -> int:
    """Submit full GA updates, dropping the tail; both models split micro-batches internally.

    ``trajs`` are the driver's trajectories (on-policy, with the batch summary's row lengths) or
    encoded dataset rows (off-policy, ``lengths`` None).
    """
    update_batch_size = train_batch_size * state.ga
    dropped = len(trajs) % update_batch_size
    if dropped:
        logger.warning(
            f"Dropping {dropped} distillation samples that cannot fill an optimizer batch of {update_batch_size}."
        )
    optimizer_steps = 0
    micro_batch_size = train_batch_size // cluster.model.device_mesh.data_world_size
    for lo, hi in optimizer_batch_slices(len(trajs), train_batch_size, state.ga):
        if state.reached_max() or state.should_stop():
            break
        update = trajs[lo:hi]
        state.micro_step += (hi - lo) // train_batch_size
        update, balance_metrics = balance_update(
            state, update, None if lengths is None else lengths[lo:hi],
            micro_batch_kwargs={'micro_batch_size': micro_batch_size})
        # The worker runs the teacher forward -- the driver never sees the full-vocab logits or the
        # logps, only the light kwargs carrying them as refs.
        signal = dict(cluster.worker.forward_kwargs(update, use_student))
        cluster.model.forward_backward(
            inputs=copy_inputs(update), gradient_accumulation_steps=1, micro_batch_size=micro_batch_size, **signal)
        cluster.model.clip_grad_and_step(max_grad_norm=state.max_grad_norm, gradient_accumulation_steps=1)
        state.record(extra_metrics=balance_metrics)
        optimizer_steps += 1
        state.step_tail(save=save)
    return optimizer_steps


def _train_loop(cluster: Cluster, driver: Any, rl_config: 'RLConfig', rollout_config: 'RolloutConfig',
                dataset_rows: List[Dict[str, Any]], state: StepState, *, train_batch_size: int,
                batch_size: int, seed: int, save: Optional[Callable[[str], Any]]) -> None:
    """One loop over the driver; off-policy blocks are drawn in between.

    Every block of ``batch_size`` rows draws its source: with probability ``lmbda`` the student's
    next driver batch (on-policy), otherwise the next ``batch_size`` dataset completions
    (off-policy, GKD adds its supervised CE term). The draws before each on-policy block run first,
    so the weight sync -- which, colocated, hands the devices to the sampler -- happens only right
    before the driver is asked for a batch.
    """
    from swift.dev.recipe.offload import sleep_for_rollout

    draws = count()
    dataset_offset = 0

    def train_off_policy() -> bool:
        """Train dataset blocks until a block draws the student; ``False`` once training must stop."""
        nonlocal dataset_offset
        while not (state.reached_max() or state.should_stop()):
            if random.Random(seed + next(draws)).random() < rl_config.lmbda:
                return True
            rows = [dataset_rows[(dataset_offset + i) % len(dataset_rows)] for i in range(batch_size)]
            dataset_offset = (dataset_offset + batch_size) % len(dataset_rows)
            _train_rows(cluster, state, rows, None, False, train_batch_size=train_batch_size, save=save)
        return False

    lent = False
    batches = 0
    try:
        if not train_off_policy():
            return
        # the initial publication: a resumed policy differs from the weights the sampler loaded
        cluster.engine.sync_weights()
        lent = cluster.colocate
        for batch, batch_facts in driver:
            if cluster.colocate:
                sleep_for_rollout(cluster.sampler, rollout_config.sleep_level)
                cluster.model.reload_to_gpu()
                lent = False
            lengths = [p + r for p, r in zip(batch_facts.get('prompt_lengths') or (),
                                            batch_facts.get('response_lengths') or ())]
            _train_rows(cluster, state, batch, lengths, True, train_batch_size=train_batch_size, save=save)
            batches += 1
            if not train_off_policy():
                break
            if batches % rollout_config.parameter_sync_step == 0:
                cluster.engine.sync_weights()
                lent = cluster.colocate
    finally:
        # Release the driver thread and its episodes before the caller's teardown shuts the rollout.
        driver.close()
        if lent:
            # ended with the devices lent: give them back so the final save finds the model resident
            sleep_for_rollout(cluster.sampler, rollout_config.sleep_level)
            cluster.model.reload_to_gpu()


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
    # One resolved value feeds BOTH the step budget and the dataloader: the derived default fills
    # whole optimizer updates (never the whole dataset -- see resolve_generation_batch_size).
    # Distillation replays each batch once (no num_iterations analogue).
    generation_batch_size = resolve_generation_batch_size(
        rollout_config.generation_batch_size, num_prompts=len(prompts),
        num_generations=rl_config.num_generations, num_iterations=1,
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps)
    batch_size = stream_batch_size(
        colocate=rollout_config.sampler_mode == "colocate", generation_batch_size=generation_batch_size,
        num_generations=max(1, rl_config.num_generations), train_batch_size=train_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps)
    num_train_epochs = math.ceil(train_config.num_train_epochs)
    # The driver combines all passes before dropping its delivery tail; off-policy blocks spend the
    # same step budget, they only replace some of the on-policy batches.
    budget = rollout_step_budget(
        num_prompts=len(prompts), num_generations=rl_config.num_generations,
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        generation_batch_size=generation_batch_size,
        stream_batch_rows=batch_size,
    )
    max_steps = resolve_rollout_max_steps(train_config.max_steps, budget, recipe='run_distill')
    # segment 3: the cluster -- every actor distillation trains with, built explicitly, nothing hidden.
    cluster = initialize_distill_cluster(
        model_config=model_config, template_config=template_config, template=template,
        train_config=train_config, distributed_config=distributed_config,
        checkpoint_config=checkpoint_config, rollout_config=rollout_config, rl_config=rl_config,
        tuner_config=tuner_config, max_steps=max_steps, signal=signal,
        train_batch_size=train_batch_size, batch_size=batch_size,
        quantize_config=quantize_config, megatron_config=megatron_config, moe_config=moe_config,
        generation_config=generation_config, resource_plan=resource_plan,
        auxiliary_meshes=auxiliary_meshes,
    )
    # segment 4: dataloader + the dissolved step cadence + the save closure.
    from swift.dev.builders import build_prompt_dataloader
    from swift.dev.builders.optimizer import resolve_max_grad_norm

    dataloader = build_prompt_dataloader(
        prompts, prompt_extras, seed=train_config.seed,
        shuffle=dataset_config.train_dataloader_shuffle,
        batch_size=generation_batch_size,
    )
    # A batch feeds batch_size / (train_batch_size * GA) optimizer steps.
    steps_per_batch = max(1, batch_size // (train_batch_size * train_config.gradient_accumulation_steps))
    driver = build_distill_driver(
        cluster=cluster, rl_config=rl_config, rollout_config=rollout_config, prompt_dataloader=dataloader,
        batch_size=batch_size, num_train_epochs=num_train_epochs, max_steps=math.ceil(max_steps / steps_per_batch))

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
        )
    from twinkle.perf import Perf

    from swift.dev.recipe.utils import filter_perf_actors

    if Perf.initialize(filter_perf_actors({
            'model': cluster.model,
            'sampler': cluster.sampler,
            **{f'teacher_{index}': teacher for index, teacher in enumerate(cluster.teachers)},
    })):
        logger.info('[perf] profiling enabled for the online %s loop.', rl_config.rl_type.upper())
    # segment 5: ONE visible loop (every placement and lmbda rides the stream).
    state.handler.on_train_begin()
    try:
        _train_loop(cluster, driver, rl_config, rollout_config, dataset_rows, state,
                    train_batch_size=train_batch_size, batch_size=batch_size, seed=train_config.seed, save=save)
    finally:
        Perf.shutdown()
        cluster.rollout.close()
        cluster.sampler.shutdown()
        state.tracker.close()
    state.handler.on_train_end()
    # The final checkpoint is synchronous; its args.json is written by initialize_rl_devices' copy.
    save_final_checkpoint(cluster.model, save, enabled=_save_final)
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
