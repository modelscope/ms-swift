# Copyright (c) ModelScope Contributors. All rights reserved.
# The offline preference family: the run_dpo main plus the family's own training loop.
from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from swift.dev.utils.logger import get_logger  # TODO(dev2): bridge until the utils batch

from swift.dev2.recipe.utils import (
    GenerativeEval, StepState, build_generative_eval, build_save_knobs, resume_training,
    save_training_step, snapshot_initial_adapter, write_ckpt_args_json)

if TYPE_CHECKING:
    from swift.dev2.recipe.preference.assembly import PreferenceCluster
    from swift.dev2.config import (
        CheckpointConfig, DatasetConfig, DistributedConfig, LoggingConfig, MegatronConfig, ModelConfig,
        MoEConfig, QuantizeConfig, RLConfig, RewardConfig, TemplateConfig, TrainConfig, TunerConfig)

logger = get_logger()

#: rl_types this recipe handles (the online types have their own recipes).
_OFFLINE_TYPES = frozenset({'dpo', 'kto', 'cpo', 'orpo', 'simpo', 'rm'})


def _epochs(num_train_epochs: float, max_steps: int) -> int:
    """Epoch bound: the ceil of num_train_epochs, or a synthetic 10**9 when max_steps caps the run."""
    return math.ceil(num_train_epochs) if max_steps <= 0 else 10**9


def _guard_nonempty_epoch(reached_max: bool, should_stop: bool, epoch: int, batches: int) -> None:
    """Fail loudly if an epoch produced no batches, instead of spinning empty epochs forever.

    When ``max_steps > 0`` the epoch bound is a synthetic ``10**9``, so only a step limit or a stop
    flag can end the loop; a dataloader that yields nothing (usually a global batch larger than the
    dataset, whose remainder ``drop_last`` discards) would burn a billion empty epochs in silence.
    """
    if batches or reached_max or should_stop:
        return
    raise ValueError(
        f'Epoch {epoch} yielded no training batches, so no optimizer step can ever be taken. This '
        'almost always means the global batch (per_device_train_batch_size * '
        'gradient_accumulation_steps * data_parallel_size) exceeds the dataset size, and drop_last '
        'discards the remainder. Enlarge the dataset or lower the batch size; otherwise training '
        'would loop empty epochs forever.')


def _evaluate_loss(model: Any, eval_dataloader: Any, *, eval_iters: int, manual_gc: bool,
                   manual_gc_eval: bool, micro_step_kwargs: Callable[..., Dict[str, Any]],
                   global_step: int) -> Optional[dict]:
    """Val-loss eval: the normalized per-token loss over eval_dataloader (None if there is none).

    Mirrors the train metric path on the eval side: each val batch runs ``forward_only`` (no grad)
    then ``calculate_loss`` (fills this batch's loss); a final ``calculate_metric(False)`` aggregates
    and resets. Megatron already produced the loss inside forward_only -- its ``calculate_loss``
    raises NotImplementedError, treated as already populated.
    """
    if eval_dataloader is None:
        return None
    if manual_gc and manual_gc_eval:
        import gc
        gc.collect()
    for index, batch in enumerate(eval_dataloader):
        if eval_iters > 0 and index >= eval_iters:
            break
        model.forward_only(**micro_step_kwargs(batch, owns_ga=True))
        try:
            model.calculate_loss()
        except NotImplementedError:
            pass
    metrics = model.calculate_metric(is_training=False)
    result = {'step': global_step}
    if metrics.get('loss') is not None:
        result['eval_loss'] = float(metrics['loss'])
    for key, value in metrics.items():
        if key.startswith('loss_'):
            result[f'eval_{key}'] = float(value)
    if manual_gc and manual_gc_eval:
        import gc
        gc.collect(0)
    return result


def _evaluate_generate(generative: Optional[GenerativeEval], *, global_step: int) -> Optional[dict]:
    """Generative eval: run an EvalScope benchmark over the resident sampler (None if unwired).

    Only rank 0 drives EvalScope (a single process owns the report files); the other ranks wait at a
    barrier so the next training step stays collective-safe. ``enter``/``exit`` bracket the run with
    the weight-sync + memory schedule for a co-resident vllm/sglang engine; they run on every rank
    (under ray the driver drives the remote workers, under torchrun they are collective).
    """
    if generative is None:
        return None
    import torch.distributed as dist

    from swift.dev2.eval import run_evaluator
    # TODO(dev2): bridge until the recipe/common batch migrates flatten_evalscope_report
    from swift.dev.recipe.common.loop_utils import flatten_evalscope_report

    distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    is_main = (not distributed) or dist.get_rank() == 0
    result: Dict[str, Any] = {'step': global_step}
    if generative.enter is not None:
        generative.enter()
    try:
        if is_main:
            summary = run_evaluator(
                generative.sampler,
                generative.template,
                datasets=generative.datasets,
                model_id=generative.model_id,
                task_config=generative.task_config)
            result.update(flatten_evalscope_report(summary))
            result['eval_report'] = summary
        if distributed:
            dist.barrier()
    finally:
        if generative.exit is not None:
            generative.exit()
    return result


def _track_best(metrics: dict, best: Tuple[Optional[float], int, Optional[str]], *,
                metric_for_best_model: Optional[str], greater_is_better: bool, global_step: int,
                output_dir: str) -> Tuple[Optional[float], int, Optional[str]]:
    """Update the ``(best_metric, best_step, best_ckpt)`` triple when the watched metric improves."""
    best_metric, best_step, best_ckpt = best
    if not metric_for_best_model:
        return best
    value = metrics.get(metric_for_best_model)
    if value is None:
        logger.warning(f'metric_for_best_model={metric_for_best_model!r} was not produced by this eval '
                       f'(got {sorted(key for key in metrics if key != "step")}); best-model tracking '
                       'skipped for this pass.')
        return best
    value = float(value)
    improved = (best_metric is None
                or (value > best_metric if greater_is_better else value < best_metric))
    if improved:
        return value, global_step, os.path.join(output_dir, f'checkpoint-{global_step}')
    return best


def _load_best_model(model: Any, best_ckpt: Optional[str], *, metric_for_best_model: Optional[str],
                     best_metric: Optional[float], best_step: int,
                     adapter_name: Optional[str]) -> None:
    """Reload the best checkpoint at train end (the caller gates on load_best_model_at_end)."""
    if not best_ckpt:
        logger.warning('load_best_model_at_end is set but no eval produced metric_for_best_model; '
                       'keeping the final weights.')
        return
    if not os.path.isdir(best_ckpt):
        raise FileNotFoundError(
            f'load_best_model_at_end wants {best_ckpt} (best {metric_for_best_model}='
            f'{best_metric} at step {best_step}), but it does not exist: save_steps must align '
            'with eval_steps so the best step is checkpointed, and save_total_limit must not have '
            'pruned it.')
    logger.info(f'load_best_model_at_end: reloading best checkpoint {best_ckpt} '
                f'({metric_for_best_model}={best_metric} at step {best_step}).')
    # LoRA checkpoints reload through their adapter group, so load() targets it rather than the
    # full-parameter weights.
    kwargs = {'adapter_name': adapter_name} if adapter_name else {}
    model.load(best_ckpt, **kwargs)


def _run_supervised(cluster: PreferenceCluster, model_config: 'ModelConfig',
                   template_config: 'TemplateConfig', dataset_config: 'DatasetConfig',
                   train_config: 'TrainConfig', distributed_config: 'DistributedConfig',
                   checkpoint_config: Optional['CheckpointConfig'],
                   tuner_config: Optional['TunerConfig'], logging_config: Optional['LoggingConfig'],
                   quantize_config: Optional['QuantizeConfig'],
                   megatron_config: Optional['MegatronConfig'], *,
                   micro_step_fn: Callable[[Any, Any], Dict[str, Any]],
                   output_dir: str = 'output', _save_final: bool = True,
                   log_formatter: Optional[Callable[[dict, float], str]] = None) -> List[dict]:
    """The preference family's training main: resume, run the epochs, then the final save.

    ``run_dpo`` assembles its cluster (:func:`build_preference_objects`) and hands it here with the
    preference encoder as ``micro_step_fn`` -- how a batch becomes forward kwargs. The loop
    scaffolding (:class:`StepState` cadence, eval + best-model tracking, save/resume) is here.
    """
    from swift.dev2.builders.optimizer import resolve_max_grad_norm
    from twinkle.perf import Perf

    model = cluster.model
    dataloader = cluster.dataloader
    ga = train_config.gradient_accumulation_steps
    max_steps = cluster.total_opt_steps
    num_train_epochs = train_config.num_train_epochs

    # The dissolved component knob sets.
    ckpt = checkpoint_config
    megatron = megatron_config
    manual_gc = bool(megatron and megatron.manual_gc)
    manual_gc_steps = megatron.manual_gc_steps if megatron else 0
    manual_gc_eval = bool(megatron and megatron.manual_gc_eval)
    eval_iters = train_config.eval_iters
    metric_for_best_model = train_config.metric_for_best_model
    greater_is_better = (train_config.greater_is_better if train_config.greater_is_better is not None else
                         not bool(metric_for_best_model and 'loss' in metric_for_best_model.lower()))
    load_best_model_at_end = train_config.load_best_model_at_end
    best_model_adapter_name = None if tuner_config is None else 'default'
    save_knobs = build_save_knobs(
        checkpoint_config, dataset_config, distributed_config, tuner_config, output_dir=output_dir)
    ignore_data_skip = bool(ckpt and ckpt.ignore_data_skip)
    predict_with_generate = train_config.predict_with_generate
    generative = build_generative_eval(
        model=model, template=cluster.template, model_config=model_config, train_config=train_config,
        distributed_config=distributed_config, template_config=template_config, output_dir=output_dir)

    # The shared step cadence + the loop-local eval state (the dissolved halves).
    state = StepState(
        model, train_config,
        gradient_accumulation_steps=ga, max_steps=max_steps,
        max_grad_norm=resolve_max_grad_norm(train_config), logging_config=logging_config,
        output_dir=output_dir, save_steps=(ckpt.save_steps if ckpt else None),
        eval_steps=train_config.eval_steps, eval_delay=int(train_config.eval_delay or 0),
        manual_gc=manual_gc, manual_gc_steps=manual_gc_steps,
        empty_cache_steps=train_config.torch_empty_cache_steps,
        log_formatter=log_formatter)
    handler = state.handler
    eval_history: List[dict] = []
    best_step = 0
    best_ckpt: Optional[str] = None
    warned_eval_dp = False

    def eval_capable() -> bool:
        """Whether an eval path is wired: val-loss needs a dataloader, generative a sampler."""
        return generative is not None if predict_with_generate else cluster.eval_dataloader is not None

    def evaluate() -> Optional[dict]:
        """Run one evaluation pass (val-loss or generative) and track the best checkpoint.

        Both paths append to ``eval_history``, write to the tracker, update best-model tracking,
        fire ``on_evaluate``, and log the scalar summary line.
        """
        nonlocal best_step, best_ckpt, warned_eval_dp
        if predict_with_generate:
            handler.on_predict()
            import torch.distributed as dist
            distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
            if distributed and dist.get_rank() == 0 and not warned_eval_dp:
                # A multi-rank torchrun launch can only be on the transformers-live sampler
                # (vllm/sglang need ray): EvalScope runs on rank 0 alone, at 1/N of the GPUs.
                world_size = dist.get_world_size()
                logger.warning(
                    f'predict_with_generate under torchrun drives EvalScope from rank 0 only '
                    f'(world_size={world_size}): generation uses 1/{world_size} of the launched GPUs '
                    'while the rest wait at the eval barrier. For full-throughput generative eval, '
                    'run under --mode ray with --eval_sampler_backend vllm|sglang.')
                warned_eval_dp = True
            result = _evaluate_generate(generative, global_step=state.global_step)
        else:
            result = _evaluate_loss(
                model, cluster.eval_dataloader, eval_iters=eval_iters, manual_gc=manual_gc,
                manual_gc_eval=manual_gc_eval, micro_step_kwargs=micro_step_fn,
                global_step=state.global_step)
        if result is None:
            return None
        eval_history.append(result)
        state.tracker.log(result, state.global_step)
        state.best_metric, best_step, best_ckpt = _track_best(
            result, (state.best_metric, best_step, best_ckpt), metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better, global_step=state.global_step, output_dir=output_dir)
        state.publish_state(metrics=result)
        handler.on_evaluate(metrics=result)
        scalars = {key: value for key, value in result.items()
                   if key != 'step' and isinstance(value, (int, float))}
        logger.info(f'step {state.global_step}  eval: {scalars}')
        return result

    def final_eval() -> None:
        """Final eval pass so training always ends with an eval point (unless just evaluated)."""
        if eval_capable() and (not eval_history or eval_history[-1].get('step') != state.global_step):
            evaluate()

    def save(name: str, is_final: bool = False) -> str:
        """One training checkpoint, with the callback state published first (its consumers read it)."""
        state.publish_state()
        # The dataloader state is read via ``get_state()`` not an attribute: under ray the driver
        # holds a proxy whose attributes live in the worker, so only the remote call answers.
        get_state = getattr(dataloader, 'get_state', None)
        dataloader_state = get_state() if get_state is not None else {}
        return save_training_step(
            model, handler, name, save_knobs,
            consumed_train_samples=int(dataloader_state.get('consumed_train_samples', 0)),
            component_states={'dataloader': dataloader_state}, is_final=is_final)

    def megatron_step(microbatch_list: list) -> None:
        """One Megatron optimizer step over a microbatch list (GA internal to the single call).

        No ``on_substep_end`` here: Megatron's GA is internal to the one ``forward_backward`` call,
        so the whole group is one optimizer window.
        """
        state.publish_state()
        handler.on_step_begin()
        model.forward_backward(**micro_step_fn(microbatch_list, owns_ga=True))
        handler.on_pre_optimizer_step()
        model.clip_grad_and_step(max_grad_norm=state.max_grad_norm)
        handler.on_optimizer_step()
        state.record()
        state.step_tail(save=save, evaluate=evaluate)

    def fit_transformers() -> None:
        """transformers GA: one micro-batch per forward_backward, twinkle's gate steps the optimizer.

        twinkle's grad-sync gate lags one micro-step, so an optimizer window here spans ga+1
        micro-batches; ``on_step_begin`` fires at each window's first micro-batch (the
        ``at_window_start`` flag, re-set after every boundary) rather than on a clean ga multiple.
        """
        at_window_start = True
        for epoch in range(start_epoch, _epochs(num_train_epochs, max_steps)):
            if state.reached_max() or state.should_stop():
                break
            state.epoch = epoch
            # Reshuffle each epoch (twinkle's EpochSampler derives each epoch's order from its
            # number, and only set_epoch advances it); resume replays it the same way.
            if hasattr(dataloader, 'set_epoch'):
                dataloader.set_epoch(epoch)
            state.publish_state()
            handler.on_epoch_begin()
            batches_this_epoch = 0
            for batch in dataloader:
                batches_this_epoch += 1
                if at_window_start:
                    state.publish_state()
                    handler.on_step_begin()
                    at_window_start = False
                state.micro_step += 1
                # forward_backward / clip_grad_and_step carry @timed at their twinkle definition
                # site, so the span is recorded on the rank that actually runs them.
                model.forward_backward(**micro_step_fn(batch, owns_ga=False))
                state.publish_state()
                handler.on_substep_end()
                # Boundary computed loop-side (backend-agnostic); clip_grad_and_step self-guards
                # via the same predicate on the worker, so calling it every micro-step is safe.
                boundary = state.is_grad_sync_boundary()
                if boundary:
                    handler.on_pre_optimizer_step()
                model.clip_grad_and_step(max_grad_norm=state.max_grad_norm,
                                         gradient_accumulation_steps=ga)
                if boundary:
                    handler.on_optimizer_step()
                    state.record()
                    state.step_tail(save=save, evaluate=evaluate)
                    at_window_start = True
                    if state.reached_max() or state.should_stop():
                        break
            _guard_nonempty_epoch(state.reached_max(), state.should_stop(), epoch, batches_this_epoch)
            state.publish_state()
            handler.on_epoch_end()
            if state.history:
                state.tracker.log(state.history[-1], state.global_step, epoch_end=True)
            if state.should_stop():
                break
        final_eval()

    def fit_megatron() -> None:
        """Megatron GA: ga dataloader batches form one microbatch list, one optimizer step per group.

        ``gradient_accumulation_steps`` is NOT passed (Megatron reads GA from ``len(inputs)``): ga
        dataloader batches concatenated form one optimizer step's microbatch list. The dataloader
        is built with ``drop_last=True`` on Megatron, so a trailing partial group cannot occur
        there; the short-group tail is still handled for safety.
        """
        for epoch in range(start_epoch, _epochs(num_train_epochs, max_steps)):
            if state.reached_max() or state.should_stop():
                break
            state.epoch = epoch
            if hasattr(dataloader, 'set_epoch'):
                dataloader.set_epoch(epoch)
            state.publish_state()
            handler.on_epoch_begin()
            group: list = []
            batches_in_group = 0
            batches_this_epoch = 0
            for batch in dataloader:
                state.micro_step += 1
                batches_this_epoch += 1
                group.extend(batch)
                batches_in_group += 1
                if batches_in_group < ga:
                    continue
                megatron_step(group)
                group, batches_in_group = [], 0
                if state.reached_max() or state.should_stop():
                    break
            if group and not state.reached_max() and not state.should_stop():  # trailing partial group
                megatron_step(group)
            _guard_nonempty_epoch(state.reached_max(), state.should_stop(), epoch, batches_this_epoch)
            state.publish_state()
            handler.on_epoch_end()
            if state.history:
                state.tracker.log(state.history[-1], state.global_step, epoch_end=True)
            if state.should_stop():
                break
        final_eval()

    # Assemble the resume state, then run.
    # Before any training step: a base-modifying LoRA init (pissa ...) needs its initial snapshot.
    snapshot_initial_adapter(model, save_knobs.initial_adapter_dir)
    state.publish_state()
    handler.on_init_end()
    start_epoch = 0
    resume_dir = ckpt.resume_from_checkpoint if ckpt else None
    if resume_dir:
        state.global_step, state.micro_step, start_epoch = resume_training(
            model, checkpoint_config=ckpt, distributed_config=distributed_config,
            megatron_config=megatron_config, tuner_config=tuner_config,
            gradient_accumulation_steps=ga, dataloader=dataloader, ignore_data_skip=ignore_data_skip)

    # TODO(dev2): bridge until the recipe/common batch migrates filter_perf_actors
    from swift.dev.recipe.common.loop_utils import filter_perf_actors

    if Perf.initialize(filter_perf_actors({
            'model': model,
            'dataloader': dataloader,
            'eval_dataloader': cluster.eval_dataloader,
            'processor': cluster.processor,
    })):
        logger.info('[perf] profiling enabled for the offline training loop.')
    # TODO(dev2): bridge until the recipe/common batch migrates the manual-gc helpers
    from swift.dev.recipe.common.loop_utils import finish_manual_gc, start_manual_gc

    gc_was_enabled = start_manual_gc(manual_gc)
    state.publish_state()
    handler.on_train_begin()
    try:
        if train_config.eval_on_start:
            evaluate()
        # TODO(dev2): bridge until the recipe/common batch migrates backend_owns_gradient_accumulation
        from swift.dev.recipe.common.loop_utils import backend_owns_gradient_accumulation

        if backend_owns_gradient_accumulation(model):
            fit_megatron()
        else:
            fit_transformers()
        if load_best_model_at_end:
            _load_best_model(
                model, best_ckpt, metric_for_best_model=metric_for_best_model,
                best_metric=state.best_metric, best_step=best_step,
                adapter_name=best_model_adapter_name)
    finally:
        Perf.shutdown()
        state.publish_state()
        handler.on_train_end()
        # The resident eval sampler is built once and lives across every eval; release it here.
        shutdown = getattr(generative.sampler if generative is not None else None, 'shutdown', None)
        if callable(shutdown):
            shutdown()
        finish_manual_gc(gc_was_enabled)
        state.tracker.close()
    if _save_final:
        # The final save is synchronous, and its args.json must land after every earlier
        # checkpoint's background write finished.
        model.wait_async_saves()
        ckpt_dir = save('checkpoint-final', is_final=True)
        write_ckpt_args_json(
            ckpt_dir, cluster.processor, model_config, template_config, tuner_config,
            quantize_config, task_type=cluster.task_type)
    return state.history


def run_dpo(model_config: 'ModelConfig', template_config: 'TemplateConfig',
            dataset_config: 'DatasetConfig', train_config: 'TrainConfig',
            distributed_config: 'DistributedConfig', checkpoint_config: 'CheckpointConfig',
            rl_config: 'RLConfig', reward_config: Optional['RewardConfig'] = None,
            tuner_config: Optional['TunerConfig'] = None,
            logging_config: Optional['LoggingConfig'] = None,
            quantize_config: Optional['QuantizeConfig'] = None,
            megatron_config: Optional['MegatronConfig'] = None,
            moe_config: Optional['MoEConfig'] = None, *, output_dir: str = 'output',
            _save_final: bool = True) -> List[dict]:
    """Assemble and run an offline preference optimisation from atomic Configs; returns loss history.

    Dispatches on ``rl_config.rl_type`` (dpo/kto/cpo/orpo/simpo/rm): the preference data pipeline
    (raw rows -> chosen/rejected interleave, the KTO rotation + z_KL, the reference-logps forward)
    is the encoder's -- the family loop's ``micro_step_fn`` seam; every other stage rides
    :func:`build_preference_objects` and :func:`_run_supervised`.
    """
    from swift.dev2.recipe.preference.assembly import build_preference_objects
    from swift.dev2.recipe.preference.encoder import PreferenceBatchEncoder, preference_log_line

    rl_type = rl_config.rl_type
    if rl_type not in _OFFLINE_TYPES:
        raise ValueError(f'run_dpo handles the offline preference types {sorted(_OFFLINE_TYPES)}, got '
                         f'rl_type={rl_type!r}. Use run_grpo (online RL), run_distill (distillation) or '
                         'run_ppo instead.')
    cluster = build_preference_objects(
        recipe='run_dpo', model_config=model_config, template_config=template_config,
        dataset_config=dataset_config, train_config=train_config,
        distributed_config=distributed_config, checkpoint_config=checkpoint_config,
        rl_config=rl_config, reward_config=reward_config, tuner_config=tuner_config,
        logging_config=logging_config, quantize_config=quantize_config,
        megatron_config=megatron_config, moe_config=moe_config, output_dir=output_dir)
    encoder = PreferenceBatchEncoder(
        cluster.model, cluster.template, rl_type=rl_type, reference=cluster.reference,
        calculate_kl=bool(rl_config.calculate_KL is not False),
        gradient_accumulation_steps=train_config.gradient_accumulation_steps)
    return _run_supervised(
        cluster, model_config, template_config, dataset_config, train_config, distributed_config,
        checkpoint_config, tuner_config, logging_config, quantize_config, megatron_config,
        micro_step_fn=encoder, output_dir=output_dir, _save_final=_save_final,
        log_formatter=preference_log_line)
