# Copyright (c) ModelScope Contributors. All rights reserved.
# The supervised family: the five visible mains plus the family's own training loop.
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from swift.dev.utils.logger import get_logger

from swift.dev.recipe.sft.assembly import SupervisedCluster, build_supervised_objects
from swift.dev.recipe.utils import (
    StepState, build_generative_eval, build_save_knobs, evaluate_generate,
    evaluate_loss, load_best_model, resume_training, save_training_step, snapshot_initial_adapter,
    track_best, write_ckpt_args_json)

if TYPE_CHECKING:
    from swift.dev.config import (
        CheckpointConfig, DatasetConfig, DistributedConfig, LoggingConfig, MegatronConfig, ModelConfig,
        MoEConfig, QuantizeConfig, TemplateConfig, TrainConfig, TunerConfig)

logger = get_logger()

#: twinkle's task name; selects the embedding pooling patch inside forward.
EMBEDDING_TASK = 'embedding'
#: twinkle's task name; the head produces logits, so forward installs no patch.
SEQ_CLS_TASK = 'seq_cls'
#: task_types the reranker recipe accepts; each threads to twinkle's forward as-is.
RERANKER_TASK_TYPES = ('reranker', 'generative_reranker')


def _supervised_micro_kwargs(task: str, ga: int) -> Callable[..., Dict[str, Any]]:
    """The forward-kwargs factory the loops and the val-loss eval share.

    The dataloader already yields ``list[InputFeature]``, so the leaf is a passthrough of ``inputs``
    plus the forward ``task``: ``owns_ga=False`` (transformers) adds ``gradient_accumulation_steps``
    so twinkle's grad-sync gate steps the optimizer every ga-th micro-batch; ``owns_ga=True``
    (Megatron) omits it, deriving GA from the microbatch-list length.
    """

    def micro_step_kwargs(unit, *, owns_ga: bool):
        kwargs = {'inputs': unit, 'task': task}
        if not owns_ga:
            kwargs['gradient_accumulation_steps'] = ga
        return kwargs

    return micro_step_kwargs


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


def _run_supervised(cluster: SupervisedCluster, model_config: 'ModelConfig',
                   template_config: 'TemplateConfig', dataset_config: 'DatasetConfig',
                   train_config: 'TrainConfig', distributed_config: 'DistributedConfig',
                   checkpoint_config: Optional['CheckpointConfig'],
                   tuner_config: Optional['TunerConfig'], logging_config: Optional['LoggingConfig'],
                   quantize_config: Optional['QuantizeConfig'],
                   megatron_config: Optional['MegatronConfig'], *,
                   micro_step_fn: Callable[[Any, Any], Dict[str, Any]],
                   output_dir: str = 'output', _save_final: bool = True,
                   log_formatter: Optional[Callable[[dict, float], str]] = None) -> List[dict]:
    """The supervised family's training main: resume, run the epochs, then the final save.

    Each entry assembles its cluster (:func:`build_supervised_objects`) and hands it here with its
    loss closure as ``micro_step_fn`` -- how a batch becomes forward kwargs. The loop scaffolding
    (:class:`StepState` cadence, eval + best-model tracking, save/resume) is this function's.
    """
    from swift.dev.builders.optimizer import resolve_max_grad_norm
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
            result = evaluate_generate(generative, global_step=state.global_step)
        else:
            result = evaluate_loss(
                model, cluster.eval_dataloader, eval_iters=eval_iters, manual_gc=manual_gc,
                manual_gc_eval=manual_gc_eval, micro_step_kwargs=micro_step_fn,
                global_step=state.global_step)
        if result is None:
            return None
        eval_history.append(result)
        state.tracker.log(result, state.global_step)
        state.best_metric, best_step, best_ckpt = track_best(
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

    from swift.dev.recipe.utils import filter_perf_actors

    if Perf.initialize(filter_perf_actors({
            'model': model,
            'dataloader': dataloader,
            'eval_dataloader': cluster.eval_dataloader,
            'processor': cluster.processor,
    })):
        logger.info('[perf] profiling enabled for the offline training loop.')
    from swift.dev.recipe.utils import finish_manual_gc, start_manual_gc

    gc_was_enabled = start_manual_gc(manual_gc)
    state.publish_state()
    handler.on_train_begin()
    try:
        if train_config.eval_on_start:
            evaluate()
        from swift.dev.recipe.utils import backend_owns_gradient_accumulation

        if backend_owns_gradient_accumulation(model):
            fit_megatron()
        else:
            fit_transformers()
        if load_best_model_at_end:
            load_best_model(
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
            quantize_config, dataset_config=dataset_config, task_type=cluster.task_type)
    return state.history


def run_sft(model_config: 'ModelConfig', template_config: 'TemplateConfig',
            dataset_config: 'DatasetConfig', train_config: 'TrainConfig',
            distributed_config: 'DistributedConfig', checkpoint_config: 'CheckpointConfig',
            tuner_config: Optional['TunerConfig'] = None,
            logging_config: Optional['LoggingConfig'] = None,
            quantize_config: Optional['QuantizeConfig'] = None,
            megatron_config: Optional['MegatronConfig'] = None,
            moe_config: Optional['MoEConfig'] = None, *, output_dir: str = 'output',
            _save_final: bool = True) -> List[dict]:
    """Assemble and run causal-LM SFT from atomic Configs; returns the loss/grad_norm history.

    SFT's own contribution is one thing, the objective; the config -> object stages run in
    :func:`build_supervised_objects` and the loop in :func:`_run_supervised`. ``_save_final``
    (internal) skips the final checkpoint for the loss-trajectory-only test path.
    """
    from swift.dev.loss import configure_loss, liger_fused_ce_enabled

    # Fused-linear-CE pairs a loss choice with the loop's forward task (one predicate feeds both):
    # configure_loss selects Liger's kernel, loop_task='fused_lm_ce' skips the lm_head GEMM.
    fused_ce = liger_fused_ce_enabled(train_config)

    def sft_loss(model) -> None:
        configure_loss(
            model,
            loss_type=train_config.loss or 'cross_entropy',
            enable_channel_loss=train_config.enable_channel_loss,
            dft=train_config.enable_dft_loss,
            fused_linear_cross_entropy=fused_ce,
            label_smoothing=train_config.label_smoothing_factor)

    cluster = build_supervised_objects(
        recipe='run_sft', model_config=model_config, template_config=template_config,
        dataset_config=dataset_config, train_config=train_config,
        distributed_config=distributed_config, checkpoint_config=checkpoint_config,
        tuner_config=tuner_config, logging_config=logging_config, quantize_config=quantize_config,
        megatron_config=megatron_config, moe_config=moe_config, task=None,
        loop_task=('fused_lm_ce' if fused_ce else None), configure_loss=sft_loss,
        configure_metric=None, output_dir=output_dir)
    return _run_supervised(
        cluster, model_config, template_config, dataset_config, train_config, distributed_config,
        checkpoint_config, tuner_config, logging_config, quantize_config, megatron_config,
        micro_step_fn=_supervised_micro_kwargs(cluster.forward_task,
                                               train_config.gradient_accumulation_steps),
        output_dir=output_dir, _save_final=_save_final)


def run_pt(model_config: 'ModelConfig', template_config: 'TemplateConfig',
           dataset_config: 'DatasetConfig', train_config: 'TrainConfig',
           distributed_config: 'DistributedConfig', checkpoint_config: 'CheckpointConfig',
           tuner_config: Optional['TunerConfig'] = None,
           logging_config: Optional['LoggingConfig'] = None,
           quantize_config: Optional['QuantizeConfig'] = None,
           megatron_config: Optional['MegatronConfig'] = None,
           moe_config: Optional['MoEConfig'] = None, *, output_dir: str = 'output',
           _save_final: bool = True) -> List[dict]:
    """Run causal-LM pretraining while enforcing the non-chat loss contract (a tightened run_sft)."""
    if model_config.task_type not in (None, 'causal_lm'):
        raise ValueError(f'PT requires task_type="causal_lm", got {model_config.task_type!r}.')
    if template_config.use_chat_template not in (None, False):
        raise ValueError('PT requires use_chat_template=false.')
    if template_config.loss_scale not in ('default', 'all'):
        raise ValueError('PT requires loss_scale="all".')
    model_config.task_type = 'causal_lm'
    template_config.use_chat_template = False
    template_config.loss_scale = 'all'
    return run_sft(
        model_config, template_config, dataset_config, train_config, distributed_config,
        checkpoint_config, tuner_config, logging_config, quantize_config, megatron_config, moe_config,
        output_dir=output_dir, _save_final=_save_final)


def run_embedding(model_config: 'ModelConfig', template_config: 'TemplateConfig',
                  dataset_config: 'DatasetConfig', train_config: 'TrainConfig',
                  distributed_config: 'DistributedConfig', checkpoint_config: 'CheckpointConfig',
                  tuner_config: Optional['TunerConfig'] = None,
                  logging_config: Optional['LoggingConfig'] = None,
                  quantize_config: Optional['QuantizeConfig'] = None, *,
                  output_dir: str = 'output', _save_final: bool = True) -> List[dict]:
    """Assemble and run embedding training from atomic Configs; returns the loss/grad_norm history.

    The loss is chosen by ``TrainConfig.loss`` (default 'infonce') and Matryoshka aggregation by
    ``TrainConfig.mrl_dims``; both are read here rather than passed separately so the same Config
    that drives legacy drives this recipe. ``_save_final`` has the same test-oriented meaning.
    """
    from swift.dev.loss import configure_embedding_loss, configure_embedding_metric

    def embedding_loss(model) -> None:
        # Runs on Megatron too: its scheduler pools the last stage's output and calls loss_instance
        # explicitly, with process_group bound to the DP group so the in-batch all-gather is safe.
        configure_embedding_loss(model, loss_type=train_config.loss or 'infonce', mrl_dims=train_config.mrl_dims)

    cluster = build_supervised_objects(
        recipe='run_embedding', model_config=model_config, template_config=template_config,
        dataset_config=dataset_config, train_config=train_config,
        distributed_config=distributed_config, checkpoint_config=checkpoint_config,
        tuner_config=tuner_config, logging_config=logging_config, quantize_config=quantize_config,
        megatron_config=None, moe_config=None, task=EMBEDDING_TASK, loop_task=None,
        configure_loss=embedding_loss, configure_metric=configure_embedding_metric,
        output_dir=output_dir)
    return _run_supervised(
        cluster, model_config, template_config, dataset_config, train_config, distributed_config,
        checkpoint_config, tuner_config, logging_config, quantize_config, None,
        micro_step_fn=_supervised_micro_kwargs(cluster.forward_task,
                                               train_config.gradient_accumulation_steps),
        output_dir=output_dir, _save_final=_save_final)


def run_reranker(model_config: 'ModelConfig', template_config: 'TemplateConfig',
                 dataset_config: 'DatasetConfig', train_config: 'TrainConfig',
                 distributed_config: 'DistributedConfig', checkpoint_config: 'CheckpointConfig',
                 tuner_config: Optional['TunerConfig'] = None,
                 logging_config: Optional['LoggingConfig'] = None,
                 quantize_config: Optional['QuantizeConfig'] = None, *,
                 output_dir: str = 'output', _save_final: bool = True) -> List[dict]:
    """Assemble and run reranker training from atomic Configs; returns the loss/grad_norm history.

    Serves a SET of task_types (``reranker`` rides a num_labels=1 SequenceClassification head;
    ``generative_reranker`` keeps the CausalLM and scores via the vocab head), so it resolves and
    validates its own rather than comparing against a single constant; the resolved value routes
    the template (reranker vs generative_reranker encode) and the model head.
    """
    from swift.dev.loss import configure_reranker_loss

    task_type = model_config.task_type or 'reranker'
    if task_type not in RERANKER_TASK_TYPES:
        raise ValueError(f'run_reranker requires ModelConfig.task_type in {list(RERANKER_TASK_TYPES)} '
                         f'(or None), got {task_type!r}.')

    def reranker_loss(model) -> None:
        # Runs on Megatron too: under task='reranker' the Megatron scheduler reduces the per-token
        # head output to the last valid token and calls loss_instance explicitly.
        configure_reranker_loss(model, loss_type=train_config.loss or 'pointwise_reranker')

    cluster = build_supervised_objects(
        recipe='run_reranker', model_config=model_config, template_config=template_config,
        dataset_config=dataset_config, train_config=train_config,
        distributed_config=distributed_config, checkpoint_config=checkpoint_config,
        tuner_config=tuner_config, logging_config=logging_config, quantize_config=quantize_config,
        megatron_config=None, moe_config=None, task=task_type, loop_task=None,
        configure_loss=reranker_loss, configure_metric=None, output_dir=output_dir)
    return _run_supervised(
        cluster, model_config, template_config, dataset_config, train_config, distributed_config,
        checkpoint_config, tuner_config, logging_config, quantize_config, None,
        micro_step_fn=_supervised_micro_kwargs(cluster.forward_task,
                                               train_config.gradient_accumulation_steps),
        output_dir=output_dir, _save_final=_save_final)


def run_seq_cls(model_config: 'ModelConfig', template_config: 'TemplateConfig',
                dataset_config: 'DatasetConfig', train_config: 'TrainConfig',
                distributed_config: 'DistributedConfig', checkpoint_config: 'CheckpointConfig',
                tuner_config: Optional['TunerConfig'] = None,
                logging_config: Optional['LoggingConfig'] = None,
                quantize_config: Optional['QuantizeConfig'] = None, *,
                output_dir: str = 'output', _save_final: bool = True) -> List[dict]:
    """Assemble and run seq_cls training from atomic Configs; returns the loss/grad_norm history.

    ``ModelConfig.problem_type`` is REQUIRED -- it selects the training objective (regression -> MSE,
    single_label -> CE, multi_label -> BCE) and is not inferred; ``num_labels`` sizes the head.
    """
    from swift.dev.loss import configure_seq_cls_loss

    problem_type = model_config.problem_type
    if problem_type is None:
        raise ValueError('run_seq_cls requires ModelConfig.problem_type '
                         '(regression / single_label_classification / multi_label_classification).')
    num_labels = model_config.num_labels
    if num_labels is None:
        raise ValueError('run_seq_cls requires ModelConfig.num_labels.')

    def seq_cls_loss(model) -> None:
        # Runs on Megatron too: under task='seq_cls' the Megatron scheduler pools the last stage's
        # per-token head output to the last valid token and calls loss_instance explicitly.
        configure_seq_cls_loss(model, problem_type=problem_type, num_labels=num_labels)

    cluster = build_supervised_objects(
        recipe='run_seq_cls', model_config=model_config, template_config=template_config,
        dataset_config=dataset_config, train_config=train_config,
        distributed_config=distributed_config, checkpoint_config=checkpoint_config,
        tuner_config=tuner_config, logging_config=logging_config, quantize_config=quantize_config,
        megatron_config=None, moe_config=None, task=SEQ_CLS_TASK, loop_task=None,
        configure_loss=seq_cls_loss, configure_metric=None, output_dir=output_dir)
    return _run_supervised(
        cluster, model_config, template_config, dataset_config, train_config, distributed_config,
        checkpoint_config, tuner_config, logging_config, quantize_config, None,
        micro_step_fn=_supervised_micro_kwargs(cluster.forward_task,
                                               train_config.gradient_accumulation_steps),
        output_dir=output_dir, _save_final=_save_final)
