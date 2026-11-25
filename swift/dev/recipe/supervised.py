# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared supervised training for SFT, classification, embedding, ranking, and preference recipes."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from swift.dev.recipe.utils import (
    StepState,
    build_generative_eval,
    build_save_knobs,
    evaluate_generate,
    evaluate_loss,
    fit_supervised,
    load_best_model,
    resume_training,
    save_final_checkpoint,
    save_training_step,
    snapshot_initial_adapter,
    track_best,
    write_ckpt_args_json,
)
from swift.dev.utils.logger import get_logger

if TYPE_CHECKING:
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        LoggingConfig,
        MegatronConfig,
        ModelConfig,
        QuantizeConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )
logger = get_logger()


def _is_vpp_megatron(distributed_config: "DistributedConfig") -> bool:
    """Whether the active backend is megatron with virtual pipeline stages enabled."""
    from swift.dev.builders import is_megatron_backend

    return (is_megatron_backend(distributed_config)
            and (distributed_config.virtual_pipeline_model_parallel_size or 1) > 1)


def _probe_batch_identity(global_step: int, samples: list) -> None:
    """Record which data one update consumed, so a resume can be compared against its source run.

    Gated by ``SWIFT_BATCH_IDENTITY_PROBE`` (a path prefix). A resumed run that trains on a different
    batch than the source run did at the same global step diverges for a reason that has nothing to do
    with weights or optimizer state, and no amount of parameter inspection can tell the two apart: both
    show up as a loss that no longer matches. Digesting the batch itself separates them -- equal
    digests with unequal loss point at the model state, unequal digests point at the dataloader
    position. The digest covers tensor bytes and the sorted scalar structure, so a reorder, a truncation
    and a different sample all change it.
    """
    import hashlib
    import json
    import os

    prefix = os.environ.get('SWIFT_BATCH_IDENTITY_PROBE')
    if not prefix:
        return
    import torch

    def digest(value) -> str:
        hasher = hashlib.blake2b(digest_size=8)
        if torch.is_tensor(value):
            tensor = value.detach().to('cpu').contiguous().flatten()
            hasher.update(str(tuple(value.shape)).encode())
            hasher.update(str(value.dtype).encode())
            # A byte view rather than ``.numpy()``: bfloat16 has no numpy counterpart, and the digest
            # must cover every dtype the dataloader can emit.
            hasher.update(tensor.view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value):
                hasher.update(str(key).encode())
                hasher.update(digest(value[key]).encode())
        elif isinstance(value, (list, tuple)):
            for item in value:
                hasher.update(digest(item).encode())
        else:
            hasher.update(repr(value).encode())
        return hasher.hexdigest()

    record = {
        'rank': int(os.environ.get('RANK') or os.environ.get('LOCAL_RANK') or 0),
        'global_step': int(global_step),
        'num_samples': len(samples),
        'batch': digest(samples),
        'per_sample': [digest(sample) for sample in samples],
    }
    with open(f'{prefix}.rank{record["rank"]}.jsonl', 'a') as handle:
        handle.write(json.dumps(record) + '\n')


def _dump_train_tensors(global_step: int, samples: list) -> None:
    """Write the trainer's actual per-sample tensors when ``SWIFT_TRAIN_TENSOR_DUMP`` names a directory.

    The batch-identity digest next to this is bit-sensitive but not invertible: it can tell that two
    runs differ, not whether the trained span is the one the template contract specifies. Reconciling
    the training scope needs the positions themselves -- labels against input_ids' single shift, which
    roles carry loss_mask, where padding sits -- so this dumps the tensors instead of a hash of them.

    At this seam the dataloader's ``InputFeature`` still holds its encoded columns as Python lists;
    the backend collates them into padded tensors inside ``forward_backward`` on the worker. The
    numeric columns are therefore materialised here, which is what carries the per-sample training
    scope. Padding the collator adds is downstream of this point and is not visible in these files.
    """
    import os
    root = os.environ.get('SWIFT_TRAIN_TENSOR_DUMP')
    if not root:
        return
    import json

    import torch
    os.makedirs(root, exist_ok=True)
    rank = int(os.environ.get('RANK') or os.environ.get('LOCAL_RANK') or 0)
    records = []
    for position, sample in enumerate(samples):
        tensors = {}
        for key, value in sample.items():
            if torch.is_tensor(value):
                tensors[key] = value.detach().to('cpu').contiguous()
            elif isinstance(value, (list, tuple)) and value and all(
                    isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
                tensors[key] = torch.tensor(value)
        scalars = {key: value for key, value in sample.items()
                   if not torch.is_tensor(value) and isinstance(value, (str, int, float, bool))}
        # Media ride as raw objects/paths, so only their COUNT is recordable here. The placeholder
        # tokens they expand to are in input_ids; pairing the two is what catches media dropped or
        # duplicated between the dataset and the trainer. After the processor runs, the raw list is
        # gone and the count lives in the grid tensor's first dimension instead.
        for key in ('images', 'videos', 'audios'):
            value = sample.get(key)
            if isinstance(value, (list, tuple)):
                scalars[f'{key}_count'] = len(value)
        grid = sample.get('image_grid_thw')
        if torch.is_tensor(grid) and grid.dim() == 2 and 'images_count' not in scalars:
            scalars['images_count'] = int(grid.shape[0])
        name = f'step{global_step:05d}.rank{rank}.s{position:03d}.pt'
        torch.save({'tensors': tensors, 'scalars': scalars}, os.path.join(root, name))
        records.append({'global_step': int(global_step), 'rank': rank, 'sample': position, 'file': name,
                        'tensor_keys': sorted(tensors), 'scalars': scalars})
    with open(os.path.join(root, f'index.rank{rank}.jsonl'), 'a', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, default=str) + '\n')


#: Forward tasks whose eval must submit each dataloader batch as ONE whole forward: their
#: labels live per record (decision) or one record expands into several sequences (embedding
#: anchor+positives, reranker query+candidates), so the forward-only micro-batcher's
#: sample-level split desynchronizes labels from logits.
_WHOLE_BATCH_EVAL_TASKS = frozenset({'decision', 'embedding', 'reranker', 'generative_reranker'})


class _Evaluation:
    def __init__(
        self,
        cluster,
        state,
        train_config,
        generative,
        micro_step_fn,
        output_dir,
        *,
        manual_gc,
        manual_gc_eval,
        greater_is_better,
        eval_group_size: int = 1,
        packed_eval: bool = False,
    ):
        self.cluster, self.state, self.model = cluster, state, cluster.model
        self.generative, self.micro_step_fn = generative, micro_step_fn
        self.output_dir = output_dir
        self.manual_gc, self.manual_gc_eval = manual_gc, manual_gc_eval
        self.predict_with_generate = train_config.predict_with_generate
        self.eval_iters = train_config.eval_iters
        self.metric_for_best_model = train_config.metric_for_best_model
        self.greater_is_better = greater_is_better
        # Tasks whose forward output is not one row per record: decision's labels/decision_meta
        # are organized per RECORD (the forward-only readout guard raises on sample-level
        # splits); embedding/reranker expand one record into anchor+candidate sequences, so the
        # merged outputs' sample axis cannot match the micro-plan's record indices. A PACKED
        # eval row is the same shape problem: one dataloader row is a group whose forward the
        # processor unpacks into N per-sequence outputs, so any sample-level micro-plan
        # desynchronizes from the merged logps' axis. All of these must run each batch whole --
        # micro_batch_size=None disables micro-batching while per_device_eval_batch_size still
        # governs the DataLoader.
        if cluster.forward_task in _WHOLE_BATCH_EVAL_TASKS or packed_eval:
            self.micro_batch_size = None
        else:
            self.micro_batch_size = train_config.per_device_eval_batch_size
        self.eval_group_size = eval_group_size
        self.history = []
        self.best_step, self.best_ckpt, self.warned_eval_dp = 0, None, False

    def eval_capable(self) -> bool:
        """Whether an eval path is wired: val-loss needs a dataloader, generative a sampler."""
        return self.generative is not None if self.predict_with_generate else self.cluster.eval_dataloader is not None

    def evaluate(self) -> Optional[dict]:
        """Run one evaluation pass (val-loss or generative) and track the best checkpoint.

        Both paths append to ``eval_history``, write to the tracker, update best-model tracking,
        fire ``on_evaluate``, and log the scalar summary line.
        """
        if self.predict_with_generate:
            self.state.handler.on_predict()
            import torch.distributed as dist

            distributed = dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1)
            if distributed and dist.get_rank() == 0 and (not self.warned_eval_dp):
                # A multi-rank torchrun launch can only be on the transformers-live sampler
                # (vllm/sglang need ray): EvalScope runs on rank 0 alone, at 1/N of the GPUs.
                world_size = dist.get_world_size()
                logger.warning(
                    "predict_with_generate under torchrun drives EvalScope from rank 0 only "
                    f"(world_size={world_size}): generation uses 1/{world_size} of the launched GPUs "
                    "while the rest wait at the eval barrier. For full-throughput generative eval, "
                    "run under --mode ray with --eval_sampler_backend vllm|sglang."
                )
                self.warned_eval_dp = True
            result = evaluate_generate(self.generative, global_step=self.state.global_step)
        else:
            result = evaluate_loss(
                self.model,
                self.cluster.eval_dataloader,
                eval_iters=self.eval_iters,
                manual_gc=self.manual_gc,
                manual_gc_eval=self.manual_gc_eval,
                micro_step_kwargs=self.micro_step_fn,
                global_step=self.state.global_step,
                micro_batch_size=self.micro_batch_size,
                eval_group_size=self.eval_group_size,
            )
        if result is None:
            return None
        self.history.append(result)
        self.state.tracker.log(result, self.state.global_step, stage='evaluation')
        self.state.best_metric, self.best_step, self.best_ckpt = track_best(
            result,
            (self.state.best_metric, self.best_step, self.best_ckpt),
            metric_for_best_model=self.metric_for_best_model,
            greater_is_better=self.greater_is_better,
            global_step=self.state.global_step,
            output_dir=self.output_dir,
        )
        self.state.publish_state(metrics=result)
        self.state.handler.on_evaluate(metrics=result)
        scalars = {key: value for key, value in result.items() if key != "step" and isinstance(value, (int, float))}
        logger.info(f"step {self.state.global_step}  eval: {scalars}")
        return result

    def final_eval(self) -> None:
        """Final eval pass so training always ends with an eval point (unless just evaluated)."""
        if self.eval_capable() and (not self.history or self.history[-1].get("step") != self.state.global_step):
            self.evaluate()


def run_supervised(
    cluster: Any,
    model_config: "ModelConfig",
    template_config: "TemplateConfig",
    dataset_config: "DatasetConfig",
    train_config: "TrainConfig",
    distributed_config: "DistributedConfig",
    checkpoint_config: Optional["CheckpointConfig"],
    tuner_config: Optional["TunerConfig"],
    logging_config: Optional["LoggingConfig"],
    quantize_config: Optional["QuantizeConfig"],
    megatron_config: Optional["MegatronConfig"],
    *,
    micro_step_fn: Callable[..., Dict[str, Any]],
    output_dir: str = "output",
    _save_final: bool = True,
    log_formatter: Optional[Callable[[dict, float], str]] = None,
    wire_callbacks: Optional[Callable[[Any], None]] = None,
) -> List[dict]:
    """The supervised family's training main: resume, run the epochs, then the final save.

    Each entry assembles its cluster (:func:`build_supervised_objects`) and hands it here with its
    loss closure as ``micro_step_fn`` -- how a batch becomes forward kwargs. The loop scaffolding
    (:class:`StepState` cadence, eval + best-model tracking, save/resume) is this function's.
    ``wire_callbacks`` runs once with the assembled ``CallbackHandler``, for task-specific wiring
    the generic loop must not know about (the decision recipe binds its calibration callback).
    """
    from twinkle.perf import Perf

    from swift.dev.builders.optimizer import resolve_max_grad_norm

    model = cluster.model
    dataloader = cluster.dataloader
    ga = train_config.gradient_accumulation_steps
    max_steps = cluster.total_opt_steps

    # The dissolved component knob sets.
    ckpt = checkpoint_config
    megatron = megatron_config
    manual_gc = bool(megatron and megatron.manual_gc)
    manual_gc_steps = megatron.manual_gc_steps if megatron else 0
    manual_gc_eval = bool(megatron and megatron.manual_gc_eval)
    metric_for_best_model = train_config.metric_for_best_model
    greater_is_better = (
        train_config.greater_is_better
        if train_config.greater_is_better is not None
        else not bool(metric_for_best_model and "loss" in metric_for_best_model.lower())
    )
    load_best_model_at_end = train_config.load_best_model_at_end
    best_model_adapter_name = None if tuner_config is None else "default"
    save_knobs = build_save_knobs(
        checkpoint_config, dataset_config, distributed_config, tuner_config, output_dir=output_dir
    )
    ignore_data_skip = bool(ckpt and ckpt.ignore_data_skip)
    generative = build_generative_eval(
        model=model,
        template=cluster.template,
        model_config=model_config,
        train_config=train_config,
        distributed_config=distributed_config,
        template_config=template_config,
        output_dir=output_dir,
    )

    # The shared step cadence + the loop-local eval state (the dissolved halves).
    state = StepState(
        model,
        train_config,
        gradient_accumulation_steps=ga,
        max_steps=max_steps,
        max_grad_norm=resolve_max_grad_norm(train_config),
        logging_config=logging_config,
        output_dir=output_dir,
        save_steps=(ckpt.save_steps if ckpt else None),
        eval_steps=train_config.eval_steps,
        eval_delay=int(train_config.eval_delay or 0),
        manual_gc=manual_gc,
        manual_gc_steps=manual_gc_steps,
        empty_cache_steps=train_config.torch_empty_cache_steps,
        log_formatter=log_formatter,
    )
    handler = state.handler
    if wire_callbacks is not None:
        wire_callbacks(handler)

    evaluation = _Evaluation(
        cluster,
        state,
        train_config,
        generative,
        micro_step_fn,
        output_dir,
        manual_gc=manual_gc,
        manual_gc_eval=manual_gc_eval,
        greater_is_better=greater_is_better,
        eval_group_size=(max(1, train_config.gradient_accumulation_steps)
                         if _is_vpp_megatron(distributed_config) else 1),
        packed_eval=bool(dataset_config.packing),
    )
    evaluate = evaluation.evaluate

    def save(name: str, is_final: bool = False) -> str:
        """One training checkpoint, with the callback state published first (its consumers read it)."""
        state.publish_state()
        # The dataloader state is read via ``get_state()`` not an attribute: under ray the driver
        # holds a proxy whose attributes live in the worker, so only the remote call answers.
        get_state = getattr(dataloader, "get_state", None)
        dataloader_state = get_state() if get_state is not None else {}
        return save_training_step(
            model,
            handler,
            name,
            save_knobs,
            consumed_train_samples=int(dataloader_state.get("consumed_train_samples", 0)),
            component_states={"dataloader": dataloader_state},
            is_final=is_final,
        )

    def update_step(samples: list) -> None:
        """Submit one update; every backend schedules its micro-batches internally."""
        state.publish_state()
        handler.on_step_begin()
        _probe_batch_identity(state.global_step, samples)
        _dump_train_tensors(state.global_step, samples)
        # forward_backward / clip_grad_and_step carry @timed at their twinkle definition
        # site, so the span is recorded on the rank that actually runs them. forward_backward is
        # declared sync=True on the model: an actor-side failure raises at this call, failing THIS
        # step instead of riding an unread handle into a green-looking failed run.
        model.forward_backward(
            **micro_step_fn(samples, micro_batch_size=train_config.per_device_train_batch_size))
        handler.on_pre_optimizer_step()
        model.clip_grad_and_step(max_grad_norm=state.max_grad_norm, gradient_accumulation_steps=1)
        handler.on_optimizer_step()
        state.record()
        state.step_tail(save=save, evaluate=evaluate)

    # Assemble the resume state, then run.
    # Before any training step: a base-modifying LoRA init (pissa ...) needs its initial snapshot.
    snapshot_initial_adapter(model, save_knobs.initial_adapter_dir)
    state.publish_state()
    handler.on_init_end()
    start_epoch = 0
    resume_dir = ckpt.resume_from_checkpoint if ckpt else None
    if resume_dir:
        state.global_step, state.micro_step, start_epoch = resume_training(
            model,
            checkpoint_config=ckpt,
            distributed_config=distributed_config,
            megatron_config=megatron_config,
            tuner_config=tuner_config,
            dataloader=dataloader,
            ignore_data_skip=ignore_data_skip,
        )

    from swift.dev.recipe.utils import filter_perf_actors

    if Perf.initialize(
        filter_perf_actors(
            {
                "model": model,
                "dataloader": dataloader,
                "eval_dataloader": cluster.eval_dataloader,
                "processor": cluster.processor,
            }
        )
    ):
        logger.info("[perf] profiling enabled for the offline training loop.")
    from swift.dev.recipe.utils import finish_manual_gc, start_manual_gc

    gc_was_enabled = start_manual_gc(manual_gc)
    try:
        state.publish_state()
        handler.on_train_begin()
        if train_config.eval_on_start:
            evaluate()
        fit_supervised(
            state,
            dataloader,
            train_config,
            start_epoch=start_epoch,
            update_step=update_step,
        )
        evaluation.final_eval()
        if load_best_model_at_end:
            load_best_model(
                model,
                evaluation.best_ckpt,
                metric_for_best_model=metric_for_best_model,
                best_metric=state.best_metric,
                best_step=evaluation.best_step,
                adapter_name=best_model_adapter_name,
            )
    finally:
        pending_error = sys.exc_info()[1]
        cleanup_error = None

        def cleanup(action):
            nonlocal cleanup_error
            try:
                action()
            except BaseException as exc:
                if pending_error is None and cleanup_error is None:
                    cleanup_error = exc
                else:
                    logger.exception('Secondary error during supervised training cleanup.')

        cleanup(Perf.shutdown)
        cleanup(state.publish_state)
        # Post-training callbacks (temperature calibration) write state only to a SUCCESSFUL run:
        # the flag carries the pending/cleanup error into on_train_end, which fires before the
        # final save, so a failed run keeps its loaded/factory temperatures and no success
        # checkpoint is written.
        handler.state.training_failed = pending_error is not None or cleanup_error is not None
        cleanup(handler.on_train_end)
        if generative is not None:
            cleanup(lambda: getattr(generative.sampler, 'shutdown', lambda: None)())
        cleanup(lambda: finish_manual_gc(gc_was_enabled))
        # Record terminal status before the tracker writes its summary.
        status = 'failed' if pending_error is not None or cleanup_error is not None else 'loop_completed'
        cleanup(lambda: state.finish_metrics(status=status))
        cleanup(state.tracker.close)
        if cleanup_error is not None:
            raise cleanup_error
    ckpt_dir = save_final_checkpoint(model, save, enabled=_save_final)
    # A remote ckpt_dir is a twinkle:// URI in the SERVER's store: the server-side checkpoint has
    # no local dir to copy args.json into, and the run-level args.json (written by the
    # assembly into output_dir) already documents the run for --load_args.
    if _save_final and distributed_config.mode != "remote":
        write_ckpt_args_json(
            ckpt_dir,
            cluster.processor,
            model_config,
            template_config,
            tuner_config,
            quantize_config,
            dataset_config=dataset_config,
            task_type=cluster.task_type,
        )
    return state.history
