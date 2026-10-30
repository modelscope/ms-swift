# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared supervised training for SFT, classification, embedding, ranking, and preference recipes."""

from __future__ import annotations
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
    ):
        self.cluster, self.state, self.model = cluster, state, cluster.model
        self.generative, self.micro_step_fn = generative, micro_step_fn
        self.output_dir = output_dir
        self.manual_gc, self.manual_gc_eval = manual_gc, manual_gc_eval
        self.predict_with_generate = train_config.predict_with_generate
        self.eval_iters = train_config.eval_iters
        self.metric_for_best_model = train_config.metric_for_best_model
        self.greater_is_better = greater_is_better
        self.micro_batch_size = train_config.per_device_eval_batch_size
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
            )
        if result is None:
            return None
        self.history.append(result)
        self.state.tracker.log(result, self.state.global_step)
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
) -> List[dict]:
    """The supervised family's training main: resume, run the epochs, then the final save.

    Each entry assembles its cluster (:func:`build_supervised_objects`) and hands it here with its
    loss closure as ``micro_step_fn`` -- how a batch becomes forward kwargs. The loop scaffolding
    (:class:`StepState` cadence, eval + best-model tracking, save/resume) is this function's.
    """
    from twinkle.perf import Perf

    from swift.dev.builders.optimizer import resolve_max_grad_norm

    model = cluster.model
    dataloader = cluster.dataloader
    ga = train_config.gradient_accumulation_steps
    train_batch_size = train_config.per_device_train_batch_size
    if distributed_config.mode == "ray":
        train_batch_size *= model.device_mesh.data_world_size
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
        model.forward_backward(**micro_step_fn(samples, micro_batch_size=train_config.per_device_train_batch_size))
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
    state.publish_state()
    handler.on_train_begin()
    try:
        if train_config.eval_on_start:
            evaluate()
        fit_supervised(
            state,
            dataloader,
            train_config,
            start_epoch=start_epoch,
            train_batch_size=train_batch_size,
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
        Perf.shutdown()
        state.publish_state()
        handler.on_train_end()
        # The resident eval sampler is built once and lives across every eval; release it here.
        shutdown = getattr(generative.sampler if generative is not None else None, "shutdown", None)
        if callable(shutdown):
            shutdown()
        finish_manual_gc(gc_was_enabled)
        state.tracker.close()
    ckpt_dir = save_final_checkpoint(model, save, enabled=_save_final)
    # Remote checkpoints are twinkle:// URIs; the assembly writes the local run args.
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
