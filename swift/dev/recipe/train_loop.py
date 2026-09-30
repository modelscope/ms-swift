"""SFTLoop: minimal transparent SFT training loop.

The thinnest possible SFT recipe: just iterate the dataloader, forward_backward,
clip_grad_and_step, periodically log/save. No RL policies, no mode hooks.
Backend-agnostic: works with any TrainableModel (twinkle-derived TransformersModel /
MegatronModel).

Cookbook users may copy this loop or write their own; CLI uses it as the default
SFT orchestration.
"""
from __future__ import annotations
import logging
import math
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from swift.dev.config import LoggingConfig
    from swift.dev.model import TrainableModel

logger = logging.getLogger(__name__)


def _backend_owns_gradient_accumulation(model) -> bool:
    """True when the backend's ``forward_backward`` internalizes gradient accumulation.

    fit() forks on exactly one axis: who schedules the GA micro-steps. Most backends (transformers) leave
    it to the loop -- one micro-batch per ``forward_backward``, the optimizer stepped every ga-th call. A
    backend that owns its own micro-batch schedule needs the loop to instead group the ga batches and call
    ``forward_backward`` once per optimizer step; in dev that is Megatron, whose 1F1B pipeline derives GA
    from the length of the micro-batch LIST it is handed. A future such backend (e.g. torchtitan/veomni
    running pipeline parallelism) is recognized HERE, so the dispatch stays a two-way fork on this
    capability rather than growing a branch per backend name.
    """
    try:
        from swift.dev.model.megatron.model import MegatronModel
    except Exception:
        return False
    return isinstance(model, MegatronModel)


def flatten_evalscope_report(summary: Any) -> Dict[str, float]:
    """Flatten EvalScope report rows into a ``{metric_name: score}`` dict.

    ``summarize`` returns one report dict per benchmark (an ``evalscope.report.Report``): its overall
    ``score`` is keyed by the benchmark ``name``, and each entry in ``metrics`` contributes its own
    ``name``/``score``. ``metric_for_best_model`` on the generative path names one of these keys, so
    best-model tracking and ``on_evaluate`` read a flat mapping rather than the nested report.
    """
    flat: Dict[str, float] = {}
    for row in summary or []:
        if not isinstance(row, dict):
            continue
        if row.get('name') is not None and row.get('score') is not None:
            flat[str(row['name'])] = float(row['score'])
        for metric in row.get('metrics') or []:
            if isinstance(metric, dict) and metric.get('name') is not None and metric.get('score') is not None:
                flat[str(metric['name'])] = float(metric['score'])
    return flat


# GA step arithmetic (twinkle's grad-sync gate lags one micro-step). Lives here because the
# only current consumer is the SFT loop; not inherently SFT-specific, so if another training
# path needs it later, lift it to a shared home.
def num_optimizer_steps(num_micro_batches: int, gradient_accumulation_steps: int) -> int:
    """Optimizer-update count for a given number of micro-batches under twinkle GA.

    twinkle's do_grad_sync fires at cur_step = ga+1, 2*ga+1, ... (one micro-step late),
    so over N micro-batches the number of optimizer steps is floor((N-1)/ga) for ga>1,
    and N for ga==1. Used to size the LR scheduler's num_training_steps so it matches the
    actual number of steps taken.
    """
    ga = max(1, gradient_accumulation_steps)
    if ga == 1:
        return num_micro_batches
    return max(0, (num_micro_batches - 1) // ga)


def start_manual_gc(enabled: bool) -> Optional[bool]:
    """Disable automatic GC for a training loop and perform one full collection."""
    if not enabled:
        return None
    import gc
    was_enabled = gc.isenabled()
    gc.disable()
    gc.collect()
    return was_enabled


def collect_manual_gc(enabled: bool, interval: int, step: int) -> None:
    """Run the configured periodic full collection after an optimizer step."""
    if enabled and interval and step % interval == 0:
        import gc
        gc.collect()


def finish_manual_gc(was_enabled: Optional[bool]) -> None:
    """Restore the interpreter's automatic-GC state after a loop exits."""
    if was_enabled:
        import gc
        gc.enable()


def save_training_checkpoint(model: 'TrainableModel', name: str, *, output_dir: str,
                             consumed_train_samples: int = 0, no_save_optim: bool = False,
                             no_save_rng: bool = False, safe_serialization: bool = True,
                             max_shard_size: str = '5GB', save_total_limit: Optional[int] = None) -> str:
    """Save one trainable model, then retain only the newest numbered/final checkpoints."""
    kwargs = {
        'consumed_train_samples': consumed_train_samples,
        'safe_serialization': safe_serialization,
        'max_shard_size': max_shard_size,
        'save_total_limit': save_total_limit,
        # TransformersModel accepts extra save kwargs; MegatronModel consumes this one.
        'no_save_rng': no_save_rng,
    }
    return model.save(name, output_dir=output_dir, save_optimizer=not no_save_optim, **kwargs)


class SFTLoop:
    """Minimal SFT training loop over a dataloader yielding list[InputFeature]."""

    def __init__(
        self,
        model: TrainableModel,
        dataloader: Any,
        *,
        max_steps: int = -1,
        num_train_epochs: float = 1.0,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        logging_steps: int = 1,
        logging_config: Optional['LoggingConfig'] = None,
        save_steps: Optional[int] = None,
        output_dir: str = 'output',
        eval_dataloader: Any = None,
        eval_steps: Optional[int] = None,
        eval_iters: int = -1,
        task: str = 'causal_lm',
        no_save_optim: bool = False,
        no_save_rng: bool = False,
        safe_serialization: bool = True,
        max_shard_size: str = '5GB',
        save_total_limit: Optional[int] = None,
        ignore_data_skip: bool = False,
        manual_gc: bool = False,
        manual_gc_eval: bool = True,
        manual_gc_steps: int = 0,
        callbacks: Optional[List[Any]] = None,
        predict_with_generate: bool = False,
        eval_sampler: Any = None,
        eval_template: Any = None,
        eval_datasets: Optional[List[str]] = None,
        eval_model_id: Optional[str] = None,
        eval_task_config: Optional[Dict[str, Any]] = None,
        eval_enter: Optional[Any] = None,
        eval_exit: Optional[Any] = None,
        eval_delay: int = 0,
        eval_on_start: bool = False,
        metric_for_best_model: Optional[str] = None,
        greater_is_better: Optional[bool] = None,
        load_best_model_at_end: bool = False,
        best_model_adapter_name: Optional[str] = None,
        hub_pusher: Optional[Any] = None,
        hub_strategy: str = 'every_save',
    ):
        self.model = model
        self.dataloader = dataloader
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.max_grad_norm = max_grad_norm
        self.logging_steps = logging_config.logging_steps if logging_config is not None else logging_steps
        self.logging_config = logging_config
        from swift.dev.recipe.tracking import RunTracker
        self.tracker = RunTracker(logging_config, output_dir)
        self.save_steps = save_steps
        self.output_dir = output_dir
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.eval_dataloader = eval_dataloader
        self.eval_steps = eval_steps
        self.eval_iters = eval_iters
        self.no_save_optim = no_save_optim
        self.no_save_rng = no_save_rng
        self.safe_serialization = safe_serialization
        self.max_shard_size = max_shard_size
        self.save_total_limit = save_total_limit
        self.ignore_data_skip = ignore_data_skip
        self.manual_gc = manual_gc
        self.manual_gc_eval = manual_gc_eval
        self.manual_gc_steps = manual_gc_steps
        if self.manual_gc_steps < 0:
            raise ValueError('manual_gc_steps must be >= 0.')
        # Forwarded verbatim to twinkle's forward_backward/forward_only. task='embedding' swaps the
        # lm_head for the pooling patch, so the loss reads outputs['embeddings'] instead of logits;
        # 'causal_lm' (the default) keeps the SFT path byte-identical.
        self.task = task
        # Who schedules the GA micro-steps (see _backend_owns_gradient_accumulation): most backends leave
        # it to the loop, but a backend that owns its own micro-batch schedule (Megatron's 1F1B pipeline
        # derives GA from the micro-batch LIST length, and stores forward kwargs for metric accumulation)
        # must be fed the whole group in one forward_backward -- passing gradient_accumulation_steps would
        # collide with its own accumulate(). Detected once so fit() forks on it.
        self._owns_gradient_accumulation = _backend_owns_gradient_accumulation(model)
        # Warn once about the 1/N generative-eval throughput under torchrun (see _evaluate_generate).
        self._warned_eval_dp = False

        # optimizer step counter (increments once per GA window)
        self.global_step = 0
        self.micro_step = 0
        self.history: list = []
        self.eval_history: list = []
        # First epoch to run; advanced by resume() so cross-epoch resume doesn't replay
        # already-consumed epochs.
        self._start_epoch = 0

        # --- callbacks (design 2.3): ONE handler fans every loop event out to the registered callbacks
        # and holds the shared CallbackState / CallbackControl the loop publishes to and honours. ---
        from swift.dev.callbacks import CallbackHandler
        self.handler = CallbackHandler(callbacks, output_dir=output_dir)

        # --- generative eval (design 1.1): predict_with_generate swaps the val-loss path for an
        # EvalScope benchmark driven through the resident sampler the assembly wired. eval_sampler is
        # None on the val-loss path, so _eval_capable() gates the generative dispatch on it. ---
        self.predict_with_generate = predict_with_generate
        self.eval_sampler = eval_sampler
        self.eval_template = eval_template
        self.eval_datasets = eval_datasets
        self.eval_model_id = eval_model_id
        self.eval_task_config = eval_task_config
        # Optional bracketing of the generative eval run with the assembly's weight-sync + memory
        # schedule for a co-resident vllm/sglang engine; both None on the transformers-live path.
        self.eval_enter = eval_enter
        self.eval_exit = eval_exit
        self.eval_delay = eval_delay
        self.eval_on_start = eval_on_start

        # --- best-model tracking (design 1.5): the watched metric, its best value/step, and the
        # checkpoint that produced it, reloaded at train end when load_best_model_at_end is set. ---
        self.metric_for_best_model = metric_for_best_model
        # An explicit flag wins; otherwise a metric with 'loss' in its name is smaller-is-better.
        self.greater_is_better = (greater_is_better if greater_is_better is not None else
                                  not bool(metric_for_best_model and 'loss' in metric_for_best_model.lower()))
        self.load_best_model_at_end = load_best_model_at_end
        # LoRA checkpoints reload through their adapter group; the assembly passes 'default' (the name
        # apply_tuner created) so load() targets it rather than the full-parameter weights.
        self.best_model_adapter_name = best_model_adapter_name
        self.best_metric: Optional[float] = None
        self.best_step = 0
        self.best_ckpt: Optional[str] = None

        # --- hub push (design 6.1): an optional callable the assembly injects to upload one checkpoint
        # dir (the assembly owns the hub credentials/target/revision). The loop fires on_push_begin then
        # calls it after a save; hub_strategy='end' defers the push to the final checkpoint only. ---
        self.hub_pusher = hub_pusher
        self.hub_strategy = hub_strategy

        # Epoch index published onto CallbackState; advanced by the fit loops.
        self._epoch = 0
        self._sync_state()
        self.handler.on_init_end()

    def _reached_max(self) -> bool:
        return self.max_steps > 0 and self.global_step >= self.max_steps

    def _is_grad_sync_boundary(self) -> bool:
        """Whether the current micro_step is an optimizer-update boundary.

        Replicates twinkle optimizer_group.do_grad_sync loop-side (using self.micro_step as
        twinkle's cur_step, which resume() keeps aligned): boundary when ga==1, else one micro-step
        late at cur_step = ga+1, 2ga+1, .... Computed here rather than read off model.optimizer_group
        so it works for BOTH the in-process transformers model and the Ray-remote Megatron model
        (whose optimizer_group lives on the workers, not the driver handle).
        """
        ga = self.gradient_accumulation_steps
        return ga == 1 or ((self.micro_step - 1) % ga == 0 and self.micro_step > 1)

    def _sync_state(self, *, loss: Optional[float] = None, metrics: Optional[dict] = None) -> None:
        """Publish the loop's current position onto the shared CallbackState before an event fires.

        ``loss``/``metrics`` are only overwritten when supplied, so a step event keeps the last
        training record while an eval event publishes its own.
        """
        state = self.handler.state
        state.global_step = self.global_step
        state.micro_step = self.micro_step
        state.epoch = self._epoch
        state.best_metric = self.best_metric
        if loss is not None:
            state.loss = loss
        if metrics is not None:
            state.metrics = metrics

    def _should_stop(self) -> bool:
        """Whether a callback (early stop / graceful exit) asked the loop to break."""
        return self.handler.control.should_training_stop

    def _is_main_process(self) -> bool:
        """True on the rank that owns single-instance side effects (the hub upload).

        Under torchrun every rank drives its own copy of this loop, so an unguarded upload would run
        once per rank; under ray the loop runs only on the driver, where ``dist`` is not initialized
        and this is trivially the single owner.
        """
        import torch.distributed as dist
        return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0

    def _eval_capable(self) -> bool:
        """Whether an eval path is actually wired: val-loss needs a dataloader, generative a sampler."""
        return self.eval_sampler is not None if self.predict_with_generate else self.eval_dataloader is not None

    def _due_for_eval(self) -> bool:
        """Whether this optimizer step should trigger a periodic evaluation pass."""
        if not self.eval_steps or not self._eval_capable():
            return False
        if self.global_step % self.eval_steps != 0:
            return False
        # eval_delay skips the first N optimizer steps (0 -> no skip).
        return self.global_step > self.eval_delay

    def evaluate(self) -> Optional[dict]:
        """Run one evaluation pass and return its metrics dict (or None when no path is wired).

        Dispatches on ``predict_with_generate``: the default val-loss path scores the eval_dataloader
        with the training loss; the generative path runs an EvalScope benchmark through the resident
        sampler. Both append to ``eval_history``, write to the tracker, update best-model tracking,
        and fire ``on_evaluate``.
        """
        result = self._evaluate_generate() if self.predict_with_generate else self._evaluate_loss()
        if result is None:
            return None
        self.eval_history.append(result)
        self.tracker.log(result, self.global_step)
        self._track_best(result)
        self._sync_state(metrics=result)
        self.handler.on_evaluate(metrics=result)
        scalars = {key: value for key, value in result.items() if key != 'step' and isinstance(value, (int, float))}
        logger.info(f'step {self.global_step}  eval: {scalars}')
        return result

    def _evaluate_loss(self) -> Optional[dict]:
        """Val-loss eval: normalized per-token loss over eval_dataloader (None if there is none).

        Mirrors the train metric path on the eval side (calculate_metrics(False)):
        for each val batch call forward_only (no grad, sets eval_status + accumulates the
        prior batch's metric) then calculate_loss (fills THIS batch's eval loss/num_tokens).
        A final calculate_metrics(False) aggregates over all batches (num_tokens-normalized)
        and resets.
        """
        if self.eval_dataloader is None:
            return None
        if self.manual_gc and self.manual_gc_eval:
            import gc
            gc.collect()
        for index, batch in enumerate(self.eval_dataloader):
            if self.eval_iters > 0 and index >= self.eval_iters:
                break
            self.model.forward_only(inputs=batch, task=self.task)
            # Fill eval_status loss/num_tokens for this batch. On the transformers backend the CE
            # loss is computed here (forward_only only stores inputs/outputs). On Megatron the
            # pipeline scheduler already produced the loss inside forward_only and calculate_loss
            # raises NotImplementedError -- treat that as "already populated" (recipes rely only
            # on forward_backward/forward_only, never the 3-way split).
            try:
                self.model.calculate_loss()
            except NotImplementedError:
                pass
        # model.calculate_metric is the driver-callable metric path (works for the in-process
        # transformers model and the Ray-remote Megatron model alike; it resolves the active
        # optimizer group internally on the worker).
        metrics = self.model.calculate_metric(is_training=False)
        result = {'step': self.global_step}
        if metrics.get('loss') is not None:
            result['eval_loss'] = float(metrics['loss'])
        for key, value in metrics.items():
            if key.startswith('loss_'):
                result[f'eval_{key}'] = float(value)
        if self.manual_gc and self.manual_gc_eval:
            import gc
            gc.collect(0)
        return result

    def _evaluate_generate(self) -> Optional[dict]:
        """Generative eval: run an EvalScope benchmark over the resident sampler (None if unwired).

        Only rank 0 drives EvalScope (a single process owns the report files); the other ranks wait at
        a barrier so the next training step stays collective-safe. The sampler -- a transformers-live
        wrapper or a weight-synced vllm/sglang engine -- is built and owned by the assembly, so this
        path only fires ``on_predict`` and runs it.
        """
        if self.eval_sampler is None:
            return None
        import torch.distributed as dist

        from swift.dev.eval import run_evalscope
        self.handler.on_predict()
        distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        is_main = (not distributed) or dist.get_rank() == 0
        if distributed and is_main and not self._warned_eval_dp:
            # A multi-rank local/torchrun launch can only be on the transformers-live sampler (vllm/sglang
            # need ray, and a param-sharding strategy is refused by _check_eval_generation), so this path
            # runs EvalScope on rank 0 alone while the peers idle at the barrier below: correct, but at 1/N
            # of the launched GPUs. Surface it once rather than let the throughput drop pass unexplained.
            world_size = dist.get_world_size()
            logger.warning(
                f'predict_with_generate under torchrun drives EvalScope from rank 0 only (world_size='
                f'{world_size}): generation uses 1/{world_size} of the launched GPUs while the rest wait at '
                'the eval barrier. For full-throughput generative eval, run under --mode ray with '
                '--eval_sampler_backend vllm|sglang.')
            self._warned_eval_dp = True
        result: Dict[str, Any] = {'step': self.global_step}
        # eval_enter/eval_exit bracket the run with the assembly's weight-sync + memory schedule for a
        # co-resident vllm/sglang engine (None on the transformers-live path, which needs no sync). They
        # run on every rank: under ray the driver drives the remote workers, under torchrun they are
        # collective. Only rank 0 drives EvalScope itself; the others wait at the barrier.
        if self.eval_enter is not None:
            self.eval_enter()
        try:
            if is_main:
                summary = run_evalscope(
                    self.eval_sampler,
                    self.eval_template,
                    datasets=self.eval_datasets,
                    model_id=self.eval_model_id,
                    task_config=self.eval_task_config)
                result.update(flatten_evalscope_report(summary))
                result['eval_report'] = summary
            if distributed:
                dist.barrier()
        finally:
            if self.eval_exit is not None:
                self.eval_exit()
        return result

    def _track_best(self, metrics: dict) -> None:
        """Update best_metric / best_step / best_ckpt when the watched metric improves."""
        if not self.metric_for_best_model:
            return
        value = metrics.get(self.metric_for_best_model)
        if value is None:
            logger.warning(f'metric_for_best_model={self.metric_for_best_model!r} was not produced by this eval '
                           f'(got {sorted(key for key in metrics if key != "step")}); best-model tracking skipped '
                           'for this pass.')
            return
        value = float(value)
        improved = (self.best_metric is None
                    or (value > self.best_metric if self.greater_is_better else value < self.best_metric))
        if improved:
            self.best_metric = value
            self.best_step = self.global_step
            self.best_ckpt = os.path.join(self.output_dir, f'checkpoint-{self.global_step}')

    def _load_best_model(self) -> None:
        """Reload the best checkpoint at train end when ``load_best_model_at_end`` is set."""
        if not self.load_best_model_at_end:
            return
        if not self.best_ckpt:
            logger.warning('load_best_model_at_end is set but no eval produced metric_for_best_model; '
                           'keeping the final weights.')
            return
        if not os.path.isdir(self.best_ckpt):
            raise FileNotFoundError(
                f'load_best_model_at_end wants {self.best_ckpt} (best {self.metric_for_best_model}='
                f'{self.best_metric} at step {self.best_step}), but it does not exist: save_steps must align '
                'with eval_steps so the best step is checkpointed, and save_total_limit must not have pruned it.')
        logger.info(f'load_best_model_at_end: reloading best checkpoint {self.best_ckpt} '
                    f'({self.metric_for_best_model}={self.best_metric} at step {self.best_step}).')
        kwargs = {'adapter_name': self.best_model_adapter_name} if self.best_model_adapter_name else {}
        self.model.load(self.best_ckpt, **kwargs)

    def fit(self) -> list:
        """Run the loop; returns the per-optimizer-step loss history.

        The two backends express gradient accumulation differently, so fit dispatches:
          - transformers: ONE micro-batch per forward_backward; twinkle's grad-sync gate no-ops on
            non-boundary micro-steps and steps the optimizer every ga-th call. The loop mirrors that
            boundary (_is_grad_sync_boundary) only to log / count.
          - Megatron: GA is internal to a single forward_backward -- it splits an inputs LIST into
            len(inputs) microbatches and accumulates their grads before one optimizer step. So the
            loop groups ga dataloader batches into one list and calls forward_backward ONCE per
            optimizer step (cross-microbatch loss normalization is handled inside twinkle/Megatron).
        """
        gc_was_enabled = start_manual_gc(self.manual_gc)
        self._sync_state()
        self.handler.on_train_begin()
        try:
            if self.eval_on_start:
                self.evaluate()
            history = (self._fit_megatron() if self._owns_gradient_accumulation else self._fit_transformers())
            self._load_best_model()
            return history
        finally:
            self._sync_state()
            self.handler.on_train_end()
            self._shutdown_eval_sampler()
            finish_manual_gc(gc_was_enabled)
            self.tracker.close()

    def _shutdown_eval_sampler(self) -> None:
        """Release the resident eval sampler (design 1.4): built once, it lives across every eval."""
        shutdown = getattr(self.eval_sampler, 'shutdown', None)
        if callable(shutdown):
            shutdown()

    def _epochs(self) -> int:
        return math.ceil(self.num_train_epochs) if self.max_steps <= 0 else 10**9

    def _record_step(self) -> None:
        """Count one completed optimizer step + log / periodic save / periodic eval (shared).

        Reads the NORMALIZED per-token loss (+ grad_norm) via the driver-callable
        model.calculate_metric (works for transformers + Ray-remote Megatron; resolves the active
        optimizer group on the worker). Raw forward_backward loss under reduction='sum' is a
        token-sum, not comparable across steps -- calculate_metric divides by num_tokens and resets.
        """
        self.global_step += 1
        collect_manual_gc(self.manual_gc, self.manual_gc_steps, self.global_step)
        metrics = self.model.calculate_metric(is_training=True)
        loss = float(metrics['loss']) if metrics.get('loss') is not None else float('nan')
        record = {'step': self.global_step, 'loss': loss}
        if metrics.get('grad_norm') is not None:
            record['grad_norm'] = float(metrics['grad_norm'])
        for key, value in metrics.items():
            if key.startswith('loss_'):
                record[key] = float(value)
        record.update(self._mtp_metrics())
        record = self.tracker.log(record, self.global_step)
        self.history.append(record)
        self._sync_state(loss=loss, metrics=record)
        should_log = (self.tracker.should_log(self.global_step) if self.logging_config is not None else
                      bool(self.logging_steps and self.global_step % self.logging_steps == 0))
        should_log = should_log or self.handler.control.should_log
        if should_log:
            gn = record.get('grad_norm')
            gn_str = f'  grad_norm={gn:.4f}' if gn is not None else ''
            mtp = record.get('mtp_loss')
            mtp_str = f'  mtp_loss={mtp:.4f}' if mtp is not None else ''
            channel_str = ''.join(f'  {key}={value:.4f}' for key, value in record.items() if key.startswith('loss_'))
            logger.info(f'step {self.global_step}  loss={loss:.4f}{gn_str}{mtp_str}{channel_str}')
            self.handler.on_log(logs=record)
        if self.save_steps and self.global_step % self.save_steps == 0:
            self.save(f'checkpoint-{self.global_step}')
        if self._due_for_eval():
            self.evaluate()
        # on_step_end is where EarlyStop / GracefulExit raise their flags; honour them, then clear the
        # transient ones so a one-shot request does not repeat on every subsequent step.
        self.handler.on_step_end()
        control = self.handler.control
        if control.should_save:
            self.save(f'checkpoint-{self.global_step}')
        if control.should_evaluate:
            self.evaluate()
        control.should_save = False
        control.should_evaluate = False
        control.should_log = False

    def _mtp_metrics(self) -> dict:
        """Per-depth MTP loss (and acceptance rate, on megatron >= 0.19) for this step, or {}.

        Kept out of ``calculate_metric``: megatron accumulates the MTP loss in a module-level tracker
        of its own rather than through twinkle's Metric objects, and it has to be *drained* each step
        or the reported value grows without bound. Absent on the transformers backend, and on megatron
        whenever MTP is off -- either way the loop is unchanged.
        """
        pop = getattr(self.model, 'pop_mtp_metrics', None)
        if pop is None:
            return {}
        return {key: float(value) for key, value in (pop() or {}).items()}

    def _final_eval(self) -> None:
        """Final eval pass so training always ends with an eval point (unless just evaluated)."""
        if self._eval_capable() and (not self.eval_history
                                     or self.eval_history[-1].get('step') != self.global_step):
            self.evaluate()

    def _fit_transformers(self) -> list:
        """transformers GA: one micro-batch per forward_backward, twinkle grad-sync gate steps."""
        ga = self.gradient_accumulation_steps
        handler = self.handler
        # twinkle's grad-sync gate lags one micro-step, so an optimizer window here spans ga+1
        # micro-batches; on_step_begin fires at each window's first micro-batch (this flag, re-set
        # after every boundary) rather than on a clean ga multiple.
        at_window_start = True
        for epoch in range(self._start_epoch, self._epochs()):
            if self._reached_max() or self._should_stop():
                break
            self._epoch = epoch
            # Reshuffle each epoch (twinkle's EpochSampler derives each epoch's order from its number,
            # and only set_epoch advances it); resume replays a given epoch's order the same way.
            if hasattr(self.dataloader, 'set_epoch'):
                self.dataloader.set_epoch(epoch)
            self._sync_state()
            handler.on_epoch_begin()
            for batch in self.dataloader:
                if at_window_start:
                    self._sync_state()
                    handler.on_step_begin()
                    at_window_start = False
                self.micro_step += 1
                self.model.forward_backward(inputs=batch, gradient_accumulation_steps=ga, task=self.task)
                self._sync_state()
                handler.on_substep_end()
                # Boundary computed loop-side (backend-agnostic); clip_grad_and_step self-guards via
                # the same predicate on the worker, so calling it every micro-step is safe.
                is_boundary = self._is_grad_sync_boundary()
                if is_boundary:
                    handler.on_pre_optimizer_step()
                self.model.clip_grad_and_step(max_grad_norm=self.max_grad_norm, gradient_accumulation_steps=ga)
                if is_boundary:
                    handler.on_optimizer_step()
                    self._record_step()
                    at_window_start = True
                    if self._reached_max() or self._should_stop():
                        break
            self._sync_state()
            handler.on_epoch_end()
            if self.history:
                self.tracker.log(self.history[-1], self.global_step, epoch_end=True)
            if self._should_stop():
                break
        self._final_eval()
        return self.history

    def _fit_megatron(self) -> list:
        """Megatron GA: group ga dataloader batches into one microbatch-list, one step per group.

        Each dataloader batch is a list[InputFeature] of size per_device_train_batch_size; ga of them
        concatenated form one optimizer step's microbatch list. forward_backward derives
        num_microbatches from that list length and accumulates internally, so exactly ONE optimizer
        step is taken per group (ga>1 is gradient-equivalent to ga=1 with a proportionally larger
        batch). gradient_accumulation_steps is NOT passed: Megatron reads GA from len(inputs) and the
        kwarg would collide with its internal metric accumulate(). micro_batch_size is NOT passed
        either: each worker picks micro_batch_size=min(2, per_rank_len) itself, and a driver-scope
        value would exceed the per-rank length (assert len(inputs) >= micro_batch_size).

        DP sharding differs by mode, and the group this loop feeds forward_backward is already the
        per-DP-rank slice in BOTH cases:
          - Ray: forward_backward's dispatch='slice_dp' splits the group across DP ranks on the
            driver before each worker runs; the loop here runs once on the driver.
          - local (torchrun): slice_dp is a no-op (no driver), so the dataloader shards by DP rank
            up front via twinkle's DeviceMeshSampler; this loop runs per rank over its own slice.

        On Megatron the dataloader is built with drop_last=True (see builders/dataset.py::_drop_last),
        so a trailing partial group cannot occur there: global_batch_size is an exact invariant and
        legacy drops the remainder too. The loop still handles a short group for the transformers
        path, which keeps drop_last=False.
        """
        ga = self.gradient_accumulation_steps
        handler = self.handler
        for epoch in range(self._start_epoch, self._epochs()):
            if self._reached_max() or self._should_stop():
                break
            self._epoch = epoch
            if hasattr(self.dataloader, 'set_epoch'):
                self.dataloader.set_epoch(epoch)
            self._sync_state()
            handler.on_epoch_begin()
            group: list = []
            batches_in_group = 0
            for batch in self.dataloader:
                self.micro_step += 1
                group.extend(batch)
                batches_in_group += 1
                if batches_in_group < ga:
                    continue
                self._megatron_step(group)
                group, batches_in_group = [], 0
                if self._reached_max() or self._should_stop():
                    break
            if group and not self._reached_max() and not self._should_stop():  # trailing partial group
                self._megatron_step(group)
            self._sync_state()
            handler.on_epoch_end()
            if self.history:
                self.tracker.log(self.history[-1], self.global_step, epoch_end=True)
            if self._should_stop():
                break
        self._final_eval()
        return self.history

    def _megatron_step(self, microbatch_list: list) -> None:
        """One Megatron optimizer step over a microbatch list (GA internal to forward_backward).

        No on_substep_end here: Megatron's GA is internal to the single forward_backward call, so the
        whole group is one optimizer window (on_step_begin -> pre/optimizer-step -> on_step_end via
        _record_step).
        """
        handler = self.handler
        self._sync_state()
        handler.on_step_begin()
        self.model.forward_backward(inputs=microbatch_list, task=self.task)
        handler.on_pre_optimizer_step()
        self.model.clip_grad_and_step(max_grad_norm=self.max_grad_norm)
        handler.on_optimizer_step()
        self._record_step()

    def save(self, name: str = 'checkpoint-final', *, is_final: bool = False) -> str:
        """Persist the model + full training state via twinkle's native save.

        Passes save_optimizer=True so twinkle writes optimizer.pt / scheduler.pt / scaler.pt /
        rng_state.pt / trainer_state.json (twinkle schema: cur_step / gradient_accumulation_steps
        / consumed_train_samples). dev does NOT hand-roll trainer_state; it feeds the dataloader's
        own consumed count so the resume position is recoverable.

        Read through ``get_state()`` rather than off an attribute: the dataloader is a twinkle
        ``remote_class``, so in ray mode the driver holds a handle whose attributes live in the worker
        and only its remote_functions answer. Same call the twinkle cookbooks use.
        """
        consumed = self._dataloader_state().get('consumed_train_samples', 0)
        result = save_training_checkpoint(
            self.model,
            name,
            output_dir=self.output_dir,
            consumed_train_samples=consumed,
            no_save_optim=self.no_save_optim,
            no_save_rng=self.no_save_rng,
            safe_serialization=self.safe_serialization,
            max_shard_size=self.max_shard_size,
            save_total_limit=self.save_total_limit)
        # on_save reports the resolved dir rather than ``result``: in Ray (Megatron) mode save()
        # returns a deferred handle, not a path, so recompute the same way save_final does.
        self._sync_state()
        ckpt_dir = os.path.join(self.output_dir, name)
        self.handler.on_save(checkpoint_dir=ckpt_dir)
        # Hub push (design 6.1): on_push_begin fires before the upload so a callback can veto/annotate.
        # 'end' pushes only the final checkpoint; the push-on-save strategies push every one. Main-process
        # only: under torchrun every rank runs this loop, so an unguarded push would upload once per rank.
        if (self.hub_pusher is not None and (self.hub_strategy != 'end' or is_final)
                and self._is_main_process()):
            self.handler.on_push_begin(checkpoint_dir=ckpt_dir)
            self.hub_pusher(ckpt_dir)
        return result

    def _dataloader_state(self) -> dict:
        """The train dataloader's ``{consumed_train_samples, resume_epoch}``, or ``{}`` if it has none."""
        get_state = getattr(self.dataloader, 'get_state', None)
        return get_state() if get_state is not None else {}

    def resume(self, state: dict) -> None:
        """Seed loop counters + dataloader position from a restored twinkle trainer_state.

        state = {'cur_step', 'consumed_train_samples', 'gradient_accumulation_steps'} (twinkle
        schema; single counting source). Weights/optimizer/scheduler/RNG are already restored by
        model.resume_from_checkpoint BEFORE this call. Here we only align the loop's own counters
        and the dataloader's skip position so training + GA phase continue exactly.
        """
        cur_step = int(state['cur_step'])
        ga = self.gradient_accumulation_steps
        # micro_step must equal twinkle's cur_step so do_grad_sync phase stays aligned
        # (twinkle's optimizer_config.cur_step was already restored to cur_step).
        self.micro_step = cur_step
        # global_step (optimizer steps taken) derived from cur_step + ga (single source).
        self.global_step = num_optimizer_steps(cur_step, ga)
        # dataloader skip: reproduce the exact epoch/offset unless the user explicitly requested
        # a fresh pass over the data while retaining checkpoint progress/optimizer state.
        if self.ignore_data_skip:
            self._start_epoch = 0
            return
        consumed = int(state['consumed_train_samples'])
        if hasattr(self.dataloader, 'skip_consumed_samples'):
            self.dataloader.skip_consumed_samples(consumed)
        # Start the epoch loop at the resume epoch so cross-epoch resume does not replay
        # already-consumed epochs (the dataloader only skips the offset within its resume epoch).
        self._start_epoch = self._dataloader_state().get('resume_epoch', 0)
