"""The step cadence of a flat offline recipe: counters, metrics, logging, callbacks.

``StepRunner`` is the thin re-home of the per-optimizer-step half of the old ``TrainLoop``: it owns
the counters (``global_step`` / ``micro_step``), the metric ``RunTracker``, the loss ``history``, the
manual-GC / empty-cache / perf knobs and the ``CallbackHandler`` every offline event fans through. It
deliberately has NO algorithm hooks and NO subclassing -- the old seven override points are gone, and
the one offline consumer of them (the Megatron MTP-metric drain) is folded straight into
:meth:`record_step`. The epoch loop, the gradient-accumulation fork and the save/eval cadence live in
the recipe (via :func:`swift.dev.recipe.offline_loop.run_offline`), which calls :meth:`record_step` at each
optimizer boundary and drives the callback events around it.

RunTracker initialises the configured reporters (wandb / swanlab / tensorboard) as a construction side
effect, so any input validation a caller needs must run BEFORE the recorder is built.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from swift.dev.utils.logger import get_logger

if TYPE_CHECKING:
    from swift.dev.callbacks.handler import CallbackHandler
    from swift.dev.config import LoggingConfig
    from swift.dev.model import TrainableModel

logger = get_logger()

__all__ = ['StepRunner']


def _default_log_line(record: dict, loss: float) -> str:
    """The default human-readable step line: loss plus grad_norm, the MTP channel and any loss_* channels.

    ``record['step']`` is the ``global_step`` :meth:`StepRunner.record_step` stamped it with. A recipe that
    wants a different console signal (the preference family surfaces the DPO reward margin/accuracy) injects
    its own formatter via ``StepRunner(log_formatter=...)`` instead of subclassing -- the recorder has no
    hooks, so the log line is the one rendering seam it exposes.
    """
    gn = record.get('grad_norm')
    gn_str = f'  grad_norm={gn:.4f}' if gn is not None else ''
    mtp = record.get('mtp_loss')
    mtp_str = f'  mtp_loss={mtp:.4f}' if mtp is not None else ''
    channel_str = ''.join(f'  {key}={value:.4f}' for key, value in record.items() if key.startswith('loss_'))
    return f"step {record.get('step')}  loss={loss:.4f}{gn_str}{mtp_str}{channel_str}"


class StepRunner:
    """Count one optimizer step's metrics, log them, and publish loop position onto the callback state.

    The recipe constructs one recorder, hands it the :class:`CallbackHandler` it built (so the recorder
    fires ``on_log`` and reads the transient ``should_log`` flag), and links it to the :class:`Evaluator`
    once that exists so ``best_metric`` reaches the shared callback state. :meth:`record_step` is the only
    per-step entry point; everything else is loop-control state the recipe reads.
    """

    def __init__(self,
                 model: 'TrainableModel',
                 handler: 'CallbackHandler',
                 *,
                 gradient_accumulation_steps: int = 1,
                 max_grad_norm: float = 1.0,
                 max_steps: int = -1,
                 logging_steps: int = 1,
                 logging_config: Optional['LoggingConfig'] = None,
                 output_dir: str = 'output',
                 manual_gc: bool = False,
                 manual_gc_steps: int = 0,
                 torch_empty_cache_steps: Optional[int] = None,
                 log_formatter: Optional[Callable[[dict, float], str]] = None,
                 extra_metrics_fn: Optional[Callable[[], Dict[str, Any]]] = None):
        from swift.dev.recipe.tracking import RunTracker

        self.model = model
        self.handler = handler
        #: Renders the console step line from (record, loss). Defaults to :func:`_default_log_line`; a recipe
        #: injects its own (e.g. the preference family's DPO margin/accuracy) -- the recorder has no hooks, so
        #: this is the one log-rendering seam. The ``on_log`` callback fan-out stays in :meth:`_log_step`.
        self._log_formatter = log_formatter if log_formatter is not None else _default_log_line
        #: Optional driver-side per-step channels the model's ``calculate_metric`` cannot surface (an online-RL
        #: recipe's outcome reward, the streaming version-span aggregate, the driver's lifetime counters). The
        #: flat offline recipes pass ``None``; an online recipe injects a closure folded into each step's record
        #: in :meth:`record_step`, the re-home of the old ``TrainLoop._extra_step_metrics`` hook.
        self._extra_metrics_fn = extra_metrics_fn
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.max_grad_norm = max_grad_norm
        self.max_steps = max_steps
        self.logging_steps = logging_config.logging_steps if logging_config is not None else logging_steps
        self.logging_config = logging_config
        self.tracker = RunTracker(logging_config, output_dir)
        self.output_dir = output_dir
        self.manual_gc = manual_gc
        self.manual_gc_steps = manual_gc_steps
        if self.manual_gc_steps < 0:
            raise ValueError('manual_gc_steps must be >= 0.')
        self.torch_empty_cache_steps = torch_empty_cache_steps
        if torch_empty_cache_steps is not None and torch_empty_cache_steps <= 0:
            raise ValueError('torch_empty_cache_steps must be > 0, or None to disable.')
        # optimizer-step counter (increments once per GA window) and the micro-step counter the recipe
        # advances once per forward_backward; CheckpointSaver.resume keeps both aligned with twinkle's
        # cur_step. ``epoch`` is published onto CallbackState and advanced by run_offline.
        self.global_step = 0
        self.micro_step = 0
        self.epoch = 0
        self.history: list = []
        # Linked after construction (the recipe builds the recorder first, then the evaluator): the source
        # of ``best_metric`` published onto the callback state. None until then, so state.best_metric stays
        # None exactly as the old loop's did before the first eval improved it.
        self.evaluator: Any = None

    # --- loop-control state the recipe reads ------------------------------------------------------
    def reached_max(self) -> bool:
        """Whether the run hit its ``max_steps`` cap (a non-positive cap never trips)."""
        return self.max_steps > 0 and self.global_step >= self.max_steps

    def should_stop(self) -> bool:
        """Whether a callback (early stop / graceful exit) asked the loop to break."""
        return self.handler.control.should_training_stop

    def is_grad_sync_boundary(self) -> bool:
        """Whether the current ``micro_step`` is an optimizer-update boundary (see is_grad_sync_boundary)."""
        from swift.dev.recipe._loop_utils import is_grad_sync_boundary
        return is_grad_sync_boundary(self.micro_step, self.gradient_accumulation_steps)

    def publish_state(self, *, loss: Optional[float] = None, metrics: Optional[dict] = None) -> None:
        """Publish the recorder's position onto the shared CallbackState before an event fires.

        ``loss`` / ``metrics`` are only overwritten when supplied, so a step event keeps the last training
        record while an eval event publishes its own. ``best_metric`` is read off the linked evaluator (the
        component that owns it); ``epoch`` is set by run_offline.
        """
        state = self.handler.state
        state.global_step = self.global_step
        state.micro_step = self.micro_step
        state.epoch = self.epoch
        if self.evaluator is not None:
            state.best_metric = self.evaluator.best_metric
        if loss is not None:
            state.loss = loss
        if metrics is not None:
            state.metrics = metrics

    # --- the one per-step entry point -------------------------------------------------------------
    def record_step(self) -> dict:
        """Count one completed optimizer step, then log it and publish it onto the callback state.

        Reads the NORMALIZED per-token loss (+ every algorithm metric) via the driver-callable
        ``model.calculate_metric`` (works for transformers + Ray-remote Megatron; resolves the active
        optimizer group on the worker). Raw ``forward_backward`` loss under ``reduction='sum'`` is a
        token-sum, not comparable across steps -- ``calculate_metric`` divides by ``num_tokens`` and resets.
        The Megatron MTP channel (tracked outside twinkle's Metric objects, drained per step) is folded in
        here; it is absent on transformers and whenever MTP is off, leaving the record unchanged.

        Returns the record appended to :attr:`history`. The periodic save / eval / ``on_step_end`` tail is
        the recipe's (run_offline), not this method's -- it owns only the metric + log half.
        """
        from swift.dev.recipe._loop_utils import collect_empty_cache, collect_manual_gc
        from swift.dev.recipe.tracking import coerce_metric_scalars
        from twinkle.perf import Perf

        self.global_step += 1
        collect_manual_gc(self.manual_gc, self.manual_gc_steps, self.global_step)
        collect_empty_cache(self.model, self.torch_empty_cache_steps, self.global_step)
        # Tick every registered actor's profiler step counter once per optimizer step (non-blocking; a no-op
        # when perf is off and the registry is empty). Collection is separate and windowed, below.
        Perf.advance_step()
        metrics = self.model.calculate_metric(is_training=True)
        loss = float(metrics['loss']) if metrics.get('loss') is not None else float('nan')
        # twinkle.metric is the single source of every algorithm metric: merge the WHOLE component output
        # (loss / grad_norm / lr / iters / accuracy, plus whatever the recipe registered via add_metric)
        # into this step's record, coerced to numeric scalars so both the tracker and the console line can
        # consume it. ``loss`` is then re-pinned as the normalized float (authoritative for nan-filtering +
        # the console); _mtp_metrics adds the one channel the model cannot surface through calculate_metric.
        should_log = self._should_log()
        record = {'step': self.global_step}
        record.update(coerce_metric_scalars(metrics))
        record['loss'] = loss
        record.update(self._mtp_metrics())
        if self._extra_metrics_fn is not None:
            record.update(self._extra_metrics_fn())
        if should_log:
            # Fold the window's merged per-rank perf summary in only on a logging step: collect_step_summary
            # fans a blocking perf_dump out to every registered actor, so keeping it to the window (not every
            # step) leaves that RPC off the hot path while each summary still aggregates every step since the
            # last collect.
            record.update(self._perf_step())
        record = self.tracker.log(record, self.global_step)
        self.history.append(record)
        self.publish_state(loss=loss, metrics=record)
        if should_log:
            self._log_step(record, loss)
        return record

    def _should_log(self) -> bool:
        """The tracker's cadence (else a logging_steps gate), plus a callback's one-shot should_log request."""
        base = (self.tracker.should_log(self.global_step) if self.logging_config is not None else
                bool(self.logging_steps and self.global_step % self.logging_steps == 0))
        return base or self.handler.control.should_log

    def _log_step(self, record: dict, loss: float) -> None:
        """Emit the human-readable step line and fan ``on_log`` out to the callbacks.

        The line rendering is the injected ``log_formatter`` (default :func:`_default_log_line`); the
        ``on_log`` fan-out is unconditional so a callback sees the same record regardless of formatter.
        """
        logger.info(self._log_formatter(record, loss))
        self.handler.on_log(logs=record)

    def _mtp_metrics(self) -> Dict[str, float]:
        """Per-depth MTP loss (and acceptance rate, on megatron >= 0.19) for this step, or {}.

        Kept out of ``calculate_metric``: megatron accumulates the MTP loss in a module-level tracker of
        its own rather than through twinkle's Metric objects, and it must be *drained* each step or the
        reported value grows without bound. Absent on the transformers backend, and on megatron whenever
        MTP is off -- either way the record is unchanged.
        """
        pop = getattr(self.model, 'pop_mtp_metrics', None)
        if pop is None:
            return {}
        return {key: float(value) for key, value in (pop() or {}).items()}

    # --- perf profiling (opt-in via TWINKLE_PERF; a single attribute read when off) ---------------
    def _perf_step(self) -> dict:
        """This logging window's merged ``perf/stage/*`` + ``perf/module/*`` metrics (``{}`` when off).

        ``Perf.collect_step_summary`` self-gates on perf being live and takes NO argument: it drains the
        driver's own recorder plus every actor ``Perf.initialize`` registered (in run_offline), snapshotting and
        resetting each so the summary is exactly the window since the last collect. Per-rank module summaries are
        tagged ``rank<k>`` when an actor spans more than one rank, so parallel imbalance stays visible. Called
        only on a logging step -- the perf_dump fan-out is blocking, so it stays off the per-step hot path.
        """
        from twinkle.perf import Perf
        return Perf.collect_step_summary()
