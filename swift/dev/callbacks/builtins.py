# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import signal
import time
from typing import Optional

from swift.dev.utils import get_logger
from swift.dev.utils.plugin import PluginRegistry

from swift.dev.callbacks.base import CallbackControl, CallbackState, TrainCallback

logger = get_logger()

__all__ = ['EarlyStopCallback', 'PerfLogCallback', 'GracefulExitCallback']


def _greater_is_better(metric_name: Optional[str], explicit: Optional[bool]) -> bool:
    """An explicit flag wins; else a metric with 'loss' in its name means smaller is better."""
    if explicit is not None:
        return explicit
    return not (metric_name and 'loss' in metric_name.lower())


@PluginRegistry.register('callback', 'early_stop')
class EarlyStopCallback(TrainCallback):
    """Stop once the watched metric has not improved for ``early_stop_interval`` evaluations.

    The watched metric is ``metric_for_best_model`` (``eval_loss`` on the val-loss path).
    """

    def __init__(self, args=None, **kwargs):
        super().__init__(args=args, **kwargs)
        self.patience = int(getattr(args, 'early_stop_interval', 0) or 0)
        self.metric_name = getattr(args, 'metric_for_best_model', None) or 'eval_loss'
        self.greater_is_better = _greater_is_better(self.metric_name, getattr(args, 'greater_is_better', None))
        self._best: Optional[float] = None
        self._stale_evals = 0

    def on_evaluate(self, state: CallbackState, control: CallbackControl, metrics=None) -> None:
        if self.patience <= 0 or not metrics:
            return
        value = metrics.get(self.metric_name)
        if value is None:
            logger.warning(f'EarlyStopCallback watches {self.metric_name!r}, which this eval did not produce '
                           f'(got {sorted(metrics)}); skipping the early-stop check for this pass.')
            return
        value = float(value)
        improved = (self._best is None
                    or (value > self._best if self.greater_is_better else value < self._best))
        if improved:
            self._best = value
            self._stale_evals = 0
        else:
            self._stale_evals += 1
            if self._stale_evals >= self.patience:
                logger.info(f'EarlyStopCallback: {self.metric_name} did not improve for {self._stale_evals} '
                            f'evaluations (best={self._best}); stopping training.')
                control.should_training_stop = True


@PluginRegistry.register('callback', 'perf_log')
class PerfLogCallback(TrainCallback):
    """Log throughput (steps/sec and mean step time) per ``logging_steps`` window."""

    def __init__(self, args=None, **kwargs):
        super().__init__(args=args, **kwargs)
        self.interval = int(getattr(args, 'logging_steps', 0) or 0) or 50
        self._window_start: Optional[float] = None
        self._window_steps = 0

    def on_step_end(self, state: CallbackState, control: CallbackControl) -> None:
        now = time.monotonic()
        if self._window_start is None:
            self._window_start = now
        self._window_steps += 1
        if self._window_steps < self.interval:
            return
        elapsed = max(now - self._window_start, 1e-9)
        speed = self._window_steps / elapsed
        logger.info(f'train_speed={speed:.4f} steps/sec  mean_step_time={elapsed / self._window_steps * 1000:.1f}ms '
                    f'(over {self._window_steps} steps)')
        self._window_start = now
        self._window_steps = 0


@PluginRegistry.register('callback', 'graceful_exit')
class GracefulExitCallback(TrainCallback):
    """Turn SIGTERM/SIGINT into a clean stop: finish the step, save a checkpoint, then exit."""

    def __init__(self, args=None, **kwargs):
        super().__init__(args=args, **kwargs)
        self._requested = False
        self._saved_handlers = {}

    def _handle_signal(self, signum, frame) -> None:
        # Async-signal-safe: only set a flag; the loop acts at the next on_step_end.
        if not self._requested:
            logger.warning(f'Received signal {signum}; finishing the current step and saving before exit. '
                           'Send it again to abort immediately.')
        self._requested = True

    def on_train_begin(self, state: CallbackState, control: CallbackControl) -> None:
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                self._saved_handlers[sig] = signal.signal(sig, self._handle_signal)
            except (ValueError, OSError):
                # Signals only land on the main thread; some platforms lack these.
                pass

    def on_train_end(self, state: CallbackState, control: CallbackControl) -> None:
        for sig, handler in self._saved_handlers.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass
        self._saved_handlers = {}

    def on_step_end(self, state: CallbackState, control: CallbackControl) -> None:
        if self._requested:
            control.should_training_stop = True
            control.should_save = True
