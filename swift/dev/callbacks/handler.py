"""Fan-out of loop events to the registered callbacks.

The loop holds ONE handler and calls its event methods; the handler forwards each to every callback
in registration order, sharing a single :class:`CallbackState` / :class:`CallbackControl` pair. The
control is merged by construction: callbacks only ever flip a flag False -> True, and they all mutate
the same object, so ``should_training_stop`` set by any callback is seen by the loop (an OR without an
explicit reduce).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from swift.dev.callbacks.base import CallbackControl, CallbackState, TrainCallback

__all__ = ['CallbackHandler']


class CallbackHandler:
    """Dispatch loop events to a list of callbacks over shared state/control records."""

    def __init__(self, callbacks: Optional[List[TrainCallback]] = None, *, output_dir: str = 'output'):
        self.callbacks: List[TrainCallback] = list(callbacks or [])
        self.state = CallbackState(output_dir=output_dir)
        self.control = CallbackControl()

    def add(self, callback: TrainCallback) -> None:
        self.callbacks.append(callback)

    def _fan(self, event: str, **payload: Any) -> None:
        """Call ``event`` on every callback with the shared state/control plus any event payload."""
        for callback in self.callbacks:
            getattr(callback, event)(self.state, self.control, **payload)

    # --- lifecycle ---
    def on_init_end(self) -> None:
        self._fan('on_init_end')

    def on_train_begin(self) -> None:
        self._fan('on_train_begin')

    def on_train_end(self) -> None:
        self._fan('on_train_end')

    def on_epoch_begin(self) -> None:
        self._fan('on_epoch_begin')

    def on_epoch_end(self) -> None:
        self._fan('on_epoch_end')

    # --- step ---
    def on_step_begin(self) -> None:
        self._fan('on_step_begin')

    def on_step_end(self) -> None:
        self._fan('on_step_end')

    def on_substep_end(self) -> None:
        self._fan('on_substep_end')

    def on_pre_optimizer_step(self) -> None:
        self._fan('on_pre_optimizer_step')

    def on_optimizer_step(self) -> None:
        self._fan('on_optimizer_step')

    # --- logging / eval / save / push ---
    def on_log(self, logs: Optional[Dict[str, Any]] = None) -> None:
        self._fan('on_log', logs=logs)

    def on_evaluate(self, metrics: Optional[Dict[str, Any]] = None) -> None:
        self._fan('on_evaluate', metrics=metrics)

    def on_save(self, checkpoint_dir: Optional[str] = None) -> None:
        self._fan('on_save', checkpoint_dir=checkpoint_dir)

    def on_push_begin(self, checkpoint_dir: Optional[str] = None) -> None:
        self._fan('on_push_begin', checkpoint_dir=checkpoint_dir)

    def on_predict(self) -> None:
        self._fan('on_predict')

    def on_prediction_step(self) -> None:
        self._fan('on_prediction_step')
