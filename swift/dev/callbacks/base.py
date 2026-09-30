"""The dev training callback contract and its state/control records.

A callback observes -- and may veto -- the training loop at the same points the transformers
``TrainerCallback`` does, but with dev's own record shapes rather than HF's ``args``/``state``/
``control`` triple. Every event method is a no-op here, so a concrete callback overrides only the
points it cares about.

The two records passed to every event:

- :class:`CallbackState` is read-only in spirit: the loop publishes where training is (step, epoch,
  last loss, last metrics, best-so-far). A callback reads it to decide.
- :class:`CallbackControl` is the write side: a callback flips a flag to ask the loop to stop, save,
  evaluate, or log. The loop honours the flags after the event returns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from swift.dev.plugin import PluginRegistry, SwiftPlugin

__all__ = ['CallbackState', 'CallbackControl', 'TrainCallback', 'CALLBACK']


@dataclass
class CallbackState:
    """Where training currently is, published by the loop before each event fires."""

    #: Optimizer steps completed (one per gradient-accumulation window), the loop's own counter.
    global_step: int = 0
    #: Zero-based epoch index currently running.
    epoch: int = 0
    #: Micro-batches consumed within the current optimizer window; resets implicitly via global_step.
    micro_step: int = 0
    #: Most recent training loss (per optimizer step), or None before the first step.
    loss: Optional[float] = None
    #: Most recent metric dict -- training metrics on ``on_log``, eval metrics on ``on_evaluate``.
    metrics: Dict[str, Any] = field(default_factory=dict)
    #: Best value of ``metric_for_best_model`` seen so far, or None when no best-model tracking is on.
    best_metric: Optional[float] = None
    #: Run output directory (where checkpoints are written).
    output_dir: str = 'output'


@dataclass
class CallbackControl:
    """What a callback asks the loop to do next. Flags only ever go False -> True within an event."""

    #: Break out of the training loop after the current step.
    should_training_stop: bool = False
    #: Force a checkpoint save at this point.
    should_save: bool = False
    #: Force an evaluation pass at this point.
    should_evaluate: bool = False
    #: Force a log emission at this point.
    should_log: bool = False


class TrainCallback(SwiftPlugin):
    """Base of every training callback: all events no-op, override only what you need.

    Constructed as ``cls(args=config)`` like every swift plugin, so a callback may read its own
    hyperparameters off the run's ``TrainConfig``. Event names mirror the transformers
    ``TrainerCallback`` set; the arguments are dev's :class:`CallbackState` / :class:`CallbackControl`
    rather than HF's triple, and the few events that carry a payload (logs / metrics / a checkpoint
    dir) take it as an optional keyword.
    """

    def on_init_end(self, state: CallbackState, control: CallbackControl) -> None:
        """After the loop is constructed and before training starts."""

    def on_train_begin(self, state: CallbackState, control: CallbackControl) -> None:
        """Once, before the first epoch."""

    def on_train_end(self, state: CallbackState, control: CallbackControl) -> None:
        """Once, after the last epoch (including on early stop)."""

    def on_epoch_begin(self, state: CallbackState, control: CallbackControl) -> None:
        """At the start of each epoch."""

    def on_epoch_end(self, state: CallbackState, control: CallbackControl) -> None:
        """At the end of each epoch."""

    def on_step_begin(self, state: CallbackState, control: CallbackControl) -> None:
        """At the start of an optimizer window, before its first micro-batch."""

    def on_step_end(self, state: CallbackState, control: CallbackControl) -> None:
        """After an optimizer step is recorded (loss/metrics published on ``state``)."""

    def on_substep_end(self, state: CallbackState, control: CallbackControl) -> None:
        """After each micro-batch's backward, i.e. once per gradient-accumulation substep."""

    def on_pre_optimizer_step(self, state: CallbackState, control: CallbackControl) -> None:
        """After grad clipping is about to run, before the optimizer steps."""

    def on_optimizer_step(self, state: CallbackState, control: CallbackControl) -> None:
        """Right after the optimizer stepped."""

    def on_log(self, state: CallbackState, control: CallbackControl, logs: Optional[Dict[str, Any]] = None) -> None:
        """When a training record is logged; ``logs`` is that record."""

    def on_evaluate(self,
                    state: CallbackState,
                    control: CallbackControl,
                    metrics: Optional[Dict[str, Any]] = None) -> None:
        """After an evaluation pass; ``metrics`` are the eval results."""

    def on_save(self, state: CallbackState, control: CallbackControl, checkpoint_dir: Optional[str] = None) -> None:
        """After a checkpoint is written; ``checkpoint_dir`` is where."""

    def on_push_begin(self,
                      state: CallbackState,
                      control: CallbackControl,
                      checkpoint_dir: Optional[str] = None) -> None:
        """Before a checkpoint is uploaded to the hub."""

    def on_predict(self, state: CallbackState, control: CallbackControl) -> None:
        """Optional: generative-evaluation path only (see the predict_with_generate dispatch)."""

    def on_prediction_step(self, state: CallbackState, control: CallbackControl) -> None:
        """Optional: per-batch hook of the generative-evaluation path."""


#: The ``callback`` extension point, declared beside its base class -- the same pattern as the ``reward``
#: kind (``rewards/orm.py``) and the ``tool`` kind (``rollout/sandbox.py``). ``config_field`` is
#: ``TrainConfig.callbacks``. Declared here rather than in the package ``__init__`` so the built-ins,
#: which ``@PluginRegistry.register('callback', ...)`` against it at import time, always find it declared.
CALLBACK = PluginRegistry.register_kind('callback', TrainCallback, config_field='callbacks')
