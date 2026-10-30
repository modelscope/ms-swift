# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from swift.dev.utils.plugin import PluginRegistry, SwiftPlugin

__all__ = ["CallbackState", "CallbackControl", "TrainCallback", "CALLBACK"]


@dataclass
class CallbackState:
    """Loop position published before each event; read-only for callbacks."""

    global_step: int = 0  # completed optimizer steps
    epoch: int = 0  # zero-based epoch index currently running
    micro_step: int = 0  # Cumulative micro-batch statistic; GRPO counts configured GA even under DBS.
    loss: Optional[float] = None  # most recent training loss
    metrics: Dict[str, Any] = field(default_factory=dict)  # last log record / eval metrics
    best_metric: Optional[float] = None  # best metric_for_best_model so far
    output_dir: str = "output"


@dataclass
class CallbackControl:
    """Write side: a callback flips a flag to ask the loop to stop / save / evaluate / log."""

    should_training_stop: bool = False
    should_save: bool = False
    should_evaluate: bool = False
    should_log: bool = False


class TrainCallback(SwiftPlugin):
    """Base of every training callback: all events no-op, override what you need.

    Constructed as ``cls(args=TrainConfig)`` so a callback reads its own hyperparameters.
    """

    def on_init_end(self, state: CallbackState, control: CallbackControl) -> None:
        """After the loop is constructed, before training starts."""

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
        """After each micro-batch's backward."""

    def on_pre_optimizer_step(self, state: CallbackState, control: CallbackControl) -> None:
        """After grad clipping is about to run, before the optimizer steps."""

    def on_optimizer_step(self, state: CallbackState, control: CallbackControl) -> None:
        """Right after the optimizer stepped."""

    def on_log(self, state: CallbackState, control: CallbackControl, logs: Optional[Dict[str, Any]] = None) -> None:
        """When a training record is logged; ``logs`` is that record."""

    def on_evaluate(
        self, state: CallbackState, control: CallbackControl, metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """After an evaluation pass; ``metrics`` are the eval results."""

    def on_save(self, state: CallbackState, control: CallbackControl, checkpoint_dir: Optional[str] = None) -> None:
        """After a checkpoint is written; ``checkpoint_dir`` is where."""

    def on_push_begin(
        self, state: CallbackState, control: CallbackControl, checkpoint_dir: Optional[str] = None
    ) -> None:
        """Before a checkpoint is uploaded to the hub."""

    def on_predict(self, state: CallbackState, control: CallbackControl) -> None:
        """Generative-evaluation path only: once per eval pass with generation."""


#: The ``callback`` extension point; ``config_field`` is ``TrainConfig.callbacks``.
CALLBACK = PluginRegistry.register_kind("callback", TrainCallback, config_field="callbacks")
