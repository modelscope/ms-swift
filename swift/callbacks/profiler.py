# Copyright (c) ModelScope Contributors. All rights reserved.
from swift.utils.profiler import ProfilerSession
from .base import TrainerCallback


class SwiftProfilerCallback(TrainerCallback):
    """Collect one profiling window using completed Transformers training steps."""

    def __init__(self, args, trainer):
        super().__init__(args, trainer)
        self.session = ProfilerSession(args, 'transformers')

    def on_step_begin(self, args, state, control, **kwargs):
        self.session.begin_step(state.global_step)

    def on_step_end(self, args, state, control, **kwargs):
        self.session.end_step()

    def on_train_end(self, args, state, control, **kwargs):
        self.session.train_end()
