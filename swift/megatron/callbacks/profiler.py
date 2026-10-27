# Copyright (c) ModelScope Contributors. All rights reserved.
from swift.utils.profiler import ProfilerSession
from .base import MegatronCallback


class MegatronProfilerCallback(MegatronCallback):
    """Collect one profiling window using completed Megatron training iterations."""

    def __init__(self, trainer):
        super().__init__(trainer)
        self.session = ProfilerSession(self.args, 'megatron')

    def on_step_begin(self):
        self.session.begin_step(self.trainer.state.iteration)

    def on_step_end(self):
        self.session.end_step()

    def on_train_end(self):
        self.session.train_end()
