# Copyright (c) ModelScope Contributors. All rights reserved.
from swift.utils.profiler import DistProfiler
from .base import MegatronCallback
from swift.utils import get_logger
logger = get_logger()


class ProfilerCallback(MegatronCallback):
    def __init__(self, trainer):
        super().__init__(trainer)
        self.args = trainer.args
        self.trainer = trainer
        self.trainer.profiler = DistProfiler(global_config=self.args)

    def on_step_begin(self):
        if self.args.profiler_steps and self.state.global_step in self.args.profiler_steps:
            self.trainer.profiler.start()

    def on_step_end(self):
        if self.args.profiler_steps and self.state.global_step + 1 not in self.args.profiler_steps:
            self.trainer.profiler.stop()
