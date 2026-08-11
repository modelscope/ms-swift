# Copyright (c) ModelScope Contributors. All rights reserved.
from transformers.trainer_callback import ProgressCallback, TrainerControl, TrainerState

from swift.utils import get_logger
from swift.utils.profiler import DistProfiler

logger = get_logger()


class ProfilerCallback(ProgressCallback):

    def __init__(self, args, trainer):
        super().__init__()
        self.args = args
        self.trainer = trainer
        self.trainer.profiler = DistProfiler(global_config=args)

    def on_step_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if self.args.profiler_steps and state.global_step in self.args.profiler_steps:
            self.trainer.profiler.start()
        super().on_step_begin(args, state, control, **kwargs)

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if self.args.profiler_steps and state.global_step + 1 not in self.args.profiler_steps:
            self.trainer.profiler.stop()
        super().on_step_end(args, state, control, **kwargs)
