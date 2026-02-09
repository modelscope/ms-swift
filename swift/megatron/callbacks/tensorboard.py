# Copyright (c) ModelScope Contributors. All rights reserved.
import os

from swift.utils import check_json_format
from .base import MegatronCallback
from .utils import rewrite_logs


class TensorboardCallback(MegatronCallback):

    def __init__(self, trainer):
        super().__init__(trainer)
        args = self.args
        self.config = check_json_format(vars(args))
        self.save_dir = args.tensorboard_dir
        if self.save_dir is None:
            self.save_dir = f'{args.output_dir}/runs'
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=self.save_dir, max_queue=args.tensorboard_queue_size)
        for k, v in self.config.items():
            self.writer.add_text(k, v, global_step=self.state.iteration)

    def on_log(self, logs):
        logs = rewrite_logs(logs)
        logs['iteration'] = self.state.iteration
        for k, v in logs.items():
            self.writer.add_scalar(k, v, self.state.iteration)

    def on_train_end(self, args, state, control, **kwargs):
        if self.tb_writer:
            self.tb_writer.close()
            self.tb_writer = None
