# Copyright (c) ModelScope Contributors. All rights reserved.

import os

from swift.utils import check_json_format
from .base import MegatronCallback
from .utils import rewrite_logs


class SwanlabCallback(MegatronCallback):

    def __init__(self, trainer):
        super().__init__(trainer)
        args = self.args
        self.config = check_json_format(vars(args))
        if args.swanlab_exp_name is None:
            args.swanlab_exp_name = args.output_dir
        self.save_dir = os.path.join(args.output_dir, 'swanlab')
        self.setup()

    def setup(self):
        import swanlab
        args = self.args
        swanlab.init(
            logdir=self.save_dir,
            experiment_name=args.swanlab_exp_name,
            project=args.swanlab_project,
            config=self.config)
        self.writer = swanlab

    def on_log(self, logs):
        logs = rewrite_logs(logs)
        logs['iteration'] = self.state.iteration
        self.writer.log(logs, step=self.state.iteration)
