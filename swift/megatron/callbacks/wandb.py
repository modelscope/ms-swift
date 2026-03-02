# Copyright (c) ModelScope Contributors. All rights reserved.
import os

from swift.utils import check_json_format, get_logger, is_last_rank
from .base import MegatronCallback
from .utils import rewrite_logs

logger = get_logger()


class WandbCallback(MegatronCallback):

    def __init__(self, trainer):
        super().__init__(trainer)
        args = self.args
        self.config = check_json_format(vars(args))
        if args.wandb_exp_name is None:
            args.wandb_exp_name = args.output_dir
        self.save_dir = os.path.join(args.output_dir, 'wandb')
        self.writer = None
        self.setup()

    def setup(self):
        import wandb
        args = self.args
        if is_last_rank():
            wandb.init(dir=self.save_dir, name=args.wandb_exp_name, project=args.wandb_project, config=self.config)
            self.writer = wandb
            if wandb.run is not None and getattr(wandb.run, "url", None):
                logger.info(f'wandb_url: {wandb.run.url}')
                print(f'wandb_url: {wandb.run.url}', flush=True)

    def on_log(self, logs):
        logs = rewrite_logs(logs)
        if is_last_rank():
            self.writer.log(logs, step=self.state.iteration)
