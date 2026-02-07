# Copyright (c) ModelScope Contributors. All rights reserved.
from .base import MegatronCallback


class DefaultFlowCallback(MegatronCallback):

    def on_step_end(self):
        args = self.args
        state = self.state

        state.iteration += 1
        state.consumed_train_samples += args.global_batch_size

        if state.iteration == 1 or state.iteration % args.log_interval == 0:
            self.state.should_log = True
        if args.eval_interval and state.iteration % args.eval_interval == 0:
            self.state.should_eval = True
        if args.save_interval and state.iteration % args.save_interval == 0:
            self.state.should_save = True

        if state.iteration >= args.train_iters:
            self.state.should_eval = True
            self.state.should_save = True
