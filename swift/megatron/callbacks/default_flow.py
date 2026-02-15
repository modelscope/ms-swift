# Copyright (c) ModelScope Contributors. All rights reserved.
import gc

from .base import MegatronCallback


class DefaultFlowCallback(MegatronCallback):

    def on_train_begin(self):
        args = self.args
        if args.manual_gc:
            gc.disable()
            gc.collect()

    def on_step_end(self):
        args = self.args
        state = self.state

        state.consumed_train_samples += args.global_batch_size

        if state.iteration == 1 or state.iteration % args.log_interval == 0:
            state.should_log = True
        if args.eval_interval and state.iteration % args.eval_interval == 0 and args.eval_iters > 0:
            state.should_eval = True
        if args.save_interval and state.iteration % args.save_interval == 0:
            state.should_save = True

        if state.iteration >= args.train_iters:
            if args.eval_iters > 0:
                state.should_eval = True
            state.should_save = True
        if args.manual_gc and args.manual_gc_interval != 0 and state.iteration % args.manual_gc_interval == 0:
            gc.collect()

    def on_eval_begin(self):
        args = self.args
        if args.manual_gc and args.manual_gc_eval:
            gc.collect()

    def on_eval_end(self):
        args = self.args
        if args.manual_gc and args.manual_gc_eval:
            gc.collect(generation=0)
