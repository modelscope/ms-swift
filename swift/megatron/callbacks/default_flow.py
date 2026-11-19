# Copyright (c) ModelScope Contributors. All rights reserved.
import gc
import os

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

        if state.iteration == 1 or state.iteration % args.logging_steps == 0:
            state.should_log = True
        if args.eval_steps and state.iteration % args.eval_steps == 0 and args.eval_iters > 0:
            state.should_eval = True
        if args.save_steps and state.iteration % args.save_steps == 0:
            state.should_save = True

        if state.iteration >= args.train_iters:
            if args.eval_iters > 0:
                state.should_eval = True
            state.should_save = True

        if args.manual_gc and args.manual_gc_steps != 0 and state.iteration % args.manual_gc_steps == 0:
            gc.collect()

        # Prevent GPU memory growth caused by cached causal masks of varying sequence lengths.
        if os.environ.get('SWIFT_CLEAR_MASK_CACHE', '1').lower() not in ('0', '', 'false', 'no'):
            try:
                from transformer_engine.pytorch.attention.dot_product_attention import softmax as te_softmax
            except ImportError:
                pass
            else:
                mask_cache = getattr(te_softmax, '_default_causal_mask', None)
                if mask_cache is not None:
                    mask_cache.clear()

    def on_eval_begin(self):
        args = self.args
        if args.manual_gc and args.manual_gc_eval:
            gc.collect()

    def on_eval_end(self):
        args = self.args
        if args.manual_gc and args.manual_gc_eval:
            gc.collect(generation=0)
