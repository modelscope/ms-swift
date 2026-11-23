# A custom training callback plugin, loaded with --external_plugins and selected with --callbacks.
#
# The `callback` extension point (swift.dev.callbacks) fires at transformers-Trainer-equivalent points
# in dev's training loop. A callback subclasses TrainCallback, overrides only the events it needs (all
# no-op by default), and is registered by name with @PluginRegistry.register('callback', <name>). The
# loop passes a CallbackState (where training is: global_step / epoch / loss / metrics / best_metric)
# and a CallbackControl (what to ask the loop to do next: should_training_stop / should_save /
# should_evaluate / should_log). Set a control flag to steer the run -- e.g. should_save forces a
# checkpoint at that step.
#
# This plugin just observes and logs progress; see EarlyStopCallback (swift.dev.callbacks.builtins) for a
# built-in that reads the same state and sets control.should_training_stop.
from typing import Any, Dict, Optional

from swift.dev.callbacks import CallbackControl, CallbackState, TrainCallback
from swift.dev.utils.plugin import PluginRegistry
from swift.dev.utils import get_logger

logger = get_logger()


@PluginRegistry.register('callback', 'progress_printer')
class ProgressPrinterCallback(TrainCallback):
    """Log a heartbeat every 10 optimizer steps, plus the eval and end-of-run summaries."""

    def on_train_begin(self, state: CallbackState, control: CallbackControl) -> None:
        logger.info('[progress_printer] training started')

    def on_step_end(self, state: CallbackState, control: CallbackControl) -> None:
        if state.global_step % 10 == 0:
            logger.info(f'[progress_printer] step={state.global_step} epoch={state.epoch} loss={state.loss}')

    def on_evaluate(self,
                    state: CallbackState,
                    control: CallbackControl,
                    metrics: Optional[Dict[str, Any]] = None) -> None:
        logger.info(f'[progress_printer] eval at step {state.global_step}: {metrics}')

    def on_train_end(self, state: CallbackState, control: CallbackControl) -> None:
        logger.info(f'[progress_printer] training ended at step {state.global_step}; best={state.best_metric}')
