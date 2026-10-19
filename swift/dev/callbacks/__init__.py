# Copyright (c) ModelScope Contributors. All rights reserved.
from swift.dev.utils.plugin import PluginRegistry

from swift.dev.callbacks.base import CALLBACK, CallbackControl, CallbackState, TrainCallback
from swift.dev.callbacks.builtins import EarlyStopCallback, GracefulExitCallback, PerfLogCallback
from swift.dev.callbacks.handler import CallbackHandler

__all__ = [
    'CALLBACK', 'TrainCallback', 'CallbackState', 'CallbackControl', 'CallbackHandler', 'EarlyStopCallback',
    'PerfLogCallback', 'GracefulExitCallback', 'build_callbacks'
]


def build_callbacks(train_config) -> list:
    """Resolve ``TrainConfig.callbacks`` to instances, then append the warranted built-ins.

    User callbacks run first (so an ``on_evaluate`` sees the metric before early stop may veto);
    built-ins dedupe by class, early_stop only when ``early_stop_interval`` is set.
    """
    callbacks = [
        PluginRegistry.resolve('callback', name, config=train_config) for name in (train_config.callbacks or [])
    ]
    present = {type(cb) for cb in callbacks}

    def _append(cls):
        if cls not in present:
            callbacks.append(cls(args=train_config))
            present.add(cls)

    if getattr(train_config, 'early_stop_interval', None):
        _append(EarlyStopCallback)
    _append(PerfLogCallback)
    _append(GracefulExitCallback)
    return callbacks
