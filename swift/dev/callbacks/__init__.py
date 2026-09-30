# Copyright (c) ModelScope Contributors. All rights reserved.
"""Training callbacks: the ``callback`` extension point and its built-ins.

Importing this package declares the ``callback`` plugin kind (so a user's ``--external_plugins`` file
may ``@PluginRegistry.register('callback', ...)`` against it) and registers the three built-ins. The
kind's ``config_field`` is ``TrainConfig.callbacks``: a run names custom callbacks there, and
:func:`build_callbacks` resolves those names to instances and appends the built-ins their own config
knobs warrant.
"""
from swift.dev.callbacks.base import CALLBACK, CallbackControl, CallbackState, TrainCallback
from swift.dev.callbacks.builtins import EarlyStopCallback, GracefulExitCallback, PerfLogCallback
from swift.dev.callbacks.handler import CallbackHandler
from swift.dev.plugin import PluginRegistry

__all__ = [
    'CALLBACK', 'TrainCallback', 'CallbackState', 'CallbackControl', 'CallbackHandler', 'EarlyStopCallback',
    'PerfLogCallback', 'GracefulExitCallback', 'build_callbacks'
]


def build_callbacks(train_config) -> list:
    """Resolve ``TrainConfig.callbacks`` to instances and append the warranted built-ins.

    Order matters: user-named callbacks run first (in the order given), then the built-ins, so a
    user's ``on_evaluate`` sees the metric before :class:`EarlyStopCallback` may veto the run. A
    built-in already named explicitly is not appended twice (deduped by class), so naming ``perf_log``
    keeps the user's single instance and position.

    - ``EarlyStopCallback`` only when ``early_stop_interval`` is set (otherwise it could never fire).
    - ``PerfLogCallback`` / ``GracefulExitCallback`` always: throughput visibility and a clean
      SIGTERM/SIGINT stop are the defaults a training run wants, mirroring legacy swift.
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
