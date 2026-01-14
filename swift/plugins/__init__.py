# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .callback import extra_callbacks, EarlyStopCallback
    from .tuner import Tuner, extra_tuners, PeftTuner
    from .prm import prms, PRM
    from .orm import orms, ORM, AsyncORM
    from .multi_turn import multi_turns, RolloutScheduler, MultiTurnScheduler
    from .rm_plugin import rm_plugins
    from .env import envs, Env
    from .context_manager import context_managers, ContextManager

else:
    _import_structure = {
        'callback': ['extra_callbacks', 'EarlyStopCallback'],
        'tuner': ['Tuner', 'extra_tuners', 'PeftTuner'],
        'prm': ['prms', 'PRM'],
        'orm': ['orms', 'ORM', 'AsyncORM'],
        'multi_turn': ['multi_turns', 'RolloutScheduler', 'MultiTurnScheduler'],
        'rm_plugin': ['rm_plugins'],
        'env': ['envs', 'Env'],
        'context_manager': ['context_managers', 'ContextManager'],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
