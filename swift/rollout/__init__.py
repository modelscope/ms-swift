# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .multi_turn import multi_turns, RolloutScheduler, MultiTurnScheduler
    from .gym_env import envs, Env, context_managers, ContextManager

else:
    _import_structure = {
        'multi_turn': ['multi_turns', 'RolloutScheduler', 'MultiTurnScheduler'],
        'gym_env': ['envs', 'Env', 'context_managers', 'ContextManager'],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
