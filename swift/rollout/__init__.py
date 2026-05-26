# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .agent_loop import extract_logprobs_from_choice, invoke_async_hook, run_multi_turn
    from .gym_env import Env, envs
    from .multi_turn import MultiTurnScheduler, RolloutScheduler, multi_turns

else:
    _import_structure = {
        'multi_turn': ['multi_turns', 'RolloutScheduler', 'MultiTurnScheduler'],
        'gym_env': ['envs', 'Env'],
        'agent_loop': ['run_multi_turn', 'extract_logprobs_from_choice', 'invoke_async_hook'],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
