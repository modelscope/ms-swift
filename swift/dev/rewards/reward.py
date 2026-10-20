# Copyright (c) ModelScope Contributors. All rights reserved.
"""Reward-rule resolution: the ORM plugin base (twinkle's Reward protocol) and name resolution."""
from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable, List, Optional, Sequence, Tuple

from swift.dev.utils.plugin import PluginRegistry, SwiftPlugin
# ground_truth/last_assistant_text moved out of twinkle.reward.base into twinkle.data_format
# (twinkle now keeps the trajectory helpers with the data-format types, mirroring its own
# reward modules); Reward stays the protocol identity in reward.base.
from twinkle.data_format import ground_truth, last_assistant_text
from twinkle.reward.base import Reward

# A reward rule: (trajectories, **kwargs) -> one score per trajectory.
RewardFunc = Callable[..., List[float]]

__all__ = ['ORM', 'RewardFunc', 'get_reward_funcs', 'is_registered_reward']


class ORM(SwiftPlugin, Reward):
    """A swift reward rule over trajectory rows: ``__call__(trajectories, **kwargs)``.

    A row IS a trajectory dict -- ``messages`` plus the sample's flat extras and
    ``response_token_ids`` -- so the runtime protocol is one shape for every reward kind
    (rule, scalar RM, judge). ``SwiftPlugin`` keeps the registry construction contract
    (``cls(args=config)``); ``Reward`` is twinkle's protocol identity.
    """

    @staticmethod
    def completion(trajectory: Any) -> str:
        """The newest assistant turn's text -- the completion a rule scores."""
        return last_assistant_text(trajectory)

    @staticmethod
    def column(trajectory: Any, key: str, default: Any = None) -> Any:
        """A row's dataset column: the packed ``user_data`` label first, the flat extra second."""
        value = ground_truth(trajectory, key)
        if value is None:
            value = trajectory.get(key)
        return default if value is None else value


def is_registered_reward(spec: Any) -> bool:
    """True when ``spec`` is a reward RULE rather than a reward-model id.

    The merged ``--orm`` / ``--prm`` selectors mix rule names with reward-model ids in one list, so a
    caller must tell them apart before resolving: a registered name, a plugin class, or an
    already-callable reward is a rule scored on completions; anything else is a reward-model id.
    Registered-name-first, so an id that happens to equal a rule name is taken as the rule.
    """
    from .orm import REWARD  # imported lazily (orm imports this module for the ORM base)

    if isinstance(spec, str):
        return spec in PluginRegistry.kind(REWARD).entries
    return isinstance(spec, type) or callable(spec)


class _AwaitableToSync:
    """Call ``func`` and run a returned coroutine to completion -- the async-reward seam.

    The scoring loop invokes each resolved reward synchronously (twinkle's grouped runner), but a
    plugin may be written ``async def __call__`` when its scoring does I/O; the shell detects the
    returned awaitable and resolves it with one ``asyncio.run``, so the reward's own internal
    gather still overlaps its batch's round trips. A sync reward's list passes through untouched.
    """

    def __init__(self, func: RewardFunc):
        self._func = func
        # display_name() reads __name__ first, so the shell keeps the reward's metric label.
        self.__name__ = PluginRegistry.display_name(func)

    def __call__(self, *args, **kwargs):
        result = self._func(*args, **kwargs)
        if inspect.isawaitable(result):
            result = asyncio.run(result)
        return result


def get_reward_funcs(reward_funcs: Sequence[Any],
                     config: Optional[Any] = None) -> Tuple[List[RewardFunc], List[str]]:
    """Resolve reward specs to callables + display names.

    A spec is a name registered at the ``reward`` extension point (instantiated as ``cls(args=config)``
    so the plugin reads its own hyperparameters off the Config), a plugin class, or an already-callable
    reward function passed through unchanged. ``names`` are suitable for per-reward metric keys.
    """
    funcs: List[RewardFunc] = []
    names: List[str] = []
    from .orm import REWARD

    for spec in reward_funcs:
        func = PluginRegistry.resolve(REWARD, spec, config=config)
        funcs.append(_AwaitableToSync(func))
        names.append(PluginRegistry.display_name(func))
    return funcs, names
