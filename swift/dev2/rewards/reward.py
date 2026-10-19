# Copyright (c) ModelScope Contributors. All rights reserved.
"""Reward-rule resolution: the ORM plugin base (twinkle's Reward protocol) and name resolution."""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple

from swift.dev.utils.plugin import PluginRegistry, SwiftPlugin  # TODO(dev2): bridge until the utils batch
from twinkle.reward.base import Reward, ground_truth, last_assistant_text

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
    from .orm import REWARD  # the kind bridge (dev-owned singleton), imported lazily

    if isinstance(spec, str):
        return spec in PluginRegistry.kind(REWARD).entries
    return isinstance(spec, type) or callable(spec)


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
        funcs.append(func)
        names.append(PluginRegistry.display_name(func))
    return funcs, names
