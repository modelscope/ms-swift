# Copyright (c) ModelScope Contributors. All rights reserved.
"""The 'harness' plugin kind: an external agent program the rollout runs per episode.

A harness plugin builds the CLI agent a rollout episode starts inside its env session --
codex, ms_agent, or a program of your own. The plugin's :meth:`build` returns the twinkle
:class:`~twinkle_agentic.harness.CliAgent` instance every episode runs: its ``command`` is
what starts the program, its ``noise_reply`` answers the program's housekeeping traffic.
The rollout owns the rest -- the endpoint the program calls back, the per-episode env
session, the account of every generation it makes -- so a plugin declares the program and
nothing about training.

Example::

    from swift.dev.utils.plugin import PluginRegistry
    from swift.dev.harness import HarnessPlugin

    @PluginRegistry.register('harness', 'my_agent')
    class MyAgent(HarnessPlugin):
        def build(self):
            return MyCliAgent(...)  # any twinkle_agentic.harness.CliAgent subclass
"""
from twinkle_agentic.harness.base import CliAgent

from swift.dev.utils.plugin import PluginRegistry, SwiftPlugin

__all__ = ['HarnessPlugin', 'harness']


class HarnessPlugin(SwiftPlugin):
    """Build the :class:`CliAgent` a rollout episode runs.

    Constructed once per run (``args=RolloutConfig``), asked for the agent once; the
    rollout starts one instance of the program per episode, in that episode's session.
    """

    def build(self) -> CliAgent:
        raise NotImplementedError


#: The kind: selected by ``RolloutConfig.harness``.
harness = PluginRegistry.register_kind('harness', HarnessPlugin, config_field='harness')


# The built-in programs, under the names the config accepts.
@PluginRegistry.register('harness', 'codex')
class _Codex(HarnessPlugin):
    def build(self) -> CliAgent:
        from twinkle_agentic.harness.codex import CodexAgent
        return CodexAgent()


@PluginRegistry.register('harness', 'ms_agent')
class _MSAgent(HarnessPlugin):
    def build(self) -> CliAgent:
        from twinkle_agentic.harness.ms_agent import MsAgent
        return MsAgent()
