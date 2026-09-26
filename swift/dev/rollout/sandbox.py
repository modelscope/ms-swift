# Copyright (c) ModelScope Contributors. All rights reserved.
"""Sandbox envs and tool plugins for a multi-turn rollout.

A multi-turn rollout can let the model act inside a sandbox: each concurrent episode leases one env
from a pool (a ``LocalEnv`` workspace by default, or an ``AgentEnv`` microVM when a template is named),
and the tools it may call come from tool plugins bound to that env. Tools are opt-in -- with no tool
plugins configured (``RolloutConfig.tools`` empty, the default) the rollout runs with no tools at all,
exactly as before this module existed.

This mirrors twinkle's own layout: the env pool + per-episode lease is the cookbook ``rsi_grpo`` pattern
(one workspace slot per concurrent job, because a single ``LocalEnv`` holds one workspace and sharing it
across episodes would cross their files), and a tool plugin's ``build(env)`` returns twinkle ``Tool``
instances (usually ``EnvTool``s) so an episode acts in -- and is judged in -- the same env it leased.

Importing this module registers the ``tool`` extension point and its built-in ``sandbox`` plugin; the
twinkle env/tool classes are imported lazily inside the functions so importing it stays cheap.
"""
from __future__ import annotations
import os
import tempfile
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from swift.dev.plugin import PluginRegistry, ToolPlugin

if TYPE_CHECKING:
    from swift.dev.config import RolloutConfig

__all__ = ['TOOL', 'SandboxTools', 'build_env_pool', 'build_tool_sandbox', 'resolve_tool_plugins', 'tool_manager_for']

#: The ``tool`` extension point. Implementations subclass :class:`ToolPlugin` and a run selects them by
#: name through ``RolloutConfig.tools`` (the ``config_field``), resolved via :meth:`resolve_tool_plugins`.
TOOL = PluginRegistry.register_kind('tool', ToolPlugin, config_field='tools')


@PluginRegistry.register('tool', 'sandbox')
class SandboxTools(ToolPlugin):
    """Expose the sandbox env's own default tools (``run_command`` / ``write_file`` / ``read_file``).

    The env advertises its tool schemas through ``env.tools()``; ``EnvTool.from_env`` wraps each into a
    callable bound to that exact env, so every call an episode makes lands in the workspace it leased.
    """

    def build(self, env: Any) -> List[Any]:
        from twinkle_agentic.envs import EnvTool
        return EnvTool.from_env(env)


def build_env_pool(rollout_config: 'RolloutConfig') -> Any:
    """RolloutConfig -> an ``EnvLeases`` pool of sandbox environments.

    ``sandbox_num_envs`` envs are built; one is leased per concurrent episode (a lease never blocks when
    the pool has at least as many members as there are workers). With no ``sandbox_template`` the pool is
    local ``LocalEnv`` slots under ``sandbox_workspace_root`` (a fresh temp dir when unset); naming a
    template builds ``AgentEnv`` microVMs instead, which boot on first use from the pool's ``clear()``.
    """
    from twinkle_agentic.envs import AgentEnv, EnvLeases, LocalEnv

    num_envs = rollout_config.sandbox_num_envs
    if num_envs < 1:
        raise ValueError(f'RolloutConfig.sandbox_num_envs must be >= 1, got {num_envs}.')
    template = rollout_config.sandbox_template
    if template:
        kwargs: dict = {}
        if rollout_config.sandbox_api_url:
            kwargs['api_url'] = rollout_config.sandbox_api_url
        if rollout_config.sandbox_timeout is not None:
            kwargs['sandbox_timeout'] = rollout_config.sandbox_timeout
        envs = [
            AgentEnv(
                template=template,
                command_timeout=rollout_config.sandbox_command_timeout,
                metadata={'run': 'swift', 'slot': str(i)},
                **kwargs) for i in range(num_envs)
        ]
    else:
        root = rollout_config.sandbox_workspace_root or tempfile.mkdtemp(prefix='swift_sandbox_')
        envs = [
            LocalEnv(
                workspace=os.path.join(root, f'slot_{i}'),
                command_timeout=rollout_config.sandbox_command_timeout,
                memory_limit_gb=rollout_config.sandbox_memory_limit_gb) for i in range(num_envs)
        ]
    return EnvLeases(envs)


def resolve_tool_plugins(names: Optional[List[str]], rollout_config: 'RolloutConfig') -> List[ToolPlugin]:
    """Registered ``tool`` plugin names -> constructed :class:`ToolPlugin` instances.

    Each is built with the RolloutConfig as its ``args`` (``cls(args=config)``), the same way a reward
    plugin reads its hyperparameters off the run's config. An empty/None list yields no plugins, i.e. a
    rollout with no tools.
    """
    return [PluginRegistry.resolve(TOOL, name, config=rollout_config) for name in (names or [])]


def tool_manager_for(env: Any, plugins: List[ToolPlugin]) -> Optional[Any]:
    """The ``ToolManager`` exposing every plugin's tools for one leased env, or None when there are none.

    Built per env, not per run: the tools are bound to the exact workspace this episode leased, and
    crossing managers between envs is the failure twinkle's ``Env.tool_manager`` docstring warns about
    (an episode acting in one workspace and being judged in another). None means "no tools this episode".
    """
    from twinkle_agentic.tools.tool_manager import ToolManager

    tools = [tool for plugin in plugins for tool in plugin.build(env)]
    return ToolManager(tools) if tools else None


def build_tool_sandbox(rollout_config: Optional['RolloutConfig']) -> Tuple[Any, List[ToolPlugin]]:
    """The ``(env_pool, tool_plugins)`` for a multi-turn rollout, or ``(None, [])`` when tools are off.

    The single entry point both recipes call. Tools are opt-in: an empty ``RolloutConfig.tools`` (the
    default), or no RolloutConfig at all, yields no plugins and no env pool, so the rollout behaves
    exactly as it did before tools existed. When tools are named, a sandbox env pool is built (a local
    ``LocalEnv`` by default, an ``AgentEnv`` microVM when a template is set) and the plugins resolved
    against the RolloutConfig; each episode then leases its own env.
    """
    if rollout_config is None or not getattr(rollout_config, 'tools', None):
        return None, []
    return build_env_pool(rollout_config), resolve_tool_plugins(rollout_config.tools, rollout_config)
