# Copyright (c) ModelScope Contributors. All rights reserved.
# Sandbox env and tool plugins for a multi-turn rollout (config -> twinkle Env / ToolManager).
from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from swift.dev.utils.plugin import PluginRegistry, ToolPlugin

if TYPE_CHECKING:
    from swift.dev.config import RolloutConfig

__all__ = ['TOOL', 'SandboxTools', 'build_env_pool', 'build_tools', 'resolve_tool_plugins']

#: The ``tool`` extension point. Implementations subclass :class:`ToolPlugin` and a run selects them
#: by name through ``RolloutConfig.tools`` (the ``config_field``), resolved via :meth:`resolve_tool_plugins`.
TOOL = PluginRegistry.register_kind('tool', ToolPlugin, config_field='tools')

#: The tool name that turns the sandbox env on.
SANDBOX = 'sandbox'


@PluginRegistry.register('tool', SANDBOX)
class SandboxTools(ToolPlugin):
    """Expose the sandbox env's own tools (``run_command`` / ``write_file`` / ``read_file``).

    Naming it builds the run's env (:func:`build_env_pool`); the rollout binds the env's tools to each
    episode's session itself, so this plugin contributes nothing to the shared tool manager.
    """

    def build(self) -> List[Any]:
        return []


def build_env_pool(rollout_config: 'RolloutConfig', *, remote_group: Optional[str] = None,
                   inflight: Optional[int] = None) -> Any:
    """RolloutConfig -> one sandbox ``Env`` component, a session per episode.

    ``sandbox_num_envs`` is the session cap over all replicas; each of the ``sandbox_num_workers``
    replicas (the ``remote_group``'s ranks) takes ``ceil(sandbox_num_envs / sandbox_num_workers)``.
    ``inflight`` bounds the episodes that may call the env at once: a replica then takes that many
    calls beyond its sessions, so the ``reset`` calls waiting for a session never hold every thread
    while ``close`` waits to free one. The family is picked by ``sandbox_kind``: 'local' sessions
    under ``sandbox_workspace_root`` (a fresh temp dir when unset); 'agentenv' microVMs
    (``sandbox_template``); 'openenv' embeds an OpenEnv environment package in-process;
    'openenv_client' connects to a running OpenEnv server. Leaving the kind at 'local' while naming a
    template still infers agentenv (back-compat).
    """
    from twinkle_agentic.envs import AgentEnv, LocalEnv, OpenEnv, OpenEnvClient

    num_envs = rollout_config.sandbox_num_envs
    if num_envs < 1:
        raise ValueError(f'RolloutConfig.sandbox_num_envs must be >= 1, got {num_envs}.')
    workers = rollout_config.sandbox_num_workers if remote_group is not None else 1
    max_sessions = math.ceil(num_envs / workers)
    common: dict = {'max_sessions': max_sessions}
    if remote_group is not None:
        common['remote_group'] = remote_group
    if inflight is not None:
        common['max_concurrency'] = max_sessions + inflight
    template = rollout_config.sandbox_template
    kind = rollout_config.sandbox_kind
    if kind in ('openenv', 'openenv_client'):
        env_name = rollout_config.sandbox_env_name
        if not env_name:
            raise ValueError(
                f"sandbox_kind='{kind}' needs sandbox_env_name (the OpenEnv environment package, "
                "e.g. 'coding_env'; pip install openenv-<name>).")
        if kind == 'openenv':
            return OpenEnv(env_name=env_name, **common)
        base_url = rollout_config.sandbox_api_url
        if not base_url and not os.environ.get('OPENENV_BASE_URL'):
            raise ValueError(
                "sandbox_kind='openenv_client' needs sandbox_api_url (or the OPENENV_BASE_URL "
                'env var) pointing at a running OpenEnv server.')
        return OpenEnvClient(env_name=env_name, base_url=base_url, **common)
    if kind == 'agentenv' or (kind == 'local' and template):
        if template is None:
            raise ValueError(
                "sandbox_kind='agentenv' needs sandbox_template (the microVM template name/ID); "
                'pick another sandbox_kind or name a template.')
        kwargs: dict = {}
        if rollout_config.sandbox_api_url:
            kwargs['api_url'] = rollout_config.sandbox_api_url
        if rollout_config.sandbox_timeout is not None:
            kwargs['sandbox_timeout'] = rollout_config.sandbox_timeout
        return AgentEnv(
            template=template,
            command_timeout=rollout_config.sandbox_command_timeout,
            metadata={'run': 'swift'},
            **kwargs,
            **common)
    return LocalEnv(
        root=rollout_config.sandbox_workspace_root,
        command_timeout=rollout_config.sandbox_command_timeout,
        memory_limit_gb=rollout_config.sandbox_memory_limit_gb,
        **common)


def resolve_tool_plugins(names: Optional[List[str]], rollout_config: 'RolloutConfig') -> List[ToolPlugin]:
    """Registered ``tool`` plugin names -> constructed :class:`ToolPlugin` instances.

    Each is built with the RolloutConfig as its ``args`` (``cls(args=config)``), the same way a
    reward plugin reads its hyperparameters off the run's config; an empty/None list yields no
    plugins, i.e. a rollout with no tools.
    """
    return [PluginRegistry.resolve(TOOL, name, config=rollout_config) for name in (names or [])]


def build_tools(rollout_config: Optional['RolloutConfig'], *, env_group: Optional[str] = None,
                inflight: Optional[int] = None) -> Tuple[Any, Optional[Any]]:
    """The ``(env, tool_manager)`` a multi-turn rollout is built with; ``(None, None)`` when tools are off.

    Tools are opt-in through ``RolloutConfig.tools``: naming ``sandbox`` builds the env (see
    :func:`build_env_pool`, ``env_group`` / ``inflight`` go there); every other plugin's tools go into
    one ``ToolManager`` the rollout copies per episode.
    """
    from twinkle_agentic.tools.tool_manager import ToolManager

    names = list(getattr(rollout_config, 'tools', None) or [])
    if not names:
        return None, None
    env = (build_env_pool(rollout_config, remote_group=env_group, inflight=inflight)
           if SANDBOX in names else None)
    tools = [tool for plugin in resolve_tool_plugins(names, rollout_config) for tool in plugin.build()]
    return env, (ToolManager(tools) if tools else None)
