# Copyright (c) ModelScope Contributors. All rights reserved.
"""Sandbox / tool-plugin wiring tests (no GPU, no model, no network).

A multi-turn rollout can let the model act inside a sandbox: it leases an env from a pool and the tools
it may call come from tool plugins bound to that env. Tools are opt-in -- with ``RolloutConfig.tools``
empty (the default) the rollout runs with no tools at all. These tests pin that default-off contract, the
plugin resolution through the ``tool`` extension point, and the env/tool binding, monkeypatching the
lazily-imported twinkle env classes so nothing heavy (or networked) is constructed.
"""
import pytest

from swift.dev.config import RolloutConfig
from swift.dev.rollout import sandbox as sandbox_mod
from swift.dev.rollout.sandbox import (SandboxTools, build_env_pool, build_tool_sandbox, resolve_tool_plugins,
                                       tool_manager_for)


# --- default-off contract -------------------------------------------------------------


def test_build_tool_sandbox_none_config_is_off():
    assert build_tool_sandbox(None) == (None, [])


def test_build_tool_sandbox_empty_tools_is_off():
    """The default RolloutConfig has no tools, so no env pool is built and no plugins resolved."""
    assert build_tool_sandbox(RolloutConfig()) == (None, [])


def test_resolve_tool_plugins_none_is_empty():
    assert resolve_tool_plugins(None, RolloutConfig()) == []
    assert resolve_tool_plugins([], RolloutConfig()) == []


# --- plugin resolution ----------------------------------------------------------------


def test_resolve_tool_plugins_sandbox_by_name():
    """'sandbox' names the built-in plugin registered under the 'tool' kind on import."""
    plugins = resolve_tool_plugins(['sandbox'], RolloutConfig())
    assert len(plugins) == 1
    assert isinstance(plugins[0], SandboxTools)


def test_sandbox_tools_build_wraps_env_tools(monkeypatch):
    """SandboxTools.build(env) returns EnvTool.from_env(env) -- tools bound to the exact leased env."""
    env = object()
    calls = {}

    class _FakeEnvTool:

        @staticmethod
        def from_env(e):
            calls['env'] = e
            return ['run_command', 'write_file']

    import twinkle_agentic.envs as envs_mod
    monkeypatch.setattr(envs_mod, 'EnvTool', _FakeEnvTool, raising=False)
    tools = SandboxTools().build(env)
    assert tools == ['run_command', 'write_file']
    assert calls['env'] is env  # the env is threaded through, not a fresh one


# --- env pool -------------------------------------------------------------------------


def test_build_env_pool_rejects_zero_envs(monkeypatch):
    cfg = RolloutConfig(sandbox_num_envs=0)
    with pytest.raises(ValueError, match='sandbox_num_envs'):
        build_env_pool(cfg)


def test_build_env_pool_local_builds_one_slot_per_env(monkeypatch, tmp_path):
    """No template -> local LocalEnv slots under the workspace root, wrapped in an EnvLeases pool."""
    built = {}

    class _FakeLocalEnv:

        def __init__(self, workspace, command_timeout, memory_limit_gb):
            built.setdefault('workspaces', []).append(workspace)
            built['command_timeout'] = command_timeout
            built['memory_limit_gb'] = memory_limit_gb

    class _FakeEnvLeases:

        def __init__(self, envs):
            built['envs'] = envs

    import twinkle_agentic.envs as envs_mod
    monkeypatch.setattr(envs_mod, 'LocalEnv', _FakeLocalEnv, raising=False)
    monkeypatch.setattr(envs_mod, 'EnvLeases', _FakeEnvLeases, raising=False)

    cfg = RolloutConfig(sandbox_num_envs=2, sandbox_workspace_root=str(tmp_path))
    pool = build_env_pool(cfg)
    assert isinstance(pool, _FakeEnvLeases)
    assert len(built['envs']) == 2
    # each slot gets its own workspace dir so concurrent episodes never cross files
    assert built['workspaces'] == [str(tmp_path / 'slot_0'), str(tmp_path / 'slot_1')]


def test_build_tool_sandbox_with_tools_builds_pool_and_plugins(monkeypatch, tmp_path):
    """Naming a tool turns the sandbox on: an env pool is built and the plugins resolved against the cfg."""

    class _FakeLocalEnv:

        def __init__(self, *a, **k):
            pass

    class _FakeEnvLeases:

        def __init__(self, envs):
            self.envs = envs

    import twinkle_agentic.envs as envs_mod
    monkeypatch.setattr(envs_mod, 'LocalEnv', _FakeLocalEnv, raising=False)
    monkeypatch.setattr(envs_mod, 'EnvLeases', _FakeEnvLeases, raising=False)

    cfg = RolloutConfig(tools=['sandbox'], sandbox_num_envs=1, sandbox_workspace_root=str(tmp_path))
    pool, plugins = build_tool_sandbox(cfg)
    assert isinstance(pool, _FakeEnvLeases) and len(pool.envs) == 1
    assert len(plugins) == 1 and isinstance(plugins[0], SandboxTools)


# --- tool manager ---------------------------------------------------------------------


def test_tool_manager_for_no_plugins_is_none(monkeypatch):
    import twinkle_agentic.tools.tool_manager as tm_mod
    monkeypatch.setattr(tm_mod, 'ToolManager', lambda tools: ('TM', tools), raising=False)
    assert tool_manager_for(object(), []) is None


def test_tool_manager_for_binds_tools_to_the_leased_env(monkeypatch):
    """Tools are built per env (not per run) so an episode acts in the workspace it leased."""
    captured = {}

    class _FakeToolManager:

        def __init__(self, tools):
            captured['tools'] = tools

    class _Plugin:

        def build(self, env):
            captured['env'] = env
            return [f'tool-for-{id(env)}']

    import twinkle_agentic.tools.tool_manager as tm_mod
    monkeypatch.setattr(tm_mod, 'ToolManager', _FakeToolManager, raising=False)
    env = object()
    manager = tool_manager_for(env, [_Plugin()])
    assert isinstance(manager, _FakeToolManager)
    assert captured['env'] is env
    assert captured['tools'] == [f'tool-for-{id(env)}']


def test_tool_kind_registers_sandbox_against_the_tools_field():
    """Importing the module declares the 'tool' extension point (selected via RolloutConfig.tools) and
    registers the built-in 'sandbox' plugin against it."""
    from swift.dev.rollout.sandbox import TOOL
    assert TOOL.name == 'tool'
    assert TOOL.config_field == 'tools'
    assert 'sandbox' in TOOL.entries
    assert hasattr(sandbox_mod, 'build_tool_sandbox')
