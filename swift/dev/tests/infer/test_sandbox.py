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
from swift.dev.builders import sandbox as sandbox_mod
from swift.dev.builders.sandbox import SandboxTools, build_env_pool, build_tools, resolve_tool_plugins


# --- default-off contract -------------------------------------------------------------


def test_build_tools_none_config_is_off():
    assert build_tools(None) == (None, None)


def test_build_tools_empty_tools_is_off():
    """The default RolloutConfig has no tools, so no env pool is built and no tool manager created."""
    assert build_tools(RolloutConfig()) == (None, None)


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
    """The rollout binds the env's tools to each episode's session itself, so the plugin
    contributes nothing to the run's shared tool manager."""
    assert SandboxTools().build() == []


# --- env pool -------------------------------------------------------------------------


def test_build_env_pool_rejects_zero_envs(monkeypatch):
    cfg = RolloutConfig(sandbox_num_envs=0)
    with pytest.raises(ValueError, match='sandbox_num_envs'):
        build_env_pool(cfg)


def test_build_env_pool_local_builds_one_slot_per_env(monkeypatch, tmp_path):
    """No template -> ONE LocalEnv under the workspace root, its session cap the whole budget."""
    built = {}

    class _FakeLocalEnv:

        def __init__(self, root, command_timeout, memory_limit_gb, max_sessions):
            built.update(root=root, command_timeout=command_timeout,
                         memory_limit_gb=memory_limit_gb, max_sessions=max_sessions)

    import twinkle_agentic.envs as envs_mod
    monkeypatch.setattr(envs_mod, 'LocalEnv', _FakeLocalEnv, raising=False)

    cfg = RolloutConfig(sandbox_num_envs=2, sandbox_workspace_root=str(tmp_path))
    env = build_env_pool(cfg)
    assert isinstance(env, _FakeLocalEnv)
    assert built['root'] == str(tmp_path)
    # one worker in-process, so the cap is the whole budget: ceil(2 / 1)
    assert built['max_sessions'] == 2


def test_build_tools_with_tools_builds_pool_and_manager(monkeypatch, tmp_path):
    """Naming a tool turns the sandbox on: an env pool is built and the plugins' tools go into one manager."""

    class _FakeLocalEnv:

        def __init__(self, *a, **k):
            pass

    import twinkle_agentic.envs as envs_mod
    monkeypatch.setattr(envs_mod, 'LocalEnv', _FakeLocalEnv, raising=False)

    cfg = RolloutConfig(tools=['sandbox'], sandbox_num_envs=1, sandbox_workspace_root=str(tmp_path))
    env, tool_manager = build_tools(cfg)
    assert isinstance(env, _FakeLocalEnv)
    # 'sandbox' builds the env only; with no other plugin there is nothing to manage
    assert tool_manager is None


def test_tool_kind_registers_sandbox_against_the_tools_field():
    """Importing the module declares the 'tool' extension point (selected via RolloutConfig.tools) and
    registers the built-in 'sandbox' plugin against it."""
    from swift.dev.builders.sandbox import TOOL
    assert TOOL.name == 'tool'
    assert TOOL.config_field == 'tools'
    assert 'sandbox' in TOOL.entries
    assert hasattr(sandbox_mod, 'build_tools')
