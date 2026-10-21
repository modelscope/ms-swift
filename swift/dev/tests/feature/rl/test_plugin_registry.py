"""S8c — swift plugin registry mechanics (L1).

The registry is swift's own extension-point mechanism: it owns the kinds, the base classes and the
entries; twinkle owns loading (``load_module``). These pin the three contracts a user plugin relies on:
a kind is a singleton that may adopt an existing dict, ``@register`` shape-checks at registration time,
and ``load_external`` declares the built-in kinds BEFORE importing a user file (so the file can
``@register`` against a built-in kind) across all three source forms.

Maps to ut_plan S8-58/59/60.
"""
import os

import pytest


@pytest.fixture
def clean_env():
    """Undo a test's registry/env side effects WITHOUT dropping the built-in kinds.

    The built-in kinds are declared once, at their module's first import (``register_kind`` runs at
    module top level); ``_declare_builtin_kinds`` is a no-op on an already-cached module, so clearing
    KINDS here would leave them permanently gone. Snapshot the kind set + each kind's entries, then on
    teardown drop only kinds added during the test and restore each surviving kind's entries.
    """
    from swift.dev.utils.plugin import PluginRegistry, _declare_builtin_kinds

    _declare_builtin_kinds()  # ensure the built-ins are present before snapshotting
    env = dict(os.environ)
    kinds_snapshot = {name: dict(kind.entries) for name, kind in PluginRegistry.KINDS.items()}
    yield
    os.environ.clear()
    os.environ.update(env)
    for name in list(PluginRegistry.KINDS):
        if name not in kinds_snapshot:
            PluginRegistry.KINDS.pop(name, None)  # a probe kind added by the test
    for name, entries in kinds_snapshot.items():
        kind = PluginRegistry.KINDS.get(name)
        if kind is not None:
            kind.entries.clear()
            kind.entries.update(entries)  # removes probe entries, keeps the built-ins


# ----------------------------------------------------------------------
# S8-59 — kind singleton (re-declare raises) + entries adopts an existing dict by reference
# ----------------------------------------------------------------------
def test_builtin_kinds_are_singletons_and_redeclare_raises(clean_env):
    """S8-59: the five built-in kinds exist; re-declaring one (exist_ok=False) is a ValueError."""
    from swift.dev.utils.plugin import PluginRegistry, SwiftPlugin, _declare_builtin_kinds

    _declare_builtin_kinds()
    for name in ('reward', 'prm_scorer', 'tool', 'callback', 'param_groups'):
        assert name in PluginRegistry.KINDS
    with pytest.raises(ValueError, match='already registered'):
        PluginRegistry.register_kind('reward', SwiftPlugin)


def test_register_kind_adopts_existing_dict_and_register_shape_checks(clean_env):
    """S8-59: a kind adopts the passed ``entries`` dict BY REFERENCE (one registry per kind); @register
    writes into it, rejects a non-subclass, and rejects a duplicate name."""
    from swift.dev.utils.plugin import PluginRegistry, SwiftPlugin

    adopted = {}
    kind = PluginRegistry.register_kind('_s8_59_probe', SwiftPlugin, entries=adopted)
    assert kind.entries is adopted  # same object, not a copy

    @PluginRegistry.register('_s8_59_probe', 'x')
    class X(SwiftPlugin):
        pass

    assert adopted['x'] is X  # the decorator wrote into the adopted dict

    # a class that does not subclass the kind's base is rejected at REGISTRATION time
    with pytest.raises(TypeError, match='must subclass'):

        @PluginRegistry.register('_s8_59_probe', 'bad')
        class NotAPlugin:  # not a SwiftPlugin
            pass

    # a duplicate name is rejected (exist_ok=False)
    with pytest.raises(ValueError, match='already registered'):

        @PluginRegistry.register('_s8_59_probe', 'x')
        class X2(SwiftPlugin):
            pass


# ----------------------------------------------------------------------
# S8-58 — load_external declares built-in kinds BEFORE importing a user file
# ----------------------------------------------------------------------
def test_load_external_declares_builtin_kinds_before_user_file(tmp_path, clean_env):
    """S8-58: a user file that ``@register``s against a BUILT-IN kind loads cleanly -- load_external
    declares the built-in kinds first, so the file never hits an empty KINDS (the registration-timing
    pitfall). The registered reward is reachable afterwards."""
    from swift.dev.rewards.orm import REWARD
    from swift.dev.utils.plugin import PluginRegistry

    plugin_file = tmp_path / 'my_s8_58_plugin.py'
    plugin_file.write_text(
        'from swift.dev.utils.plugin import PluginRegistry\n'
        'from swift.dev.rewards.reward import ORM\n'
        "@PluginRegistry.register('reward', 's8_58_probe')\n"
        'class S858Reward(ORM):\n'
        '    def __call__(self, trajectories, **kw):\n'
        '        return [0.0] * len(trajectories)\n')

    loaded = PluginRegistry.load_external([str(plugin_file)])
    assert loaded == [str(plugin_file)]
    entries = PluginRegistry.kind(REWARD).entries
    assert 's8_58_probe' in entries  # the user file's @register('reward', ...) ran
    # the worker-publish channel records the resolved absolute path + the reload hook
    assert os.environ.get('SWIFT_EXTERNAL_PLUGINS', '').find('my_s8_58_plugin.py') >= 0
    entries.pop('s8_58_probe', None)


# ----------------------------------------------------------------------
# S8-60 — external_plugins three source forms (file / folder / hub id)
# ----------------------------------------------------------------------
def test_load_external_three_source_forms(tmp_path, monkeypatch, clean_env):
    """S8-60: a local file, a local folder, and a hub id all route through twinkle's single loader;
    hub ids (hf:// / ms://) pass through untouched (mocked here -- no network)."""
    from swift.dev.utils import plugin as plugin_mod

    # folder form: a package directory (twinkle's _load_local imports a folder holding __init__.py)
    folder = tmp_path / 'plugin_dir'
    folder.mkdir()
    (folder / '__init__.py').write_text('VALUE = 1\n')
    assert plugin_mod.PluginRegistry.load_external([str(folder)]) == [str(folder)]

    # hub ids: mock twinkle's loader and assert the ids are forwarded verbatim (no path resolution)
    calls = []
    import twinkle.utils
    monkeypatch.setattr(twinkle.utils, 'load_module', lambda src: calls.append(src))
    plugin_mod.PluginRegistry.load_external(['hf://user/repo', 'ms://user/repo'])
    assert calls == ['hf://user/repo', 'ms://user/repo']
    # hub ids are published to workers untouched (not resolved to a local path)
    published = os.environ.get('SWIFT_EXTERNAL_PLUGINS', '')
    assert 'hf://user/repo' in published and 'ms://user/repo' in published
