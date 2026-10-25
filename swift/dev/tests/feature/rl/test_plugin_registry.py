"""S8c — the plugin registry: kind declaration, singletons, external sources (L1).

Two builtin kinds exist now -- ``reward`` (base ORM, 9 entries) and ``prm_scorer`` (base PRMScorer,
2 entries) -- declared when the rewards package imports. The registry checks shape at REGISTRATION
time (a reward not subclassing ORM is rejected there, not halfway through a rollout), adopts an
existing entries dict by reference, and loads external plugin sources from a file, a folder, or a
hub id. The fixtures snapshot and restore global state; builtin kinds are never cleared (re-declaring
them is a no-op once cached, so clearing would brick the process).

Maps to ut_plan S8-58/59/60.
"""
import os

import pytest

from swift.dev.utils.plugin import PluginRegistry

_PLUGIN_SOURCE = (
    "from swift.dev.utils.plugin import PluginRegistry\n"
    "from swift.dev.rewards.reward import ORM\n"
    "@PluginRegistry.register('reward', '{name}')\n"
    "class {cls}(ORM):\n"
    "    def __call__(self, completions, **kw): return [0.0 for _ in completions]\n"
)


@pytest.fixture
def clean_env(tmp_path):
    """Snapshot KINDS/entries; restore after. Builtin kinds are never removed -- only kinds and
    entries THIS test creates are torn down."""
    import swift.dev.rewards.orm  # noqa: F401 -- declares the builtin kinds
    import swift.dev.rewards.prm  # noqa: F401

    kinds_snapshot = dict(PluginRegistry.KINDS)
    entries_snapshot = {name: dict(kind.entries) for name, kind in kinds_snapshot.items()}
    yield tmp_path
    PluginRegistry.KINDS.clear()
    PluginRegistry.KINDS.update(kinds_snapshot)
    for name, kind in PluginRegistry.KINDS.items():
        if name in entries_snapshot:
            kind.entries.clear()
            kind.entries.update(entries_snapshot[name])


def _write_plugin(path, name, cls):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(_PLUGIN_SOURCE.format(name=name, cls=cls))


class TestBuiltinKinds:
    """S8-58: the builtin kinds are declared (lazily, on the rewards package import) BEFORE any
    external plugin can register into them -- an external source hitting an undeclared kind crashes."""

    def test_builtin_kinds_declared_with_expected_shape(self):
        import swift.dev.rewards.orm  # noqa: F401
        import swift.dev.rewards.prm  # noqa: F401

        assert set(PluginRegistry.KINDS) >= {'reward', 'prm_scorer'}
        reward = PluginRegistry.kind('reward')
        from swift.dev.rewards.reward import ORM

        assert reward.base is ORM  # the base flipped to ORM (SwiftPlugin is long gone)
        assert 'accuracy' in reward.entries and 'format' in reward.entries
        assert set(PluginRegistry.kind('prm_scorer').entries) == {'delimiter', 'turn'}


class TestRegistrySingleton:
    """S8-59: kinds are singletons -- a second register_kind of the same name raises unless
    exist_ok; an entries dict passed at declaration is adopted BY REFERENCE (the module that owns the
    dict keeps writing into the live registry)."""

    def test_duplicate_kind_rejected(self, clean_env):
        with pytest.raises(ValueError, match='already registered'):
            PluginRegistry.register_kind('reward', dict)

    def test_duplicate_kind_exist_ok_redeclares(self, clean_env):
        before = PluginRegistry.kind('reward')
        after = PluginRegistry.register_kind('reward', before.base, exist_ok=True)
        assert after.name == 'reward'

    def test_duplicate_entry_name_rejected(self, clean_env):
        from swift.dev.rewards.reward import ORM

        with pytest.raises(ValueError, match='accuracy.*is already registered by MathAccuracy'):
            @PluginRegistry.register('reward', 'accuracy')
            class Dup(ORM):
                def __call__(self, completions, **kw):
                    return [0.0 for _ in completions]

    def test_registration_checks_base_at_registration_time(self, clean_env):
        """A class that does not subclass the kind's base is rejected AT REGISTRATION -- never
        halfway through a rollout when the plugin is first instantiated."""
        with pytest.raises(TypeError, match='must subclass ORM'):
            @PluginRegistry.register('reward', 'bad_base_rule')
            class NotAPlugin:
                pass

    def test_entries_dict_adopted_by_reference(self, clean_env):
        entries = {}
        PluginRegistry.register_kind('s8_tmp_kind', object, entries=entries)
        try:
            entries['late'] = int  # the owning module writes into its own dict...
            assert 'late' in PluginRegistry.kind('s8_tmp_kind').entries  # ...and the live registry sees it
        finally:
            del PluginRegistry.KINDS['s8_tmp_kind']


class TestResolve:
    """resolve: a registered name -> a no-arg instance; a class -> an instance; a callable passes
    through; anything else is rejected."""

    def test_resolve_name_to_instance(self, clean_env):
        assert type(PluginRegistry.resolve('reward', 'accuracy')).__name__ == 'MathAccuracy'

    def test_resolve_class_instantiates_no_arg(self, clean_env):
        from swift.dev.rewards.orm import MathAccuracy

        assert isinstance(PluginRegistry.resolve('reward', MathAccuracy), MathAccuracy)

    def test_resolve_callable_passes_through(self, clean_env):
        fn = lambda completions, **kw: [0.0]  # noqa: E731
        assert PluginRegistry.resolve('reward', fn) is fn

    def test_resolve_unknown_name_rejected(self, clean_env):
        with pytest.raises(ValueError, match='not registered'):
            PluginRegistry.resolve('reward', 'no_such_rule')

    def test_resolve_plain_value_rejected(self, clean_env):
        with pytest.raises(ValueError, match='registered name, a class, or a callable'):
            PluginRegistry.resolve('reward', 42)


class TestLoadExternal:
    """S8-60: external plugin sources load from a single file, a folder package (with __init__.py),
    or a hub id; the @register calls inside them run at import."""

    def test_single_file(self, clean_env):
        path = clean_env / 'my_ext_plugin.py'
        _write_plugin(path, 's8_file_rule', 'S8FileRule')
        assert PluginRegistry.load_external(str(path)) == [str(path)]
        assert type(PluginRegistry.resolve('reward', 's8_file_rule')).__name__ == 'S8FileRule'

    def test_folder_package(self, clean_env):
        folder = clean_env / 'pkg'
        folder.mkdir()
        with open(folder / '__init__.py', 'w', encoding='utf-8') as f:
            f.write('from . import rule\n')
        _write_plugin(folder / 'rule.py', 's8_folder_rule', 'S8FolderRule')
        assert PluginRegistry.load_external(str(folder)) == [str(folder)]
        assert type(PluginRegistry.resolve('reward', 's8_folder_rule')).__name__ == 'S8FolderRule'

    def test_none_loads_nothing(self, clean_env):
        assert PluginRegistry.load_external(None) == []
