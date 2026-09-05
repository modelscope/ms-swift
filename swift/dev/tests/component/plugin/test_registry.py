# Copyright (c) ModelScope Contributors. All rights reserved.
"""The plugin mechanism itself: registration, resolution, and loading a user's own ``.py``.

These are the behaviors an extension author depends on, so they are asserted rather than described:
a name resolves, a wrong shape is refused at registration, the legacy ``orms[...] = cls`` idiom and
the decorator write to the same registry, and two plugin files never shadow each other.
"""
import textwrap

import pytest

from swift.dev.plugin import AsyncRewardPlugin, PluginKind, PluginRegistry, RewardPlugin, SwiftPlugin


@pytest.fixture
def kind():
    """A throwaway kind, so a test never registers into (or pollutes) a real extension point."""
    name = 'test_kind'
    PluginRegistry.KINDS.pop(name, None)
    yield PluginRegistry.register_kind(name, RewardPlugin, config_field='test_funcs')
    PluginRegistry.KINDS.pop(name, None)


def test_the_legacy_dict_is_the_registry():
    """``orms`` must be the reward kind's own ``entries``, not a copy of it.

    The whole point of adopting an existing dict: a plugin file written the legacy way
    (``orms['my'] = MyORM``) and one written with the decorator end up in the same place, so a
    consumer never has to ask which mechanism a reward came from. A copy would silently split them --
    exactly the legacy/dev split that made user rewards unresolvable on dev.
    """
    from swift.dev.rewards import REWARD, orms

    assert REWARD.entries is orms
    assert set(orms) >= {'accuracy', 'format', 'cosine', 'repetition', 'soft_overlong'}


def test_orm_is_the_reward_plugin_base():
    """``ORM`` is an alias, not a parallel class: a legacy plugin file subclassing ORM registers."""
    from swift.dev.rewards import ORM, AsyncORM

    assert ORM is RewardPlugin
    assert AsyncORM is AsyncRewardPlugin


def test_register_then_get(kind):

    @PluginRegistry.register(kind, 'mine')
    class Mine(RewardPlugin):

        def __call__(self, completions, **kwargs):
            return [1.0] * len(completions)

    assert PluginRegistry.get(kind, 'mine') is Mine
    assert kind.entries['mine'] is Mine


def test_register_reads_the_class_attribute_name(kind):

    @PluginRegistry.register(kind)
    class Mine(RewardPlugin):
        name = 'from_attr'

    assert PluginRegistry.get(kind, 'from_attr') is Mine


def test_register_refuses_the_wrong_shape(kind):
    """A shape error surfaces at registration, not halfway through a rollout."""
    with pytest.raises(TypeError, match='must subclass RewardPlugin'):

        @PluginRegistry.register(kind, 'not_a_plugin')
        class NotAPlugin:
            pass


def test_register_refuses_a_silent_overwrite(kind):

    @PluginRegistry.register(kind, 'dup')
    class First(RewardPlugin):
        pass

    with pytest.raises(ValueError, match='already registered'):

        @PluginRegistry.register(kind, 'dup')
        class Second(RewardPlugin):
            pass

    assert PluginRegistry.get(kind, 'dup') is First


def test_a_kind_may_accept_more_than_one_contract():
    """sync and async rewards are one extension point, so ``base`` takes a tuple.

    ``AsyncRewardPlugin`` is deliberately NOT a subclass of ``RewardPlugin`` (an awaited ``__call__``
    is a different contract, not a refinement of one), so a single base would have rejected it.
    """
    from swift.dev.rewards import REWARD

    assert isinstance(REWARD.base, tuple)
    assert set(REWARD.base) == {RewardPlugin, AsyncRewardPlugin}
    assert 'RewardPlugin' in REWARD.base_names and 'AsyncRewardPlugin' in REWARD.base_names


def test_unknown_name_lists_what_is_available(kind):
    """The error must be actionable: what exists, and how to add your own."""
    with pytest.raises(ValueError) as excinfo:
        PluginRegistry.get(kind, 'nope')
    message = str(excinfo.value)
    assert 'not registered' in message and '--external_plugins' in message


def test_resolve_accepts_a_name_a_class_an_instance_and_a_callable(kind):
    """The four things a user can put in a Config field, and what each becomes."""

    @PluginRegistry.register(kind, 'named')
    class Named(RewardPlugin):

        def __call__(self, completions, **kwargs):
            return [0.5] * len(completions)

    config = object()
    from_name = PluginRegistry.resolve(kind, 'named', config=config)
    assert isinstance(from_name, Named) and from_name.args is config  # built with the run's Config

    assert isinstance(PluginRegistry.resolve(kind, Named, config=config), Named)

    instance = Named()
    assert PluginRegistry.resolve(kind, instance) is instance  # already built: passed through

    def plain(completions, **kwargs):
        return [0.0] * len(completions)

    assert PluginRegistry.resolve(kind, plain) is plain  # a bare function needs no base class


def test_display_name_labels_both_classes_and_functions(kind):

    @PluginRegistry.register(kind, 'labelled')
    class Labelled(RewardPlugin):
        pass

    assert PluginRegistry.display_name(Labelled()) == 'Labelled'

    def scorer(completions, **kwargs):
        return []

    assert PluginRegistry.display_name(scorer) == 'scorer'


_PLUGIN_FILE = textwrap.dedent("""
    from swift.dev.plugin import PluginRegistry, RewardPlugin

    @PluginRegistry.register('test_kind', {name!r})
    class {cls}(RewardPlugin):
        def __call__(self, completions, **kwargs):
            return [{score}] * len(completions)
""")


def test_two_plugin_files_with_the_same_stem_do_not_shadow_each_other(kind, tmp_path):
    """Each file gets a module name derived from its path.

    twinkle's loader imports every plugin as ``__init__``, so the second file hits ``sys.modules``
    and hands back the FIRST one's classes -- which reads as "my plugin was ignored". Two files named
    ``plugin.py`` in different directories is the ordinary case (one per experiment), so it is the
    case worth pinning.
    """
    files = []
    for i, (name, score) in enumerate((('first', 1.0), ('second', 2.0))):
        directory = tmp_path / f'exp{i}'
        directory.mkdir()
        path = directory / 'plugin.py'
        path.write_text(_PLUGIN_FILE.format(name=name, cls=f'Reward{i}', score=score))
        files.append(str(path))

    PluginRegistry.load_external(files)
    assert set(kind.entries) == {'first', 'second'}
    assert PluginRegistry.resolve(kind, 'first')(completions=['a']) == [1.0]
    assert PluginRegistry.resolve(kind, 'second')(completions=['a']) == [2.0]


def test_loading_the_same_file_twice_is_a_no_op(kind, tmp_path):
    """Idempotent by path: a re-import would re-run ``@register`` and raise 'already registered'."""
    path = tmp_path / 'plugin.py'
    path.write_text(_PLUGIN_FILE.format(name='once', cls='Once', score=1.0))

    assert PluginRegistry.load_external([str(path)]) == [str(path)]
    assert PluginRegistry.load_external([str(path)]) == []  # second call loads nothing
    assert set(kind.entries) == {'once'}


def test_a_missing_plugin_file_fails_immediately(tmp_path):
    """A typo'd --external_plugins path must not be silently skipped."""
    with pytest.raises(FileNotFoundError, match='is not a file'):
        PluginRegistry.load_external([str(tmp_path / 'nope.py')])


def test_a_broken_plugin_file_does_not_leave_a_half_imported_module(kind, tmp_path):
    """The exception propagates AND the module is unregistered, so a fixed file can be re-loaded."""
    path = tmp_path / 'plugin.py'
    path.write_text('raise RuntimeError("boom")\n')
    with pytest.raises(RuntimeError, match='boom'):
        PluginRegistry.load_external([str(path)])

    path.write_text(_PLUGIN_FILE.format(name='fixed', cls='Fixed', score=1.0))
    PluginRegistry.load_external([str(path)])
    assert set(kind.entries) == {'fixed'}


def test_load_configured_reads_both_config_fields(kind, tmp_path):
    """external_plugins and custom_register_path are loaded together, as legacy does."""
    from swift.dev.config import ModelConfig

    first = tmp_path / 'a.py'
    first.write_text(_PLUGIN_FILE.format(name='a', cls='A', score=1.0))
    second = tmp_path / 'b.py'
    second.write_text(_PLUGIN_FILE.format(name='b', cls='B', score=1.0))

    model_config = ModelConfig(model='/m', external_plugins=[str(first)], custom_register_path=[str(second)])
    PluginRegistry.load_configured(model_config)
    assert set(kind.entries) == {'a', 'b'}


def test_a_third_party_can_declare_a_whole_new_kind():
    """The mechanism is itself extensible: a new extension point needs no edit to swift."""
    PluginRegistry.KINDS.pop('third_party', None)
    try:

        class ThirdPartyBase(SwiftPlugin):
            pass

        new_kind = PluginRegistry.register_kind('third_party', ThirdPartyBase, config_field='third_party_impl')
        assert isinstance(new_kind, PluginKind)
        assert PluginRegistry.kind('third_party') is new_kind  # reachable by name from anywhere

        @PluginRegistry.register('third_party', 'mine')
        class Mine(ThirdPartyBase):
            pass

        assert PluginRegistry.get(new_kind, 'mine') is Mine
    finally:
        PluginRegistry.KINDS.pop('third_party', None)


def test_declaring_the_same_kind_twice_is_refused():
    """Two owners for one extension point is a mistake, not a merge."""
    PluginRegistry.KINDS.pop('twice', None)
    try:
        PluginRegistry.register_kind('twice', RewardPlugin)
        with pytest.raises(ValueError, match='already registered'):
            PluginRegistry.register_kind('twice', RewardPlugin)
    finally:
        PluginRegistry.KINDS.pop('twice', None)
