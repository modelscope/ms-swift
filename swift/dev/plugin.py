# Copyright (c) ModelScope Contributors. All rights reserved.
"""swift's own plugin mechanism: the product's extension points, with swift's own base classes.

An extension point is a *product* concern, so swift declares it -- twinkle's roster is not a
substitute, and the two are not the same thing:

  - The signatures differ because the semantics differ. A swift reward plugin scores strings against
    dataset columns (``(completions, **columns) -> List[float]``, which is also what every legacy
    ``ORM`` and every user plugin file already implements); twinkle's ``Reward`` scores ``Trajectory``
    objects. twinkle's own base is not consumed anywhere inside twinkle -- it exists for its cookbook.
  - twinkle's string-to-class path (``construct_class`` -> ``Plugin.load_plugin`` -> ``load_module``)
    resolves a name in a kernel namespace, a class, or a source to import. Its loader now accepts a
    local ``.py`` file, a local folder, or a ``hf://`` / ``ms://`` id under one unique-module-name
    scheme, so swift delegates loading to it instead of keeping a parallel loader of its own.

So: swift owns the base classes, the registry and the kind declarations; twinkle owns loading, and is
handed constructed objects.

Two levels, so the mechanism itself is extensible -- a third party adds a whole new *kind* of plugin
without editing swift:

    kind  = PluginRegistry.register_kind('reward', RewardPlugin, config_field='orm')
    @PluginRegistry.register('reward', 'my_reward')
    class MyReward(RewardPlugin): ...

A kind may adopt an *existing* dict as its ``entries``, which is how "one registry per kind" is kept
literal: ``orms`` stays the very dict it has always been, so the legacy idiom
``orms['my'] = MyReward`` and the decorator above write to the same place, and no caller has to learn
which one a plugin came from.

Deliberately NOT extension points: the optimizer (dev refuses ``--optimizer`` outright, see
``cli/sft.py``) and the tuner (a capability, not a hook).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Tuple, Type, Union

from swift.dev.utils import get_logger

logger = get_logger()

__all__ = ['SwiftPlugin', 'RewardPlugin', 'ToolPlugin', 'PluginKind', 'PluginRegistry']


class SwiftPlugin:
    """Base of every swift plugin.

    ``args`` is whatever Config the recipe is running with, so a plugin can read its own
    hyperparameters off it (``cosine_*`` / ``repetition_*`` and friends) instead of needing its own
    plumbing. The name is deliberately ``args`` rather than ``config``: every legacy plugin -- and
    every user plugin file written against legacy -- is constructed as ``cls(args=...)``.
    """

    #: Registry key. Optional: ``PluginRegistry.register`` also takes the name explicitly.
    name: ClassVar[Optional[str]] = None

    def __init__(self, args: Optional[Any] = None, **kwargs):
        self.args = args


class RewardPlugin(SwiftPlugin):
    """Score model completions against dataset columns.

    ``__call__`` may be written ``async def`` when its scoring does I/O (an API call, a database
    query): the scoring loop (:func:`swift.dev.reward.compute_rewards_per_func`) detects the returned
    coroutines and resolves them with one ``asyncio.gather``, so a batch's round trips overlap instead
    of running one after another. There is no separate async base to choose -- the same class covers
    both, and a run may mix sync and async rewards freely.

    Example::

        class MyReward(RewardPlugin):
            def __call__(self, completions, **kwargs) -> List[float]:
                return [1.0 if len(c) > 100 else 0.0 for c in completions]
    """

    def __call__(self, **kwargs) -> List[float]:
        raise NotImplementedError


class ToolPlugin(SwiftPlugin):
    """Build the tools a multi-turn rollout exposes to the model, bound to one sandbox env.

    A tool plugin is constructed once per run (``cls(args=RolloutConfig)``) and then asked for its tools
    per episode: :meth:`build` receives the leased twinkle ``Env`` and returns the twinkle ``Tool``
    instances (usually ``EnvTool``s) the agent may call that turn. Returning ``[]`` means the plugin
    contributes nothing for that env, so a run with no tool plugins -- the default -- rolls out with no
    tools at all.

    Example::

        @PluginRegistry.register('tool', 'my_tools')
        class MyTools(ToolPlugin):
            def build(self, env):
                return [MyTool(env)]
    """

    def build(self, env: Any) -> List[Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class PluginKind:
    """One extension point: what it is called, what it must subclass, and what selects it."""

    name: str
    #: Every implementation must subclass this. Usually a class from this module; a tuple when one point
    #: accepts more than one contract. It may be a twinkle base when the product's plugin *is* a kernel
    #: part (loss), which avoids forking twinkle's roster.
    base: Union[type, Tuple[type, ...]]
    #: The Config field a run picks an implementation with, e.g. ``orm``. Recorded so the
    #: "no plugin field is silently ignored" test can pair kinds with Config fields generically.
    config_field: Optional[str] = None
    #: name -> implementation. Shared by reference when a kind adopts a pre-existing registry.
    entries: Dict[str, type] = field(default_factory=dict)

    @property
    def base_names(self) -> str:
        """The accepted base classes, for error messages."""
        bases = self.base if isinstance(self.base, tuple) else (self.base, )
        return ' / '.join(base.__name__ for base in bases)


#: Modules whose import declares swift's built-in plugin kinds (each runs ``register_kind`` at module
#: top level). A user's ``--external_plugins`` file may ``@register`` against any built-in kind -- the
#: ``tool`` kind (``swift.dev.rollout.sandbox``), the ``reward`` kind (``swift.dev.rewards.orm``), the
#: ``prm_scorer`` kind (``swift.dev.rewards.prm``) or the ``callback`` kind (``swift.dev.callbacks``) -- so
#: the loader must guarantee those kinds are declared BEFORE it imports the user file. Nothing in the CLI's
#: config lifecycle imports them that early (``process_configs`` calls ``load_configured`` before any recipe
#: runs), so a plugin registering a built-in kind would otherwise hit an empty ``KINDS``.
_BUILTIN_KIND_MODULES = ('swift.dev.rewards.orm', 'swift.dev.rewards.prm', 'swift.dev.rollout.sandbox',
                         'swift.dev.callbacks')


def _declare_builtin_kinds() -> None:
    """Import the modules that declare swift's built-in plugin kinds. Idempotent (Python caches modules)
    and lazy (a function-level import) so it runs after this module is fully loaded, avoiding the cycle
    those modules create by importing :class:`PluginRegistry` at their own top level."""
    import importlib
    for name in _BUILTIN_KIND_MODULES:
        importlib.import_module(name)


class PluginRegistry:
    """The registry of kinds, and of the implementations of each kind."""

    KINDS: ClassVar[Dict[str, PluginKind]] = {}

    @staticmethod
    def register_kind(name: str,
                      base: Union[type, Tuple[type, ...]],
                      *,
                      config_field: Optional[str] = None,
                      entries: Optional[Dict[str, type]] = None,
                      exist_ok: bool = False) -> PluginKind:
        """Declare an extension point. Pass ``entries`` to adopt an existing registry dict as-is."""
        if not exist_ok and name in PluginRegistry.KINDS:
            raise ValueError(f'plugin kind `{name}` is already registered with base '
                             f'{PluginRegistry.KINDS[name].base_names}.')
        kind = PluginKind(name, base, config_field, entries if entries is not None else {})
        PluginRegistry.KINDS[name] = kind
        return kind

    @staticmethod
    def kind(kind: Union[str, PluginKind]) -> PluginKind:
        """A kind name (or a kind, passed straight through) -> the kind.

        Accepting the object means the module that declares a point can hand it around directly, so its
        name is written once, at ``register_kind``, and never spelled again at a call site.
        """
        if isinstance(kind, PluginKind):
            return kind
        if kind not in PluginRegistry.KINDS:
            raise ValueError(f'plugin kind `{kind}` is not registered. Available: {sorted(PluginRegistry.KINDS)}')
        return PluginRegistry.KINDS[kind]

    @staticmethod
    def register(kind: Union[str, PluginKind], name: Optional[str] = None, *, exist_ok: bool = False):
        """Register an implementation. Usable as ``@register('reward', 'my_reward')`` or bare.

        Unlike a plain dict assignment this checks the shape at *registration* time: a reward that
        does not subclass ``RewardPlugin`` is rejected here rather than halfway through a rollout.
        """

        def _register(cls: type) -> type:
            entry = name or cls.name
            assert entry, f'{cls.__name__} must set `name` or be registered with an explicit name.'
            registry = PluginRegistry.kind(kind)
            if not (isinstance(cls, type) and issubclass(cls, registry.base)):
                raise TypeError(f'{cls.__name__} must subclass {registry.base_names} '
                                f'to be registered as a `{registry.name}` plugin.')
            if not exist_ok and entry in registry.entries:
                raise ValueError(f'`{registry.name}` plugin `{entry}` is already registered '
                                 f'by {registry.entries[entry].__name__}.')
            registry.entries[entry] = cls
            return cls

        return _register

    @staticmethod
    def get(kind: Union[str, PluginKind], name: str) -> type:
        """The implementation class registered under ``name``."""
        registry = PluginRegistry.kind(kind)
        if name not in registry.entries:
            raise ValueError(f'`{registry.name}` plugin {name!r} is not registered '
                             f'(available: {sorted(registry.entries)}). Pass a registered name, or '
                             f'load your own with --external_plugins.')
        return registry.entries[name]

    @staticmethod
    def resolve(kind: Union[str, PluginKind], spec: Union[str, type, Any], *, config: Optional[Any] = None) -> Any:
        """A name / class / instance / plain callable -> something ready to be called.

        Kinds whose construction needs arguments of its own (``loss``) use :meth:`get` and instantiate
        themselves; this is for the plugins that are built from a Config and nothing else.
        """
        if isinstance(spec, str):
            spec = PluginRegistry.get(kind, spec)
        if isinstance(spec, type):
            return spec(args=config) if issubclass(spec, SwiftPlugin) else spec()
        if callable(spec):
            return spec
        raise ValueError(f'`{PluginRegistry.kind(kind).name}` plugin {spec!r} must be a registered name, '
                         f'a class, or a callable.')

    @staticmethod
    def display_name(plugin: Any) -> str:
        """How a resolved plugin is labelled in metrics and error messages."""
        return getattr(plugin, '__name__', None) or plugin.__class__.__name__

    @staticmethod
    def load_configured(plugin_config: Optional[Any]) -> List[str]:
        """Load every plugin source a run's ``PluginConfig`` names -- the one object that knows which
        field that is, so a recipe cannot load half of them.

        ``None`` loads nothing, for a caller that holds no PluginConfig (the plugin sources were then
        already imported earlier in the run's config lifecycle).
        """
        if plugin_config is None:
            return []
        return PluginRegistry.load_external(plugin_config.external_plugins)

    @staticmethod
    def load_external(paths: Union[str, Iterable[str], None]) -> List[str]:
        """Import user plugin sources so the ``@register`` calls inside them run.

        Each entry may be a local ``.py`` file, a local folder, or a ``hf://`` / ``ms://`` id; all three
        go through twinkle's single loader (:func:`twinkle.utils.load_module`), which imports under a
        unique path-derived module name and caches the result, so loading is idempotent and two plugins
        never collide on a shared name. swift keeps no parallel loader of its own.
        """
        raw_paths = [paths] if isinstance(paths, str) else list(paths or [])
        if not raw_paths:
            return []
        # Declare swift's own extension points first: a user plugin file may @register against any
        # built-in kind, which fails if that kind's declaring module has not been imported yet.
        _declare_builtin_kinds()
        from twinkle.utils import load_module
        for raw in raw_paths:
            load_module(raw)
        logger.info(f'Loaded {len(raw_paths)} external plugin source(s): {raw_paths}')
        return list(raw_paths)
