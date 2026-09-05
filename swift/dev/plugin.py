# Copyright (c) ModelScope Contributors. All rights reserved.
"""swift's own plugin mechanism: the product's extension points, with swift's own base classes.

An extension point is a *product* concern, so swift declares it -- twinkle's roster is not a
substitute, and the two are not the same thing:

  - The signatures differ because the semantics differ. A swift reward plugin scores strings against
    dataset columns (``(completions, **columns) -> List[float]``, which is also what every legacy
    ``ORM`` and every user plugin file already implements); twinkle's ``Reward`` scores ``Trajectory``
    objects. twinkle's own base is not consumed anywhere inside twinkle -- it exists for its cookbook.
  - twinkle's string-to-class path (``construct_class`` -> ``Plugin.load_plugin``) only accepts
    ``hf://`` / ``ms://`` ids and demands trust_remote_code, so "a name plus a local ``.py``" -- the
    only interface a CLI can offer -- cannot be expressed there at all.

So: swift owns the base classes, the registry and the loading; twinkle is handed constructed objects.

Two levels, so the mechanism itself is extensible -- a third party adds a whole new *kind* of plugin
without editing swift:

    kind  = PluginRegistry.register_kind('reward', RewardPlugin, config_field='reward_funcs')
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
import hashlib
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Tuple, Type, Union

from swift.dev.utils import get_logger

logger = get_logger()

__all__ = ['SwiftPlugin', 'RewardPlugin', 'AsyncRewardPlugin', 'PluginKind', 'PluginRegistry']


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

    Example::

        class MyReward(RewardPlugin):
            def __call__(self, completions, **kwargs) -> List[float]:
                return [1.0 if len(c) > 100 else 0.0 for c in completions]
    """

    def __call__(self, **kwargs) -> List[float]:
        raise NotImplementedError


class AsyncRewardPlugin(SwiftPlugin):
    """A reward whose scoring does I/O (an API call, a database query).

    Async rewards are gathered with ``asyncio.gather``, so the round trips of one batch overlap
    instead of running one after another.
    """

    async def __call__(self, **kwargs) -> List[float]:
        raise NotImplementedError


@dataclass(frozen=True)
class PluginKind:
    """One extension point: what it is called, what it must subclass, and what selects it."""

    name: str
    #: Every implementation must subclass this. Usually a class from this module; a tuple when one point
    #: accepts more than one contract (a reward may be sync or async). It may be a twinkle base when the
    #: product's plugin *is* a kernel part (loss), which avoids forking twinkle's roster.
    base: Union[type, Tuple[type, ...]]
    #: The Config field a run picks an implementation with, e.g. ``reward_funcs``. Recorded so the
    #: "no plugin field is silently ignored" test can pair kinds with Config fields generically.
    config_field: Optional[str] = None
    #: name -> implementation. Shared by reference when a kind adopts a pre-existing registry.
    entries: Dict[str, type] = field(default_factory=dict)

    @property
    def base_names(self) -> str:
        """The accepted base classes, for error messages."""
        bases = self.base if isinstance(self.base, tuple) else (self.base, )
        return ' / '.join(base.__name__ for base in bases)


class PluginRegistry:
    """The registry of kinds, and of the implementations of each kind."""

    KINDS: ClassVar[Dict[str, PluginKind]] = {}
    #: abspath -> module, so loading the same file twice is a no-op rather than a re-registration error.
    LOADED: ClassVar[Dict[str, ModuleType]] = {}

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
    def load_configured(model_config: Any) -> List[str]:
        """Load every plugin file a run's ``ModelConfig`` names -- the only place that knows which fields
        those are, so a recipe cannot load half of them.

        ``custom_register_path`` is loaded alongside ``external_plugins`` because legacy concatenates the
        two before importing (base_args.py): both mean "import this .py first", one named for the
        hooks (rewards, losses) and one for registrations (models, datasets), and nothing tells them
        apart at load time.
        """
        return PluginRegistry.load_external([*model_config.external_plugins, *model_config.custom_register_path])

    @staticmethod
    def load_external(paths: Union[str, Iterable[str], None]) -> List[str]:
        """Import user ``.py`` files so the ``@register`` calls inside them run.

        Idempotent, and every file gets a module name derived from its path. twinkle's loader instead
        imports every plugin as ``__init__``, so a second plugin hits ``sys.modules`` and silently
        hands back the first one's classes -- a name collision that reads as "my plugin was ignored".
        The file's directory joins ``sys.path`` so a plugin may import its own neighbours.
        """
        loaded: List[str] = []
        for raw in ([paths] if isinstance(paths, str) else list(paths or [])):
            path = Path(raw).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f'external plugin {raw!r} is not a file (resolved to {path}).')
            key = str(path)
            if key in PluginRegistry.LOADED:
                continue
            parent = str(path.parent)
            if parent not in sys.path:
                sys.path.insert(0, parent)
            module_name = f'swift_dev_plugin_{hashlib.sha1(key.encode()).hexdigest()[:8]}_{path.stem}'
            spec = importlib.util.spec_from_file_location(module_name, key)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(module_name, None)
                raise
            PluginRegistry.LOADED[key] = module
            loaded.append(key)
        if loaded:
            logger.info(f'Loaded {len(loaded)} external plugin file(s): {loaded}')
        return loaded
