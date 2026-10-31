# Copyright (c) ModelScope Contributors. All rights reserved.
"""The thin adaptation protocol between canvas declarations and real objects.

A component here is a binding, never a wrapper: it holds the resident native
object (a twinkle model, a DataLoader, a StreamingDriver) and exposes the
canvas-facing call surface (build once / advance / invoke / run task). The
native object never learns about the canvas -- no inheritance, no port
concepts, no GraphSession in twinkle or swift.dev proper.

Lifecycle classes mirror the descriptor's ``lifecycle`` and describe only
construction and call counts:

- ``ResourceComponent``: built once, held resident, never rebuilt on re-click.
- ``IteratorComponent``: the native iterator is created once (``iter`` is
  called exactly once -- StreamingDriver is single-use) and later calls only
  advance it; stopping a graph call never exits the iterator.
- ``StageComponent``: invoked per batch with named inputs; heavy payloads stay
  infra references passed as top-level arguments.
- ``TaskComponent``: invoked once per request (merge / infer / deploy...).

Plugins and code components register through the same ``ComponentRegistry``
and get the same validation and scheduling rules as built-ins; external plugin
sources load via ``twinkle.utils.load_module`` (no new plugin downloader).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Type

from .descriptor import ComponentDescriptor
from .registry import REGISTRY


class ComponentError(RuntimeError):
    """A component-level failure the session reports as the node's error."""


def slot_values(entries: Any) -> List[Any]:
    """The values of a many-port input, in slot declaration order.

    Many-port inputs arrive from the loop as ``[(slot, value), ...]``: the
    slot is the consuming parameter name (a loss kwarg, a named lane). The
    slot names stay with the consumers that need them; everyone else reads
    positional values through this helper."""
    if entries is None:
        return []
    return [value for _slot, value in entries]


class ResourceComponent:
    """Base for built-once resources: model, sampler, template, trainer..."""

    #: The descriptor type this component binds; set by subclasses.
    type: str = ''

    def __init__(self, node_id: str, params: Dict[str, str]):
        self.node_id = node_id
        self.params = dict(params)
        self.obj: Any = None  #: the resident native object

    def build(self, inputs: Dict[str, Any]) -> Any:
        """Construct the native object once. Inputs carry upstream residents."""
        raise NotImplementedError

    def release(self) -> None:
        """Close the resident resource; idempotent and best-effort."""
        obj = getattr(self, 'obj', None)
        if obj is None:
            return
        close = getattr(obj, 'close', None)  # samplers, envs, endpoints
        if callable(close):
            try:
                close()
            except Exception:  # release must never mask the primary error
                pass
        self.obj = None


class IteratorComponent(ResourceComponent):
    """Base for on-demand batch sources: the iterator starts once, calls advance.

    The source keeps the CURRENT delivered batch and its acknowledgement state:
    a debug re-click reuses this batch; only a fully consumed batch is
    acknowledged exactly once. See the plan's batch-handover contract.
    """

    def start(self, inputs: Dict[str, Any]) -> None:
        """Create the native iterator (called at most once per source)."""
        raise NotImplementedError

    def take_batch(self) -> Any:
        """Advance the iterator: the next batch, as a native reference."""
        raise NotImplementedError

    def commit(self, steps: int) -> None:
        """Acknowledge full consumption of the current batch (exactly once)."""
        raise NotImplementedError

    def skip_batch(self) -> None:
        """Explicitly drop the current batch: commit(0), cursor advances."""
        self.commit(0)


class StageComponent:
    """Base for per-batch invocations: encode, score, advantage, train_step..."""

    type: str = ''

    def __init__(self, node_id: str, params: Dict[str, str]):
        self.node_id = node_id
        self.params = dict(params)

    def invoke(self, inputs: Dict[str, Any], batch: Any) -> Dict[str, Any]:
        """Run the stage on one batch; inputs are named top-level arguments.

        Returns the named outputs. Heavy payloads must stay references: this
        method receives them and passes them on without materialising.
        """
        raise NotImplementedError


class TaskComponent:
    """Base for one-shot actions over training products: merge, deploy..."""

    type: str = ''

    def __init__(self, node_id: str, params: Dict[str, str]):
        self.node_id = node_id
        self.params = dict(params)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


#: Component class per descriptor type. Filled by the components package and by
#: plugin registration; the session looks bindings up here first and falls back
#: to the descriptor's factory import path.
COMPONENT_CLASSES: Dict[str, Type] = {}


def register_component(cls: Type, descriptor: Optional[ComponentDescriptor] = None) -> None:
    """Register a component binding class (built-in or plugin) under its type.

    The descriptor must already exist in the registry (plugins extend a
    declared generic type or add their own descriptor first); the class must
    carry ``type`` matching it.
    """
    type_key = getattr(cls, 'type', '')
    if not type_key:
        raise ComponentError(f'{cls.__name__}: component classes must set .type')
    descriptor = descriptor or REGISTRY.get(type_key)
    if descriptor is None:
        raise ComponentError(f'{cls.__name__}: no descriptor for type {type_key!r}')
    COMPONENT_CLASSES[type_key] = cls


def load_code_component(code: str, entry: str = 'fn') -> Callable:
    """Load the canvas code blob's entry callable.

    The blob IS the payload (the canvas wrote it); this mirrors what a plugin
    module provides. The entry name defaults to ``fn``; when absent, the LAST
    callable defined in the blob is the entry -- the same contract the previous
    single-run path used.
    """
    namespace: Dict[str, Any] = {}
    exec(code, namespace)  # noqa: S102
    fn = namespace.get(entry)
    if fn is None:
        fn = next((v for v in reversed(list(namespace.values())) if callable(v)), None)
    if fn is None:
        raise ComponentError('the code defines no callable (name it fn, or make the last def the entry)')
    return fn


def load_plugin_source(source: str) -> None:
    """Load one external plugin source; its register_component calls take effect.

    Reuses twinkle's module loading -- the webui adds no plugin downloader.
    """
    from twinkle.utils import load_module
    load_module(source)
