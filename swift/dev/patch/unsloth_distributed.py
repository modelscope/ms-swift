"""Make ``unsloth_zoo.distributed_function`` run inline.

unsloth_zoo funnels one-off work (writing a compiled kernel to the cache dir) through
``distributed_function``, which elects rank 0, runs the work there and ``broadcast_object_list`` +
``barrier`` the result -- it assumes unsloth owns the process group and that every rank reaches the
call together. Recent versions short-circuit to a local call when the process group is *not*
initialized (unslothai/unsloth#3703), but under twinkle/dev the group belongs to the strategy and is
already up while these call sites are not collective, so the election still hangs. Letting every rank
do the work itself is both correct and cheap here, since the work is local.

As of unsloth_zoo 2026.8.12 the call sites are all in ``unsloth_zoo/compiler.py`` (compiled-source
writes plus the cache-location decision); ``utils`` is patched too because it owns the definition.

Ported from legacy swift's ``load_by_unsloth`` (``swift/model/register.py``), where the same swap is
an inline contextmanager. As a ``Patch`` it is idempotent, reversible, and no-op when unsloth_zoo is
absent, and it can be scoped with twinkle's ``apply_context``::

    from twinkle.patch import apply_context
    from swift.dev.patch import UnslothDistributedFunctionPatch

    with apply_context(None, UnslothDistributedFunctionPatch()):
        model, tokenizer = FastLanguageModel.from_pretrained(...)
"""
from __future__ import annotations

import importlib.util
import os

from twinkle.patch import Patch
from twinkle.utils import get_logger

logger = get_logger()

_MARKER = '_swift_origin_distributed_function'


def _inline_distributed_function(n=1, function=None, *args, **kwargs):
    """Drop the rank election: run ``function`` on the calling rank."""
    return function(*args, **kwargs)


def _unsloth_zoo_modules():
    """The unsloth_zoo modules that hold their own ``distributed_function`` reference.

    Both import it by value (``from .utils import distributed_function``), so patching one does not
    affect the other -- both have to be swapped. Returns an empty tuple when unsloth_zoo is not
    installed, or when a future version no longer exposes the symbol.

    ``unsloth_zoo/__init__.py`` refuses to import unless ``UNSLOTH_IS_PRESENT`` is in the env (it is
    meant to be reached through ``import unsloth``). Without setting it here the ImportError below
    would swallow that refusal and the patch would silently do nothing whenever it is applied before
    the first unsloth import. Gated on unsloth actually being installed so the flag is not faked.
    """
    if importlib.util.find_spec('unsloth') is not None:
        os.environ.setdefault('UNSLOTH_IS_PRESENT', '1')
    try:
        from unsloth_zoo import compiler, utils
    except ImportError:
        return ()
    return tuple(m for m in (utils, compiler) if hasattr(m, 'distributed_function'))


class UnslothDistributedFunctionPatch(Patch):
    """Run unsloth_zoo's ``distributed_function`` on every rank instead of electing one.
    Idempotent, reversible, no-op without unsloth_zoo."""

    def __call__(self, module=None, *args, **kwargs):
        modules = _unsloth_zoo_modules()
        if not modules:
            return module
        patched = []
        for mod in modules:
            if hasattr(mod, _MARKER):
                continue
            setattr(mod, _MARKER, mod.distributed_function)
            mod.distributed_function = _inline_distributed_function
            patched.append(mod.__name__)
        if patched:
            logger.info(f'Patched {patched} distributed_function to run inline for unsloth.')
        return module

    def unpatch(self, module=None, *args, **kwargs):
        for mod in _unsloth_zoo_modules():
            origin = getattr(mod, _MARKER, None)
            if origin is not None:
                mod.distributed_function = origin
                delattr(mod, _MARKER)
        return module
