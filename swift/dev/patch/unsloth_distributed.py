# Copyright (c) ModelScope Contributors. All rights reserved.
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

    Both compiler and utils import it by value, so both have to be swapped. Returns an empty tuple
    when unsloth_zoo is not installed or no longer exposes the symbol. Sets UNSLOTH_IS_PRESENT first:
    unsloth_zoo/__init__ refuses to import without it, and the ImportError below would swallow that
    refusal, silently disabling the patch whenever it is applied before the first unsloth import.
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

    Ported from legacy swift's load_by_unsloth (an inline contextmanager); as a Patch it can be
    scoped with twinkle's apply_context.
    """

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
