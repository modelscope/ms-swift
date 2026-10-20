# Copyright (c) ModelScope Contributors. All rights reserved.
"""Skip the dependency pre-flight check for ``trust_remote_code`` models.

``transformers.dynamic_module_utils.check_imports`` statically scans a downloaded
``modeling_*.py`` for its top-level imports, runs ``importlib.import_module`` on
each one, and raises ``ImportError: This modeling file requires the following
packages that were not found in your environment: ...`` if any is missing. The
scan is purely lexical -- it does not care whether the code path actually taken at
runtime needs the package. Remote-code models that unconditionally
``import flash_attn`` at module top level while still shipping an eager-attention
fallback are therefore rejected before a single weight is read.

This patch swaps the function for just its return value,
``get_relative_imports(filename)`` -- the list the dynamic module loader consumes
to recursively pull in sibling files -- dropping only the verification loop. The
one call site (``get_cached_module_file``) resolves ``check_imports`` through the
module namespace, so rebinding the attribute is enough.

The trade-off is deliberate: a genuinely missing *required* dependency stops being
a clean ImportError at load time and becomes one raised from inside forward, with
a much worse traceback. Apply it per-model around the actual load, never globally.

Usage:
    with apply_context(None, TransformersIgnoreCheckImportsPatch()):
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
"""
from typing import List

from twinkle.patch import Patch

_MARKER = '_swift_ignore_check_imports'


class TransformersIgnoreCheckImportsPatch(Patch):
    """Neuter ``dynamic_module_utils.check_imports`` for the duration of a model load."""

    def __init__(self):
        self._origin = None

    def __call__(self, module=None, *args, **kwargs):
        import transformers.dynamic_module_utils as td

        # Nested apply, or another instance owns the active replacement: leave it in
        # place so whoever installed it stays responsible for restoring the real one.
        if getattr(td.check_imports, _MARKER, False):
            return module

        def check_imports(filename) -> List[str]:
            from transformers.dynamic_module_utils import get_relative_imports
            return get_relative_imports(filename)

        setattr(check_imports, _MARKER, True)
        self._origin = td.check_imports
        td.check_imports = check_imports
        return module

    def unpatch(self, module=None, *args, **kwargs):
        if self._origin is None:
            return module
        import transformers.dynamic_module_utils as td

        # Only take back what we installed; a later patcher owns anything else.
        if getattr(td.check_imports, _MARKER, False):
            td.check_imports = self._origin
        self._origin = None
        return module
