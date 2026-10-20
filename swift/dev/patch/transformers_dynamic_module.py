# Copyright (c) ModelScope Contributors. All rights reserved.
"""Serialize the remote-code module cache write for ``trust_remote_code`` models.

Loading a remote-code model makes every rank call
``transformers.dynamic_module_utils.get_cached_module_file`` for the same repo: it downloads
``modeling_*.py`` into the hub cache, then copies it into
``~/.cache/huggingface/modules/transformers_modules/<repo>/<commit>/`` and recursively does the
same for the sibling files it imports.

That path is not safe against concurrent processes. ``dynamic_module_utils`` guards it with
``_HF_REMOTE_CODE_LOCK = threading.Lock()``, which only serializes threads inside one process, and
the copy itself is an ``if not target.exists(): shutil.copyfile(...)`` -- a TOCTOU window plus a
non-atomic write. Two ranks hitting it together can have one importing a file the other is still
writing, which surfaces as a truncated-source ``SyntaxError`` or a partially defined module,
usually only under cold cache and therefore rarely in testing.

This patch wraps the function in ``processing_lock`` so one rank populates the cache while the
others wait, then all of them read it. The key is per (repo, module file) rather than per repo,
because the loader calls this once for each remote file it pulls in, and each of those is a
separate write. ``sticky=True`` because the result is identified by that key and the work is
idempotent: a rank arriving after the file is in place proceeds instead of waiting for a new round.

Usage:
    with apply_context(None, TransformersDynamicModulePatch()):
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
"""
from twinkle.patch import Patch
from twinkle.utils.parallel import processing_lock

_MARKER = '_swift_locked_dynamic_module'


class TransformersDynamicModulePatch(Patch):
    """Put ``dynamic_module_utils.get_cached_module_file`` behind ``processing_lock``."""

    def __init__(self):
        self._origin = None

    def __call__(self, module=None, *args, **kwargs):
        import transformers.dynamic_module_utils as td

        # Nested apply, or another instance owns the active replacement: leave it in place so
        # whoever installed it stays responsible for restoring the real one.
        if getattr(td.get_cached_module_file, _MARKER, False):
            return module

        origin = td.get_cached_module_file

        def get_cached_module_file(pretrained_model_name_or_path, module_file, *args, **kwargs):
            key = f'dynamic_module:{pretrained_model_name_or_path}:{module_file}'
            with processing_lock(key, sticky=True):
                return origin(pretrained_model_name_or_path, module_file, *args, **kwargs)

        setattr(get_cached_module_file, _MARKER, True)
        self._origin = origin
        td.get_cached_module_file = get_cached_module_file
        return module

    def unpatch(self, module=None, *args, **kwargs):
        if self._origin is None:
            return module
        import transformers.dynamic_module_utils as td

        # Only take back what we installed; a later patcher owns anything else.
        if getattr(td.get_cached_module_file, _MARKER, False):
            td.get_cached_module_file = self._origin
        self._origin = None
        return module
