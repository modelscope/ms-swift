# Copyright (c) ModelScope Contributors. All rights reserved.
"""Skip accelerate's offload hooks so `device_map` keeps real tensors instead of meta stubs.

When a `device_map` sends part of a model to ``cpu``/``disk``, ``accelerate.dispatch_model`` ends by
calling ``attach_align_device_hook_on_blocks``, which hangs an ``AlignDevicesHook`` on each block.
Those hooks make offloading work *at forward time*: the real weights live in a ``weights_map`` and
are fetched onto the execution device right before a block runs, then dropped afterwards. The
parameters left on the module are meta stubs -- accelerate says as much, warning "Some parameters are
on the meta device because they were offloaded".

That is exactly wrong for code that reads weights without ever running the model, e.g. converting a
checkpoint to another format: it walks ``named_parameters()`` and would copy empty meta tensors. This
patch replaces the function with a no-op (the original mutates the module in place and returns
``None``, so a bare ``return`` is equivalent), leaving the parameters as real tensors on whichever
device the map assigned.

Scope: apply this only around loading a model you will *not* run a forward pass on. Keep the hooks
whenever the model is actually executed -- with them gone, an offloaded model's forward would read
meta tensors. ``dispatch_model`` resolves the name through the ``big_modeling`` module namespace, so
rebinding that attribute is enough; a caller importing it straight from ``accelerate.hooks`` is not
affected.

Usage:
    with apply_context(None, AccelerateSkipOffloadHookPatch()):
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map)
"""
from twinkle.patch import Patch

_MARKER = '_swift_skip_offload_hook'


class AccelerateSkipOffloadHookPatch(Patch):
    """Neuter ``big_modeling.attach_align_device_hook_on_blocks`` for the duration of a model load."""

    def __init__(self):
        self._origin = None

    def __call__(self, module=None, *args, **kwargs):
        from accelerate import big_modeling

        # Nested apply, or another instance owns the active replacement: leave it in place so
        # whoever installed it stays responsible for restoring the real one.
        if getattr(big_modeling.attach_align_device_hook_on_blocks, _MARKER, False):
            return module

        def attach_align_device_hook_on_blocks(*args, **kwargs):
            # The original attaches hooks in place and returns None; doing nothing is equivalent
            # except that the parameters keep their real data.
            return

        setattr(attach_align_device_hook_on_blocks, _MARKER, True)
        self._origin = big_modeling.attach_align_device_hook_on_blocks
        big_modeling.attach_align_device_hook_on_blocks = attach_align_device_hook_on_blocks
        return module

    def unpatch(self, module=None, *args, **kwargs):
        if self._origin is None:
            return module
        from accelerate import big_modeling

        # Only take back what we installed; a later patcher owns anything else.
        if getattr(big_modeling.attach_align_device_hook_on_blocks, _MARKER, False):
            big_modeling.attach_align_device_hook_on_blocks = self._origin
        self._origin = None
        return module
