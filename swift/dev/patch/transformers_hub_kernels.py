# Copyright (c) ModelScope Contributors. All rights reserved.
"""Route transformers Hub-kernel downloads through ModelScope.

When a model requests kernelization, transformers auto-downloads the kernel repo via
``transformers.integrations.hub_kernels.get_kernel`` during model load. That pull targets
HuggingFace Hub and can fail on networks without HF access.

This patch rewrites ``get_kernel`` to fetch the kernel repo from ModelScope
(``swift.dev.utils.safe_snapshot_download``) and load it locally via ``kernels.get_local_kernel``,
falling back to the original HF-based ``get_kernel`` on any error. Mirrors legacy swift's
``patch_kernels`` (swift/utils/hub_utils.py), minus the build-variant pre-resolution -- the whole
repo is downloaded, which is simpler and avoids coupling to ModelScope's file-listing API.

The ``USE_HF=1`` env keeps the original HF-based loading, checked on each ``get_kernel`` call, so
the runtime switch matches legacy semantics.

No-op when transformers has no ``hub_kernels`` integration or the optional ``kernels`` package
is missing.

Usage (persistent, idempotent)::

    from twinkle.patch import apply_patch
    from swift.dev.patch.transformers_hub_kernels import TransformersHubKernelPatch
    apply_patch(None, TransformersHubKernelPatch())
"""
from twinkle.patch import Patch
from twinkle.utils import get_logger

logger = get_logger()

_MARKER = '_swift_origin_get_kernel'


def _hub_kernels_module():
    """Return transformers' ``hub_kernels`` integration module, or ``None`` when unavailable
    (older transformers, or the optional ``kernels`` package is missing)."""
    try:
        from kernels import get_local_kernel  # noqa: F401  (require the local loader)
        from transformers.integrations import hub_kernels
        return hub_kernels
    except ImportError:
        return None


class TransformersHubKernelPatch(Patch):
    """Download transformers Hub kernels from ModelScope instead of HuggingFace.
    Idempotent, reversible, no-op without the hub_kernels integration / ``kernels``."""

    def __call__(self, module=None, *args, **kwargs):
        hub_kernels = _hub_kernels_module()
        if hub_kernels is None or hasattr(hub_kernels, _MARKER):
            return module

        origin_get_kernel = hub_kernels.get_kernel

        def get_kernel(repo_id, *args, **kwargs):
            from pathlib import Path

            from swift.dev.utils import safe_snapshot_download, use_hf_hub

            if use_hf_hub():
                return origin_get_kernel(repo_id, *args, **kwargs)
            try:
                from kernels import get_local_kernel

                model_dir = safe_snapshot_download(repo_id, use_hf=False)
                package_name = repo_id.split('/')[-1].replace('-', '_')
                kernel = get_local_kernel(Path(model_dir), package_name)
                logger.info(f'Loaded kernel `{repo_id}` from ModelScope: {model_dir}')
                return kernel
            except Exception as e:
                logger.warning(f'Failed to load kernel `{repo_id}` from ModelScope ({e}), fallback to HuggingFace.')
                return origin_get_kernel(repo_id, *args, **kwargs)

        setattr(hub_kernels, _MARKER, origin_get_kernel)
        hub_kernels.get_kernel = get_kernel
        logger.info('Patched transformers hub_kernels.get_kernel to source kernels from ModelScope.')
        return module

    def unpatch(self, module=None, *args, **kwargs):
        hub_kernels = _hub_kernels_module()
        if hub_kernels is None:
            return module
        origin = getattr(hub_kernels, _MARKER, None)
        if origin is not None:
            hub_kernels.get_kernel = origin
            delattr(hub_kernels, _MARKER)
        return module
