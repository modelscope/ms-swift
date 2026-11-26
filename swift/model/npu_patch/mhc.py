# Copyright (c) ModelScope Contributors. All rights reserved.
"""Load the optional mHC operators used by the Transformers and Megatron NPU patches."""
import importlib
import importlib.util
import os
import torch
from functools import cache
from types import ModuleType

from swift.utils.logger import get_logger


def get_mhc_mode(mode=None):
    mode = (os.environ.get('GLM_MHC', 'auto') if mode is None else mode).lower()
    if mode not in ('auto', 'off', 'pre', 'post', 'both'):
        raise ValueError(f'Expected GLM_MHC to be auto, off, pre, post or both, got {mode!r}')
    return mode


@cache
def _supported_device(device):
    return torch.npu.get_device_name(device).startswith(('Ascend910B', 'Ascend910_93'))


def supports_auto_mhc(x, streams, width):
    # CANN A2/A3 shapes, including the GLM-5.3-Flash width.
    return (x.device.type == 'npu' and x.dtype == torch.bfloat16 and x.numel() > 0 and streams == 4
            and width in (1280, 1920, 2560, 4096) and _supported_device(x.device))


def cann_package():
    import torch_npu  # Register the NPU backend before importing either provider.
    for name in ('cann_ops_transformer', 'npu_ops_transformer'):
        if importlib.util.find_spec(name) is not None:
            return importlib.import_module(name)
    return None


def lookup_op(package, name):
    """Find a named Python wrapper. Keep errors from its own dependencies visible."""
    ops_path = package.__name__ + '.ops'
    exports = getattr(package, 'ops', None)
    if exports is None:
        try:
            exports = importlib.import_module(ops_path)
        except ModuleNotFoundError as exc:
            if exc.name != ops_path:
                raise
    for namespace in (exports, package):
        op = getattr(namespace, name, None)
        if isinstance(op, ModuleType):
            op = getattr(op, name, None)
        if callable(op):
            return op
    if exports is not None and hasattr(exports, '__path__'):
        module_name = ops_path + '.' + name
        if importlib.util.find_spec(module_name) is not None:
            op = getattr(importlib.import_module(module_name), name, None)
            if callable(op):
                return op
    return None


def resolve_op(package, name, backend):
    op = lookup_op(package, name) if package is not None else None
    if op is not None:
        return op, f'{package.__name__}.{name}'
    if backend == 'cann':
        provider = package.__file__ if package is not None else 'no installed CANN transformer package'
        raise ImportError(f'GLM_MHC_BACKEND=cann requires {name}, unavailable in {provider}')
    # An explicitly selected auto backend can use a separately supplied Triton implementation.
    glm_mhc_triton = importlib.import_module('glm_mhc_triton')

    if name == 'mhc_pre_sinkhorn':
        return glm_mhc_triton.pre_sinkhorn, 'Triton Ascend Sinkhorn + PyTorch projection'
    return glm_mhc_triton.mhc_post, 'Triton Ascend mHC post'


@cache
def get_mhc_ops(mode, backend='cann'):
    native = (None, 'native')
    if mode == 'off':
        return native, native
    try:
        package = cann_package()
        pre = resolve_op(package, 'mhc_pre_sinkhorn', backend) if mode != 'post' else native
        post = resolve_op(package, 'mhc_post', backend) if mode != 'pre' else native
        return pre, post
    except (ImportError, OSError, RuntimeError) as exc:
        if mode != 'auto':
            raise
        get_logger().debug('Using native GLM mHC: %s', exc)
        return native, native
