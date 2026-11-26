# Copyright (c) ModelScope Contributors. All rights reserved.
"""Optional mHC forward and backward fusion for the GLM-5.3-Flash text decoder."""
import ast
import importlib
import importlib.util
import inspect
import linecache
import os
import textwrap
import torch
from functools import wraps
from types import ModuleType

from swift.utils.logger import get_logger
from .utils import apply_patch_map

logger = get_logger()


def native_post(residual, comb, h_out, post):
    # Preserve the HF rounding points for the reference comparison.
    dtype = residual.dtype
    mixed = comb.to(dtype).transpose(-1, -2) @ residual
    return post.to(dtype).unsqueeze(-1) * h_out.unsqueeze(-2) + mixed


def with_fused_post(forward, merge):
    """Replace only the two known residual assignments in the installed decoder."""
    source = textwrap.dedent(inspect.getsource(forward))
    tree = ast.parse(source)
    fn = tree.body[0]
    if not isinstance(fn, ast.FunctionDef) or fn.decorator_list or forward.__closure__:
        raise RuntimeError('Unexpected decoder wrapper. Inspect the loaded decoder forward first.')
    reference = ast.parse('hidden_states = post.to(dtype).unsqueeze(-1) * hidden_states.unsqueeze(-2)'
                          ' + torch.matmul(comb.to(dtype).transpose(-1, -2), residual)').body[0]
    matches = [node for node in fn.body if ast.dump(node) == ast.dump(reference)]
    if len(matches) != 2:
        raise RuntimeError(f'Expected two HF residual assignments in {inspect.getfile(forward)}, found {len(matches)}. '
                           'Use GLM_MHC=pre to enable only pre fusion.')
    # Splice the original lines so all attention branches, FFN, and return values
    # come directly from the installed implementation.
    lines = source.splitlines(keepends=True)
    for node in reversed(matches):
        lines[node.lineno - 1:node.end_lineno] = [
            ' ' * node.col_offset + 'hidden_states = _glm_mhc_merge(residual, comb, hidden_states, post)\n'
        ]
    source = ''.join(lines)
    filename = f'<GLM mHC post patch of {inspect.getfile(forward)}>'
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = dict(forward.__globals__, _glm_mhc_merge=merge)
    exec(compile(source, filename, 'exec'), namespace)
    patched = namespace[fn.name]
    return wraps(forward)(patched)


def make_pre(cann_pre, original, backend):
    # CANN releases expose either an autograd wrapper with eight arguments or
    # a wrapper with an explicit backward-intermediates flag.
    sig = inspect.signature(cann_pre)
    flags = {'outflag', 'needbackward'}
    flag = next((name for name in sig.parameters if name.replace('_', '').lower() in flags), None)
    if flag:
        sig.bind(*([None] * 8), **{flag: True})
    else:
        sig.bind(*([None] * 8))

    @wraps(original)
    def fused(self, hidden_streams):
        x = hidden_streams
        if x.device.type != 'npu' or x.dtype != torch.bfloat16:
            if backend == 'cann':
                raise ValueError('CANN mHC pre requires BF16 hidden streams on NPU')
            return original(self, x)
        batch, seq, streams, _ = x.shape
        phi, alpha, bias = self.fn.float(), self.scale.float(), self.base.float()
        kwargs = {}
        if flag:
            kwargs[flag] = torch.is_grad_enabled() and any(t.requires_grad for t in (x, phi, alpha, bias))
        values = cann_pre(x.contiguous(), phi, alpha, bias, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps,
                          self.input_norm.eps, **kwargs)
        h_in, post, comb = values[:3]
        return post, comb.reshape(batch, seq, streams, streams), h_in

    return fused


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


def selected_backend():
    backend = os.environ.get('GLM_MHC_BACKEND', 'cann').lower()
    if backend not in ('auto', 'cann'):
        raise ValueError('GLM_MHC_BACKEND must be auto or cann')
    return backend


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


def make_forwards(mode, original, backend):
    package = cann_package()
    pre = original
    pre_backend = post_backend = 'HF'
    if mode in ('pre', 'both'):
        pre_op, pre_backend = resolve_op(package, 'mhc_pre_sinkhorn', backend)
        pre = make_pre(pre_op, original, backend)
    if mode in ('post', 'both'):
        mhc_post, post_backend = resolve_op(package, 'mhc_post', backend)

        def post(residual, comb, h_out, post_weight):
            if residual.device.type != 'npu' or residual.dtype != torch.bfloat16:
                if backend == 'cann':
                    raise ValueError('CANN mHC post requires BF16 residuals on NPU')
                return native_post(residual, comb, h_out, post_weight)
            return mhc_post(residual.contiguous(),
                            comb.float().contiguous(), h_out.contiguous(),
                            post_weight.float().contiguous())
    else:
        post = native_post

    return pre, post, pre_backend, post_backend


def apply_patch(mode):
    """Resolve the selected operators and validate both decoder call sites before patching."""
    if mode == 'off':
        return
    if mode not in ('pre', 'post', 'both'):
        raise ValueError('GLM_MHC must be off, pre, post, or both')

    from transformers.models.glm5_next import modeling_glm5_next as glm

    backend = selected_backend()
    installed = getattr(glm, '_glm_mhc_swift_mode', None)
    if installed == mode and getattr(glm, '_glm_mhc_swift_backend', None) == backend:
        return
    if installed:
        raise RuntimeError(f'GLM mHC is already installed in mode {installed}. Restart to change mode or backend.')

    pre, post, pre_name, post_name = make_forwards(mode, glm.Glm5NextTextHyperConnection.forward, backend)
    patches = {}
    if mode in ('post', 'both'):
        patches['Glm5NextTextDecoderLayer.forward'] = with_fused_post(glm.Glm5NextTextDecoderLayer.forward, post)
    if mode in ('pre', 'both'):
        patches['Glm5NextTextHyperConnection.forward'] = pre
    apply_patch_map(glm, patches)
    glm._glm_mhc_swift_mode = mode
    glm._glm_mhc_swift_backend = backend
    logger.info(f'[glm-mhc] mode={mode} pre={pre_name} post={post_name}')
