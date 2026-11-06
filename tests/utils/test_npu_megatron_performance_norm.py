# Copyright (c) ModelScope Contributors. All rights reserved.
"""Frozen RMSNorm parity and delegation for unsupported execution modes."""

import pytest
import torch
from transformers.utils import is_torch_npu_available
from types import SimpleNamespace

from swift.model.npu_patch import megatron_norm as norm


def test_cpu_delegation(monkeypatch):
    calls = []
    monkeypatch.setattr(norm, '_ORIGINAL_FORWARD', lambda *a, **kw: calls.append((a, kw)) or 'original')
    x = torch.ones(2, 8)
    assert norm.rmsnorm_forward(SimpleNamespace(), input=x) == 'original'
    assert len(calls) == 1
    assert calls[0][0][1] is x
    assert calls[0][1] == {}


@pytest.mark.skipif(not is_torch_npu_available(), reason='Requires Ascend NPU')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('zero_gamma', [False, True])
@pytest.mark.parametrize('shape', [(17, 32), (7, 2, 4, 8), (17, 128), (17, 2048)])
def test_frozen_outputs_and_gradients(dtype, zero_gamma, shape):
    from transformer_engine.pytorch.module.rmsnorm import RMSNorm

    import swift.megatron  # noqa: F401

    normalized = shape[-2:] if len(shape) == 4 else shape[-1]
    module = RMSNorm(normalized, dtype=dtype, device='npu', zero_centered_gamma=zero_gamma)
    module.weight.requires_grad_(False)
    module.weight.copy_(torch.linspace(0.3, 1.7, module.weight.numel(), device='npu').reshape_as(module.weight))
    x = torch.randn(shape, dtype=dtype, device='npu', requires_grad=True)
    dy = torch.randn_like(x)
    results = []
    for fn in (norm._ORIGINAL_FORWARD, norm.rmsnorm_forward):
        y = fn(module, x)
        results.append((y, torch.autograd.grad(y, x, dy)[0]))
    for original, patched in zip(*results):
        torch.testing.assert_close(original, patched, rtol=0, atol=0)
    assert module.weight.grad is None


@pytest.mark.skipif(not is_torch_npu_available(), reason='Requires Ascend NPU')
@pytest.mark.parametrize(
    'feature', ['trainable', 'fp8', 'calibration', 'offload', 'debug', 'compile', 'autocast', 'empty', 'subclass'])
def test_feature_delegation(monkeypatch, feature):
    from transformer_engine.pytorch.module.rmsnorm import RMSNorm

    import swift.megatron  # noqa: F401

    class CustomRMSNorm(RMSNorm):
        pass

    module = (CustomRMSNorm if feature == 'subclass' else RMSNorm)(8, dtype=torch.bfloat16, device='npu')
    module.weight.requires_grad_(feature == 'trainable')
    x = torch.ones((0 if feature == 'empty' else 2, 8), device='npu', dtype=torch.bfloat16)
    monkeypatch.setattr(norm, '_ORIGINAL_FORWARD', lambda *a, **kw: 'original')
    monkeypatch.setattr(
        norm, '_FP8_MANAGER',
        SimpleNamespace(is_fp8_enabled=lambda: feature == 'fp8', is_fp8_calibration=lambda: feature == 'calibration'))
    monkeypatch.setattr(norm, '_CPU_OFFLOAD', lambda: feature == 'offload')
    monkeypatch.setattr(norm, '_DEBUG_STATE', SimpleNamespace(debug_enabled=feature == 'debug'))
    monkeypatch.setattr(torch.compiler, 'is_compiling', lambda: feature == 'compile')
    monkeypatch.setattr(torch, 'is_autocast_enabled', lambda *a: feature == 'autocast')
    assert norm.rmsnorm_forward(module, input=x) == 'original'
