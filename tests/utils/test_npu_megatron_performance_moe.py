# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Stable permutation parity and delegation to the original implementation."""

import pytest
import torch
from transformers.utils import is_torch_npu_available
from types import SimpleNamespace

from swift.model.npu_patch import megatron_moe as moe


@pytest.mark.parametrize('options', [{}, {'fused': True}, {'drop_and_pad': True}, {'align_size': 32}])
def test_fallback(monkeypatch, options):
    calls = []
    monkeypatch.setattr(moe, '_ORIGINAL_PERMUTE', lambda *args: calls.append(args) or 'original')
    device = 'cpu' if not options else 'npu'
    tokens = SimpleNamespace(device=SimpleNamespace(type=device))
    assert moe.permute(tokens, None, **options) == 'original'
    assert len(calls) == 1
    assert calls[0][0] is tokens


@pytest.mark.skipif(not is_torch_npu_available(), reason='Requires Ascend NPU')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('shape', [(17, 8, 3), (1, 4, 1), (0, 4, 1), (8, 4, 0)])
@pytest.mark.parametrize('with_probs', [False, True])
def test_permutation_and_gradients(dtype, shape, with_probs):
    import swift.megatron  # noqa: F401

    n, experts, topk = shape
    torch.manual_seed(42)
    indices = torch.rand(n, experts, device='npu').topk(topk, dim=-1).indices
    mask = torch.zeros(n, experts, device='npu', dtype=torch.bool).scatter_(1, indices, True)
    tokens = torch.randn(n, 7, device='npu', dtype=dtype, requires_grad=True)
    probs = torch.randn(n, experts, device='npu', requires_grad=True) if with_probs else None
    counts = mask.sum(dim=0)
    results = [
        fn(tokens, mask, probs, n * topk, tokens_per_expert=counts) for fn in (moe._ORIGINAL_PERMUTE, moe.permute)
    ]
    for old, new in zip(*results):
        if old is None:
            assert new is None
        else:
            torch.testing.assert_close(old, new, rtol=0, atol=0)
    assert results[1][-1] is counts
    gradients = []
    for result in results:
        loss = result[0].float().sum()
        if probs is not None:
            loss = loss + result[1].sum()
        gradients.append(torch.autograd.grad(loss, (tokens, probs) if with_probs else (tokens, ), retain_graph=True))
    for old, new in zip(*gradients):
        torch.testing.assert_close(old, new, rtol=0, atol=0)
