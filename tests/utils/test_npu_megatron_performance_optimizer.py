# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Overflow semantics and absence of per-tensor host synchronization."""

import pytest
import torch
from transformers.utils import is_torch_npu_available

from swift.model.npu_patch.megatron_optimizer import (multi_tensor_l2norm, multi_tensor_scale_tensor,
                                                      multi_tensor_unscale_l2norm)

DEVICES = [
    'cpu',
    pytest.param('npu', marks=pytest.mark.skipif(not is_torch_npu_available(), reason='Requires Ascend NPU'))
]


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('unscale_factor', [0.25, 1e-20, float('inf'), float('nan')])
@pytest.mark.parametrize('values', [[1.0, -2.0, 3.0], [float('nan')], [float('inf')], [], [1e30, 1e30]])
@pytest.mark.parametrize('initial_flag', [0, 1])
def test_norm_and_unscale_overflow(device, dtype, values, initial_flag, unscale_factor):
    inputs = [
        torch.tensor(values, dtype=dtype, device=device),
        torch.tensor([4.0], dtype=dtype, device=device),
    ]
    for unscale in (False, True):
        flag = torch.tensor([initial_flag], dtype=torch.int32, device=device)
        factor = unscale_factor if unscale else 1.0
        expected_parts = torch.stack([(x.float() * factor).norm() for x in inputs])
        if unscale:
            result = multi_tensor_unscale_l2norm(65536, flag, [inputs], torch.tensor(factor, device=device), True)
        else:
            result = multi_tensor_l2norm(65536, flag, [inputs], True)
        torch.testing.assert_close(result[0], expected_parts.norm(), equal_nan=True)
        torch.testing.assert_close(result[1], expected_parts, equal_nan=True)
        expected_flag = int(initial_flag or not torch.isfinite(expected_parts).all())
        assert flag.item() == expected_flag


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('inplace', [False, True])
@pytest.mark.parametrize(
    'values,expected_flag',
    [([1.0, 2.0], 0), ([float('nan')], 1), ([float('inf')], 1), ([3e38], 0), ([], 0)],
)
def test_scale_checks_input_overflow(device, inplace, values, expected_flag):
    src = torch.tensor(values, device=device)
    expected = src * 2
    dst = src if inplace else torch.empty_like(src)
    flag = torch.zeros(1, dtype=torch.int32, device=device)
    multi_tensor_scale_tensor(65536, flag, [[src], [dst]], torch.tensor(2.0, device=device))
    torch.testing.assert_close(dst, expected, equal_nan=True)
    assert flag.item() == expected_flag


@pytest.mark.parametrize('device', DEVICES)
def test_norm_and_scale_do_not_read_device_scalars(device):
    inputs = [torch.arange(8, dtype=torch.float32, device=device) for _ in range(16)]
    flag = torch.zeros(1, dtype=torch.int32, device=device)
    inv_scale = torch.tensor(0.25, device=device)
    scale = torch.tensor(0.5, device=device)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
        multi_tensor_l2norm(65536, flag, [inputs])
        multi_tensor_unscale_l2norm(65536, flag, [inputs], inv_scale)
        multi_tensor_scale_tensor(65536, flag, [inputs, inputs], scale)
    names = {event.key for event in trace.key_averages()}
    assert not names.intersection({'aten::item', 'aten::_local_scalar_dense', 'aten::is_nonzero'})


@pytest.mark.parametrize('device', DEVICES)
def test_overlapping_scale_views_keep_sequential_semantics(device):
    storage = torch.tensor([1.0, 3e38, 2.0], device=device)
    views = [storage[:2], storage[1:]]
    flag = torch.zeros((), dtype=torch.int32, device=device)
    multi_tensor_scale_tensor(65536, flag, [views, views], torch.tensor(2.0, device=device))
    torch.testing.assert_close(storage, torch.tensor([2.0, float('inf'), 4.0], device=device))
    assert flag.item() == 1


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('strided', [False, True])
def test_inplace_scale_layouts_and_existing_flag(device, dtype, strided):
    storage = torch.arange(16, dtype=dtype, device=device)
    src = storage[::2] if strided else storage[:8]
    expected = src * 0.25
    flag = torch.ones((), dtype=torch.int32, device=device)
    multi_tensor_scale_tensor(65536, flag, [[src], [src]], torch.tensor(0.25, device=device))
    torch.testing.assert_close(src, expected)
    assert flag.item() == 1


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('groups', [[], [[]], [[], []]])
@pytest.mark.parametrize('per_tensor', [False, True])
def test_empty_norm_groups(device, groups, per_tensor):
    flag = torch.ones(1, dtype=torch.int32, device=device)
    total, parts = multi_tensor_l2norm(65536, flag, groups, per_tensor)
    torch.testing.assert_close(total, torch.zeros(1, dtype=torch.float32))
    assert (parts is not None and parts.numel() == 0) if per_tensor else parts is None
    assert flag.item() == 1


@pytest.mark.parametrize('device', DEVICES)
def test_unscale_accepts_cpu_factor_without_tracking_its_gradient(device):
    grad = torch.tensor([3.0, 4.0], device=device)
    before = grad.clone()
    factor = torch.tensor(0.1, dtype=torch.float64, requires_grad=True, device='cpu')
    flag = torch.zeros(1, dtype=torch.int32, device=device)
    total, _ = multi_tensor_unscale_l2norm(65536, flag, [[grad]], factor)
    torch.testing.assert_close(total, (before * factor.item()).norm())
    torch.testing.assert_close(grad, before)
    assert not total.requires_grad
