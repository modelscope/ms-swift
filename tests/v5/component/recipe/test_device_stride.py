"""defect80 (5feb7e9eb): the device plan's nproc_per_node is the per-node device stride, not the
global rank span. ResourceManager maps rank->(node,device) via rank//stride / rank%stride and
filters nodes by device_count >= stride; passing the span made every plan whose disjoint auxiliary
ranks overflow the visible devices fail with a misleading "available: 0".

PLAN (case | input | expected | kills):
 1 span-fits | span3 per_node4 | 3 | always-per_node over-reservation
 2 span-equals | span4 per_node4 | 4 | off-by-one (< instead of <=)
 3 span-overflows | span5 per_node4 | 4 | old stride=span bug (defect80 itself)
 4 span1 | span1 per_node8 | 1 | single-GPU minimal reservation lost
 5 no-device-module | module None | ValueError | silent stride 0
 6 zero-devices | device_count 0 | ValueError | silent stride 0
 7 span0 | span0 | 0 (no device query) | guard removed -> spurious device access

Oracle = the ResourceManager contract (rank//stride node mapping), hand-computed min().
Reverse-verify: revert min() to `return rank_span` -> case 3 RED (5 != 4), cases 1/2/4 stay GREEN.
"""

from types import SimpleNamespace as NS

import pytest

from swift.dev.recipe.utils import _allocation_nproc_per_node


def _patch_devices(monkeypatch, count):
    module = None if count is None else NS(device_count=lambda: count)
    monkeypatch.setattr('twinkle.Platform.get_platform', lambda: NS(device_module=lambda: module))


@pytest.mark.parametrize('span,per_node,expected', [
    (3, 4, 3),   # span-fits: minimal reservation, all ranks on node 0
    (4, 4, 4),   # span-equals
    (5, 4, 4),   # span-overflows: honest per-node stride, not the misleading span
    (1, 8, 1),   # span1
])
def test_stride_is_min_of_span_and_per_node(monkeypatch, span, per_node, expected):
    _patch_devices(monkeypatch, per_node)
    assert _allocation_nproc_per_node(span) == expected


def test_no_device_module_fails_loudly(monkeypatch):
    _patch_devices(monkeypatch, None)
    with pytest.raises(ValueError, match='no accelerator devices'):
        _allocation_nproc_per_node(3)


def test_zero_devices_fails_loudly(monkeypatch):
    _patch_devices(monkeypatch, 0)
    with pytest.raises(ValueError, match='no accelerator devices'):
        _allocation_nproc_per_node(3)


def test_nonpositive_span_short_circuits(monkeypatch):
    def _boom():
        raise AssertionError('device platform must not be queried for a nonpositive span')

    monkeypatch.setattr('twinkle.Platform.get_platform', _boom)
    assert _allocation_nproc_per_node(0) == 0
    assert _allocation_nproc_per_node(-2) == 0
