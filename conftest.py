# Copyright (c) ModelScope Contributors. All rights reserved.
"""Test tiering: ``accel(n)`` needs n accelerators, ``slow`` is daily-only.

CI runs one command -- ``pytest -m "not slow"`` on a PR, ``pytest`` nightly. Everything runs inside
an accelerator container, so ``accel(n)`` is not about *whether* the box has a device but *how many*:
a 2-card CI box skips the 4-card megatron cases instead of failing them, while an 8-card dev box
runs them. This replaces the ``TEST_LEVEL`` env var, which CI set and no test ever read.
"""
import pytest


class Accelerators:
    """The box's accelerators, whichever backend it has."""

    @staticmethod
    def count() -> int:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        npu = getattr(torch, 'npu', None)
        return npu.device_count() if npu is not None and npu.is_available() else 0


def pytest_collection_modifyitems(config, items):
    """Skip -- never fail -- tests asking for more accelerators than this box has."""
    available = Accelerators.count()
    for item in items:
        marker = item.get_closest_marker('accel')
        if marker is None:
            continue
        needed = marker.args[0] if marker.args else 1
        if available < needed:
            item.add_marker(pytest.mark.skip(reason=f'needs {needed} accelerator(s), found {available}'))
