# Copyright (c) ModelScope Contributors. All rights reserved.
"""This tree is legacy end-to-end scripts, so the whole tree is ``slow`` -- daily CI only.

Measured: of the 102 test files here, 80 hardcode a real model id, 54 assign ``CUDA_VISIBLE_DEVICES``
at import time, and 94 end in a ``__main__`` block. They were written to be run by hand, one file at
a time, with a warm model cache -- not to be collected together on every PR.

Marking the tree rather than each file keeps the rule stateless: a test earns its way into the PR
tier by moving to ``tests/v5``, where it declares what hardware it needs with
``@pytest.mark.accel(n)`` instead of pinning devices at import time.

``testpaths`` keeps ``tests`` in the default collection, so the scheduled CI's bare ``pytest`` still
runs this tree; PR CI passes ``-m "not slow"`` and collects only ``tests/v5``.
"""
from pathlib import Path

import pytest

HERE = Path(__file__).parent
V5 = HERE / 'v5'


def pytest_collection_modifyitems(items):
    # ``items`` is the whole session, not just this directory -- filter, or the maintained suites
    # under tests/v5 get marked slow too and the PR tier collects nothing.
    for item in items:
        if HERE in item.path.parents and V5 not in item.path.parents:
            item.add_marker(pytest.mark.slow)
