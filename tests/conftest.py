# Copyright (c) ModelScope Contributors. All rights reserved.
"""This tree is legacy end-to-end scripts, so the whole tree is ``slow`` -- daily CI only.

Measured: of the 102 test files here, 80 hardcode a real model id, 54 assign ``CUDA_VISIBLE_DEVICES``
at import time, and 94 end in a ``__main__`` block. They were written to be run by hand, one file at
a time, with a warm model cache -- not to be collected together on every PR.

Marking the tree rather than each file keeps the rule stateless: a test earns its way into the PR
tier by moving to ``swift/dev/tests``, where it declares what hardware it needs with
``@pytest.mark.accel(n)`` instead of pinning devices at import time.
"""
from pathlib import Path

import pytest

HERE = Path(__file__).parent


def pytest_collection_modifyitems(items):
    # ``items`` is the whole session, not just this directory -- filter, or the maintained suites
    # under swift/dev/tests get marked slow too and the PR tier collects nothing.
    for item in items:
        if HERE in item.path.parents:
            item.add_marker(pytest.mark.slow)
