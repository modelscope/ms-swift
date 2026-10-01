# Copyright (c) ModelScope Contributors. All rights reserved.
"""Pytest config for all swift/dev tests (now consolidated under swift/dev/tests/).

Tiering lives in the root ``pyproject.toml`` / ``conftest.py``: mark heavy tests ``slow`` and CI
selects with ``-m``. Run just the heavy ones with ``pytest swift/dev/tests -m slow``.
"""
import shutil

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Stash each phase's report so fixtures can tell whether the test passed."""
    outcome = yield
    setattr(item, f'_report_{outcome.get_result().when}', outcome.get_result())


@pytest.fixture(autouse=True)
def _reclaim_heavy_tmp(request):
    """Delete a PASSING slow test's tmp_path; keep it when the test fails.

    One Megatron run writes 12-15 GB of fp32 checkpoints and pytest keeps the last 3 basetemp
    trees, so two runs fill a 30 GB /tmp and the next save fails with ENOSPC -- which reads like a
    checkpointing bug rather than a full disk (it cost this project one such misdiagnosis).
    """
    # Resolve the path during SETUP: by teardown time the tmp_path fixture may already be gone,
    # and a plain Path needs no live fixture to delete.
    heavy_tmp = None
    if 'slow' in request.keywords and 'tmp_path' in request.fixturenames:
        heavy_tmp = request.getfixturevalue('tmp_path')
    yield
    report = getattr(request.node, '_report_call', None)
    if heavy_tmp is None or report is None or not report.passed:
        return
    shutil.rmtree(heavy_tmp, ignore_errors=True)


@pytest.fixture(autouse=True)
def _isolate_twinkle_runtime():
    """Reset twinkle's Ray cluster and infra globals after every test.

    A pytest session drives MANY recipes through ONE process, so the second ``mode='ray'`` test would
    otherwise reuse the first test's live cluster and collide on twinkle's deterministic worker-actor
    name. The mechanics (and why each step is needed) live in ``reset_twinkle_runtime``; this fixture
    is the between-tests caller, and a multi-session test that runs ``run_sft`` twice in-process calls
    the same helper between its own runs.
    """
    from swift.dev.tests._twinkle_runtime import reset_twinkle_runtime
    yield
    reset_twinkle_runtime()
