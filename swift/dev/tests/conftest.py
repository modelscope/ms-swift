# Copyright (c) ModelScope Contributors. All rights reserved.
"""Pytest config for all swift/dev tests (now consolidated under swift/dev/tests/).

Registers the `slow` marker (real vLLM/Megatron engine, heavy GPU / multi-GPU Ray) and skips it
by default so the normal `pytest swift/dev/tests` regression stays fast. Run slow tests explicitly:
    pytest swift/dev/tests/... -m slow --runslow
"""
import pytest
import shutil


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


def pytest_configure(config):
    config.addinivalue_line('markers',
                            'slow: heavy test (real vLLM/Megatron engine, multi-GPU); skipped unless --runslow')


def pytest_addoption(parser):
    parser.addoption('--runslow', action='store_true', default=False, help='run slow tests (real vLLM/Megatron engine)')


def pytest_collection_modifyitems(config, items):
    if config.getoption('--runslow'):
        return
    skip_slow = pytest.mark.skip(reason='slow test; pass --runslow to run')
    for item in items:
        if 'slow' in item.keywords:
            item.add_marker(skip_slow)
