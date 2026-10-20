# Copyright (c) ModelScope Contributors. All rights reserved.
"""Pytest config for all swift/dev tests.

Tiering + capability gating live here because the repo root carries no pytest ini the dev tree can
lean on: markers are self-registered and deselection happens at collection, so ``pytest -m "not
slow"`` on any box -- CPU runner, CUDA box, NPU box -- stays green without CI carrying test lists.

Levels (the ut_plan contract):
  * unmarked / L1: must finish in minutes, runs with every PR (``-m "not slow"``).
  * ``@pytest.mark.slow`` / L2: daily. Add ``@pytest.mark.accel(N)`` to pin the GPU count; a box with
    fewer accelerators skips rather than fails.

Optional engines (vllm / sglang / megatron / unsloth / sentence-transformers) are NOT hard
requirements of every environment: a case that needs one carries ``@pytest.mark.requires_<engine>``
and is skipped where the import is missing, instead of erroring mid-run.
"""
import importlib.util
import shutil

import pytest

#: marker -> the import whose absence means the engine is unavailable on this box.
REQUIREMENTS = {
    'requires_megatron': ('megatron.core', ),
    'requires_vllm': ('vllm', ),
    'requires_sglang': ('sglang', ),
    'requires_unsloth': ('unsloth', ),
    'requires_st': ('sentence_transformers', ),
}
_requirement_cache: dict = {}


def _has(module: str) -> bool:
    if module not in _requirement_cache:
        try:
            _requirement_cache[module] = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):  # a broken install also counts as absent
            _requirement_cache[module] = False
    return _requirement_cache[module]


def pytest_configure(config):
    config.addinivalue_line('markers', 'slow: L2 -- daily tier, may run long')
    config.addinivalue_line('markers', 'accel(N): needs at least N accelerators (skip when fewer)')
    for marker in REQUIREMENTS:
        engine = marker.removeprefix('requires_')
        config.addinivalue_line('markers', f'{marker}: needs the {engine} engine installed (skip when absent)')


def pytest_collection_modifyitems(config, items):
    """Skip -- never fail -- tests asking for more accelerators or absent engines than this box has."""
    available = _accelerator_count()
    for item in items:
        marker = item.get_closest_marker('accel')
        if marker is not None:
            needed = marker.args[0] if marker.args else 1
            if available < needed:
                item.add_marker(pytest.mark.skip(reason=f'needs {needed} accelerator(s), found {available}'))
        for marker_name, modules in REQUIREMENTS.items():
            if item.get_closest_marker(marker_name) is None:
                continue
            missing = [module for module in modules if not _has(module)]
            if missing:
                item.add_marker(pytest.mark.skip(reason=f'needs {"/".join(missing)}, which is not installed'))


def _accelerator_count() -> int:
    """The box's accelerators, whichever backend it has (CUDA today, NPU on an ascend box)."""
    try:
        import torch
    except ImportError:
        return 0
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    npu = getattr(torch, 'npu', None)
    return npu.device_count() if npu is not None and npu.is_available() else 0


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
