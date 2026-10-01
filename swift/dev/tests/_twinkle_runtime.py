# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tear down twinkle's Ray cluster + infra globals so the next run re-initializes cleanly.

Production Ray runs are one-shot -- the process exits after training -- so twinkle keeps no public
shutdown counterpart to ``initialize`` (the ``twinkle.shutdown()`` calls sprinkled through the Megatron
component tests raise AttributeError and are swallowed by their bare ``except``, i.e. silent no-ops).
A pytest process is NOT one-shot: it drives MANY recipes through ONE interpreter. Two callers need an
explicit reset:

  * the autouse ``_isolate_twinkle_runtime`` conftest fixture, between tests;
  * a multi-session test that calls ``run_sft`` (mode='ray') more than once in-process -- e.g.
    ``test_run_sft_megatron_ga_equivalence`` and ``test_run_sft_megatron_two_bridges_bit_identical``
    each run it twice -- which must reset BETWEEN its own runs, where the fixture cannot reach.

Without a reset the second ``mode='ray'`` run reuses the first run's live cluster and collides on
twinkle's deterministic worker-actor name ``{group}-{cls}-{caller_file}_{line}-{rank}`` (every run_sft
Ray run builds the model at the same call site, so they all want the same name), and stale infra
globals (``_mode``/``_device_group``/``_device_mesh``) leak between the local and Ray runs.

``RayHelper.teardown`` drops the placement group + registry actor and clears the class-level
``resource_manager`` (so the next ``initialize`` rebuilds it instead of pointing at dead actors);
``ray.shutdown`` then kills the worker actors and frees their GPUs. Safe to call after ``run_sft``
returns: ``SFTLoop.fit`` materializes the history into plain dicts (``float(metrics['loss'])`` per
step) before returning, so no live Ray future is dropped here. Every step is best-effort and guarded
-- a run that never touched Ray leaves ``ray_inited()`` False and this is a cheap no-op.
"""
import os


def reset_twinkle_runtime() -> None:
    """Reset twinkle's Ray session and infra globals to their import-time defaults."""
    try:
        from twinkle.infra._ray import RayHelper
        if RayHelper.ray_inited():
            RayHelper.teardown()
    except Exception:  # noqa: BLE001 -- teardown is best-effort; never mask a caller's own result
        pass
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
    except Exception:  # noqa: BLE001
        pass
    try:
        import twinkle.infra as infra
        # Restore _mode to twinkle's import-time default, NOT None: a run that never calls
        # initialize() (e.g. constructing InputProcessor) relies on that default, and the
        # @remote_class __init__ raises "Unsupported mode: None" for any other value. Only a
        # preceding ray run's _mode='ray' is the stale state we actually need to clear.
        infra._mode = 'ray' if os.environ.get('TWINKLE_MODE', 'local') == 'ray' else 'local'
        infra._device_group, infra._device_mesh = None, None
    except Exception:  # noqa: BLE001
        pass
    try:
        # accelerate's AcceleratorState/PartialState are process-global singletons keyed on the FIRST
        # setting that built them: once one test initializes them (a TransformersModel with
        # strategy='accelerate', mixed_precision='no'), a later test that requests a DIFFERENT setting
        # (e.g. mixed_precision='bf16') dies with "AcceleratorState has already been initialized and
        # cannot be changed, restart your runtime completely". Production is one-shot per process so
        # this never bites there; a pytest process drives many runs, so clear the singletons between
        # tests. reset_partial_state=True also clears PartialState (device/distributed), whose shared
        # state AcceleratorState._reset_state alone leaves behind.
        from accelerate.state import AcceleratorState
        AcceleratorState._reset_state(reset_partial_state=True)
    except Exception:  # noqa: BLE001
        pass
