"""defect499: GRPO teardown must not mask the original training exception.

PLAN (case | input | expected | kills):
 1 orig+stats-raise | E in flight, driver.stats raises C | E propagates, later steps run | dead-actor RPC masks E
 2 orig+finish-raise | E, finish_rollout_metrics raises C | E propagates, all run | first-cleanup masking
 3 orig+rollout-raise | E, rollout.close raises C (middle) | E propagates, tracker runs | middle-cleanup masking
 4 orig+tracker-raise | E, tracker.close raises C (innermost) | E propagates | innermost finally masking
 5 orig+all-raise | E, every cleanup raises | E propagates, all attempted | cumulative masking
 6 success+raise | no E, sampler.shutdown raises C | C propagates | over-suppression
 7 success+ok | no E, all ok | no raise, full order | control GREEN
 8 orig+ok | E, cleanups ok | E propagates | clean failure path
 9 order | no E, all ok | stats..tracker order | order regression
10 remote | remote_run=True | stats/finish/sampler/reap skip | remote branch
11 no-job-id | engine_core_job_id=None | reap skipped | reap condition

Oracle = Python exception semantics (which type propagates) + recorded call order.
Fakes are unit fixtures locating the defect; reverse-verify removes suppression and 1-5 go RED.
"""

from types import SimpleNamespace as NS

import pytest

from swift.dev.recipe.grpo.run_grpo import _finalize_grpo_resources


class OrigError(Exception):
    pass


class CleanupError(Exception):
    pass


def _build(raisers=None):
    raisers = raisers or {}
    calls = []

    def mk(label, ret=None):
        def fn(*a, **k):
            calls.append(label)
            if label in raisers:
                raise raisers[label]
            return ret

        return fn

    state = NS(finish_rollout_metrics=mk("finish"), tracker=NS(close=mk("tracker")))
    driver = NS(stats=mk("stats", ret={"s": 1}))
    cluster = NS(rollout=NS(close=mk("rollout")), sampler=NS(shutdown=mk("sampler")))
    return state, driver, cluster, mk("perf"), mk("reap"), calls


def _finalize(state, driver, cluster, perf, reap, *, remote_run=False, job_id="job123"):
    _finalize_grpo_resources(
        state, driver, cluster, remote_run=remote_run, engine_core_job_id=job_id, perf_shutdown=perf, reap=reap
    )


def _with_orig(fn):
    try:
        raise OrigError("training failed")
    finally:
        fn()


@pytest.mark.parametrize("label", ["stats", "finish", "rollout", "tracker"])
def test_original_exception_preserved_when_cleanup_raises(label):
    args = _build({label: CleanupError(f"{label} cleanup died")})
    with pytest.raises(OrigError):
        _with_orig(lambda: _finalize(*args[:5]))
    # later steps still ran despite the failing cleanup
    assert "tracker" in args[5]
    assert "perf" in args[5]


def test_original_exception_preserved_when_all_cleanups_raise():
    raisers = {k: CleanupError(k) for k in ["stats", "finish", "perf", "rollout", "sampler", "reap", "tracker"]}
    args = _build(raisers)
    with pytest.raises(OrigError):
        _with_orig(lambda: _finalize(*args[:5]))
    # every guarded step was attempted even though each raised
    for label in ["perf", "rollout", "sampler", "reap", "tracker"]:
        assert label in args[5]


def test_success_path_cleanup_error_still_propagates():
    args = _build({"sampler": CleanupError("sampler teardown died")})
    with pytest.raises(CleanupError):
        _finalize(*args[:5])


def test_success_path_all_ok_control():
    args = _build()
    _finalize(*args[:5])  # must not raise
    assert args[5] == ["stats", "finish", "perf", "rollout", "sampler", "reap", "tracker"]


def test_orig_error_cleanups_ok_control():
    args = _build()
    with pytest.raises(OrigError):
        _with_orig(lambda: _finalize(*args[:5]))


def test_cleanup_order():
    args = _build()
    _finalize(*args[:5])
    assert args[5] == ["stats", "finish", "perf", "rollout", "sampler", "reap", "tracker"]


def test_remote_run_skips_local_steps():
    args = _build()
    _finalize(*args[:5], remote_run=True)
    assert args[5] == ["perf", "rollout", "tracker"]


def test_reap_skipped_without_job_id():
    args = _build()
    _finalize(*args[:5], job_id=None)
    assert "reap" not in args[5]
    assert args[5] == ["stats", "finish", "perf", "rollout", "sampler", "tracker"]
