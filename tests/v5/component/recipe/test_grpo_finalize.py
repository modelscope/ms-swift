"""defect499: RL teardown must not mask the original training exception.

Contract updated for eeedc364c ("ray-boundary plan batch"): GRPO's private
``_finalize_grpo_resources`` was replaced by the shared
``swift.dev.recipe.utils.finalize_rl_resources``. The defect499 guarantee survives unchanged --
an in-flight training exception wins and every cleanup step still runs -- while the success
path now aggregates ALL teardown failures into one ``BaseExceptionGroup`` instead of letting
the first one propagate. The old stats/reap/engine_core_job_id steps are gone: driver.close
replaces the stats RPC, Perf.shutdown is called directly, and remote_run only skips the
server-owned sampler.

PLAN (case | input | expected | kills):
 1 orig+one-raise   | E in flight, one cleanup raises C | E propagates, later steps run | cleanup masking E
 2 orig+all-raise   | E, every cleanup raises | E propagates, all attempted | cumulative masking
 3 success+raise    | no E, sampler.close raises C | BaseExceptionGroup([C]) | over-suppression
 4 success+multi    | no E, two cleanups raise | group carries BOTH errors | errors[0] dropping the rest
 5 success+ok       | no E, all ok | no raise, full order | control GREEN
 6 orig+ok          | E, cleanups ok | E propagates | clean failure path
 7 order            | no E, all ok | finish..tracker order | order regression
 8 remote           | remote_run=True | sampler.close skipped | closing a server-owned sampler
 9 alias-dedup      | same handle in two slots | closed once | double-close of an aliased handle
10 none-slots       | None handles | skipped, no raise | None.close crash

Oracle = Python exception semantics (which type propagates) + recorded call order.
Fakes are unit fixtures locating the defect; reverse-verify removes suppression and 1-2 go RED.
"""

from types import SimpleNamespace as NS
from typing import Any, NamedTuple

import pytest

from swift.dev.recipe.utils import finalize_rl_resources
from twinkle.perf import Perf


class OrigError(Exception):
    pass


class CleanupError(Exception):
    pass


class Cluster(NamedTuple):
    model: Any
    sampler: Any
    rollout: Any
    engine: Any


_FULL_ORDER = ['finish', 'driver', 'perf', 'engine', 'rollout', 'model', 'sampler', 'tracker']


def _build(monkeypatch, raisers=None):
    raisers = raisers or {}
    calls = []

    def mk(label):
        def fn(*a, **k):
            calls.append(label)
            if label in raisers:
                raise raisers[label]

        return fn

    def handle(label):
        return NS(close=mk(label))

    # finalize_rl_resources calls twinkle.perf.Perf.shutdown directly; record it in the same log.
    monkeypatch.setattr(Perf, 'shutdown', staticmethod(mk('perf')))
    state = NS(tracker=NS(close=mk('tracker')))
    driver = NS(close=mk('driver'))
    cluster = Cluster(
        model=handle('model'), sampler=handle('sampler'), rollout=handle('rollout'), engine=handle('engine'))
    return state, driver, cluster, mk('finish'), calls


def _finalize(args, *, remote_run=False):
    state, driver, cluster, finish, calls = args
    finalize_rl_resources(state, cluster, remote_run=remote_run, finish=finish, driver=driver, recipe='GRPO')


def _with_orig(fn):
    try:
        raise OrigError('training failed')
    finally:
        fn()


@pytest.mark.parametrize('label', ['finish', 'driver', 'perf', 'engine', 'rollout', 'model', 'sampler', 'tracker'])
def test_original_exception_preserved_when_cleanup_raises(label, monkeypatch):
    args = _build(monkeypatch, {label: CleanupError(f'{label} cleanup died')})
    with pytest.raises(OrigError):
        _with_orig(lambda: _finalize(args))
    # later steps still ran despite the failing cleanup
    assert 'tracker' in args[4]
    assert 'perf' in args[4]


def test_original_exception_preserved_when_all_cleanups_raise(monkeypatch):
    labels = ['finish', 'driver', 'perf', 'engine', 'rollout', 'model', 'sampler', 'tracker']
    args = _build(monkeypatch, {k: CleanupError(k) for k in labels})
    with pytest.raises(OrigError):
        _with_orig(lambda: _finalize(args))
    assert args[4] == _FULL_ORDER  # every guarded step was attempted even though each raised


def test_success_path_cleanup_error_still_propagates(monkeypatch):
    err = CleanupError('sampler teardown died')
    args = _build(monkeypatch, {'sampler': err})
    with pytest.raises(BaseExceptionGroup) as caught:
        _finalize(args)
    assert caught.value.exceptions == (err, )


def test_success_path_group_carries_every_failure(monkeypatch):
    e1, e2 = CleanupError('engine died'), CleanupError('tracker died')
    args = _build(monkeypatch, {'engine': e1, 'tracker': e2})
    with pytest.raises(BaseExceptionGroup) as caught:
        _finalize(args)
    assert caught.value.exceptions == (e1, e2)
    assert 'tracker' in args[4]  # the later cleanup ran despite the earlier failure


def test_success_path_all_ok_control(monkeypatch):
    args = _build(monkeypatch)
    _finalize(args)  # must not raise
    assert args[4] == _FULL_ORDER


def test_orig_error_cleanups_ok_control(monkeypatch):
    args = _build(monkeypatch)
    with pytest.raises(OrigError):
        _with_orig(lambda: _finalize(args))


def test_cleanup_order(monkeypatch):
    args = _build(monkeypatch)
    _finalize(args)
    assert args[4] == _FULL_ORDER


def test_remote_run_skips_server_owned_sampler(monkeypatch):
    args = _build(monkeypatch)
    _finalize(args, remote_run=True)
    assert args[4] == [step for step in _FULL_ORDER if step != 'sampler']


def test_aliased_handle_closed_once(monkeypatch):
    args = _build(monkeypatch)
    state, driver, cluster, finish, calls = args
    aliased = cluster._replace(engine=cluster.sampler)  # engine and sampler are the SAME object
    finalize_rl_resources(state, aliased, remote_run=False, finish=finish, driver=driver, recipe='GRPO')
    assert calls.count('sampler') == 1
    assert 'engine' not in calls


def test_none_handles_skipped(monkeypatch):
    args = _build(monkeypatch)
    state, driver, cluster, finish, calls = args
    sparse = cluster._replace(engine=None, model=None)
    finalize_rl_resources(state, sparse, remote_run=False, finish=finish, driver=driver, recipe='GRPO')
    assert calls == [step for step in _FULL_ORDER if step not in ('engine', 'model')]