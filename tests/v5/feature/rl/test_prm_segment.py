# Copyright (c) ModelScope Contributors. All rights reserved.
"""C2-G39 (QUEUE1005): the PRM delimiter segmentation predicate, guard and selection.

``swift/dev/reward/prm.py`` decides WHERE a PRM segment closes. The delimiter plugin's
predicate is ``delimiter in last_assistant_text(traj)``: a step closes the running segment
only when its newest assistant text carries the marker, turns without one accumulate.
``build_prm_segment`` selects the plugin by config and rejects a delimiter knob paired with
the 'turn' scorer.

Contract updated for eeedc364c ("ray-boundary plan batch"): ``materialize_traj`` is gone --
content-based predicates no longer resolve stashed references themselves. Instead
``build_prm_segment`` wraps EVERY resolved predicate (plugin-built or plain callable) in a
``PRMSegmentWorker`` remote class that evaluates the predicate at the compute boundary and
publishes only the boolean verdict; locally the worker call is synchronous and zero-ray.

PLAN (case | input | expected | failure it kills):
  closes_only_marked_turn   | assistant text with/without '\n' | True only when marker present | predicate closing every turn / never closing
  custom_delimiter          | prm_step_delimiter='###'          | True only for '###', not bare '\n' | custom delimiter ignored (hardcoded '\n')
  default_delimiter_variants| args None / knob None / ''        | delimiter falls back to '\n' | default not falling back
  last_assistant_only       | several assistant turns           | only the LAST turn's text decides | scanning any assistant turn
  guard_delimiter_turn      | prm_scorer=turn + delimiter knob  | ValueError | delimiter silently ignored under 'turn'
  guard_delimiter_delimiter | prm_scorer=delimiter + knob       | worker returned, verdict True | over-eager guard rejecting the valid combo
  build_prm_off             | reward_config None / prm=False    | None | a predicate returned when PRM is off
  build_default_by_shape    | spec None +/- max_turns           | turn->None; no max_turns->worker | wrong shape default
  build_callable_wrapped    | prm_scorer = plain callable       | worker carries the callable, verdict flows | callable dropped or bypassed
  worker_local_verdict      | worker(None, dict)                | bool verdict, no ray needed | worker unusable on the local path
  worker_bool_coercion      | predicate returns truthy non-bool | verdict is a real bool | non-bool leaking across the ray boundary
  single_turn_degenerate    | one assistant turn                | marker->True (one segment at done), else False | the documented single-turn limit

Reverse-verified (tools/probe_reverse_verify_g39.py): forcing the predicate constant-True,
removing the turn+delimiter guard, and hardcoding the delimiter each turn the matching case RED.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from swift.dev.reward.prm import DelimiterPRMScorer, PRMSegmentWorker, build_prm_segment


def _traj(*assistant_texts):
    messages = []
    for i, text in enumerate(assistant_texts):
        messages.append({'role': 'user', 'content': f'q{i}'})
        messages.append({'role': 'assistant', 'content': text})
    return {'messages': messages}


def _predicate(delimiter=None, args=None):
    if args is None:
        args = SimpleNamespace(prm_step_delimiter=delimiter)
    return DelimiterPRMScorer(args).build()


def _reward(prm=True, prm_scorer=None, prm_step_delimiter=None):
    return SimpleNamespace(prm=prm, prm_scorer=prm_scorer, prm_step_delimiter=prm_step_delimiter)


def test_closes_only_marked_turn():
    pred = _predicate()  # default '\n'
    assert pred(None, _traj('step one\n')) is True  # marker present -> closes
    assert pred(None, _traj('step one')) is False  # no marker -> accumulates


def test_custom_delimiter():
    pred = _predicate(delimiter='###')
    assert pred(None, _traj('reasoning###')) is True
    assert pred(None, _traj('reasoning\n')) is False  # bare newline is not the custom marker


@pytest.mark.parametrize('args', [None, SimpleNamespace(prm_step_delimiter=None),
                                  SimpleNamespace(prm_step_delimiter='')])
def test_default_delimiter_variants(args):
    pred = DelimiterPRMScorer(args).build()
    assert pred(None, _traj('done\n')) is True
    assert pred(None, _traj('done')) is False


def test_last_assistant_only():
    pred = _predicate()
    # An earlier turn carried the marker, but the newest did not -> the segment stays open.
    assert pred(None, _traj('first\n', 'second')) is False
    assert pred(None, _traj('first', 'second\n')) is True


def test_guard_delimiter_turn_rejected():
    with pytest.raises(ValueError, match='only shapes the delimiter'):
        build_prm_segment(_reward(prm_scorer='turn', prm_step_delimiter='\n'), SimpleNamespace(max_turns=4))


def test_guard_delimiter_delimiter_ok():
    worker = build_prm_segment(_reward(prm_scorer='delimiter', prm_step_delimiter='###'),
                               SimpleNamespace(max_turns=None))
    assert isinstance(worker, PRMSegmentWorker)
    assert worker(None, _traj('x###')) is True


def test_build_prm_off_returns_none():
    assert build_prm_segment(None, SimpleNamespace(max_turns=4)) is None
    assert build_prm_segment(_reward(prm=False), SimpleNamespace(max_turns=4)) is None


def test_build_default_by_shape():
    # multi-turn (max_turns set) -> 'turn' -> the driver default (None predicate)
    assert build_prm_segment(_reward(), SimpleNamespace(max_turns=4)) is None
    # single-turn shape -> 'delimiter' -> a worker around a real predicate
    worker = build_prm_segment(_reward(), SimpleNamespace(max_turns=None))
    assert isinstance(worker, PRMSegmentWorker)
    assert worker(None, _traj('a\n')) is True


def test_build_callable_wrapped():
    sentinel = lambda sample, traj: True  # noqa: E731 - a plain predicate plugin
    worker = build_prm_segment(_reward(prm_scorer=sentinel), SimpleNamespace(max_turns=None))
    # eeedc364c: a plain callable is no longer returned as-is; the worker carries it to the
    # compute boundary and only its boolean verdict crosses back.
    assert isinstance(worker, PRMSegmentWorker)
    assert worker.predicate is sentinel
    assert worker(None, _traj('anything')) is True


def test_worker_local_verdict():
    worker = build_prm_segment(_reward(), SimpleNamespace(max_turns=None))
    # local direct call: synchronous verdict on the in-process dict, no ray involvement
    assert worker(None, _traj('a\n')) is True
    assert worker(None, _traj('a')) is False


def test_worker_bool_coercion():
    truthy = lambda sample, traj: ['non-bool truthy']  # noqa: E731
    worker = build_prm_segment(_reward(prm_scorer=truthy), SimpleNamespace(max_turns=None))
    verdict = worker(None, _traj('a'))
    assert verdict is True  # bool() at the boundary, not the raw list


def test_single_turn_degenerate():
    pred = _predicate()
    assert pred(None, _traj('only turn\n')) is True  # one segment, closed at done
    assert pred(None, _traj('only turn')) is False  # no marker -> never closes mid-rollout