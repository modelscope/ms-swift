# Copyright (c) ModelScope Contributors. All rights reserved.
"""C2-G39 (QUEUE1005): the PRM delimiter segmentation predicate, guard and selection.

``swift/dev/rewards/prm.py`` decides WHERE a PRM segment closes. The delimiter plugin's
predicate is ``delimiter in last_assistant_text(materialize_traj(traj))``: a step closes the
running segment only when its newest assistant text carries the marker, turns without one
accumulate. ``build_prm_segment`` selects the plugin by config and rejects a delimiter knob
paired with the 'turn' scorer. The ray materialization integration face was signed separately
(task#135); this matrix covers the pure predicate/guard/selection/local-zero-cost face.

PLAN (case | input | expected | failure it kills):
  closes_only_marked_turn   | assistant text with/without '\\n' | True only when marker present | predicate closing every turn / never closing
  custom_delimiter          | prm_step_delimiter='###'          | True only for '###', not bare '\\n' | custom delimiter ignored (hardcoded '\\n')
  default_delimiter_variants| args None / knob None / ''        | delimiter falls back to '\\n' | default not falling back
  last_assistant_only       | several assistant turns           | only the LAST turn's text decides | scanning any assistant turn
  guard_delimiter_turn      | prm_scorer=turn + delimiter knob  | ValueError | delimiter silently ignored under 'turn'
  guard_delimiter_delimiter | prm_scorer=delimiter + knob       | predicate returned, no raise | over-eager guard rejecting the valid combo
  build_prm_off             | reward_config None / prm=False    | None | a predicate returned when PRM is off
  build_default_by_shape    | spec None +/- max_turns           | turn->None predicate; no max_turns->delimiter callable | wrong shape default
  build_callable_passthrough| prm_scorer = plain callable       | returned as-is | callable not passed through
  materialize_dict_identity | a dict                            | the same dict, no ray | local direct call paying ray.get
  materialize_lazycollect   | wrapper with .result()            | the resolved dict | LazyCollect wrapper not unwrapped
  single_turn_degenerate    | one assistant turn                | marker->True (one segment at done), else False | the documented single-turn limit

Reverse-verified (tools/probe_reverse_verify_g39.py): forcing the predicate constant-True,
removing the turn+delimiter guard, and hardcoding the delimiter each turn the matching case RED.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from swift.dev.rewards.prm import (DelimiterPRMScorer, TurnPRMScorer, build_prm_segment,
                                   materialize_traj)


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
    pred = build_prm_segment(_reward(prm_scorer='delimiter', prm_step_delimiter='###'),
                             SimpleNamespace(max_turns=None))
    assert callable(pred)
    assert pred(None, _traj('x###')) is True


def test_build_prm_off_returns_none():
    assert build_prm_segment(None, SimpleNamespace(max_turns=4)) is None
    assert build_prm_segment(_reward(prm=False), SimpleNamespace(max_turns=4)) is None


def test_build_default_by_shape():
    # multi-turn (max_turns set) -> 'turn' -> the driver default (None predicate)
    assert build_prm_segment(_reward(), SimpleNamespace(max_turns=4)) is None
    # single-turn shape -> 'delimiter' -> a real predicate
    pred = build_prm_segment(_reward(), SimpleNamespace(max_turns=None))
    assert callable(pred)
    assert pred(None, _traj('a\n')) is True


def test_build_callable_passthrough():
    sentinel = lambda sample, traj: True  # noqa: E731 - a plain predicate plugin
    assert build_prm_segment(_reward(prm_scorer=sentinel), SimpleNamespace(max_turns=None)) is sentinel


def test_materialize_dict_identity():
    traj = _traj('a\n')
    assert materialize_traj(traj) is traj  # local direct call: zero cost, no ray


def test_materialize_lazycollect():
    traj = _traj('a\n')

    class Lazy:
        def result(self):
            return traj

    assert materialize_traj(Lazy()) is traj


def test_single_turn_degenerate():
    pred = _predicate()
    assert pred(None, _traj('only turn\n')) is True  # one segment, closed at done
    assert pred(None, _traj('only turn')) is False  # no marker -> never closes mid-rollout