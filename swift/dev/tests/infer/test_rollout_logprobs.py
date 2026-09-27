# Copyright (c) ModelScope Contributors. All rights reserved.
"""Regression test for the single-turn old_logps alignment guard (no GPU, no model).

``run_infer``'s plain path drives ``RolloutEngine.generate(..., force_logprobs=False)``, so the sampler
returns ``sequence.logprobs=None`` (the transformers backend only computes logprobs when asked). The
single-turn ``samples_from_responses`` used to require old_logbs to align with the response tokens
unconditionally, which made a plain ``swift infer --infer_backend pt`` run crash with "rollout logprobs
misaligned". ``require_logprobs`` now gates that check: GRPO (which forces logprobs) keeps the fatal
guard, plain inference tolerates an empty old_logps.
"""
from types import SimpleNamespace

import pytest

from swift.dev.rollout import RolloutEngine


def _response(tokens, logprobs, prompt_tokens=(1, 2)):
    sequence = SimpleNamespace(tokens=list(tokens), logprobs=logprobs, decoded='x', stop_reason='stop')
    return SimpleNamespace(prompt_token_ids=list(prompt_tokens), sequences=[sequence])


def test_plain_infer_tolerates_missing_logprobs():
    """force_logprobs=False -> require_logprobs=False: logprobs=None yields an empty old_logps, no raise."""
    samples = RolloutEngine._samples_from_responses([_response([7, 8, 9], None)], require_logprobs=False)
    assert len(samples) == 1
    assert samples[0].old_logps == []  # no logprobs requested, so none carried
    assert samples[0].response_token_ids == [7, 8, 9]  # the tokens/training feature are still built
    assert samples[0].encoded['input_ids'] == [1, 2, 7, 8, 9]


def test_grpo_still_guards_missing_logprobs():
    """The default (require_logprobs=True, what GRPO uses) keeps the missing-logprob case fatal."""
    with pytest.raises(RuntimeError, match='logprobs misaligned'):
        RolloutEngine._samples_from_responses([_response([7, 8, 9], None)])
    # a short (partial) logprob list is equally fatal under the guard
    with pytest.raises(RuntimeError, match='logprobs misaligned'):
        RolloutEngine._samples_from_responses([_response([7, 8], [-0.5])])


def test_aligned_logprobs_pass_under_both_settings():
    """When logprobs do align, both settings accept them and old_logps is populated."""
    for require in (True, False):
        samples = RolloutEngine._samples_from_responses([_response([7, 8], [-0.5, -1.5])], require_logprobs=require)
        assert samples[0].old_logps == [-0.5, -1.5]
