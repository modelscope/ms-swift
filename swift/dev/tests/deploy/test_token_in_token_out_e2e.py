# Copyright (c) ModelScope Contributors. All rights reserved.
"""token-in-token-out sampler surface of ``swift deploy`` (requirement 6).

Drives the *same* ``twinkle_client`` path ``examples/v5/deploy/token_in_token_out_client.py`` uses, so a
green suite means the example runs as written. Raw token ids go in; each sequence comes back with a
``new_input_feature`` -- the full prompt+completion as trainable features -- and the assertions here are
about **field correctness** (``input_ids`` / ``labels`` / ``completion_mask`` all present and the same
length) and **offset correctness** (``input_ids`` = prompt then completion; the trainable positions are
exactly the completion; their ``labels`` are the completion tokens). The multi-turn case feeds turn one's
whole output back as turn two's prompt and checks the prior turns become an exact, non-trainable prefix.

The server is started with ``enable_data_plane`` so both sampler paths the example shows are live: the
lightweight ``sample`` and the RL ``asample_to_data_plane`` rollout that keeps the generated group
server-side and hands back a ``DataRef``.
"""
from __future__ import annotations

import asyncio
import time

import pytest
from twinkle.data_format import InputFeature

from swift.dev.tests.deploy.conftest import SERVED_NAME, encode_prompt_ids

pytestmark = [pytest.mark.slow, pytest.mark.accel(1)]

#: A short, deterministic-ish prompt: the model reliably emits a few tokens, so the completion is non-empty.
_PROMPT_MESSAGES = [{'role': 'user', 'content': 'Count from 1 to 5.'}]
#: Mirrors the example's sampling params; ``logprobs`` exercises the per-token logprob path too.
_SAMPLING = {'max_tokens': 16, 'temperature': 1.0, 'logprobs': 1}


@pytest.fixture(scope='module')
def server(live_server):
    # route_prefix MUST be /api/v1: twinkle_client's get_base_url() appends '/api/v1' to whatever base_url
    # it is given, so the gateway has to be mounted there or the client dials a doubled '/v1/api/v1/...' and
    # 404s. enable_data_plane derives sampler_type=vllm_async, which serves both the lightweight sample and
    # the data-plane rollout; the plain sample path is unchanged on the async sampler.
    with live_server(
            backend='vllm', deploy_overrides={
                'enable_data_plane': True,
                'route_prefix': '/api/v1',
            }) as srv:
        yield srv


@pytest.fixture(scope='module')
def sampler(server):
    """A ``vLLMSampler`` bound to the live deployment, through the same global client the example builds.

    ``run_deploy_process`` yields once the *gateway* answers ``/models``, but the sampler replica (which
    loads the checkpoint and starts vLLM) registers its own HTTP route only once it is healthy -- a minute
    or so later. ``vLLMSampler.__init__`` dials that route directly (``{base}/sampler/{model}/twinkle/create``)
    with no retry, so constructing it immediately would race a 404. Poll the construction until the sampler
    route goes live, exactly as a patient client would.
    """
    from twinkle_client import init_twinkle_client
    from twinkle_client.sampler import vLLMSampler
    client = init_twinkle_client(base_url=server.base_url, api_key=server.api_key)
    try:
        deadline = time.time() + 600.0
        last_error: Exception | None = None
        connected = None
        while time.time() < deadline:
            try:
                connected = vLLMSampler(SERVED_NAME)
                break
            except Exception as error:  # noqa: BLE001 - the sampler 404s/503s until its replica is healthy
                last_error = error
                time.sleep(5.0)
        if connected is None:
            raise RuntimeError(f'the sampler route never came up under {server.base_url}') from last_error
        yield connected
    finally:
        client.close()


def _sample_one(sampler, prompt_ids: list):
    """One prompt in, one sequence out -- the lightweight token-in-token-out call the example makes."""
    responses = sampler.sample([InputFeature(input_ids=list(prompt_ids))], sampling_params=dict(_SAMPLING))
    assert len(responses) == 1
    assert len(responses[0].sequences) == 1
    return responses[0].sequences[0]


def _assert_offsets(feature: dict, prompt_ids: list, tokens: list) -> None:
    """Field + offset correctness of one ``new_input_feature`` against the prompt fed and tokens sampled."""
    for key in ('input_ids', 'labels', 'completion_mask'):
        assert key in feature, f'new_input_feature is missing {key!r}'
    prompt_len, completion_len = len(prompt_ids), len(tokens)
    input_ids, labels, mask = feature['input_ids'], feature['labels'], feature['completion_mask']
    # All three fields span the whole sequence: prompt then completion.
    assert len(input_ids) == len(labels) == len(mask) == prompt_len + completion_len
    assert input_ids[:prompt_len] == list(prompt_ids)
    assert input_ids[prompt_len:] == list(tokens)
    # The trainable positions are exactly the completion, and completion_mask marks precisely the
    # positions whose label is a real token (not the -100 ignore index).
    assert sum(mask) == completion_len
    assert list(mask) == [0 if label == -100 else 1 for label in labels]
    assert [label for label, m in zip(labels, mask) if m == 1] == list(tokens)
    # The final position generates nothing trainable, so its label is the ignore index.
    assert labels[-1] == -100


def test_lightweight_sample_returns_aligned_trainable_features(sampler):
    prompt_ids = encode_prompt_ids(_PROMPT_MESSAGES)
    sequence = _sample_one(sampler, prompt_ids)
    assert len(sequence.tokens) > 0, 'the model produced no completion tokens to check offsets against'
    assert sequence.stop_reason in ('length', 'stop')
    _assert_offsets(sequence.new_input_feature, prompt_ids, sequence.tokens)


def test_multi_turn_token_in_token_out_preserves_offsets(sampler):
    # Turn 1: a fresh prompt.
    prompt_ids = encode_prompt_ids(_PROMPT_MESSAGES)
    seq1 = _sample_one(sampler, prompt_ids)
    assert len(seq1.tokens) > 0
    turn1_ids = seq1.new_input_feature['input_ids']
    assert turn1_ids == list(prompt_ids) + list(seq1.tokens)

    # Turn 2: feed turn 1's entire output back as raw token ids -- a real loop's next-turn continuation.
    seq2 = _sample_one(sampler, turn1_ids)
    assert len(seq2.tokens) > 0
    feature2 = seq2.new_input_feature
    # Turn 1 is now an exact prefix of turn 2's sequence, and only turn 2's completion is trainable:
    # the earlier turn became context. This is the offset invariant a multi-turn RL loop relies on.
    assert feature2['input_ids'][:len(turn1_ids)] == turn1_ids
    assert sum(feature2['completion_mask']) == len(seq2.tokens)
    _assert_offsets(feature2, turn1_ids, seq2.tokens)


def test_data_plane_rollout_stores_group_and_accepts_rewards(sampler):
    """RL path: keep the generated group server-side, read it back, append the training targets, release."""
    from twinkle_client import DataPlaneClient
    prompt_ids = encode_prompt_ids(_PROMPT_MESSAGES)

    async def _rollout():
        data_plane = DataPlaneClient()
        ref = await sampler.asample_to_data_plane(
            [InputFeature(input_ids=list(prompt_ids))],
            sampling_params={'max_tokens': 16, 'temperature': 1.0, 'logprobs': 1},
            num_samples=2,
            group_ids=['group-0'],
        )
        rows = await data_plane.aget(ref, fields=['decoded', 'tokens', 'sampled_logprobs'])
        # num_samples sequences were generated and stored server-side; only the ref crossed the wire.
        assert len(rows) == 2
        assert all('decoded' in row for row in rows)

        # Score on the client, then append the per-sequence training targets back onto the same ref.
        rewards = [1.0 if (row.get('decoded') or '').strip() else 0.0 for row in rows]
        mean = sum(rewards) / len(rewards)
        scored_ref = await data_plane.aappend(
            ref, [{'reward': float(r), 'advantage': float(r - mean)} for r in rewards])
        scored = await data_plane.aget(scored_ref, fields=['reward', 'advantage'])
        assert len(scored) == 2
        assert all('reward' in row and 'advantage' in row for row in scored)
        await data_plane.arelease(scored_ref)

    asyncio.run(_rollout())
