# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end OpenAI-surface coverage for ``swift deploy`` (real Qwen3.5-4B + vLLM + Ray Serve).

One server is brought up for the whole module (``run_deploy_process`` really spawns ``run_deploy``: a
local Ray head, Ray Serve, and a vLLM engine on the real checkpoint), then every request goes over HTTP
through twinkle's gateway -- the exact path an OpenAI SDK client and ``examples/v5/deploy`` use. A green
module therefore means the served surface works as documented, not that a stub returned a canned shape.

The weights are the real checkpoint but the assertions are about *shape and contract*, not what the model
said: correct ``object`` discriminators, the choice/usage/message fields OpenAI clients read, the auth
and error status codes, and the token-level fields ``/infer`` hands a rollout.

All tests are ``@pytest.mark.slow`` + ``@pytest.mark.accel(1)``; run with ``-m slow``.
"""
import json

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.accel(1)]

#: The served name / api_key the conftest deploys with; the client's ``model`` must equal the former.
SERVED = 'policy'
API_KEY = 'EMPTY'


@pytest.fixture(scope='module')
def server(live_server):
    """One plain vLLM deployment (no data plane) shared by every test in this module."""
    with live_server(backend='vllm') as srv:
        yield srv


# --- liveness -------------------------------------------------------------------------


def test_health_and_ping(server):
    """/health and /ping (GET and POST) answer 200 with an empty body once the gateway is up."""
    for method, path in (('GET', '/health'), ('GET', '/ping'), ('POST', '/ping')):
        resp = server.get(path) if method == 'GET' else server.post(path, {})
        assert resp.status_code == 200, (path, resp.text)
        assert resp.json() == {}


# --- model listing --------------------------------------------------------------------


def test_models_lists_served_name(server):
    """/models is the OpenAI model-list shape, carrying the served name and ``owned_by``."""
    body = server.get('/models').json()
    assert body['object'] == 'list'
    ids = [m['id'] for m in body['data']]
    assert SERVED in ids
    entry = next(m for m in body['data'] if m['id'] == SERVED)
    assert entry['object'] == 'model' and entry['owned_by'] == 'swift'


# --- chat completions -----------------------------------------------------------------


def test_chat_completion_non_streaming(server):
    """A non-streaming chat completion is a ``chat.completion`` with an assistant message and usage."""
    body = server.chat([{'role': 'user', 'content': 'Say hello in one short sentence.'}], max_tokens=32)
    assert body['object'] == 'chat.completion'
    assert body['model'] == SERVED and body['id'].startswith('chatcmpl-')
    choice = body['choices'][0]
    assert choice['index'] == 0
    assert choice['message']['role'] == 'assistant'
    assert isinstance(choice['message']['content'], str) and choice['message']['content']
    assert choice['finish_reason'] in ('stop', 'length')
    usage = body['usage']
    assert usage['prompt_tokens'] > 0 and usage['completion_tokens'] > 0
    assert usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens']


def test_chat_completion_streaming(server):
    """A streaming chat completion is a sequence of ``chat.completion.chunk`` SSE frames ending in [DONE]."""
    lines = list(server.stream('/chat/completions', {
        'model': SERVED,
        'messages': [{'role': 'user', 'content': 'Count from one to three.'}],
        'max_tokens': 32,
        'stream': True,
    }))
    assert lines[-1].strip() == 'data: [DONE]'
    chunks = [json.loads(line[len('data: '):]) for line in lines if line.startswith('data: ')
              and line.strip() != 'data: [DONE]']
    assert chunks, 'no streamed chunks'
    assert all(c['object'] == 'chat.completion.chunk' for c in chunks)
    # the first frame announces the assistant role; content deltas accumulate to a non-empty completion
    assert chunks[0]['choices'][0]['delta'].get('role') == 'assistant'
    text = ''.join(c['choices'][0]['delta'].get('content', '') for c in chunks)
    assert text


# --- text completions -----------------------------------------------------------------


def test_text_completion(server):
    """/completions is the ``text_completion`` shape: the prompt is served as one user turn."""
    resp = server.post('/completions', {'model': SERVED, 'prompt': 'The capital of France is', 'max_tokens': 16})
    resp.raise_for_status()
    body = resp.json()
    assert body['object'] == 'text_completion' and body['id'].startswith('cmpl-')
    choice = body['choices'][0]
    assert isinstance(choice['text'], str) and choice['finish_reason'] in ('stop', 'length')
    # OpenAI completion logprobs are deliberately null here; /infer carries the raw per-token logprobs.
    assert choice['logprobs'] is None


# --- rollout /infer (token-in-token-out over HTTP) ------------------------------------


def test_infer_returns_token_level_records(server):
    """/infer returns per-prompt token ids, logprobs and each sequence's ``new_input_feature``."""
    resp = server.post('/infer', {
        'infer_requests': [{'messages': [{'role': 'user', 'content': 'Hi'}]}],
        'request_config': {
            'model': SERVED,
            'max_tokens': 16,
            'temperature': 0.0
        },
    })
    resp.raise_for_status()
    records = resp.json()
    assert isinstance(records, list) and len(records) == 1
    record = records[0]
    assert record['prompt_token_ids'], 'the served prompt token ids must come back'
    seq = record['sequences'][0]
    assert seq['tokens'], 'generated token ids must come back'
    assert seq['logprobs'] is not None and len(seq['logprobs']) == len(seq['tokens'])
    assert seq['finish_reason'] in ('stop', 'length')
    feature = seq['new_input_feature']
    assert feature and feature.get('input_ids'), 'new_input_feature must carry the full prompt+completion ids'


# --- auth -----------------------------------------------------------------------------


def test_api_key_gates_inference_routes(server):
    """With an api_key configured, the inference routes reject a missing or wrong Bearer token with a
    401 and accept the right one. ``/models`` is deliberately left unauthenticated (the OpenAI
    convention: discovery is open, generation is gated), so it stays reachable without a key."""
    import httpx
    chat_body = {'model': SERVED, 'messages': [{'role': 'user', 'content': 'Hi'}], 'max_tokens': 8}
    no_auth = server.post('/chat/completions', chat_body, auth=False)
    assert no_auth.status_code == 401
    assert no_auth.json()['error']['type'] == 'authentication_error'
    assert no_auth.json()['error']['code'] == 'invalid_api_key'
    wrong = httpx.post(
        server.url('/chat/completions'),
        json=chat_body,
        headers={'Authorization': 'Bearer not-the-key'},
        timeout=60.0,
    )
    assert wrong.status_code == 401
    # the configured key is accepted, and /models stays open without any key
    assert server.post('/chat/completions', chat_body).status_code == 200
    assert httpx.get(server.url('/models'), timeout=30.0).status_code == 200


# --- errors ---------------------------------------------------------------------------


def test_bad_request_on_missing_fields(server):
    """A translator ValueError becomes a 400 naming the offending field, not a 500."""
    no_messages = server.post('/chat/completions', {'model': SERVED})
    assert no_messages.status_code == 400
    assert 'messages' in no_messages.json()['error']['message']
    no_prompt = server.post('/completions', {'model': SERVED})
    assert no_prompt.status_code == 400
    assert 'prompt' in no_prompt.json()['error']['message']


def test_unknown_model_falls_back_to_the_single_served_model(server):
    """With exactly one supported model, the gateway routes any ``model`` to it (documented fallback)
    rather than 404-ing -- a 404 needs two or more supported models to disambiguate against."""
    body = server.chat([{'role': 'user', 'content': 'Hi'}], model='not-a-real-model', max_tokens=8)
    assert body['object'] == 'chat.completion' and body['choices'][0]['message']['content'] is not None


def test_embeddings_is_unsupported_on_a_generation_model(server):
    """/embeddings proxies to the sampler's pooling forward; a generation-only sampler has no pooling
    head, so the failure is surfaced as an error status rather than crashing the gateway."""
    resp = server.post('/embeddings', {'model': SERVED, 'input': 'hello'})
    assert resp.status_code != 200
    assert 'error' in resp.json()
