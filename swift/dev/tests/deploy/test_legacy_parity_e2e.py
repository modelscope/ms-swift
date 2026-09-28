# Copyright (c) ModelScope Contributors. All rights reserved.
"""Capability parity between the twinkle-server-backed ``swift deploy`` and legacy ``SwiftDeploy`` (req. 4).

Legacy ``swift deploy`` (``swift/pipelines/infer/deploy.py`` -> ``SwiftDeploy``) hand-rolled a FastAPI app
whose response objects were the dataclasses in ``swift/infer_engine/protocol.py``. The dev rewrite serves
through twinkle-server's gateway instead, so this module checks the new deployment still honours the
*same OpenAI contract a legacy client depends on* -- and pins the places it intentionally diverges.

The comparison is tied to the legacy code rather than to prose: the discriminators and required fields are
read straight off the legacy dataclasses (``ChatCompletionResponse`` / ``CompletionResponse`` / ``Model`` /
``ModelList`` / ``UsageInfo`` / ``ChatCompletionStreamResponse``), so if the legacy contract is what a
client was written against, these assertions are exactly what that client still gets.

Known, deliberate divergences (the new gateway follows OpenAI convention where legacy did not):

- **auth failure**: legacy ``_check_api_key`` folded a bad key into ``create_error_response(BAD_REQUEST)``
  -> HTTP **400** with ``{'message', 'object': 'error'}``; the gateway returns HTTP **401** with an
  ``{'error': {'type': 'authentication_error', ...}}`` body.
- **unknown model**: legacy ``_check_model`` rejected any model not in the list with **400**; the gateway,
  with a single served model, routes any ``model`` to it (a documented fallback).
- **owned_by**: legacy ``Model.owned_by`` defaulted to ``'ms-swift'``; the gateway reports ``'swift'``.
- **route mount**: legacy exposed ``/health`` and ``/infer/`` at the root; the gateway mounts everything
  under ``route_prefix`` (``/v1`` here), so it is ``/v1/health`` and ``/v1/infer``.
"""
from __future__ import annotations

import dataclasses
import json

import pytest
from swift.infer_engine.protocol import (ChatCompletionResponse, ChatCompletionStreamResponse, ChatMessage,
                                         CompletionResponse, Model, ModelList, UsageInfo)

from swift.dev.tests.deploy.conftest import SERVED_NAME

pytestmark = [pytest.mark.slow, pytest.mark.accel(1)]

#: The OpenAI-standard fields legacy always emitted and a legacy client reads back. Each is asserted to be
#: a genuine legacy dataclass field first, so this set cannot silently drift from the legacy contract.
_MODEL_CORE = {'id', 'object', 'created', 'owned_by'}
_CHAT_CORE = {'object', 'id', 'created', 'model', 'choices', 'usage'}
_CMPL_CORE = {'object', 'id', 'created', 'model', 'choices', 'usage'}
_CHOICE_CORE = {'index', 'message', 'finish_reason'}
_MESSAGE_CORE = {'role', 'content'}


@pytest.fixture(scope='module')
def server(live_server):
    """One plain vLLM deployment (no data plane) shared by every parity check in this module."""
    with live_server(backend='vllm') as srv:
        yield srv


def _legacy_default(dataclass_type, field_name: str):
    """The legacy dataclass's hardcoded default for ``field_name`` (its discriminator value)."""
    return dataclass_type.__dataclass_fields__[field_name].default


def _legacy_fields(dataclass_type) -> set:
    return {field.name for field in dataclasses.fields(dataclass_type)}


# --- model listing --------------------------------------------------------------------


def test_model_list_matches_legacy_schema(server):
    """/models carries the legacy ``ModelList`` / ``Model`` shape, down to the legacy discriminators."""
    # The fields we require are proven to be legacy's own fields, so this is a real parity check.
    assert _MODEL_CORE <= _legacy_fields(Model)
    assert {'data', 'object'} <= _legacy_fields(ModelList)

    body = server.get('/models').json()
    assert body['object'] == _legacy_default(ModelList, 'object')  # legacy: 'list'
    assert SERVED_NAME in [m['id'] for m in body['data']]
    entry = next(m for m in body['data'] if m['id'] == SERVED_NAME)
    assert _MODEL_CORE <= set(entry)
    assert entry['object'] == _legacy_default(Model, 'object')  # legacy: 'model'
    assert isinstance(entry['created'], int)
    # Divergence: legacy Model.owned_by defaulted to 'ms-swift'; the gateway reports 'swift'.
    assert entry['owned_by'] == 'swift' != _legacy_default(Model, 'owned_by')


# --- chat completions -----------------------------------------------------------------


def test_chat_completion_matches_legacy_schema(server):
    """A non-streaming chat completion carries every field legacy's ``ChatCompletionResponse`` defined."""
    assert _CHAT_CORE <= _legacy_fields(ChatCompletionResponse)
    assert set(UsageInfo.__dataclass_fields__) >= {'prompt_tokens', 'completion_tokens', 'total_tokens'}

    body = server.chat([{'role': 'user', 'content': 'Say hello in one short sentence.'}], max_tokens=32)
    assert _CHAT_CORE <= set(body)
    assert body['object'] == _legacy_default(ChatCompletionResponse, 'object')  # legacy: 'chat.completion'
    assert body['id'].startswith('chatcmpl-')  # legacy id factory prefix
    assert body['model'] == SERVED_NAME
    # usage matches the legacy UsageInfo field-for-field
    assert _legacy_fields(UsageInfo) <= set(body['usage'])
    usage = body['usage']
    assert usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens']
    # choice + message match legacy ChatCompletionResponseChoice / ChatMessage cores
    choice = body['choices'][0]
    assert _CHOICE_CORE <= set(choice)
    assert choice['finish_reason'] in ('stop', 'length')
    assert _MESSAGE_CORE <= _legacy_fields(ChatMessage)
    assert _MESSAGE_CORE <= set(choice['message'])
    assert choice['message']['role'] == 'assistant'
    assert isinstance(choice['message']['content'], str) and choice['message']['content']


# --- text completions -----------------------------------------------------------------


def test_text_completion_matches_legacy_schema(server):
    """/completions carries the legacy ``CompletionResponse`` shape (``text_completion``, ``choices[].text``)."""
    assert _CMPL_CORE <= _legacy_fields(CompletionResponse)
    resp = server.post('/completions', {'model': SERVED_NAME, 'prompt': 'The capital of France is', 'max_tokens': 16})
    resp.raise_for_status()
    body = resp.json()
    assert _CMPL_CORE <= set(body)
    assert body['object'] == _legacy_default(CompletionResponse, 'object')  # legacy: 'text_completion'
    assert body['id'].startswith('cmpl-')  # legacy id factory prefix
    assert _legacy_fields(UsageInfo) <= set(body['usage'])
    choice = body['choices'][0]
    assert {'index', 'text', 'finish_reason'} <= set(choice)
    assert isinstance(choice['text'], str) and choice['finish_reason'] in ('stop', 'length')


# --- streaming ------------------------------------------------------------------------


def test_streaming_matches_legacy_schema(server):
    """Streaming frames carry the legacy ``chat.completion.chunk`` shape and legacy's ``data: [DONE]`` end."""
    assert _legacy_default(ChatCompletionStreamResponse, 'object') == 'chat.completion.chunk'
    lines = list(
        server.stream(
            '/chat/completions', {
                'model': SERVED_NAME,
                'messages': [{'role': 'user', 'content': 'Count from one to three.'}],
                'max_tokens': 32,
                'stream': True,
            }))
    # legacy's _gen_wrapper terminated the stream with exactly this sentinel
    assert lines[-1].strip() == 'data: [DONE]'
    chunks = [
        json.loads(line[len('data: '):]) for line in lines
        if line.startswith('data: ') and line.strip() != 'data: [DONE]'
    ]
    assert chunks, 'no streamed chunks'
    assert all(c['object'] == _legacy_default(ChatCompletionStreamResponse, 'object') for c in chunks)
    assert {'index', 'delta', 'finish_reason'} <= set(chunks[0]['choices'][0])
    text = ''.join(c['choices'][0]['delta'].get('content', '') for c in chunks)
    assert text


# --- intentional divergences ----------------------------------------------------------


def test_auth_failure_diverges_from_legacy_400_to_401(server):
    """Legacy folded a bad api_key into a 400 ``{'message','object':'error'}``; the gateway uses 401 + ``error``."""
    chat_body = {'model': SERVED_NAME, 'messages': [{'role': 'user', 'content': 'Hi'}], 'max_tokens': 8}
    resp = server.post('/chat/completions', chat_body, auth=False)
    # New behaviour: 401 with the OpenAI-style error envelope (legacy returned 400 with {'message','object'}).
    assert resp.status_code == 401
    error = resp.json()['error']
    assert error['type'] == 'authentication_error' and error['code'] == 'invalid_api_key'
    assert 'message' not in resp.json() and resp.json().get('object') != 'error'


def test_unknown_model_diverges_from_legacy_rejection(server):
    """Legacy ``_check_model`` 400-ed on a model absent from the list; the gateway falls back to the one model."""
    body = server.chat([{'role': 'user', 'content': 'Hi'}], model='not-a-legacy-known-model', max_tokens=8)
    assert body['object'] == _legacy_default(ChatCompletionResponse, 'object')
    assert body['choices'][0]['message']['content'] is not None
