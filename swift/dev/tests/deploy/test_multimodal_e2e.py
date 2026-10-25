# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end multimodal (vision-language) coverage for ``swift deploy``'s OpenAI surface.

``Qwen3.5-4B`` is a unified vision-language checkpoint, so the same deployment that serves text also
serves image-text. These tests drive it over HTTP through twinkle's gateway -- the exact path an OpenAI
SDK client uses -- and assert the two things that prove an image really reached the model rather than
being silently dropped:

1. The request succeeds and returns the OpenAI ``chat.completion`` shape (a VL model handed an image
   placeholder with no pixels raises on the token/patch mismatch, so a clean completion means the image
   tensors reached ``generate``).
2. The image measurably entered the prompt: ``usage.prompt_tokens`` for the image request exceeds the
   *same* text prompt sent without the image, and ``/infer``'s ``prompt_token_ids`` for the image request
   carries vision-marker token ids absent from the text-only prompt. Qwen VL collapses each image's
   ``<vision_start><image_pad>...<vision_end>`` run to a single pad in the returned ids, so the delta is
   the marker tokens rather than the expanded patch count -- but it is nonzero only if the image was
   tokenized into the prompt, which is exactly what a dropped image would fail to do.

Both the OpenAI-standard content-parts shape (``{"type": "image_url", "image_url": {"url": ...}}``) and
twinkle's native shape (a string ``<image>`` placeholder plus a per-message ``images`` list) are covered,
because both are valid ways a caller attaches an image to a chat message.

All tests are ``@pytest.mark.slow`` + ``@pytest.mark.accel(1)``; run with ``-m slow``.
"""
import pytest

from swift.dev.tests.deploy.conftest import image_data_url

pytestmark = [pytest.mark.slow, pytest.mark.accel(1)]

SERVED = 'policy'
#: One shared prompt so the only difference between the text and image requests is the image itself.
PROMPT_TEXT = 'Describe this image in one short sentence.'


@pytest.fixture(scope='module')
def server(live_server):
    """One plain vLLM deployment of the VL checkpoint, shared by every test in this module."""
    with live_server(backend='vllm') as srv:
        yield srv


def _openai_vision_messages(data_url: str):
    """The OpenAI-standard vision shape: a text part and an ``image_url`` part in one user turn."""
    return [{
        'role': 'user',
        'content': [{
            'type': 'text',
            'text': PROMPT_TEXT
        }, {
            'type': 'image_url',
            'image_url': {
                'url': data_url
            }
        }],
    }]


def _native_vision_messages(data_url: str):
    """twinkle's native shape: a string ``<image>`` placeholder plus a parallel per-message images list."""
    return [{'role': 'user', 'content': f'<image>{PROMPT_TEXT}', 'images': [data_url]}]


def _infer(server, messages, max_tokens=16):
    """One /infer call, returning the single record's prompt_token_ids and first sequence."""
    resp = server.post('/infer', {
        'infer_requests': [{
            'messages': messages
        }],
        'request_config': {
            'model': SERVED,
            'max_tokens': max_tokens,
            'temperature': 0.0
        },
    })
    resp.raise_for_status()
    records = resp.json()
    assert isinstance(records, list) and len(records) == 1
    return records[0]


def test_text_only_prompt_token_baseline(server):
    """Sanity: the text-only form of the same prompt completes, establishing the token baseline."""
    body = server.chat([{'role': 'user', 'content': PROMPT_TEXT}], max_tokens=16)
    assert body['object'] == 'chat.completion'
    assert body['usage']['prompt_tokens'] > 0


def test_openai_image_url_parts_reach_the_model(server):
    """The OpenAI-standard ``image_url`` content part is ingested as vision tokens, not dropped."""
    text_only = server.chat([{'role': 'user', 'content': PROMPT_TEXT}], max_tokens=16)
    data_url = image_data_url(size=224, seed=7)
    with_image = server.chat(_openai_vision_messages(data_url), max_tokens=16)

    assert with_image['object'] == 'chat.completion'
    choice = with_image['choices'][0]
    assert choice['message']['role'] == 'assistant' and choice['message']['content']
    assert choice['finish_reason'] in ('stop', 'length')
    assert with_image['usage']['prompt_tokens'] > text_only['usage']['prompt_tokens'], (
        f'the image did not enlarge the prompt ({text_only["usage"]["prompt_tokens"]} -> '
        f'{with_image["usage"]["prompt_tokens"]}); it was not tokenized into the prompt')


def test_native_image_placeholder_reaches_the_model(server):
    """twinkle's native ``<image>`` + per-message ``images`` shape is ingested too."""
    text_only = server.chat([{'role': 'user', 'content': PROMPT_TEXT}], max_tokens=16)
    data_url = image_data_url(size=224, seed=11)
    with_image = server.chat(_native_vision_messages(data_url), max_tokens=16)

    assert with_image['object'] == 'chat.completion'
    assert with_image['choices'][0]['message']['content']
    assert with_image['usage']['prompt_tokens'] > text_only['usage']['prompt_tokens']


def test_image_chat_streams(server):
    """A streaming image chat still yields ``chat.completion.chunk`` frames ending in [DONE]."""
    import json
    data_url = image_data_url(size=224, seed=3)
    lines = list(
        server.stream(
            '/chat/completions', {
                'model': SERVED,
                'messages': _openai_vision_messages(data_url),
                'max_tokens': 16,
                'stream': True,
            }))
    assert lines[-1].strip() == 'data: [DONE]'
    chunks = [
        json.loads(line[len('data: '):]) for line in lines
        if line.startswith('data: ') and line.strip() != 'data: [DONE]'
    ]
    assert chunks and all(c['object'] == 'chat.completion.chunk' for c in chunks)
    assert ''.join(c['choices'][0]['delta'].get('content', '') for c in chunks)


def test_infer_with_image_carries_vision_tokens_and_serializable_feature(server):
    """/infer on an image request proves ingestion at the token level and returns a JSON-safe feature.

    The image request's ``prompt_token_ids`` must carry token ids the identical text-only prompt does not
    (the vision markers the template injects for the image), which is the definitive proof the image was
    tokenized into the prompt rather than dropped. A multimodal encode also leaves raw ``PIL.Image``
    objects inside the trajectory's ``messages``; the sampler must strip what cannot cross the HTTP
    boundary rather than fail response serialization, so a 200 with a populated ``new_input_feature`` is
    the assertion that the media did not leak into JSON.
    """
    text_record = _infer(server, [{'role': 'user', 'content': PROMPT_TEXT}])
    data_url = image_data_url(size=224, seed=5)
    image_record = _infer(server, _native_vision_messages(data_url))

    text_ids = set(text_record['prompt_token_ids'])
    image_ids = image_record['prompt_token_ids']
    assert set(image_ids) - text_ids, (
        'the image prompt introduced no token ids beyond the text-only prompt; the image was dropped')

    seq = image_record['sequences'][0]
    assert seq['tokens'] and seq['finish_reason'] in ('stop', 'length')
    feature = seq['new_input_feature']
    assert feature and feature.get('input_ids'), 'new_input_feature must carry the prompt+completion ids'
