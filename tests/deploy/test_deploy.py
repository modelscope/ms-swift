import asyncio
import os
import pytest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from swift.infer_engine.protocol import (ChatCompletionRequest, ChatCompletionResponseStreamChoice,
                                         ChatCompletionStreamResponse, DeltaMessage, InferRequest)
from swift.pipelines.infer.deploy import SwiftDeploy


def test_post_process_reasoning_only_stream_chunk():
    deploy = object.__new__(SwiftDeploy)
    deploy.args = SimpleNamespace()
    request_info = {'response': ''}
    response = ChatCompletionStreamResponse(
        model='test',
        choices=[
            ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(content=None, reasoning_content='thinking'),
                finish_reason=None,
            )
        ],
    )

    processed = deploy._post_process(request_info, response)

    assert processed is response
    assert processed.choices[0].delta.content is None
    assert processed.choices[0].delta.reasoning_content == 'thinking'
    assert request_info['response'] == ''


def test_prepare_request_media_materializes_all_url_locations():
    url = 'https://example.com/media'
    request = InferRequest(
        messages=[{
            'role': 'user',
            'content': [{
                'type': 'image_url',
                'image_url': {
                    'url': url
                }
            }]
        }],
        images=[url],
        audios=[url],
        videos=[[url, url]],
    )

    with patch('swift.pipelines.infer.deploy.SafeUrlFetcher.read', return_value=b'media') as read:
        with SwiftDeploy._prepare_request_media(request):
            paths = [
                request.images[0], request.audios[0], *request.videos[0],
                request.messages[0]['content'][0]['image_url']['url']
            ]
            assert all(os.path.isfile(path) for path in paths)
            assert all(Path(path).read_bytes() == b'media' for path in paths)
        assert read.call_count == 5
    assert all(not os.path.exists(path) for path in paths)


def test_prepare_request_media_blocks_unsafe_urls_and_schemes():
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(ValueError):
            SwiftDeploy._materialize_media('http://127.0.0.1/private', 'image', temp_dir)
        for value in ('ftp://example.com/file', 'file:///etc/' + 'x' * 300):
            with pytest.raises(ValueError, match='only http://, https://, and data:'):
                SwiftDeploy._materialize_media(value, 'image', temp_dir)


def test_prepare_request_media_preserves_inline_media():
    with tempfile.TemporaryDirectory() as temp_dir:
        for value in ('aGVsbG8=', 'data:image/png;base64,aGVsbG8='):
            assert SwiftDeploy._materialize_media(value, 'image', temp_dir) == value


def test_prepare_request_media_materializes_bytes_path_dict():
    url = 'https://example.com/image.jpg'
    media_objects = [
        ({
            'bytes': None,
            'path': url
        }, 'path'),
        ({
            'bytes': url,
            'path': None
        }, 'bytes'),
        ({
            'path': url
        }, 'path'),
    ]
    with tempfile.TemporaryDirectory() as temp_dir:
        for media, source_key in media_objects:
            with patch('swift.pipelines.infer.deploy.SafeUrlFetcher.read', return_value=b'image'):
                result = SwiftDeploy._materialize_media(media, 'image', temp_dir)
            assert result is media
            assert os.path.isfile(result[source_key])
            assert Path(result[source_key]).read_bytes() == b'image'

        with pytest.raises(ValueError, match='expected a url, bytes, or path field'):
            SwiftDeploy._materialize_media({'unexpected': url}, 'image', temp_dir)


async def _test_streaming_response_keeps_media_until_stream_finishes():
    deploy = object.__new__(SwiftDeploy)
    deploy.args = SimpleNamespace(
        api_key=None,
        adapter_mapping={},
        max_logprobs=20,
        get_request_config=lambda: None,
    )
    deploy.infer_kwargs = {}
    captured_paths = []

    async def _check_model(request):
        return None

    async def _stream():
        if False:
            yield None

    async def _infer_async(infer_request, request_config, **kwargs):
        captured_paths.extend(infer_request.images)
        assert all(os.path.isfile(path) for path in captured_paths)
        return _stream()

    deploy._check_model = _check_model
    deploy.infer_async = _infer_async
    request = ChatCompletionRequest(
        model='test',
        messages=[{
            'role': 'user',
            'content': '<image>describe'
        }],
        images=['https://example.com/image.jpg'],
        stream=True,
    )

    with patch('swift.pipelines.infer.deploy.SafeUrlFetcher.read', return_value=b'image'):
        response = await deploy.create_chat_completion(request, SimpleNamespace(headers={}))
        assert all(os.path.isfile(path) for path in captured_paths)
        chunks = [chunk async for chunk in response.body_iterator]

    assert chunks == ['data: [DONE]\n\n']
    assert all(not os.path.exists(path) for path in captured_paths)


def test_streaming_response_keeps_media_until_stream_finishes():
    asyncio.run(_test_streaming_response_keeps_media_until_stream_finishes())


def test_media_is_not_fetched_before_authentication():
    asyncio.run(_test_media_is_not_fetched_before_authentication())


def test_infer_handler_materializes_media_for_the_whole_batch():
    asyncio.run(_test_infer_handler_materializes_media_for_the_whole_batch())


def test_infer_handler_checks_api_key_before_parsing_body():
    asyncio.run(_test_infer_handler_checks_api_key_before_parsing_body())


def test_infer_handler_rejects_streaming_before_fetching_media():
    asyncio.run(_test_infer_handler_rejects_streaming_before_fetching_media())


async def _test_media_is_not_fetched_before_authentication():
    deploy = object.__new__(SwiftDeploy)
    deploy.args = SimpleNamespace(
        api_key='secret',
        adapter_mapping={},
        max_logprobs=20,
        get_request_config=lambda: None,
    )

    async def _check_model(request):
        return None

    deploy._check_model = _check_model
    request = ChatCompletionRequest(
        model='test',
        messages=[{
            'role': 'user',
            'content': '<image>describe'
        }],
        images=['https://example.com/image.jpg'],
    )

    with patch('swift.pipelines.infer.deploy.SafeUrlFetcher.read') as read:
        response = await deploy.create_chat_completion(request, SimpleNamespace(headers={}))

    assert response.status_code == 400
    read.assert_not_called()


async def _test_infer_handler_checks_api_key_before_parsing_body():
    deploy = object.__new__(SwiftDeploy)
    deploy.args = SimpleNamespace(api_key='secret')

    class RawRequest:
        headers = {}

        async def json(self):
            raise AssertionError('Unauthorized requests must be rejected before parsing the body.')

    response = await deploy.infer_handler(RawRequest())
    assert response.status_code == 400


async def _test_infer_handler_rejects_streaming_before_fetching_media():
    deploy = object.__new__(SwiftDeploy)
    deploy.args = SimpleNamespace(api_key=None)

    async def _infer_async(infer_request, request_config):
        raise AssertionError('Streaming requests must be rejected before inference.')

    class RawRequest:
        headers = {}

        async def json(self):
            return {
                'infer_requests': [{
                    'messages': [{
                        'role': 'user',
                        'content': '<image>describe'
                    }],
                    'images': ['https://example.com/image.jpg'],
                }],
                'request_config': {
                    'stream': True
                },
            }

    deploy.infer_async = _infer_async
    with patch('swift.pipelines.infer.deploy.SafeUrlFetcher.read') as read:
        response = await deploy.infer_handler(RawRequest())

    assert response.status_code == 400
    read.assert_not_called()


async def _test_infer_handler_materializes_media_for_the_whole_batch():
    deploy = object.__new__(SwiftDeploy)
    deploy.args = SimpleNamespace(api_key=None)
    captured_paths = []

    async def _infer_async(infer_request, request_config):
        captured_paths.extend(infer_request.images)
        assert all(os.path.isfile(path) for path in captured_paths)
        return 'ok'

    class RawRequest:
        headers = {}

        async def json(self):
            return {
                'infer_requests': [{
                    'messages': [{
                        'role': 'user',
                        'content': '<image>describe'
                    }],
                    'images': ['https://example.com/image.jpg'],
                }]
            }

    deploy.infer_async = _infer_async
    with patch('swift.pipelines.infer.deploy.SafeUrlFetcher.read', return_value=b'image'):
        assert await deploy.infer_handler(RawRequest()) == ['ok']

    assert all(not os.path.exists(path) for path in captured_paths)
