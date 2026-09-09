# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import unittest
from aiohttp import web
from aiohttp.test_utils import TestServer
from requests.exceptions import HTTPError

from swift.infer_engine import InferClient, InferRequest, RequestConfig


class TestInferClientStream(unittest.IsolatedAsyncioTestCase):

    async def _consume(self, payload):

        async def handle(request):
            body = await request.json()
            self.assertTrue(body['stream'])
            response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
            await response.prepare(request)
            # Split SSE lines and UTF-8 characters across HTTP chunks.
            for index in range(0, len(payload), 7):
                await response.write(payload[index:index + 7])
            await response.write_eof()
            return response

        app = web.Application()
        app.router.add_post('/v1/chat/completions', handle)
        async with TestServer(app) as server:
            client = InferClient(base_url=str(server.make_url('/v1')))
            stream = await client.infer_async(
                InferRequest(messages=[{
                    'role': 'user',
                    'content': 'Hello'
                }]), RequestConfig(stream=True), model='test')
            return [chunk async for chunk in stream]

    @staticmethod
    def _event(content):
        return 'data: ' + json.dumps(
            {
                'object': 'chat.completion.chunk',
                'model': 'test',
                'choices': [{
                    'index': 0,
                    'delta': {
                        'content': content
                    },
                    'finish_reason': None
                }]
            },
            ensure_ascii=False)

    async def test_stream_with_heartbeat_comments(self):
        for newline in ('\n', '\r\n'):
            with self.subTest(newline=newline):
                lines = [
                    ': OPENROUTER PROCESSING', '',
                    self._event('你好'), '', ': ping', ':', '',
                    self._event(': world'), '', 'data: [DONE]', '', 'invalid after done'
                ]
                chunks = await self._consume(newline.join(lines).encode('utf-8'))
                self.assertEqual([chunk.choices[0].delta.content for chunk in chunks], ['你好', ': world'])

    async def test_stream_without_comments(self):
        chunks = await self._consume((self._event('hello') + '\n\ndata: [DONE]\n\n').encode())
        self.assertEqual([chunk.choices[0].delta.content for chunk in chunks], ['hello'])

    async def test_error_after_comment(self):
        payload = b': ping\n\ndata: {"object": "error", "message": "generation failed"}\n\n'
        with self.assertRaisesRegex(HTTPError, 'generation failed'):
            await self._consume(payload)

    async def test_usage_after_done_shaped_comment(self):
        event = {
            'object': 'chat.completion.chunk',
            'model': 'test',
            'choices': [],
            'usage': {
                'prompt_tokens': 2,
                'completion_tokens': 3,
                'total_tokens': 5
            }
        }
        payload = ': data: [DONE]\n\ndata: ' + json.dumps(event) + '\n\ndata: [DONE]\n\n'
        chunks = await self._consume(payload.encode())
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].choices, [])
        self.assertEqual(chunks[0].usage.total_tokens, 5)

    async def test_invalid_json_after_comment(self):
        with self.assertRaises(json.JSONDecodeError):
            await self._consume(b': ping\n\ndata: {invalid json}\n\n')

    async def test_comments_without_content(self):
        chunks = await self._consume(b': ping\n\n: heartbeat\n\ndata: [DONE]\n\n')
        self.assertEqual(chunks, [])

    def test_comment_lines(self):
        for line in (b':\n', b': ping\r\n', b': OPENROUTER PROCESSING\n', b': data: [DONE]\n'):
            with self.subTest(line=line):
                self.assertIsNone(InferClient._parse_stream_data(line))

    def test_data_and_invalid_lines(self):
        self.assertIsNone(InferClient._parse_stream_data(b'\n'))
        self.assertEqual(InferClient._parse_stream_data(b'data: [DONE]\n'), '[DONE]')
        self.assertEqual(InferClient._parse_stream_data(b'data: :content\n'), ':content')
        with self.assertRaises(AssertionError):
            InferClient._parse_stream_data(b'invalid\n')


if __name__ == '__main__':
    unittest.main()
