# Copyright (c) ModelScope Contributors. All rights reserved.
import httpx
import json
import parallel_search
import unittest
from openai import AsyncOpenAI
from types import SimpleNamespace
from unittest.mock import patch


class ParallelSearchTest(unittest.IsolatedAsyncioTestCase):

    async def test_agent_dispatches_search_and_fetch_over_http(self):
        mcp_requests = []
        model_requests = []
        tool_ids = ['search-call', 'fetch-call']

        def mcp_handler(request):
            self.assertEqual(str(request.url), parallel_search.MCP_URL)
            self.assertEqual(request.headers['User-Agent'], parallel_search.USER_AGENT)
            self.assertNotIn('Authorization', request.headers)
            if request.method != 'POST':
                return httpx.Response(405)
            body = json.loads(request.content)
            mcp_requests.append(body)
            if 'id' not in body:
                return httpx.Response(202)
            method = body['method']
            if method == 'initialize':
                result = {
                    'protocolVersion': body['params']['protocolVersion'],
                    'capabilities': {
                        'tools': {}
                    },
                    'serverInfo': {
                        'name': 'parallel-fixture',
                        'version': '1'
                    }
                }
            elif method == 'tools/list':
                # Exercise pagination and preserve the schemas discovered at runtime.
                name = 'web_fetch' if body.get('params', {}).get('cursor') else 'web_search'
                result = {
                    'tools': [{
                        'name': name,
                        'description': name,
                        'inputSchema': {
                            'type': 'object',
                            'properties': {
                                'fixture_field': {
                                    'type': 'string'
                                }
                            }
                        }
                    }]
                }
                if name == 'web_search':
                    result['nextCursor'] = 'page-2'
            elif method == 'tools/call':
                name = body['params']['name']
                result = {'content': [{'type': 'text', 'text': f'{name}: https://example.com/swift Deployment guide'}]}
            else:
                self.fail(f'Unexpected MCP method: {method}')
            return httpx.Response(200, json={'jsonrpc': '2.0', 'id': body['id'], 'result': result})

        def model_handler(request):
            self.assertEqual(request.url.path, '/v1/chat/completions')
            body = json.loads(request.content)
            model_requests.append(body)
            for tool in body['tools']:
                self.assertIn('fixture_field', tool['function']['parameters']['properties'])
            step = len(model_requests)
            if step < 3:
                name = 'web_search' if step == 1 else 'web_fetch'
                arguments = {'objective': 'SWIFT deployment', 'search_queries': ['ms-swift deployment agent']}
                if step == 2:
                    arguments = {'urls': ['https://example.com/swift']}
                arguments.update(session_id='model-generated-id', model_name='invented-model')
                message = {
                    'role':
                    'assistant',
                    'content':
                    None,
                    'tool_calls': [{
                        'id': tool_ids[step - 1],
                        'type': 'function',
                        'function': {
                            'name': name,
                            'arguments': json.dumps(arguments)
                        }
                    }]
                }
            else:
                message = {'role': 'assistant', 'content': 'Serve SWIFT: https://example.com/swift'}
            return httpx.Response(
                200,
                json={
                    'id': 'fixture',
                    'created': 0,
                    'object': 'chat.completion',
                    'model': 'served-model',
                    'choices': [{
                        'index': 0,
                        'finish_reason': 'stop',
                        'message': message
                    }]
                })

        # Exercise connect(), including the actual SDK transport, with no network.
        http = httpx.AsyncClient(
            headers={'User-Agent': parallel_search.USER_AGENT}, transport=httpx.MockTransport(mcp_handler))
        async with AsyncOpenAI(
                api_key='EMPTY',
                base_url='http://127.0.0.1:8000/v1',
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(model_handler))) as client:
            with patch.object(parallel_search.httpx, 'AsyncClient', return_value=http) as factory:
                async with parallel_search.connect() as session:
                    answer = await parallel_search.run_agent(client, session, 'served-model', 'How to deploy SWIFT?')
                factory.assert_called_once_with(headers={'User-Agent': parallel_search.USER_AGENT}, timeout=60)
        self.assertIn('https://example.com/swift', answer)
        calls = [body['params'] for body in mcp_requests if body['method'] == 'tools/call']
        self.assertEqual([call['name'] for call in calls], ['web_search', 'web_fetch'])
        identities = {call['arguments']['session_id'] for call in calls}
        self.assertEqual(len(identities), 1)
        self.assertNotIn('model-generated-id', identities)
        self.assertTrue(all(call['arguments']['model_name'] == 'served-model' for call in calls))
        history = model_requests[-1]['messages']
        self.assertEqual([message['tool_call_id'] for message in history if message['role'] == 'tool'], tool_ids)
        self.assertTrue(
            all('Deployment guide' in message['content'] for message in history if message['role'] == 'tool'))
        self.assertEqual(history[2]['tool_calls'][0]['id'], 'search-call')

    async def test_tool_errors_empty_results_and_unknown_tools(self):

        class Session:

            async def call_tool(self, name, arguments):
                return SimpleNamespace(isError=True, content=[SimpleNamespace(type='text', text='rate limited')])

        with self.assertRaisesRegex(ValueError, 'Unknown tool'):
            await parallel_search.call_tool(Session(), 'unexpected', {}, 'session')
        with self.assertRaisesRegex(RuntimeError, 'rate limited'):
            await parallel_search.call_tool(Session(), 'web_search', {}, 'session')

        class EmptySession:

            async def call_tool(self, name, arguments):
                return SimpleNamespace(isError=False, content=[])

        with self.assertRaisesRegex(RuntimeError, 'no text'):
            await parallel_search.call_tool(EmptySession(), 'web_fetch', {}, 'session')

    async def test_missing_tools_fail_before_inference(self):

        class Session:

            async def list_tools(self, cursor=None):
                return SimpleNamespace(tools=[], nextCursor=None)

        with self.assertRaisesRegex(RuntimeError, 'must expose'):
            await parallel_search.run_agent(None, Session(), 'served-model', 'Question')


if __name__ == '__main__':
    unittest.main()
