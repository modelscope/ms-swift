# Copyright (c) ModelScope Contributors. All rights reserved.
import asyncio
import importlib.util
import os
import threading
import time
import unittest
from aiohttp import web
from email.utils import formatdate
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLUGIN_PATH = _REPO_ROOT / 'examples' / 'train' / 'grpo' / 'plugin' / 'plugin.py'


def _run_coro(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _load_grpo_plugin():
    spec = importlib.util.spec_from_file_location('grpo_plugin_judge0_test', _PLUGIN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _LoopbackJudge0Server:
    """In-process real HTTP server built with aiohttp.web on 127.0.0.1."""

    def __init__(self, handler):
        self._handler = handler
        self._loop = None
        self._thread = None
        self._runner = None
        self.url = None
        self._ready = threading.Event()
        self._error = None

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=10)
        if self._error is not None:
            raise self._error
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._loop is not None and self._runner is not None:
            fut = asyncio.run_coroutine_threadsafe(self._runner.cleanup(), self._loop)
            fut.result(timeout=5)
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._start())
        except Exception as exc:
            self._error = exc
            self._ready.set()
            return
        self._ready.set()
        self._loop.run_forever()
        self._loop.close()

    async def _start(self):
        app = web.Application()
        app.router.add_post('/submissions/', self._handler)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, '127.0.0.1', 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.url = f'http://127.0.0.1:{port}'


class TestJudge0CodeReward(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.plugin = _load_grpo_plugin()
        cls.CodeRewardByJudge0 = cls.plugin.CodeRewardByJudge0

    def _make_reward(self, endpoint, extra_env=None):
        env = {'JUDGE0_ENDPOINT': endpoint, 'JUDGE0_MAX_CONCURRENCY': ''}
        if extra_env:
            env.update(extra_env)
        with patch.dict(os.environ, env, clear=False):
            return self.CodeRewardByJudge0(SimpleNamespace())

    def test_batch_recovers_from_429_rate_limit_responses(self):
        lock = threading.Lock()
        state = {'in_flight': 0, 'rate_limited': 0, 'tokens_seen': set()}

        async def handle(request):
            state['tokens_seen'].add(request.headers.get('X-Auth-Token'))
            payload = await request.json()
            with lock:
                state['in_flight'] += 1
                if state['in_flight'] > 4:
                    state['rate_limited'] += 1
                    state['in_flight'] -= 1
                    return web.json_response(
                        {'message': 'Too many concurrent requests'},
                        status=429,
                        headers={'Retry-After': '0.01'},
                    )
            try:
                await asyncio.sleep(0.02)
            finally:
                with lock:
                    state['in_flight'] -= 1
            status_desc = 'Accepted' if 'ok' in payload['source_code'] else 'Wrong Answer'
            return web.json_response({'status': {'description': status_desc}}, status=201)

        completions = [f"```python\nprint({'ok' if i % 2 == 0 else 'bad'})\n```" for i in range(14)]
        verification_info = [{
            'language': 'python',
            'test_cases': [{
                'input': '',
                'output': 'ok\n'
            } for _ in range(3)],
        } for _ in range(14)]
        expected = [1.0 if i % 2 == 0 else 0.0 for i in range(14)]

        with _LoopbackJudge0Server(handle) as server:
            reward = self._make_reward(server.url, {'JUDGE0_X_AUTH_TOKEN': 'secret-token'})
            reward.retry_backoff = 0.01
            scores = reward(completions, verification_info=verification_info)

        self.assertGreater(state['rate_limited'], 0)
        self.assertEqual(state['tokens_seen'], {'secret-token'})
        self.assertEqual(scores, expected)

    def test_503_text_and_json_bodies_retry_before_success(self):
        attempts = 0

        async def handle(request):
            nonlocal attempts
            attempts += 1
            await request.json()
            if attempts == 1:
                return web.Response(text='Service Unavailable', status=503)
            if attempts == 2:
                return web.json_response({'error': 'queue is full'}, status=503)
            return web.json_response({'status': {'description': 'Accepted'}}, status=200)

        completions = ['```python\nprint(1)\n```']
        verification_info = [{'language': 'python', 'test_cases': [{'input': '', 'output': '1\n'}]}]

        with _LoopbackJudge0Server(handle) as server:
            reward = self._make_reward(server.url)
            reward.retry_backoff = 0.01
            scores = reward(completions, verification_info=verification_info)

        self.assertEqual(attempts, 3)
        self.assertEqual(scores, [1.0])

    def test_max_concurrency_caps_in_flight_requests(self):
        lock = threading.Lock()
        state = {'in_flight': 0, 'peak': 0, 'calls': 0}

        async def handle(request):
            await request.json()
            with lock:
                state['in_flight'] += 1
                state['calls'] += 1
                if state['in_flight'] > state['peak']:
                    state['peak'] = state['in_flight']
            try:
                await asyncio.sleep(0.02)
            finally:
                with lock:
                    state['in_flight'] -= 1
            return web.json_response({'status': {'description': 'Accepted'}}, status=200)

        completions = ['```python\nprint(1)\n```' for _ in range(8)]
        verification_info = [{
            'language': 'python',
            'test_cases': [{
                'input': '',
                'output': '1\n'
            }, {
                'input': '',
                'output': '1\n'
            }],
        } for _ in range(8)]

        with _LoopbackJudge0Server(handle) as server:
            reward = self._make_reward(server.url, {'JUDGE0_MAX_CONCURRENCY': '2'})
            scores = reward(completions, verification_info=verification_info)

        self.assertEqual(state['calls'], 16)
        self.assertLessEqual(state['peak'], 2)
        self.assertEqual(scores, [1.0] * 8)

    def test_non_retryable_4xx_fails_fast_and_exhaustion_returns_zero(self):
        calls = {'401': 0, '503': 0}

        async def handle(request):
            payload = await request.json()
            if payload['source_code'].strip() == 'unauthorized':
                calls['401'] += 1
                return web.json_response({'message': 'invalid token'}, status=401)
            calls['503'] += 1
            return web.json_response({'error': 'overloaded'}, status=503, headers={'Retry-After': '0.01'})

        completions = [
            '```python\nunauthorized\n```',
            '```python\nalways_503\n```',
        ]
        verification_info = [
            {
                'language': 'python',
                'test_cases': [{
                    'input': '',
                    'output': '1\n'
                }]
            },
            {
                'language': 'python',
                'test_cases': [{
                    'input': '',
                    'output': '1\n'
                }]
            },
        ]

        with _LoopbackJudge0Server(handle) as server:
            reward = self._make_reward(server.url)
            reward.max_retries = 2
            reward.retry_backoff = 0.01
            scores = reward(completions, verification_info=verification_info)

        self.assertEqual(calls['401'], 1)
        self.assertEqual(calls['503'], 3)
        self.assertEqual(scores, [0.0, 0.0])

    def test_retry_after_parsing_and_backoff_bounds(self):
        reward = self._make_reward('http://127.0.0.1:9')

        self.assertIsNone(reward._parse_retry_after(None))
        self.assertIsNone(reward._parse_retry_after(''))
        self.assertIsNone(reward._parse_retry_after('not-a-date'))
        self.assertIsNone(reward._parse_retry_after('-1.5'))
        self.assertIsNone(reward._parse_retry_after('nan'))
        self.assertIsNone(reward._parse_retry_after('inf'))
        self.assertEqual(reward._parse_retry_after('0'), 0.0)
        self.assertAlmostEqual(reward._parse_retry_after('2.5'), 2.5)

        future_date = formatdate(time.time() + 5, usegmt=True)
        parsed_future = reward._parse_retry_after(future_date)
        self.assertIsNotNone(parsed_future)
        self.assertGreater(parsed_future, 0.0)
        self.assertLessEqual(parsed_future, 6.0)

        past_date = formatdate(time.time() - 10, usegmt=True)
        self.assertEqual(reward._parse_retry_after(past_date), 0.0)

        self.assertEqual(reward._compute_retry_delay(0, None), reward.retry_backoff)
        self.assertEqual(reward._compute_retry_delay(1, None), reward.retry_backoff * 2)
        self.assertEqual(reward._compute_retry_delay(20, None), reward.max_retry_backoff)
        self.assertEqual(reward._compute_retry_delay(0, '9999'), reward.max_retry_backoff)
        self.assertEqual(reward._compute_retry_delay(0, '0'), 0.0)

    def test_invalid_max_concurrency_raises_value_error(self):
        for bad_value in ('0', '-3', 'abc'):
            env = {'JUDGE0_ENDPOINT': 'http://127.0.0.1:9', 'JUDGE0_MAX_CONCURRENCY': bad_value}
            with patch.dict(os.environ, env, clear=False):
                with self.assertRaises(ValueError):
                    self.CodeRewardByJudge0(SimpleNamespace())

    def test_direct_evaluate_code_and_empty_inputs(self):
        attempts = 0

        async def handle(request):
            nonlocal attempts
            attempts += 1
            await request.json()
            if attempts == 1:
                return web.Response(text='rate limited', status=429, headers={'Retry-After': '0.01'})
            return web.json_response({'status': {'description': 'Accepted'}}, status=200)

        with _LoopbackJudge0Server(handle) as server:
            reward = self._make_reward(server.url)
            reward.retry_backoff = 0.01
            score = _run_coro(reward._evaluate_code('print(1)', [{'input': '', 'output': '1\n'}], 71))
            empty_code_score = _run_coro(reward._evaluate_code('', [{'input': '', 'output': '1\n'}], 71))
            empty_cases_score = _run_coro(reward._evaluate_code('print(1)', [], 71))

        self.assertEqual(attempts, 2)
        self.assertEqual(score, 1.0)
        self.assertEqual(empty_code_score, 0.0)
        self.assertEqual(empty_cases_score, 0.0)


if __name__ == '__main__':
    unittest.main()
