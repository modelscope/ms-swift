# Copyright (c) ModelScope Contributors. All rights reserved.
import asyncio
import unittest
from queue import Queue
from threading import Event
from transformers import GenerationConfig

from swift.infer_engine import AdapterRequest, RequestConfig, TransformersEngine
from swift.utils import shutdown_event_loop_in_daemon


class _WorkerEngine(TransformersEngine):

    def __init__(self):
        self._queue = Queue()
        self._task_pool = {}
        self._task_thread = None
        self.max_batch_size = 0
        self.stopped = Event()
        self.batches = []

    def _fetch_infer_requests(self):
        if self.stopped.is_set():
            raise SystemExit
        return super()._fetch_infer_requests()

    def _infer_worker(self):
        try:
            super()._infer_worker()
        except SystemExit:
            pass

    def _infer(self, infer_requests, request_config, **kwargs):
        self.batches.append(infer_requests)
        if request_config.stream and request_config.num_beams == 2:
            # Exercise the real backend rejection before model generation starts.
            return self._infer_stream({},
                                      generation_config=GenerationConfig(num_beams=2),
                                      adapter_request=None,
                                      request_config=request_config,
                                      template_inputs=[])
        if infer_requests[0] == 'bad':
            raise ValueError('encoding failed')
        if request_config.stream:

            def stream():
                yield ['partial'] * len(infer_requests)
                if infer_requests[0] == 'partial_failure':
                    raise RuntimeError('stream iteration failed')

            return stream()
        return infer_requests


class TestTransformersBatching(unittest.TestCase):

    def enqueue(self, engine, text, config, adapter):
        queue = Queue()
        engine._queue.put((text, {'request_config': config, 'adapter_request': adapter}, queue))
        return queue

    def test_different_adapters_keep_their_request_options(self):
        adapter = AdapterRequest('a', '/adapters/a')
        for other in (None, AdapterRequest('b', '/adapters/b'), AdapterRequest('a', '/adapters/other')):
            for reverse in (False, True):
                for batch_size in (0, 1):
                    with self.subTest(other=other, reverse=reverse, batch_size=batch_size):
                        engine = _WorkerEngine()
                        engine.max_batch_size = batch_size
                        adapters = [adapter, other]
                        if reverse:
                            adapters.reverse()
                        queues = [
                            self.enqueue(engine, str(i), RequestConfig(), value) for i, value in enumerate(adapters)
                        ]
                        for i, expected in enumerate(adapters):
                            kwargs, batch_queues = engine._fetch_infer_requests()
                            self.assertEqual(kwargs['infer_requests'], [str(i)])
                            self.assertEqual(kwargs['adapter_request'], expected)
                            self.assertEqual(batch_queues, [queues[i]])
                        self.assertIsNone(engine._fetch_infer_requests())

    def test_equivalent_requests_still_batch_and_split(self):
        for adapter in (None, AdapterRequest('a', '/adapters/a')):
            for batch_size in (0, 2):
                with self.subTest(adapter=adapter, batch_size=batch_size):
                    engine = _WorkerEngine()
                    engine.max_batch_size = batch_size
                    queues = []
                    for i in range(3):
                        # Equivalent requests need not share the same config or adapter object.
                        value = None if adapter is None else AdapterRequest(adapter.name, adapter.path)
                        queues.append(self.enqueue(engine, str(i), RequestConfig(), value))
                    other_queue = self.enqueue(engine, 'other_config', RequestConfig(max_tokens=7), adapter)
                    offset = 0
                    while offset < 3:
                        kwargs, batch_queues = engine._fetch_infer_requests()
                        end = min(offset + (batch_size or 3), 3)
                        self.assertEqual(kwargs['infer_requests'], [str(i) for i in range(offset, end)])
                        self.assertEqual(kwargs['adapter_request'], adapter)
                        self.assertEqual(batch_queues, queues[offset:end])
                        offset = end
                    kwargs, batch_queues = engine._fetch_infer_requests()
                    self.assertEqual(kwargs['infer_requests'], ['other_config'])
                    self.assertEqual(kwargs['request_config'].max_tokens, 7)
                    self.assertEqual(batch_queues, [other_queue])
                    self.assertIsNone(engine._fetch_infer_requests())


class TestTransformersWorker(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.engine = _WorkerEngine()

    async def asyncTearDown(self):
        self.engine.stopped.set()
        if self.engine._task_thread is not None:
            self.engine._task_thread.join(timeout=2)
            self.assertFalse(self.engine._task_thread.is_alive())

    async def request(self, text, **kwargs):
        return await asyncio.wait_for(self.engine.infer_async(text, RequestConfig(**kwargs)), timeout=2)

    async def assert_recovery(self):
        self.assertEqual(await self.request('healthy'), 'healthy')
        self.assertTrue(self.engine._task_thread.is_alive())

    async def test_nonstream_batch_failure_reaches_every_caller(self):
        results = await asyncio.gather(self.request('bad'), self.request('also_bad'), return_exceptions=True)
        self.assertEqual(self.engine.batches[0], ['bad', 'also_bad'])
        for error in results:
            self.assertIsInstance(error, ValueError)
            self.assertEqual(str(error), 'encoding failed')
        await self.assert_recovery()

    async def test_stream_beam_search_error_reaches_caller(self):
        streams = await asyncio.gather(
            self.request('query', stream=True, num_beams=2), self.request('another_query', stream=True, num_beams=2))
        for stream in streams:
            with self.assertRaisesRegex(ValueError, 'does not support beam search'):
                await asyncio.wait_for(anext(stream), timeout=2)
        await self.assert_recovery()

    async def test_error_after_partial_stream_and_normal_completion(self):
        stream = await self.request('partial_failure', stream=True)
        self.assertEqual(await asyncio.wait_for(anext(stream), timeout=2), 'partial')
        with self.assertRaisesRegex(RuntimeError, 'stream iteration failed'):
            await asyncio.wait_for(anext(stream), timeout=2)
        await self.assert_recovery()
        stream = await self.request('healthy', stream=True)
        self.assertEqual(await asyncio.wait_for(anext(stream), timeout=2), 'partial')
        with self.assertRaises(StopAsyncIteration):
            await asyncio.wait_for(anext(stream), timeout=2)


class TestTransformersWorkerStrictMode(unittest.TestCase):

    def test_outer_infer_preserves_strict_policy(self):
        for strict in (True, False):
            for stream in (True, False):
                with self.subTest(strict=strict, stream=stream):
                    engine = _WorkerEngine()
                    engine.strict = strict
                    config = RequestConfig(stream=stream)

                    def request():
                        results = engine.infer(['bad'], config, use_tqdm=False)
                        return list(results[0]) if stream else results

                    try:
                        # Non-streaming infer bypasses the worker and raises directly.
                        if strict or not stream:
                            with self.assertRaisesRegex(ValueError, 'encoding failed'):
                                request()
                        else:
                            self.assertEqual(request(), [])
                        self.assertEqual(engine.infer(['healthy'], RequestConfig(), use_tqdm=False), ['healthy'])
                    finally:
                        engine.stopped.set()
                        if engine._task_thread is not None:
                            engine._task_thread.join(timeout=2)
                            self.assertFalse(engine._task_thread.is_alive())
                        loop = getattr(engine, '_event_loop', None)
                        if loop is not None:
                            shutdown_event_loop_in_daemon(engine._event_loop_thread, loop)
                            self.assertFalse(engine._event_loop_thread.is_alive())
                            self.assertTrue(loop.is_closed())


if __name__ == '__main__':
    unittest.main()
