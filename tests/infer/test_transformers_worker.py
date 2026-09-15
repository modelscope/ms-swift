# Copyright (c) ModelScope Contributors. All rights reserved.
import asyncio
import torch
import unittest
from queue import Queue
from threading import Event
from transformers import GenerationConfig
from types import SimpleNamespace
from unittest.mock import patch

from swift.infer_engine import AdapterRequest, RequestConfig, TransformersEngine
from swift.infer_engine.utils import TokensIteratorStreamer


class _WorkerEngine(TransformersEngine):

    def __init__(self):
        self._queue = Queue()
        self._task_pool = {}
        self._task_thread = None
        self._adapters_pool = {}
        self.max_batch_size = 0
        self.stopped = Event()
        self.batches = []
        self.model = SimpleNamespace()
        self.model_name = 'test'
        self.processor = SimpleNamespace(pad_token_id=0, decode=lambda token: str(token))
        self.template = SimpleNamespace(
            tokenizer=self.processor,
            prepare_generate_kwargs=lambda kwargs, **_: kwargs,
            generate=self._generate,
            get_generate_ids=lambda ids, prompt_length: ids[:, prompt_length:],
            decode_generate_ids=lambda ids, **_: 'partial ' * len(ids))

    @staticmethod
    def _generate(model, *, input_ids, streamer, failure, logits_processor=(), **kwargs):
        if failure == 'generation_fail':
            raise ValueError('generation failed')
        streamer.put(input_ids)
        for processor in logits_processor:
            processor(input_ids, torch.zeros(input_ids.shape[0], 4))
        streamer.put(torch.full((input_ids.shape[0], ), 2))
        if failure == 'generation_partial':
            raise ValueError('generation failed')
        streamer.end()

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
        if request_config.stream and infer_requests[0].startswith('generation_'):
            inputs = torch.ones(len(infer_requests), 1, dtype=torch.long)
            return self._infer_stream({
                'input_ids': inputs,
                'attention_mask': inputs,
                'failure': infer_requests[0]
            },
                                      generation_config=GenerationConfig(
                                          num_beams=1, max_new_tokens=2, output_logits=request_config.logprobs),
                                      adapter_request=None,
                                      request_config=request_config,
                                      template_inputs=[None] * len(infer_requests))
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
        # These fixtures use CPU tensors and do not model accelerator placement.
        npu_available = patch('swift.infer_engine.transformers_engine.is_torch_npu_available', return_value=False)
        npu_available.start()
        self.addCleanup(npu_available.stop)
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

    async def test_npu_device_error_reaches_caller(self):
        error = RuntimeError('device setup failed')
        self.engine.model.device = 'npu:0'
        with patch(
                'swift.infer_engine.transformers_engine.is_torch_npu_available', return_value=True), patch.object(
                    torch, 'npu', create=True) as npu, patch.object(self.engine.template, 'generate') as generate:
            npu.set_device.side_effect = error
            stream = await self.request('generation_ok', stream=True)
            with self.assertRaises(RuntimeError) as context:
                await asyncio.wait_for(anext(stream), timeout=2)
            self.assertIs(context.exception, error)
            npu.set_device.assert_called_once_with('npu:0')
            generate.assert_not_called()
        await self.assert_recovery()

    async def test_generation_thread_errors_reach_batched_callers(self):
        for failure in ('generation_fail', 'generation_partial'):
            for logprobs in (False, True):
                with self.subTest(failure=failure, logprobs=logprobs):
                    streams = await asyncio.gather(
                        *[self.request(failure, stream=True, logprobs=logprobs, top_logprobs=1) for _ in range(2)])
                    for stream in streams:
                        if failure == 'generation_partial':
                            partial = await asyncio.wait_for(anext(stream), timeout=2)
                            self.assertEqual(partial.choices[0].delta.content, 'partial ')
                            if logprobs:
                                self.assertIsNotNone(partial.choices[0].logprobs)
                        with self.assertRaisesRegex(ValueError, 'generation failed'):
                            await asyncio.wait_for(anext(stream), timeout=2)
                    self.assertEqual(self.engine.batches[-1], [failure, failure])
                    await self.assert_recovery()
                    stream = await self.request('generation_ok', stream=True, logprobs=logprobs)

                    async def collect():
                        return [chunk async for chunk in stream]

                    chunks = await asyncio.wait_for(collect(), timeout=2)
                    self.assertEqual(''.join(c.choices[0].delta.content for c in chunks), 'partial ')
                    self.assertEqual(chunks[-1].choices[0].finish_reason, 'stop')


class TestTokensIteratorStreamer(unittest.TestCase):

    def test_error_preserves_queued_tokens(self):
        streamer = TokensIteratorStreamer()
        tokens = torch.tensor([1, 2])
        error = ValueError('generation failed')
        streamer.put(tokens)
        streamer.queue.put(error)
        streamer.end()
        self.assertIs(next(streamer), tokens)
        with self.assertRaises(ValueError) as context:
            next(streamer)
        self.assertIs(context.exception, error)
        with self.assertRaises(StopIteration):
            next(streamer)


class TestTransformersWorkerStrictMode(unittest.TestCase):

    def setUp(self):
        # These fixtures use CPU tensors and do not model accelerator placement.
        npu_available = patch('swift.infer_engine.transformers_engine.is_torch_npu_available', return_value=False)
        npu_available.start()
        self.addCleanup(npu_available.stop)

    def test_generation_error_preserves_partial_output_and_strict_policy(self):
        for strict in (True, False):
            with self.subTest(strict=strict):
                engine = _WorkerEngine()
                engine.strict = strict
                try:
                    stream = engine.infer(['generation_partial'], RequestConfig(stream=True), use_tqdm=False)[0]
                    self.assertEqual(next(stream).choices[0].delta.content, 'partial ')
                    if strict:
                        with self.assertRaisesRegex(ValueError, 'generation failed'):
                            next(stream)
                    else:
                        with self.assertRaises(StopIteration):
                            next(stream)
                    chunks = list(engine.infer(['generation_ok'], RequestConfig(stream=True), use_tqdm=False)[0])
                    self.assertEqual(chunks[-1].choices[0].finish_reason, 'stop')
                finally:
                    engine.stopped.set()
                    if engine._task_thread is not None:
                        engine._task_thread.join(timeout=2)
                        self.assertFalse(engine._task_thread.is_alive())
                    loop = getattr(engine, '_event_loop', None)
                    if loop is not None:
                        loop.close()

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
                            loop.close()


if __name__ == '__main__':
    unittest.main()
