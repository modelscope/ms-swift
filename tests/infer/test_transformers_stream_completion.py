# Copyright (c) ModelScope Contributors. All rights reserved.
import asyncio
import torch
import unittest
from queue import Queue
from threading import Event
from transformers import GenerationConfig
from transformers.utils import is_torch_npu_available
from types import SimpleNamespace
from unittest.mock import Mock, patch

from swift.infer_engine import RequestConfig, TransformersEngine
from swift.infer_engine.utils import TokensIteratorStreamer
from swift.template import Template


class _Tokenizer:

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def decode(self, ids, **kwargs):
        if isinstance(ids, int):
            ids = [ids]
        return ''.join({7: '甲', 8: '乙', 9: '结', 10: '束', 11: 'hel', 12: 'lo'}.get(int(token), '') for token in ids)

    def batch_decode(self, ids, **kwargs):
        return [self.decode(row) for row in ids]


class _StreamEngine(TransformersEngine):

    def __init__(self, steps, *, pad=0, eos=2, stop_words=(), pause_after=None, logprobs=False):
        self.steps = steps
        self.pause_after = pause_after
        self.paused, self.resume, self.generated, self.stopped = (Event() for _ in range(4))
        self._queue = Queue()
        self._task_pool, self._adapters_pool = {}, {}
        self._task_thread = None
        self.max_batch_size = 0
        self.model_name = 'controlled-stream'
        self.model = SimpleNamespace(device=torch.npu.current_device() if is_torch_npu_available() else 'cpu')
        self.processor = _Tokenizer(pad)
        self._get_toolcall = Mock(return_value=None)
        self.config = GenerationConfig(
            max_new_tokens=len(steps), eos_token_id=eos, pad_token_id=pad, output_logits=logprobs, num_beams=1)
        self.template = SimpleNamespace(
            tokenizer=self.tokenizer,
            template_meta=SimpleNamespace(stop_words=list(stop_words)),
            generate=self._generate,
            get_generate_ids=lambda ids, length: ids[:, length:],
            decode_generate_ids=self.tokenizer.decode)
        self.template.prepare_generate_kwargs = lambda kwargs, **kw: Template.prepare_generate_kwargs(
            self.template, kwargs, **kw)

    def _generate(self, model, input_ids, streamer, stopping_criteria, logits_processor=(), **kwargs):
        try:
            streamer.put(input_ids)
            for step, tokens in enumerate(self.steps, 1):
                for processor in logits_processor:
                    processor(input_ids, torch.zeros(len(tokens), 16))
                tokens = torch.tensor(tokens)
                input_ids = torch.cat([input_ids, tokens[:, None]], dim=1)
                streamer.put(tokens)
                # Match HF's ordering: tokens are queued before stopping criteria run.
                stopping_criteria(input_ids, None)
                if step == self.pause_after:
                    self.paused.set()
                    self.resume.wait(timeout=5)
        finally:
            streamer.end()
            self.generated.set()

    def _infer(self, infer_requests, request_config, **kwargs):
        inputs = {'input_ids': torch.tensor([[4, 2], [0, 5]]), 'attention_mask': torch.tensor([[1, 1], [0, 1]])}
        return self._infer_stream(
            inputs,
            generation_config=self.config,
            adapter_request=None,
            request_config=request_config,
            template_inputs=[None, None])

    def _fetch_infer_requests(self):
        if self.stopped.is_set():
            raise SystemExit
        return super()._fetch_infer_requests()

    def _infer_worker(self):
        try:
            super()._infer_worker()
        except SystemExit:
            pass


class TestStreamCompletion(unittest.TestCase):

    def test_eos_finishes_once_before_batch_end(self):
        for pad, eos, end in ((0, 2, 2), (2, 2, 2), (0, [2, 3], 3)):
            with self.subTest(pad=pad, eos=eos):
                engine = _StreamEngine([[7, 7], [end, 8], [pad, 8], [pad, 8]], pad=pad, eos=eos, logprobs=True)
                chunks = list(engine._infer(['a', 'b'], RequestConfig(stream=True, logprobs=True, top_logprobs=1)))
                a_chunks = [batch[0] for batch in chunks if batch[0] is not None]
                self.assertEqual([r.choices[0].finish_reason for r in a_chunks], [None, 'stop'])
                self.assertEqual(a_chunks[-1].usage.completion_tokens, 2)
                self.assertEqual(len(a_chunks[-1].choices[0].logprobs['content']), 1)
                self.assertEqual(chunks[1][1].choices[0].finish_reason, None)
                self.assertTrue(all(batch[0] is None for batch in chunks[2:]))
                self.assertEqual(chunks[-1][1].choices[0].finish_reason, 'length')
                self.assertEqual(engine._get_toolcall.call_count, 2)

    def test_pad_alone_is_not_a_stop(self):
        engine = _StreamEngine([[7, 7], [0, 8], [8, 8], [2, 8]])
        chunks = list(engine._infer(['a', 'b'], RequestConfig(stream=True)))
        self.assertIsNone(chunks[1][0])
        a_chunks = [batch[0] for batch in chunks if batch[0] is not None]
        self.assertEqual([r.choices[0].finish_reason for r in a_chunks], [None, None, 'stop'])
        self.assertEqual(''.join(r.choices[0].delta.content for r in a_chunks), '甲乙')

    def test_final_event_flushes_buffered_text(self):
        engine = _StreamEngine([[11, 7], [12, 8], [2, 8], [0, 8]])
        chunks = list(engine._infer(['a', 'b'], RequestConfig(stream=True)))
        self.assertIsNone(chunks[0][0])
        self.assertIsNone(chunks[1][0])
        a_chunks = [batch[0] for batch in chunks if batch[0] is not None]
        self.assertEqual(len(a_chunks), 1)
        self.assertEqual(a_chunks[0].choices[0].delta.content, 'hello')
        self.assertEqual(a_chunks[0].choices[0].finish_reason, 'stop')

    def test_eos_on_first_token_or_at_length_limit(self):
        for steps in ([[2, 7], [0, 8]], [[7, 7], [2, 8]]):
            with self.subTest(steps=steps):
                engine = _StreamEngine(steps)
                chunks = list(engine._infer(['a', 'b'], RequestConfig(stream=True)))
                finals = [
                    batch[0] for batch in chunks
                    if batch[0] is not None and batch[0].choices[0].finish_reason is not None
                ]
                self.assertEqual(len(finals), 1)
                self.assertEqual(finals[0].choices[0].finish_reason, 'stop')
                self.assertEqual(chunks[-1][1].choices[0].finish_reason, 'length')

    def test_sampled_pad_counts_towards_length_limit(self):
        engine = _StreamEngine([[7, 7], [0, 8]], eos=None)
        chunks = list(engine._infer(['a', 'b'], RequestConfig(stream=True)))
        self.assertEqual(chunks[-1][0].choices[0].finish_reason, 'length')
        self.assertEqual(chunks[-1][0].usage.completion_tokens, 2)

    def test_streamed_prompt_is_not_a_generated_eos(self):
        engine = _StreamEngine([[7, 7], [2, 8]])
        engine.template.get_generate_ids = lambda ids, length: ids
        chunks = list(engine._infer(['a', 'b'], RequestConfig(stream=True)))
        a_chunks = [batch[0] for batch in chunks if batch[0] is not None]
        self.assertEqual([r.choices[0].finish_reason for r in a_chunks], [None, 'stop'])
        self.assertEqual(a_chunks[-1].usage.completion_tokens, 2)

    def test_stop_words_have_independent_stream_state(self):
        for stop in ('结束', [9, 10]):
            with self.subTest(stop=stop):
                engine = _StreamEngine([[7, 7], [9, 8], [10, 8], [0, 8]], stop_words=[stop])
                original_next = TokensIteratorStreamer.__next__

                def read_after_generation(streamer):
                    self.assertTrue(engine.generated.wait(timeout=2))
                    return original_next(streamer)

                # The producer has already reached the stop word before any tokens are consumed.
                with patch.object(TokensIteratorStreamer, '__next__', read_after_generation):
                    chunks = list(engine._infer(['a', 'b'], RequestConfig(stream=True)))
                a_chunks = [batch[0] for batch in chunks if batch[0] is not None]
                self.assertEqual([r.choices[0].finish_reason for r in a_chunks], [None, None, 'stop'])
                self.assertEqual(chunks[2][1].choices[0].finish_reason, None)
                self.assertIsNone(chunks[-1][0])


class TestStreamCompletionWorker(unittest.IsolatedAsyncioTestCase):

    async def test_request_closes_while_other_request_is_still_generating(self):
        engine = _StreamEngine([[7, 7], [0, 8], [2, 8], [0, 8]], pause_after=3)
        config = RequestConfig(stream=True)
        try:
            a, b = await asyncio.gather(engine.infer_async('a', config), engine.infer_async('b', config))
            self.assertTrue(await asyncio.to_thread(engine.paused.wait, 2))
            a_chunks = await asyncio.wait_for(self.collect(a), timeout=2)
            self.assertFalse(engine.generated.is_set())
            self.assertEqual([r.choices[0].finish_reason for r in a_chunks], [None, 'stop'])
            self.assertEqual(engine._get_toolcall.call_count, 1)
            engine.resume.set()
            b_chunks = await asyncio.wait_for(self.collect(b), timeout=2)
            self.assertTrue(all(r.choices[0].finish_reason is None for r in b_chunks[:-1]))
            self.assertEqual(b_chunks[-1].choices[0].finish_reason, 'length')
            self.assertEqual(''.join(r.choices[0].delta.content for r in b_chunks), '甲乙乙乙')
            self.assertEqual(b_chunks[-1].usage.completion_tokens, 4)
            self.assertEqual(engine._get_toolcall.call_count, 2)
        finally:
            engine.resume.set()
            await asyncio.to_thread(engine.generated.wait, 2)
            engine.stopped.set()
            if engine._task_thread is not None:
                await asyncio.to_thread(engine._task_thread.join, 2)
                self.assertFalse(engine._task_thread.is_alive())

    @staticmethod
    async def collect(stream):
        return [chunk async for chunk in stream]
