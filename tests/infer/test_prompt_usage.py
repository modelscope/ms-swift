# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import unittest
from transformers import GenerationConfig
from types import SimpleNamespace

from swift.infer_engine import RequestConfig, TransformersEngine
from swift.infer_engine.infer_engine import InferEngine


def _generate(model, input_ids, generation_config, streamer=None, **kwargs):
    input_ids = input_ids.repeat_interleave(generation_config.num_return_sequences, dim=0)
    generated = torch.full((input_ids.shape[0], 2), 9)
    if streamer is not None:
        streamer.put(input_ids)
        for token in generated.T:
            streamer.put(token)
        streamer.end()
    return {'sequences': torch.cat([input_ids, generated], dim=1)}


class TestPromptUsage(unittest.TestCase):

    def setUp(self):
        self.inputs = {
            'input_ids': torch.tensor([[0, 0, 4, 5], [4, 5, 6, 7]]),
            'attention_mask': torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
        }
        self.engine = object.__new__(TransformersEngine)
        self.engine.model = lambda **kw: SimpleNamespace(logits=torch.tensor([[1., 0.], [0., 1.]]))
        self.engine.model_name = 'test-model'
        self.engine.processor = SimpleNamespace(pad_token_id=0)
        self.engine._adapters_pool = {}
        self.engine.template = SimpleNamespace(
            tokenizer=self.engine.tokenizer,
            prepare_generate_kwargs=lambda kwargs, **kw: kwargs,
            generate=_generate,
            get_generate_ids=lambda ids, length: ids[:, length:],
            decode_generate_ids=lambda ids, **kw: '好' * len(ids),
            debug_logger=lambda data: None,
            task_type='seq_cls',
            decode_seq_cls=lambda logits, top: (logits.argmax(-1).tolist(), [None, None]))

    def test_prompt_count_keeps_padded_width_for_generation(self):
        masks = [
            torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]]),
            torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool),
        ]
        for mask in masks:
            inputs = {**self.inputs, 'attention_mask': mask}
            self.assertEqual(InferEngine._get_num_tokens(inputs), 4)
            self.assertEqual(InferEngine._get_num_tokens(inputs, batch_idx=0), 2)
            self.assertEqual(InferEngine._get_num_tokens(inputs, batch_idx=1), 4)
        embeddings = {'inputs_embeds': torch.zeros(2, 4, 8), 'attention_mask': self.inputs['attention_mask']}
        self.assertEqual(InferEngine._get_num_tokens(embeddings), 4)
        self.assertEqual(InferEngine._get_num_tokens(embeddings, batch_idx=0), 2)
        for mask in (None, torch.ones(2, 1, 4, 4)):
            inputs = {'input_ids': self.inputs['input_ids'], 'attention_mask': mask}
            self.assertEqual(InferEngine._get_num_tokens(inputs, batch_idx=0), 4)

    def test_real_pad_token_ids_and_generation_budget(self):
        # A token equal to pad_token_id is still part of the prompt when its mask is 1.
        inputs = {
            'input_ids': torch.tensor([[0, 0, 0, 5], [4, 5, 6, 7]]),
            'attention_mask': torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])
        }
        self.assertEqual(InferEngine._get_num_tokens(inputs, batch_idx=0), 3)
        self.engine.max_model_len = 10
        self.engine.max_tokens_offset = 0
        for requested in (None, 9):
            config = RequestConfig(max_tokens=requested)
            self.engine.set_default_max_tokens(config, inputs)
            self.assertEqual(config.max_tokens, 6)

    def test_full_generation_usage_and_choices(self):
        for n in (1, 2):
            config = GenerationConfig(max_new_tokens=2, num_return_sequences=n, do_sample=True)
            responses = self.engine._infer_full(
                self.inputs,
                generation_config=config,
                adapter_request=None,
                request_config=RequestConfig(),
                template_inputs=[None, None])
            self.assertEqual([r.usage.prompt_tokens for r in responses], [2, 4])
            self.assertEqual([r.usage.completion_tokens for r in responses], [2 * n, 2 * n])
            self.assertEqual([r.usage.total_tokens for r in responses], [2 + 2 * n, 4 + 2 * n])
            for response in responses:
                self.assertEqual(len(response.choices), n)
                self.assertTrue(all(c.message.content == '好好' for c in response.choices))

    def test_stream_usage_and_generation_slicing(self):
        chunks = list(
            self.engine._infer_stream(
                self.inputs,
                generation_config=GenerationConfig(
                    max_new_tokens=2, output_logits=False, num_beams=1, num_return_sequences=1),
                adapter_request=None,
                request_config=RequestConfig(stream=True),
                template_inputs=[None, None]))
        for chunk in chunks:
            for i, response in enumerate(chunk):
                if response is not None:
                    self.assertEqual(response.usage.prompt_tokens, [2, 4][i])
        self.assertEqual([r.usage.total_tokens for r in chunks[-1]], [4, 6])
        for i in range(2):
            text = ''.join(chunk[i].choices[0].delta.content for chunk in chunks if chunk[i] is not None)
            self.assertEqual(text, '好好')

    def test_forward_only_usage(self):
        responses = self.engine._infer_forward(self.inputs, adapter_request=None, request_config=RequestConfig())
        self.assertEqual([r.usage.prompt_tokens for r in responses], [2, 4])
        self.assertEqual([r.usage.total_tokens for r in responses], [3, 5])


if __name__ == '__main__':
    unittest.main()
