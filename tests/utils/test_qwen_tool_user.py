# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import unittest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from types import SimpleNamespace

from swift.template import get_template


class TestQwenToolUser(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Byte-level tokens keep this template regression independent of model downloads.
        vocab = {char: i for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
        backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        cls.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            eos_token='<|im_end|>',
            pad_token='<|endoftext|>',
            additional_special_tokens=['<|im_start|>', '<think>', '</think>'])
        cls.tokenizer.model_info = SimpleNamespace(config=SimpleNamespace(), task_type='causal_lm', max_model_len=8192)
        cls.tokenizer.model_meta = SimpleNamespace(is_multimodal=False)

    def make_template(self, **kwargs):
        template = get_template(
            self.tokenizer, template_type='qwen3_5', preserve_thinking=True, add_non_thinking_prefix=False, **kwargs)
        template.set_mode('train')
        return template

    @staticmethod
    def make_data(n_tools=1, n_users=1):
        call = '<think>\nplan\n</think>\n\n<tool_call>\n<function=weather>\n</function>\n</tool_call>'
        answer = '<think>\nreply\n</think>\n\nfinal_answer'
        messages = [{'role': 'user', 'content': 'question'}, {'role': 'assistant', 'content': call}]
        messages += [{'role': 'tool', 'content': f'result_{i}'} for i in range(n_tools)]
        messages += [{'role': 'user', 'content': f'followup_{i}'} for i in range(n_users)]
        messages.append({'role': 'assistant', 'content': answer})
        return {'messages': messages}

    def test_followup_boundaries_and_labels(self):
        for strategy in ('default', 'last_round'):
            for n_tools, n_users in ((1, 0), (2, 0), (1, 1), (2, 1), (1, 2), (2, 2)):
                with self.subTest(strategy=strategy, tools=n_tools, users=n_users):
                    template = self.make_template(loss_scale=strategy)
                    data = self.make_data(n_tools, n_users)
                    original = copy.deepcopy(data)
                    encoded = template.encode(data)
                    call = data['messages'][1]['content']
                    answer = data['messages'][-1]['content']
                    observations = '\n'.join(f'<tool_response>\nresult_{i}\n</tool_response>' for i in range(n_tools))
                    turns = [('user', 'question'), ('assistant', call), ('user', observations)]
                    turns += [('user', f'followup_{i}') for i in range(n_users)]
                    turns.append(('assistant', answer))
                    # Independent ChatML rendering: every user turn retains its own boundary.
                    text = '\n'.join(f'<|im_start|>{role}\n{content}<|im_end|>' for role, content in turns) + '\n'
                    self.assertEqual(self.tokenizer.decode(encoded['input_ids']), text)
                    self.assertEqual(data, original)
                    expected_labels = [-100] * len(encoded['input_ids'])
                    responses = [answer] if strategy == 'last_round' else [call, answer]
                    for response in responses:
                        start = text.index(response)
                        begin = len(self.tokenizer.encode(text[:start], add_special_tokens=False))
                        tokens = self.tokenizer.encode(response + '<|im_end|>\n', add_special_tokens=False)
                        expected_labels[begin:begin + len(tokens)] = tokens
                    self.assertEqual(encoded['labels'], expected_labels)

    def test_openai_and_swift_tool_calls_match(self):
        call = {'name': 'weather', 'arguments': {'city': 'Beijing'}}
        data = self.make_data()
        data['messages'][1] = {
            'role': 'assistant',
            'content': '',
            'tool_calls': [{
                'id': 'call_1',
                'type': 'function',
                'function': call
            }]
        }
        data['messages'][2]['tool_call_id'] = 'call_1'
        native = copy.deepcopy(data)
        native['messages'][1] = {'role': 'tool_call', 'content': call}
        native['messages'][2]['role'] = 'tool_response'
        template = self.make_template()
        self.assertEqual(template.encode(data), template.encode(native))

    def test_inference_after_user_followup(self):
        template = self.make_template(enable_thinking=True)
        template.set_mode('transformers')
        data = self.make_data(n_tools=2, n_users=2)
        data['messages'].pop()
        encoded = template.encode(data)
        text = self.tokenizer.decode(encoded['input_ids'])
        self.assertTrue(text.endswith('followup_1<|im_end|>\n<|im_start|>assistant\n<think>\n'))
        self.assertEqual(text.count('<|im_start|>assistant\n'), 2)
        self.assertIn('</tool_response><|im_end|>\n<|im_start|>user\nfollowup_0', text)
        self.assertNotIn('labels', encoded)

    def test_multiple_rounds_keep_last_round_supervision(self):
        data = self.make_data()
        data['messages'] += self.make_data(n_tools=2, n_users=2)['messages']
        template = self.make_template(loss_scale='last_round')
        encoded = template.encode(data)
        text = self.tokenizer.decode(encoded['input_ids'])
        self.assertEqual(text.count('<tool_response>'), 3)
        self.assertEqual(text.count('<|im_start|>assistant\n'), 4)
        supervised = [token for token in encoded['labels'] if token != -100]
        self.assertEqual(self.tokenizer.decode(supervised), data['messages'][-1]['content'] + '<|im_end|>\n')

    def test_response_loss_weight_is_preserved(self):
        template = self.make_template(is_binary_loss_scale=False)
        data = self.make_data()
        data['messages'][-1]['loss_scale'] = 0.4
        encoded = template.encode(data)
        text = self.tokenizer.decode(encoded['input_ids'])
        start = text.index(data['messages'][-1]['content'])
        begin = len(self.tokenizer.encode(text[:start], add_special_tokens=False))
        length = len(self.tokenizer.encode(data['messages'][-1]['content'], add_special_tokens=False))
        self.assertEqual(encoded['loss_scale'][begin:begin + length], [0.4] * length)
        for label, weight in zip(encoded['labels'], encoded['loss_scale']):
            if label == -100:
                self.assertEqual(weight, 0.)


if __name__ == '__main__':
    unittest.main()
