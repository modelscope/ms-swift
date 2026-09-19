# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import unittest
from types import SimpleNamespace

from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from swift.template import get_template


def _make_tokenizer():
    # Byte-level tokens keep this template regression independent of model downloads.
    vocab = {char: i for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token='<|im_end|>',
        pad_token='<|endoftext|>',
        additional_special_tokens=['<|im_start|>', '<think>', '</think>'])
    tokenizer.model_info = SimpleNamespace(config=SimpleNamespace(), task_type='causal_lm', max_model_len=8192)
    tokenizer.model_meta = SimpleNamespace(is_multimodal=False)
    return tokenizer


def _make_glm_tokenizer():
    # GLM renders tool results with its own control tokens, so a GLM-flavoured byte tokenizer keeps this
    # regression independent of model downloads while still exercising the `<|user|>` splice.
    vocab = {char: i for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token='<|endoftext|>',
        pad_token='<|endoftext|>',
        additional_special_tokens=[
            '[gMASK]', '<sop>', '<|system|>', '<|user|>', '<|assistant|>', '<|observation|>', '<think>', '</think>'
        ])
    tokenizer.model_info = SimpleNamespace(config=SimpleNamespace(), task_type='causal_lm', max_model_len=8192)
    tokenizer.model_meta = SimpleNamespace(is_multimodal=False)
    return tokenizer


class TestQwenToolUserFollowup(unittest.TestCase):
    """Reference byte-match: the root fix must reproduce independent ChatML rendering for qwen3_5."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = _make_tokenizer()

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

    def test_no_followup_is_unchanged(self):
        # The fix must be a no-op when no user turn follows the tool results.
        template = self.make_template()
        data = self.make_data(n_tools=2, n_users=0)
        encoded = template.encode(data)
        text = self.tokenizer.decode(encoded['input_ids'])
        self.assertIn('<tool_response>\nresult_0\n</tool_response>', text)

    def test_openai_and_swift_tool_calls_match(self):
        # The follow-up path must treat OpenAI-format tool_calls and native tool_call/tool_response alike.
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

    def test_response_loss_weight_is_preserved(self):
        # Splicing the follow-up into the query must not disturb the assistant response loss weights.
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


class TestHermesToolUserFollowup(unittest.TestCase):
    """The root fix generalizes: a non-qwen ChatML template stops crashing and keeps the user boundary."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = _make_tokenizer()

    def make_template(self):
        template = get_template(self.tokenizer, template_type='qwen2_5', agent_template='hermes')
        template.set_mode('train')
        return template

    @staticmethod
    def make_data():
        messages = [
            {'role': 'user', 'content': 'question'},
            {'role': 'assistant', 'content': '<tool_call>\n{"name": "weather", "arguments": {}}\n</tool_call>'},
            {'role': 'tool', 'content': 'result_0'},
            {'role': 'user', 'content': 'followup_0'},
            {'role': 'assistant', 'content': 'final_answer'},
        ]
        return {'messages': messages}

    def test_does_not_crash_and_keeps_boundary(self):
        template = self.make_template()
        encoded = template.encode(self.make_data())
        text = self.tokenizer.decode(encoded['input_ids'])
        self.assertIn('<tool_response>\nresult_0\n</tool_response>', text)
        self.assertIn('followup_0', text)
        # The follow-up user keeps its own turn boundary rather than being glued to the tool result.
        self.assertIn('</tool_response><|im_end|>\n<|im_start|>user\nfollowup_0', text)


class TestGLMToolUserFollowup(unittest.TestCase):
    """GLM renders tool results independently; a follow-up user must become a normal `<|user|>` turn."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = _make_glm_tokenizer()

    def make_template(self):
        template = get_template(self.tokenizer, template_type='glm4_5')
        template.set_mode('train')
        return template

    @staticmethod
    def make_data(n_users=1):
        call = '<tool_call>weather\n<arg_key>city</arg_key>\n<arg_value>BJ</arg_value>\n</tool_call>'
        messages = [
            {'role': 'user', 'content': 'question'},
            {'role': 'assistant', 'content': call},
            {'role': 'tool', 'content': 'result_0'},
        ]
        messages += [{'role': 'user', 'content': f'followup_{i}'} for i in range(n_users)]
        messages.append({'role': 'assistant', 'content': 'final_answer'})
        return {'messages': messages}

    def test_followup_spliced_before_assistant(self):
        template = self.make_template()
        encoded = template.encode(self.make_data(n_users=1))
        text = self.tokenizer.decode(encoded['input_ids'])
        # Matches the official jinja: `</tool_response><|user|>\n{followup}<|assistant|>`.
        self.assertIn('</tool_response><|user|>\nfollowup_0<|assistant|>', text)
        # The follow-up belongs to the query side and must not be supervised.
        supervised = self.tokenizer.decode([t for t in encoded['labels'] if t != -100])
        self.assertNotIn('followup_0', supervised)

    def test_no_followup_is_unchanged(self):
        template = self.make_template()
        encoded = template.encode(self.make_data(n_users=0))
        text = self.tokenizer.decode(encoded['input_ids'])
        self.assertIn('</tool_response><|assistant|>', text)
        self.assertNotIn('<|user|>\nfollowup', text)


if __name__ == '__main__':
    unittest.main()
