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


if __name__ == '__main__':
    unittest.main()
