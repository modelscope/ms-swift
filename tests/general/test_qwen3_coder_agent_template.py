# Copyright (c) ModelScope Contributors. All rights reserved.
import unittest
from copy import deepcopy
from itertools import product
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from types import SimpleNamespace

from swift.agent_template import agent_template_map
from swift.template import TEMPLATE_MAPPING, get_template

CALL = '<tool_call>\n<function=get_weather>\n<parameter=city>\nBeijing\n</parameter>\n</function>\n</tool_call>'
SECOND_CALL = '<tool_call>\n<function=finish>\n</function>\n</tool_call>'
USER = '<|im_start|>user\nWeather?<|im_end|>\n<|im_start|>assistant\n'
END = '<|im_end|>\n'


class TestQwen3CoderToolCallContent(unittest.TestCase):

    def setUp(self):
        # An in-memory byte tokenizer preserves whitespace without model downloads.
        vocab = {char: i for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
        backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            eos_token='<|im_end|>',
            pad_token='<|endoftext|>',
            additional_special_tokens=['<|im_start|>', '<|image_pad|>', '<|video_pad|>'])

    def make_template(self, template_type='qwen3_coder', **kwargs):
        processor = SimpleNamespace(
            tokenizer=self.tokenizer,
            model_info=SimpleNamespace(config={}, task_type='causal_lm', max_model_len=8192),
            model_meta=SimpleNamespace(is_multimodal=template_type == 'qwen3_5'))
        template = get_template(processor, template_type=template_type, **kwargs)
        self.assertIs(type(template), TEMPLATE_MAPPING[template_type].template_cls)
        return template

    def test_encode_tool_call_content(self):
        # https://modelscope.cn/models/Qwen/Qwen3-Coder-480B-A35B-Instruct/resolve/master/chat_template.jinja
        # Qwen3-Coder's chat_template.jinja renders
        # '\n' + content|trim + '\n' for nonempty assistant content, then
        # '\n<tool_call>' for each call. Empty/whitespace-only content is omitted.
        cases = [
            ('Let me check.', 'Let me check.\n\n'),
            ('Let me check.\n', 'Let me check.\n\n'),
            ('Let me check.\n\n', 'Let me check.\n\n'),
            (' \tLet me check.\n ', 'Let me check.\n\n'),
            ('First\nSecond', 'First\nSecond\n\n'),
            ('<think>reason</think>\n\n', '<think>reason</think>\n\n'),
            ('', ''),
            (' \t\n ', ''),
            (None, ''),
        ]
        for mode, message_format, call_count, (content, expected_content) in product(['train', 'transformers'],
                                                                                     ['openai', 'canonical'], [1, 2],
                                                                                     cases):
            with self.subTest(mode=mode, message_format=message_format, call_count=call_count, content=content):
                functions = [
                    {
                        'name': 'get_weather',
                        'arguments': {
                            'city': 'Beijing'
                        }
                    },
                    {
                        'name': 'finish',
                        'arguments': {}
                    },
                ][:call_count]
                messages = [{'role': 'user', 'content': 'Weather?'}]
                if message_format == 'openai':
                    tool_calls = [{'type': 'function', 'function': function} for function in functions]
                    messages.append({'role': 'assistant', 'content': content, 'tool_calls': tool_calls})
                else:
                    if content is not None:
                        messages.append({'role': 'assistant', 'content': content})
                    messages.extend({'role': 'tool_call', 'content': function} for function in functions)
                expected = USER + expected_content + CALL
                if call_count == 2:
                    expected += '\n' + SECOND_CALL
                expected += END
                if mode == 'transformers':
                    messages.append({'role': 'tool', 'content': 'Sunny'})
                    expected += '<|im_start|>user\n<tool_response>\nSunny\n</tool_response>\n' + END
                    expected += '<|im_start|>assistant\n'
                inputs = {'messages': messages}
                original = deepcopy(inputs)
                template = self.make_template()
                template.set_mode(mode)
                encoded = template.encode(inputs)
                self.assertEqual(self.tokenizer.decode(encoded['input_ids']), expected)
                self.assertEqual(inputs, original)
                if mode == 'train':
                    prompt_length = len(self.tokenizer.encode(USER, add_special_tokens=False))
                    self.assertEqual(encoded['labels'][:prompt_length], [-100] * prompt_length)
                    self.assertEqual(encoded['labels'][prompt_length:], encoded['input_ids'][prompt_length:])

    def test_plain_assistant_content_is_preserved(self):
        template = self.make_template()
        template.set_mode('train')
        content = ' \tLet me check.\n\n'
        messages = [{'role': 'user', 'content': 'Weather?'}, {'role': 'assistant', 'content': content}]
        encoded = template.encode({'messages': messages})
        self.assertEqual(self.tokenizer.decode(encoded['input_ids']), USER + content + END)

    def test_qwen3_5_keeps_its_own_prefix(self):
        cases = [
            ('<think>\nreason\n</think>\n\n', '<think>\nreason\n</think>\n\n'),
            ('<think>\nreason\n</think>\n\nLet me check.', '<think>\nreason\n</think>\n\nLet me check.\n\n'),
        ]
        for mode, (content, expected_content) in product(['train', 'transformers'], cases):
            with self.subTest(mode=mode, content=content):
                template = self.make_template('qwen3_5', enable_thinking=True, preserve_thinking=True)
                template.set_mode(mode)
                function = {'name': 'get_weather', 'arguments': {'city': 'Beijing'}}
                tool_calls = [{'type': 'function', 'function': function}]
                messages = [
                    {
                        'role': 'user',
                        'content': 'Weather?'
                    },
                    {
                        'role': 'assistant',
                        'content': content,
                        'tool_calls': tool_calls
                    },
                ]
                expected = USER + expected_content + CALL + END
                if mode == 'transformers':
                    messages.append({'role': 'tool', 'content': 'Sunny'})
                    expected += '<|im_start|>user\n<tool_response>\nSunny\n</tool_response>' + END
                    expected += '<|im_start|>assistant\n<think>\n'
                encoded = template.encode({'messages': messages})
                self.assertEqual(self.tokenizer.decode(encoded['input_ids']), expected)

    def test_non_assistant_prefix_is_unchanged(self):
        agent_template = agent_template_map['qwen3_coder']()
        for pre_message in [None, {'role': 'user', 'content': 'query'}]:
            with self.subTest(pre_message=pre_message):
                self.assertEqual(agent_template._add_tool_call_prefix(CALL, pre_message), CALL)


if __name__ == '__main__':
    unittest.main()
