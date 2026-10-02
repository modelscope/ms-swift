# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import unittest
from copy import deepcopy
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from types import SimpleNamespace

from swift.agent_template import agent_template_map
from swift.template import TEMPLATE_MAPPING, get_template


class TestGLM4_5ToolCallArguments(unittest.TestCase):
    # The GLM-4.5/4.7/5.1 chat templates render each argument with
    # `{{ v | tojson(ensure_ascii=False) if v is not string else v }}`.

    def test_non_string_arguments_are_rendered_as_json(self):
        arguments = {'city': '北京', 'days': 3, 'metric': True, 'extra': None, 'filters': {'tags': ['a', 'b']}}
        message = {'role': 'tool_call', 'content': json.dumps({'name': 'get_weather', 'arguments': arguments})}
        for name in ['glm4_5', 'glm4_7', 'glm5_1']:
            with self.subTest(agent_template=name):
                rendered = agent_template_map[name]()._format_tool_calls([message])
                self.assertIn('<arg_value>北京</arg_value>', rendered)
                self.assertIn('<arg_value>3</arg_value>', rendered)
                self.assertIn('<arg_value>true</arg_value>', rendered)
                self.assertIn('<arg_value>null</arg_value>', rendered)
                self.assertIn('<arg_value>{"tags": ["a", "b"]}</arg_value>', rendered)


class TestGLM4_5ToolCallPrefix(unittest.TestCase):
    # The GLM-4.5/4.6 chat_template renders `{{ '\n<tool_call>' + tc.name }}` for
    # every tool call: a '\n' separates the assistant's non-empty content from
    # <tool_call>. The GLM-5 style templates (glm4_7/glm5_1) have no separator.

    def test_non_empty_content_gets_newline_separator(self):
        agent_template = agent_template_map['glm4_5']()
        prefix = agent_template._add_tool_call_prefix('<tool_call>get_current_weather</tool_call>', {
            'role': 'assistant',
            'content': 'Let me check.'
        })
        self.assertEqual(prefix, '\n<tool_call>get_current_weather</tool_call>')

    def test_empty_content_has_no_separator(self):
        agent_template = agent_template_map['glm4_5']()
        for content in ['', '   ', None, ['Let me check.']]:
            with self.subTest(content=content):
                prefix = agent_template._add_tool_call_prefix('<tool_call>call</tool_call>', {
                    'role': 'assistant',
                    'content': content,
                })
                self.assertEqual(prefix, '<tool_call>call</tool_call>')

    def test_non_assistant_pre_message_has_no_separator(self):
        agent_template = agent_template_map['glm4_5']()
        for pre_message in [None, {'role': 'user', 'content': 'query'}]:
            with self.subTest(pre_message=pre_message):
                prefix = agent_template._add_tool_call_prefix('<tool_call>call</tool_call>', pre_message)
                self.assertEqual(prefix, '<tool_call>call</tool_call>')

    def test_glm5_style_templates_have_no_separator(self):
        for name in ['glm4_7', 'glm5_1']:
            with self.subTest(agent_template=name):
                agent_template = agent_template_map[name]()
                prefix = agent_template._add_tool_call_prefix('<tool_call>call</tool_call>', {
                    'role': 'assistant',
                    'content': 'Let me check.'
                })
                self.assertEqual(prefix, '<tool_call>call</tool_call>')


class TestGLM4_5ToolCallEncoding(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # A local byte tokenizer keeps whitespace visible without model downloads.
        vocab = {char: i for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
        backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        cls.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, eos_token='<|user|>', pad_token='<|endoftext|>')
        cls.processor = SimpleNamespace(
            tokenizer=cls.tokenizer,
            model_info=SimpleNamespace(config={}, task_type='causal_lm', max_model_len=8192),
            model_meta=SimpleNamespace(is_multimodal=False))

    def _encode(self, content, mode, preserve_thinking, history=False, call_count=1, canonical=False):
        template = get_template(self.processor, template_type='glm4_5', preserve_thinking=preserve_thinking)
        # GLM's registered class has additional newline handling after the base encoder.
        self.assertIs(type(template), TEMPLATE_MAPPING['glm4_5'].template_cls)
        template.set_mode(mode)
        functions = [{'name': name, 'arguments': {}} for name in ['weather', 'time'][:call_count]]
        messages = [{'role': 'user', 'content': 'query'}]
        if canonical:
            messages.append({'role': 'assistant', 'content': content, 'loss': False})
            messages.extend({'role': 'tool_call', 'content': function, 'loss': True} for function in functions)
        else:
            messages.append({
                'role': 'assistant',
                'content': content,
                'tool_calls': [{
                    'type': 'function',
                    'function': function
                } for function in functions],
            })
        messages.extend({'role': 'tool', 'content': 'result'} for _ in functions)
        messages.append({'role': 'assistant', 'content': 'done', 'loss': False})
        if history:
            messages.extend([{'role': 'user', 'content': 'thanks'}, {'role': 'assistant', 'content': 'welcome'}])
        inputs = {'messages': messages, 'add_eos': False}
        original = deepcopy(inputs)
        encoded = template.encode(inputs)
        self.assertEqual(inputs, original)
        text = self.tokenizer.decode(encoded['input_ids'])
        start = text.index('<|assistant|>')
        end = text.index('<|observation|>', start)
        return text[start:end], encoded

    def test_content_is_trimmed_before_the_tool_separator(self):
        # GLM-4.5/4.6 chat_template.jinja renders '\n' + content.strip(),
        # followed by '\n<tool_call>' for every call.
        for mode in ['train', 'transformers']:
            for preserve in [False, True]:
                for content in ['Let me check.', 'Let me check.\n', 'Let me check.\n\n', ' \tLet me check.\n ']:
                    for call_count in [1, 2]:
                        with self.subTest(mode=mode, preserve=preserve, content=content, call_count=call_count):
                            actual, _ = self._encode(content, mode, preserve, call_count=call_count)
                            # Inference with preserve_thinking=True does not synthesize an empty think block.
                            thinking = '<think></think>\n' if mode == 'train' or not preserve else ''
                            expected = '<|assistant|>\n' + thinking + 'Let me check.\n<tool_call>weather\n</tool_call>'
                            if call_count == 2:
                                expected += '\n<tool_call>time\n</tool_call>'
                            self.assertEqual(actual, expected)

    def test_thinking_and_history_use_the_registered_newline_handling(self):
        for mode in ['train', 'transformers']:
            for preserve in [False, True]:
                for content in ['<think>reason</think>', '<think>reason</think>\n\n']:
                    with self.subTest(mode=mode, preserve=preserve, content=content):
                        actual, _ = self._encode(content, mode, preserve, history=True)
                        reasoning = 'reason' if preserve else ''
                        self.assertEqual(
                            actual, '<|assistant|>\n<think>' + reasoning + '</think>\n<tool_call>weather\n</tool_call>')

    def test_canonical_messages_keep_separate_supervision(self):
        for content in ['Let me check.', ' \tLet me check.\n\n']:
            with self.subTest(content=content):
                actual, encoded = self._encode(content, 'train', True, canonical=True)
                self.assertEqual(actual,
                                 '<|assistant|>\n<think></think>\nLet me check.\n<tool_call>weather\n</tool_call>')
                supervised = self.tokenizer.decode([token for token in encoded['labels'] if token != -100])
                self.assertEqual(supervised, '\n<tool_call>weather\n</tool_call><|observation|>')


if __name__ == '__main__':
    unittest.main()
