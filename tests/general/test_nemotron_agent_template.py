# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import unittest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from types import SimpleNamespace

from swift.agent_template import agent_template_map
from swift.template import TEMPLATE_MAPPING, get_template

# The chat_template.jinja of every NVIDIA-Nemotron-3.5 checkpoint registered by SWIFT
# (Lightning-30B-A3B -> nemotron_h, Super-VL-120B-A12B -> nemotron_h_omni,
# Transcribe-30B-A4B -> nemo_speechlm), e.g.
# https://modelscope.cn/models/nv-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/resolve/master/chat_template.jinja
# describes tools as `<tools>\n<function>\n<name>...` and renders/asks for XML tool calls:
#   (content | trim) ~ '\n' ~ '<tool_call>\n<function=' ~ name ~ '>\n' ... '<parameter=' ~ k ~ '>\n' ~ v
#   ~ '\n</parameter>\n' ... '</function>\n</tool_call>\n' ~ '<|im_end|>\n'
NEMOTRON_TEMPLATE_TYPES = ['nemotron_h', 'nemotron_h_omni', 'nemo_speechlm']
SYSTEM = 'You are a helpful assistant.'
TOOLS = [{
    'type': 'function',
    'function': {
        'name': 'realtime_aqi',
        'description': 'Get real-time air quality.',
        'parameters': {
            'type': 'object',
            'properties': {
                'city': {
                    'type': 'string',
                    'description': 'City name'
                }
            },
            'required': ['city']
        }
    }
}]
NATIVE_CALL = '<tool_call>\n<function=realtime_aqi>\n<parameter=city>\nBeijing\n</parameter>\n</function>\n</tool_call>'
NATIVE_TOOLS_HEADER = ('# Tools\n\nYou have access to the following functions:\n\n'
                       '<tools>\n<function>\n<name>realtime_aqi</name>')
TOOL_CALL = {'type': 'function', 'function': {'name': 'realtime_aqi', 'arguments': {'city': 'Beijing'}}}


def _agent_template(template_type):
    return agent_template_map[TEMPLATE_MAPPING[template_type].agent_template]()


class TestNemotronAgentTemplate(unittest.TestCase):

    def test_native_tool_call_is_parsed(self):
        # The model answers in its native format; InferEngine._get_toolcall relies on get_toolcall.
        response = '<think>\nI need the AQI.\n</think>\n' + NATIVE_CALL + '\n'
        for template_type in NEMOTRON_TEMPLATE_TYPES:
            with self.subTest(template_type=template_type):
                functions = _agent_template(template_type).get_toolcall(response)
                self.assertEqual(len(functions), 1)
                self.assertEqual(functions[0].name, 'realtime_aqi')
                self.assertEqual(json.loads(functions[0].arguments), {'city': 'Beijing'})

    def test_tool_calls_are_rendered_in_native_format(self):
        content = json.dumps({'name': 'realtime_aqi', 'arguments': {'city': 'Beijing'}})
        message = {'role': 'tool_call', 'content': content}
        for template_type in NEMOTRON_TEMPLATE_TYPES:
            with self.subTest(template_type=template_type):
                rendered = _agent_template(template_type)._format_tool_calls([message, message])
                # Every call, including the last one, ends with '</tool_call>\n'.
                self.assertEqual(rendered, f'{NATIVE_CALL}\n{NATIVE_CALL}\n')

    def test_tools_are_described_in_native_format(self):
        for template_type in NEMOTRON_TEMPLATE_TYPES:
            with self.subTest(template_type=template_type):
                agent_template = _agent_template(template_type)
                system = agent_template._format_tools(TOOLS, SYSTEM)
                self.assertTrue(system.startswith(f'{SYSTEM}\n\n{NATIVE_TOOLS_HEADER}'), system)
                # Without a system message the jinja starts the system turn with '# Tools' directly.
                system = agent_template._format_tools(TOOLS, None)
                self.assertTrue(system.startswith(NATIVE_TOOLS_HEADER), system)


class TestNemotronToolCallEncoding(unittest.TestCase):

    def setUp(self):
        # A local byte tokenizer keeps the rendered text visible without model downloads.
        vocab = {char: i for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
        backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            eos_token='<|im_end|>',
            pad_token='<|endoftext|>',
            additional_special_tokens=['<|im_start|>'])

    def encode(self, template_type, assistant_content):
        processor = SimpleNamespace(
            tokenizer=self.tokenizer,
            model_info=SimpleNamespace(config={}, task_type='causal_lm', max_model_len=8192),
            model_meta=SimpleNamespace(is_multimodal=False))
        template = get_template(processor, template_type=template_type)
        template.set_mode('train')
        messages = [
            {
                'role': 'user',
                'content': 'AQI in Beijing?'
            },
            {
                'role': 'assistant',
                'content': assistant_content,
                'tool_calls': [TOOL_CALL]
            },
            {
                'role': 'tool',
                'content': '{"aqi": 10}'
            },
            {
                'role': 'assistant',
                'content': 'It is 10.'
            },
        ]
        encoded = template.encode({'messages': messages, 'tools': TOOLS})
        return self.tokenizer.decode(encoded['input_ids'])

    def test_train_encoding_uses_native_tool_call(self):
        for template_type in NEMOTRON_TEMPLATE_TYPES:
            with self.subTest(template_type=template_type):
                text = self.encode(template_type, 'Let me check. ')
                self.assertIn(f'<|im_start|>system\n{NATIVE_TOOLS_HEADER}', text)
                self.assertIn(f'Let me check.\n{NATIVE_CALL}\n<|im_end|>\n', text)
                self.assertIn('<|im_start|>user\n<tool_response>\n{"aqi": 10}\n</tool_response>\n<|im_end|>\n', text)

    def test_train_encoding_without_assistant_text(self):
        # The jinja renders '<think></think>\n' before the first call when there is no assistant text.
        for template_type in ['nemotron_h_omni', 'nemo_speechlm']:
            with self.subTest(template_type=template_type):
                text = self.encode(template_type, None)
                self.assertIn(f'<|im_start|>assistant\n<think></think>\n{NATIVE_CALL}\n<|im_end|>\n', text)


if __name__ == '__main__':
    unittest.main()
