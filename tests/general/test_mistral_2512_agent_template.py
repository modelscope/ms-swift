# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from swift.agent_template import agent_template_map
from swift.template import TEMPLATE_MAPPING, StdTemplateInputs

# The chat_template.jinja of every Ministral-3 2512 checkpoint registered by SWIFT
# (Ministral-3-{3B,8B,14B}-Instruct-2512 -> mistral_2512, Ministral-3-{3B,8B,14B}-Reasoning-2512 ->
# mistral_2512_thinking), e.g.
# https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512/blob/main/chat_template.jinja
# declares the tools as one JSON list right before the first user turn and renders every call as
# `[TOOL_CALLS]name[ARGS]arguments`:
#   available_tools = '[AVAILABLE_TOOLS]' + (tools | tojson) + '[/AVAILABLE_TOOLS]'
#   assistant: content, then for each call '[TOOL_CALLS]' + name + '[ARGS]' + arguments, then eos_token
#   tool:      '[TOOL_RESULTS]' + message['content'] | string + '[/TOOL_RESULTS]'
# The expected strings below are what that jinja renders for the same conversations.
MISTRAL_2512_TEMPLATE_TYPES = ['mistral_2512', 'mistral_2512_thinking']
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
CALL_BEIJING = '[TOOL_CALLS]realtime_aqi[ARGS]{"city": "Beijing"}'
CALL_SHANGHAI = '[TOOL_CALLS]realtime_aqi[ARGS]{"city": "Shanghai"}'
RESULT_BEIJING = '{"city": "Beijing", "aqi": "10"}'
RESULT_SHANGHAI = '{"city": "Shanghai", "aqi": "72"}'
AVAILABLE_TOOLS = f'[AVAILABLE_TOOLS]{json.dumps(TOOLS)}[/AVAILABLE_TOOLS]'
# Rendered by the official chat_template.jinja for `[system, user] + tools`.
OFFICIAL_PROMPT = f'<s>[SYSTEM_PROMPT]{SYSTEM}[/SYSTEM_PROMPT]{AVAILABLE_TOOLS}[INST]weather?[/INST]'
USER = {'role': 'user', 'content': 'weather?'}


def _tool_call(city):
    return {'role': 'tool_call', 'content': json.dumps({'name': 'realtime_aqi', 'arguments': {'city': city}})}


def _tool_response(content):
    return {'role': 'tool_response', 'content': content}


def _assistant(content):
    return {'role': 'assistant', 'content': content}


def _render(template_type, messages, mode='train', system=SYSTEM, tools=TOOLS, model_dir=None):
    meta = TEMPLATE_MAPPING[template_type]
    template = meta.template_cls(None, meta)
    # These tests exercise prompt serialization before tokenization; no model download is needed.
    template.model_meta = SimpleNamespace(is_multimodal=False)
    template.model_info = SimpleNamespace(model_dir=model_dir)
    template.init_env_args()
    template.set_mode(mode)
    if system is not None:
        messages = [{'role': 'system', 'content': system}] + messages
    inputs = StdTemplateInputs.from_dict({'messages': messages})
    inputs.tools = tools
    template._swift_prepare_inputs(inputs)
    contexts, _, _ = template._swift_encode(inputs)
    return ''.join(contexts)


class TestMistral2512AgentTemplate(unittest.TestCase):

    def _check(self, messages, expected, **kwargs):
        for template_type in MISTRAL_2512_TEMPLATE_TYPES:
            with self.subTest(template_type=template_type):
                self.assertEqual(_render(template_type, messages, **kwargs), expected)

    def test_native_tool_calls_are_parsed(self):
        # The model answers in its native format; InferEngine._get_toolcall relies on get_toolcall.
        for template_type in MISTRAL_2512_TEMPLATE_TYPES:
            for eos in ['', '</s>']:
                with self.subTest(template_type=template_type, eos=eos):
                    agent_template = agent_template_map[TEMPLATE_MAPPING[template_type].agent_template]()
                    functions = agent_template.get_toolcall(f'Let me check.{CALL_BEIJING}{CALL_SHANGHAI}{eos}')
                    self.assertEqual([function.name for function in functions], ['realtime_aqi', 'realtime_aqi'])
                    self.assertEqual([json.loads(function.arguments) for function in functions], [{
                        'city': 'Beijing'
                    }, {
                        'city': 'Shanghai'
                    }])

    def test_single_tool_call_round(self):
        messages = [USER, _tool_call('Beijing'), _tool_response(RESULT_BEIJING), _assistant('Beijing is fine.')]
        self._check(
            messages, f'{OFFICIAL_PROMPT}{CALL_BEIJING}</s>[TOOL_RESULTS]{RESULT_BEIJING}[/TOOL_RESULTS]'
            'Beijing is fine.</s>')

    def test_text_before_parallel_tool_calls(self):
        messages = [
            USER,
            _assistant('Let me check.'),
            _tool_call('Beijing'),
            _tool_call('Shanghai'),
            _tool_response(RESULT_BEIJING),
            _tool_response(RESULT_SHANGHAI)
        ]
        history = (f'{OFFICIAL_PROMPT}Let me check.{CALL_BEIJING}{CALL_SHANGHAI}</s>'
                   f'[TOOL_RESULTS]{RESULT_BEIJING}[/TOOL_RESULTS][TOOL_RESULTS]{RESULT_SHANGHAI}[/TOOL_RESULTS]')
        self._check(messages + [_assistant('Both fine.')], f'{history}Both fine.</s>')
        # Inference: the model continues right after the tool results.
        self._check(messages, history, mode='transformers')

    def test_sample_ending_with_tool_call(self):
        # Exactly one `</s>` ends the tool-call turn.
        self._check([USER, _tool_call('Beijing')], f'{OFFICIAL_PROMPT}{CALL_BEIJING}</s>')

    def test_user_turn_after_tool_results(self):
        messages = [
            USER,
            _tool_call('Beijing'),
            _tool_response(RESULT_BEIJING), {
                'role': 'user',
                'content': 'and?'
            },
            _assistant('Beijing is fine.')
        ]
        self._check(
            messages, f'{OFFICIAL_PROMPT}{CALL_BEIJING}</s>[TOOL_RESULTS]{RESULT_BEIJING}[/TOOL_RESULTS]'
            '[INST]and?[/INST]Beijing is fine.</s>')

    def test_default_system_precedes_tools(self):
        # Without a system message the jinja renders its default system prompt, then the tools.
        with tempfile.TemporaryDirectory() as model_dir:
            with open(os.path.join(model_dir, 'SYSTEM_PROMPT.txt'), 'w', encoding='utf-8') as f:
                f.write('Default system.')
            self._check([USER, _assistant('Fine.')],
                        f'<s>[SYSTEM_PROMPT]Default system.[/SYSTEM_PROMPT]{AVAILABLE_TOOLS}[INST]weather?[/INST]'
                        'Fine.</s>',
                        system=None,
                        model_dir=model_dir)

    def test_without_tools(self):
        self._check([USER, _assistant('Fine.')],
                    f'<s>[SYSTEM_PROMPT]{SYSTEM}[/SYSTEM_PROMPT][INST]weather?[/INST]Fine.</s>',
                    tools=None)


if __name__ == '__main__':
    unittest.main()
