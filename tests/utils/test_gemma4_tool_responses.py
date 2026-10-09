# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import json
import unittest

from swift.agent_template.gemma4 import Gemma4AgentTemplate
from swift.template.template_inputs import StdTemplateInputs
from swift.template.templates.gemma import Gemma4Template, Gemma4TemplateMeta


class TestGemma4ToolResponses(unittest.TestCase):

    def setUp(self):
        self.agent = Gemma4AgentTemplate()
        self.calls = [f'<|tool_call>call:{name}{{}}<tool_call|>' for name in ('weather', 'time')]
        self.tools = [{'role': 'tool', 'name': name, 'content': 'ok'} for name in ('weather', 'time')]
        self.responses = [
            f'<|tool_response>response:{name}{{value:<|"|>ok<|"|>}}<tool_response|>' for name in ('weather', 'time')
        ]

    def test_native_tool_responses(self):
        for count in (1, 2):
            call = ''.join(self.calls[:count])
            for content in (call, ['I will check.', call], ['I will check.', 'Both tools.', call]):
                for opener in ('', '<|tool_response>'):
                    with self.subTest(count=count, content=content, opener=opener):
                        source = copy.deepcopy(content)
                        if isinstance(source, list):
                            source[-1] += opener
                        else:
                            source += opener
                        original = copy.deepcopy(source)
                        assistant, responses = self.agent._format_tool_responses(source, self.tools[:count])
                        self.assertEqual(assistant, content)
                        self.assertEqual(responses, [''.join(self.responses[:count])])
                        self.assertEqual(source, original)

    def test_opener_only_segment_keeps_its_position(self):
        content = ['I will check.', self.calls[0], '<|tool_response>']
        assistant, responses = self.agent._format_tool_responses(content, self.tools[:1])
        self.assertEqual(assistant, ['I will check.', self.calls[0], ''])
        self.assertEqual(responses, self.responses[:1])

    def test_react_fallback(self):
        action = 'Action: weather\nAction Input: {}'
        for ending in ('', '\n', '\nObservation:'):
            for segmented in (False, True):
                with self.subTest(ending=ending, segmented=segmented):
                    content = action + ending
                    expected = action + '\nObservation:'
                    if segmented:
                        content = ['I will check.', content]
                        expected = ['I will check.', expected]
                    assistant, responses = self.agent._format_tool_responses(content, self.tools)
                    self.assertEqual(assistant, expected)
                    self.assertEqual(responses, ['ok', '\n', 'Observation:', 'ok', '\n'])

    def test_prepare_inputs_preserves_segment_supervision(self):
        # Exercise the real caller without downloading a model or tokenizer.
        template = Gemma4Template(None, Gemma4TemplateMeta('gemma4'), agent_template='gemma4')
        messages = [
            {
                'role': 'user',
                'content': 'Check both.'
            },
            {
                'role': 'assistant',
                'content': 'I will check.',
                'loss': False,
                'loss_scale': 0.5
            },
            {
                'role': 'assistant',
                'content': 'Both tools.'
            },
        ]
        messages.extend({
            'role': 'tool_call',
            'content': json.dumps({
                'name': name,
                'arguments': {}
            }),
            'loss': True,
            'loss_scale': 0.25,
        } for name in ('weather', 'time'))
        messages.extend(copy.deepcopy(self.tools))
        messages.append({'role': 'assistant', 'content': 'Done.', 'loss_scale': 0.75})
        inputs = StdTemplateInputs.from_dict({'messages': messages})
        template._swift_prepare_inputs(inputs)
        self.assertEqual([message['role'] for message in inputs.messages], ['user', 'assistant', 'tool', 'assistant'])
        self.assertEqual(
            inputs.messages[1], {
                'role': 'assistant',
                'content': ['I will check.', 'Both tools.', ''.join(self.calls)],
                'loss': [False, None, True],
                'loss_scale': [0.5, None, 0.25],
            })
        self.assertEqual(inputs.messages[2]['content'], [''.join(self.responses)])
        self.assertEqual(inputs.messages[3], messages[-1])


if __name__ == '__main__':
    unittest.main()
