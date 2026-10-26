# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import json
import unittest
from types import SimpleNamespace

from swift.agent_template import agent_template_map
from swift.template import TEMPLATE_MAPPING
from swift.template.template_inputs import StdTemplateInputs

TOOLS = [{'name': 'realtime_aqi', 'description': 'Get real-time air quality.', 'parameters': {'type': 'object'}}]
RESULTS = ['{"city": "Beijing", "aqi": "10"}', '{"city": "Shanghai", "aqi": "72"}']


def _messages(narration):
    # The documented shape: assistant text (e.g. reasoning) followed by tool calls in the same turn.
    messages = [{'role': 'user', 'content': 'What is the weather like in Beijing and Shanghai today?'}]
    if narration:
        messages.append({'role': 'assistant', 'content': narration})
    messages.extend({
        'role': 'tool_call',
        'content': json.dumps({
            'name': 'realtime_aqi',
            'arguments': {
                'city': city
            }
        })
    } for city in ('Beijing', 'Shanghai'))
    messages.extend({'role': 'tool_response', 'content': result} for result in RESULTS)
    messages.append({'role': 'assistant', 'content': 'Beijing is good, Shanghai is mildly polluted.'})
    return messages


def _make_template(template_type, agent_template):
    meta = TEMPLATE_MAPPING[template_type]
    return meta.template_cls(None, meta, agent_template=agent_template)


class TestMergedAssistantToolResponses(unittest.TestCase):
    """`_swift_prepare_inputs` merges adjacent assistant messages into a list of segments.

    The ReAct-family formatters and the Seed-OSS template still assume a string there
    (Gemma4 was fixed in #10203), so narration before a tool call changes the rendering.
    """

    def test_react_family_keeps_observation_separators(self):
        for name in ('react_en', 'react_zh', 'qwen_en', 'qwen_zh', 'qwen_en_parallel', 'qwen_zh_parallel', 'toolbench',
                     'react_grpo'):
            agent = agent_template_map[name]()
            observation = agent.keyword.observation
            tool_messages = [{'role': 'tool', 'content': result} for result in RESULTS]
            expected = agent._format_tool_responses(agent._format_tool_calls(_messages('')[1:3]), tool_messages)[1]
            with self.subTest(agent_template=name):
                # Agent-support.md: every result is prefixed by the observation keyword and ends with '\n'.
                self.assertEqual(expected, [RESULTS[0], '\n', observation, RESULTS[1], '\n'])
                content = ['Let me check.\n', agent._format_tool_calls(_messages('')[1:3])]
                source = copy.deepcopy(content)
                assistant, responses = agent._format_tool_responses(source, tool_messages)
                self.assertEqual(responses, expected)
                self.assertEqual(assistant, content)

    def test_react_prepare_inputs_with_narration(self):
        template = _make_template('qwen2_5', 'react_en')
        rendered = {}
        for narration in ('', 'Let me check.\n'):
            inputs = StdTemplateInputs.from_dict({'messages': _messages(narration), 'tools': TOOLS})
            template._swift_prepare_inputs(inputs)
            self.assertEqual([m['role'] for m in inputs.messages], ['user', 'assistant', 'tool', 'assistant'])
            rendered[narration] = ''.join(inputs.messages[2]['content'])
        # The tool observations must not depend on whether the assistant narrated before calling.
        self.assertEqual(rendered['Let me check.\n'], rendered[''])
        self.assertIn(f'\nObservation:{RESULTS[1]}\n', rendered['Let me check.\n'])

    def test_seed_oss_narration_before_tool_call(self):
        template = _make_template('seed_oss', 'seed_oss')
        # Exercise the real caller without downloading a model or tokenizer.
        template.model_meta = SimpleNamespace(is_multimodal=False)
        template.init_env_args()
        rendered = {}
        for narration in ('', 'Let me check.\n'):
            inputs = StdTemplateInputs.from_dict({'messages': _messages(narration), 'tools': TOOLS})
            # Must not raise: AttributeError: 'list' object has no attribute 'replace'
            template._swift_prepare_inputs(inputs)
            self.assertEqual([m['role'] for m in inputs.messages], ['user', 'assistant', 'tool', 'assistant'])
            rendered[narration] = inputs.messages[1]['content']
        merged = rendered['Let me check.\n']
        # Segments stay aligned with loss/loss_scale; the thinking block opens the turn.
        self.assertEqual(len(merged), 2)
        self.assertTrue(merged[0].startswith('<seed:think>'))
        self.assertEqual(''.join(merged), rendered[''].replace('</seed:think>', '</seed:think>Let me check.\n', 1))


if __name__ == '__main__':
    unittest.main()
