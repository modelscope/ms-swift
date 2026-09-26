# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from swift.dataset import load_dataset
from swift.loss_scale.base import LossScale
from swift.template import ContextType, get_last_user_round
from swift.template.template_inputs import TemplateInputs


class TestRejectedResponse(unittest.TestCase):

    def test_openai_tool_response_through_dataset_loader(self):
        row = {
            'messages': [
                {
                    'role': 'user',
                    'content': 'What is the weather?'
                },
                {
                    'role': 'assistant',
                    'content': None,
                    'tool_calls': [{
                        'type': 'function',
                        'function': {
                            'name': 'weather',
                            'arguments': '{}'
                        }
                    }]
                },
                {
                    'role': 'tool',
                    'content': 'Sunny'
                },
                {
                    'role': 'assistant',
                    'content': 'It is sunny'
                },
            ],
            'rejected_response':
            'Unknown',
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'preference.jsonl'
            path.write_text(json.dumps(row) + '\n', encoding='utf-8')
            dataset, _ = load_dataset(str(path), split_dataset_ratio=0, num_proc=1, strict=True)
            self.assertEqual(len(dataset), 1)
            result = TemplateInputs.from_dict(dataset[0])
        self.assertEqual(result.rejected.messages, [
            {
                'role': 'user',
                'content': 'What is the weather?'
            },
            {
                'role': 'assistant',
                'content': 'Unknown'
            },
        ])

    def test_user_only_boundary(self):
        messages = [
            {
                'role': 'user',
                'content': 'Question'
            },
            {
                'role': 'assistant',
                'content': 'Call'
            },
            {
                'role': 'tool',
                'content': 'Result'
            },
        ]
        self.assertEqual(get_last_user_round(messages), 2)
        self.assertEqual(get_last_user_round(messages, include_tool=False), 0)
        self.assertEqual(get_last_user_round(messages[1:], include_tool=False), -1)
        self.assertEqual(get_last_user_round([], include_tool=False), -1)

    def test_replace_agent_response_after_last_user(self):
        history = [
            {
                'role': 'user',
                'content': 'Earlier question'
            },
            {
                'role': 'assistant',
                'content': 'Earlier answer'
            },
            {
                'role': 'user',
                'content': 'What is the weather?'
            },
        ]
        rejected_trajectory = [
            {
                'role': 'tool_call',
                'content': '{"name": "wrong_city", "arguments": {}}'
            },
            {
                'role': 'tool',
                'content': 'Wrong city weather'
            },
            {
                'role': 'assistant',
                'content': 'Wrong answer'
            },
        ]
        for role in ('tool', 'tool_response'):
            for prefix in (history[-1:], history):
                for rejected in ('Unknown', [{'role': 'assistant', 'content': 'Unknown'}], rejected_trajectory):
                    with self.subTest(role=role, history=len(prefix), rejected=rejected):
                        messages = prefix + [
                            {
                                'role': 'tool_call',
                                'content': '{"name": "weather", "arguments": {}}'
                            },
                            {
                                'role': role,
                                'content': 'Sunny'
                            },
                            {
                                'role': 'assistant',
                                'content': 'It is sunny'
                            },
                        ]
                        row = {'messages': messages, 'rejected_response': rejected}
                        original = deepcopy(row)
                        result = TemplateInputs.from_dict(row)
                        replacement = [{
                            'role': 'assistant',
                            'content': rejected
                        }] if isinstance(rejected, str) else rejected
                        expected = TemplateInputs.from_dict({'messages': prefix + replacement}).chosen
                        self.assertEqual(result.rejected.messages, expected.messages)
                        self.assertEqual(result.chosen.messages,
                                         TemplateInputs.from_dict({
                                             'messages': messages
                                         }).chosen.messages)
                        self.assertEqual(row, original)

    def test_plain_multi_turn_response(self):
        messages = [
            {
                'role': 'user',
                'content': 'First question'
            },
            {
                'role': 'assistant',
                'content': 'First answer'
            },
            {
                'role': 'user',
                'content': 'Second question'
            },
            {
                'role': 'assistant',
                'content': 'Chosen answer'
            },
        ]
        result = TemplateInputs.from_dict({'messages': messages, 'rejected_response': 'Rejected answer'})
        self.assertEqual(result.rejected.messages, messages[:-1] + [{
            'role': 'assistant',
            'content': 'Rejected answer'
        }])

    def test_rejected_messages_remain_explicit(self):
        chosen = [{'role': 'user', 'content': 'Question'}, {'role': 'assistant', 'content': 'Chosen'}]
        rejected = [{'role': 'user', 'content': 'Other question'}, {'role': 'assistant', 'content': 'Rejected'}]
        result = TemplateInputs.from_dict({'messages': chosen, 'rejected_messages': rejected})
        self.assertEqual(result.rejected.messages, rejected)

    def test_last_round_loss_still_starts_after_tool(self):
        messages = [
            {
                'role': 'user',
                'content': 'Question'
            },
            {
                'role': 'assistant',
                'content': 'Tool call'
            },
            {
                'role': 'tool',
                'content': 'Tool result'
            },
            {
                'role': 'assistant',
                'content': 'Final answer'
            },
        ]
        contexts = [message['content'] for message in messages]
        types = [ContextType.OTHER, ContextType.RESPONSE, ContextType.OTHER, ContextType.RESPONSE]
        result, weights = LossScale('last_round')(contexts, types, messages)
        self.assertEqual(result, contexts)
        self.assertEqual(weights, [0., 0., 0., 1.])


if __name__ == '__main__':
    unittest.main()
