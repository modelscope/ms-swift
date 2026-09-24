# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import json
import tempfile
import unittest
from pathlib import Path

from swift.agent_template.gemma4 import Gemma4AgentTemplate
from swift.dataset import load_dataset
from swift.dataset.preprocessor.core import RowPreprocessor
from swift.template.template_inputs import StdTemplateInputs


class TestToolResponseName(unittest.TestCase):

    @staticmethod
    def make_row(role='tool', openai=True):
        names = ['weather', 'time']
        calls = [{'name': name, 'arguments': {'city': 'Beijing'}} for name in names]
        messages = [{'role': 'user', 'content': 'Check the weather and time.'}]
        if openai:
            messages.append({
                'role': 'assistant',
                'content': '',
                'tool_calls': [{
                    'type': 'function',
                    'function': call
                } for call in calls]
            })
        else:
            messages.extend({'role': 'tool_call', 'content': json.dumps(call)} for call in calls)
        messages.extend({'role': role, 'name': name, 'content': f'{name} result'} for name in names)
        messages.append({'role': 'assistant', 'content': 'Done.', 'loss_scale': 0.5})
        return {'messages': messages}

    @staticmethod
    def tool_messages(row):
        inputs = StdTemplateInputs.from_dict(row)
        return [message for message in inputs.messages if message['role'] == 'tool']

    def test_dataset_preserves_tool_names(self):
        agent = Gemma4AgentTemplate()
        for streaming in (False, True):
            # Native calls keep streaming Arrow inference free of mixed string/dict content.
            for role, openai in (('tool', not streaming), ('tool_response', False)):
                with self.subTest(streaming=streaming, role=role):
                    row = self.make_row(role, openai)
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / 'tools.jsonl'
                        path.write_text(json.dumps(row) + '\n', encoding='utf-8')
                        dataset, _ = load_dataset(
                            str(path), streaming=streaming, strict=True, load_from_cache_file=False)
                        loaded = next(iter(dataset))
                    tools = self.tool_messages(loaded)
                    self.assertEqual([message.get('name') for message in tools], ['weather', 'time'])
                    rendered = agent._get_tool_responses(tools)
                    self.assertEqual(rendered, agent._get_tool_responses(self.tool_messages(row)))
                    self.assertIn('response:weather{', rendered)
                    self.assertIn('response:time{', rendered)
                    self.assertEqual(loaded['messages'][-1]['loss_scale'], 0.5)

    def test_missing_and_empty_names_keep_fallback(self):
        agent = Gemma4AgentTemplate()
        rows = []
        for name_fields in ({}, {'name': None}, {'name': ''}):
            row = self.make_row(openai=False)
            for message in row['messages']:
                if message['role'] == 'tool':
                    message.pop('name')
                    message.update(name_fields)
            rows.append(row)
        # Mix named and unnamed rows to exercise Arrow's nullable message fields.
        rows.append(self.make_row(openai=False))
        for streaming in (False, True):
            with self.subTest(streaming=streaming), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'mixed.jsonl'
                path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
                dataset, _ = load_dataset(
                    str(path), streaming=streaming, strict=True, load_from_cache_file=False, shuffle=False)
                loaded_rows = list(dataset)
                self.assertEqual(len(loaded_rows), len(rows))
                for source, loaded in zip(rows, loaded_rows):
                    self.assertEqual(
                        agent._get_tool_responses(self.tool_messages(source)),
                        agent._get_tool_responses(self.tool_messages(loaded)))

    def test_other_message_metadata_is_still_filtered(self):
        for role in ('system', 'user', 'assistant', 'tool_call', 'tool', 'tool_response'):
            with self.subTest(role=role):
                message = {
                    'role': role,
                    'content': 'text',
                    'name': 'weather',
                    'loss': False,
                    'loss_scale': 0.5,
                    'unexpected': 'drop'
                }
                expected = copy.deepcopy(message)
                expected.pop('unexpected')
                if role not in ('tool', 'tool_response'):
                    expected.pop('name')
                row = {'messages': [message]}
                RowPreprocessor._check_messages(row)
                self.assertEqual(row['messages'], [expected])


if __name__ == '__main__':
    unittest.main()
