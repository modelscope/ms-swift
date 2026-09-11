# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import pyarrow as pa
import unittest
from copy import deepcopy

from swift.agent_template.base import BaseAgentTemplate
from swift.agent_template.hermes import HermesAgentTemplate
from swift.agent_template.qwen3_coder import Qwen3CoderAgentTemplate


def make_tool(name, properties):
    return {
        'type': 'function',
        'function': {
            'name': name,
            'description': name,
            'parameters': {
                'type': 'object',
                'properties': properties
            },
        },
    }


class TestAgentToolSchema(unittest.TestCase):

    def test_arrow_aligned_tools_keep_their_own_parameters(self):
        tools = [make_tool('first', {'a': {'type': 'string'}}), make_tool('second', {'b': {'type': 'integer'}})]
        aligned = pa.Table.from_pylist([{'tools': tools}]).to_pylist()[0]['tools']
        self.assertIsNone(aligned[0]['function']['parameters']['properties']['b'])
        self.assertIsNone(aligned[1]['function']['parameters']['properties']['a'])

        self.assertEqual([BaseAgentTemplate.wrap_tool(tool) for tool in aligned], tools)

    def test_nested_properties_are_cleaned_without_mutating_input(self):
        tool = make_tool(
            'nested', {
                'object': {
                    'type': 'object',
                    'properties': {
                        'keep': {
                            'type': 'string'
                        },
                        'foreign': None
                    }
                },
                'array': {
                    'type': 'array',
                    'items': {
                        'properties': {
                            'keep': {},
                            'foreign': None
                        }
                    }
                },
                'foreign': None,
            })
        original = deepcopy(tool)

        result = BaseAgentTemplate.wrap_tool(tool)

        properties = result['function']['parameters']['properties']
        self.assertNotIn('foreign', properties)
        self.assertEqual(properties['object']['properties'], {'keep': {'type': 'string'}})
        self.assertEqual(properties['array']['items']['properties'], {'keep': {}})
        self.assertEqual(tool, original)

    def test_schema_branches_and_definitions_are_cleaned(self):
        child = {'properties': {'keep': {}, 'foreign': None}}
        parameters = {
            '$defs': {
                'entry': deepcopy(child)
            },
            'anyOf': [deepcopy(child), {
                'type': 'null'
            }],
            'additionalProperties': deepcopy(child),
        }
        tool = make_tool('branches', {})
        tool['function']['parameters'] = parameters

        result = BaseAgentTemplate.wrap_tool(tool)['function']['parameters']

        expected_child = {'properties': {'keep': {}}}
        self.assertEqual(result['$defs']['entry'], expected_child)
        self.assertEqual(result['anyOf'], [expected_child, {'type': 'null'}])
        self.assertEqual(result['additionalProperties'], expected_child)

    def test_semantic_nulls_boolean_schemas_and_literal_objects_are_preserved(self):
        literal = {'properties': {'literal_null': None}}
        properties = {
            'nullable': {
                'type': ['string', 'null'],
                'default': None,
                'const': None,
                'enum': [None, 'value']
            },
            'any': True,
            'never': False,
            'unconstrained': {},
            'literal': {
                'default': literal,
                'examples': [literal],
                'const': literal,
                'x-extra': literal
            },
        }
        tool = make_tool('valid', properties)

        self.assertEqual(BaseAgentTemplate.wrap_tool(tool), tool)

    def test_unwrapped_tools_and_non_schema_parameters(self):
        tool = make_tool('raw', {'keep': {}, 'foreign': None})['function']
        result = BaseAgentTemplate.wrap_tool(tool)
        self.assertEqual(result, make_tool('raw', {'keep': {}}))

        for parameters in [None, 'a JSON string', [{'name': 'argument', 'description': None}]]:
            with self.subTest(parameters=parameters):
                tool = {'name': 'legacy', 'parameters': parameters}
                self.assertEqual(BaseAgentTemplate.wrap_tool(tool), {'type': 'function', 'function': tool})

    def test_hermes_prompt_does_not_include_foreign_parameters(self):
        tool = make_tool('first', {'a': {'type': 'string'}, 'foreign': None})

        prompt = HermesAgentTemplate()._format_tools([tool])
        rendered_tool = json.loads(prompt.split('<tools>\n', 1)[1].split('\n</tools>', 1)[0])

        self.assertEqual(rendered_tool, make_tool('first', {'a': {'type': 'string'}}))

    def test_qwen3_coder_prompt_accepts_arrow_aligned_tools(self):
        tools = [make_tool('first', {'a': {'type': 'string'}}), make_tool('second', {'b': {'type': 'integer'}})]
        aligned = pa.Table.from_pylist([{'tools': tools}]).to_pylist()[0]['tools']
        # Template preprocessing wraps each tool before formatting the system prompt.
        wrapped = [BaseAgentTemplate.wrap_tool(tool) for tool in aligned]

        prompt = Qwen3CoderAgentTemplate()._format_tools(wrapped)

        self.assertEqual(prompt.count('<name>a</name>'), 1)
        self.assertEqual(prompt.count('<name>b</name>'), 1)


if __name__ == '__main__':
    unittest.main()
