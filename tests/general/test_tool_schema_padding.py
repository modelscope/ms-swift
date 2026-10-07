# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import os
import shutil
import tempfile
import unittest
from copy import deepcopy

from swift.utils import TOOL_KEYS, remove_arrow_padding

try:
    import pyarrow as pa
except ImportError:
    pa = None


def make_tool(name, properties, description=None, **extra_parameters):
    parameters = {'type': 'object', 'properties': properties}
    parameters.update(extra_parameters)
    function = {'name': name, 'parameters': parameters}
    if description is not None:
        function['description'] = description
    return {'type': 'function', 'function': function}


# Two tools differing at every level: property names, property keywords, `required`,
# `description` and `additionalProperties`. Aligning them into one Arrow struct pads the
# missing fields of each tool with null, which used to reach the prompt as foreign
# parameters (issue #6467) and to crash the Qwen3-Coder formatter.
CONVERT_TEMPERATURE = make_tool(
    'convert_temperature', {
        'temperature': {
            'type': 'number',
            'description': 'The temperature value'
        },
        'unit': {
            'type': 'string',
            'enum': ['celsius', 'fahrenheit'],
            'description': 'The unit'
        },
    },
    description='Convert temperature from one unit to another',
    required=['temperature', 'unit'],
    additionalProperties=False)
SEARCH_WEB = make_tool('search_web', {'query': {'type': 'string'}}, required=['query'])
HETEROGENEOUS_TOOLS = [CONVERT_TEMPERATURE, SEARCH_WEB]


def make_rows(tools):
    return [{
        'messages': [{
            'role': 'user',
            'content': f'query{i}'
        }, {
            'role': 'assistant',
            'content': f'answer{i}'
        }],
        'tools': [tool],
    } for i, tool in enumerate(tools)]


class TestRemoveArrowPadding(unittest.TestCase):

    def test_arrow_aligned_tools_are_restored(self):
        if pa is None:
            self.skipTest('pyarrow is not installed')
        aligned = pa.Table.from_pylist([{'tools': HETEROGENEOUS_TOOLS}]).to_pylist()[0]['tools']
        # Sanity check: pyarrow really padded the first tool with the second one's fields.
        first = aligned[0]['function']
        self.assertIsNone(first['parameters']['properties']['query'])
        self.assertIsNone(aligned[1]['function']['description'])

        cleaned = [remove_arrow_padding(tool) for tool in aligned]

        self.assertEqual(cleaned, HETEROGENEOUS_TOOLS)

    def test_padding_is_removed_from_every_schema_position(self):
        padded = {
            'type': 'function',
            'function': {
                'name': 'f',
                'description': None,
                'strict': None,
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'keep': {
                            'type': 'string',
                            'enum': None,
                            'format': None
                        },
                        'foreign': None,
                        'nested': {
                            'type': 'object',
                            'properties': {
                                'inner': {
                                    'type': 'string'
                                },
                                'foreign': None
                            },
                            '$defs': {
                                'Entry': {
                                    'type': 'string'
                                },
                                'foreign': None
                            },
                            'patternProperties': {
                                '^x': {
                                    'type': 'string'
                                },
                                'foreign': None
                            },
                        },
                        'array': {
                            'type': 'array',
                            'items': None
                        },
                    },
                    'required': None,
                    'additionalProperties': None,
                },
            },
        }
        expected = {
            'type': 'function',
            'function': {
                'name': 'f',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'keep': {
                            'type': 'string'
                        },
                        'nested': {
                            'type': 'object',
                            'properties': {
                                'inner': {
                                    'type': 'string'
                                }
                            },
                            '$defs': {
                                'Entry': {
                                    'type': 'string'
                                }
                            },
                            'patternProperties': {
                                '^x': {
                                    'type': 'string'
                                }
                            },
                        },
                        'array': {
                            'type': 'array'
                        },
                    },
                },
            },
        }

        self.assertEqual(remove_arrow_padding(padded), expected)

    def test_semantic_nulls_and_literal_objects_are_preserved(self):
        literal = {'properties': {'literal_null': None}}
        tool = make_tool(
            'valid', {
                'nullable': {
                    'type': ['string', 'null'],
                    'default': None,
                    'const': None,
                    'enum': [None, 'value'],
                    'examples': [None, 'value']
                },
                'any': True,
                'never': False,
                'unconstrained': {},
                'literal': {
                    'default': literal,
                    'const': literal,
                    'examples': [literal],
                    'x-extra': literal
                },
            },
            description='A valid tool')
        original = deepcopy(tool)

        self.assertEqual(remove_arrow_padding(tool), tool)
        self.assertEqual(tool, original)  # the input is never mutated

    def test_dependencies_may_hold_property_names(self):
        schema = {'dependencies': {'a': ['b', 'c'], 'd': None}, 'dependentRequired': {'e': ['f']}}

        self.assertEqual(
            remove_arrow_padding(schema), {
                'dependencies': {
                    'a': ['b', 'c']
                },
                'dependentRequired': {
                    'e': ['f']
                }
            })

    def test_non_dict_values_are_returned_as_is(self):
        for value in [None, 'text', 3, 1.5, True, ['a', None]]:
            with self.subTest(value=value):
                self.assertEqual(remove_arrow_padding(value), value)


class TestToolsColumnNormalization(unittest.TestCase):
    """`tools` must survive jsonl loading -> preprocess -> Arrow cache -> concat verbatim.

    The rows are read by iterating the dataset rather than with `Dataset.to_list()`: the
    latter returns the raw storage and does not decode a `Json` feature, so it yields JSON
    strings. That is pre-existing `datasets` behaviour and applies to `messages` too.
    """

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        # One file holding both tools: `read_json` aligns them into a single struct, which
        # is how the null padded schemas of issue #6467 reach the template.
        cls.mixed_path = cls._write_jsonl('mixed', make_rows(HETEROGENEOUS_TOOLS))
        cls.split_paths = [
            cls._write_jsonl(f'single{i}', make_rows([tool])) for i, tool in enumerate(HETEROGENEOUS_TOOLS)
        ]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    @classmethod
    def _write_jsonl(cls, name, rows):
        path = os.path.join(cls.temp_dir, f'{name}.jsonl')
        with open(path, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(f'{json.dumps(row, ensure_ascii=False)}\n')
        return path

    @staticmethod
    def _load(paths):
        try:
            from swift.dataset import load_dataset
        except ImportError as e:
            raise unittest.SkipTest(str(e))
        return load_dataset(paths)[0]

    @staticmethod
    def _preprocess(rows, batch_size=1000):
        try:
            from datasets import Dataset

            from swift.dataset import RowPreprocessor
        except ImportError as e:
            raise unittest.SkipTest(str(e))
        dataset = Dataset.from_list(deepcopy(rows))
        return RowPreprocessor()(dataset, num_proc=1, batch_size=batch_size, load_from_cache_file=False)

    @staticmethod
    def _normalize_tools(row):
        try:
            from swift.dataset import RowPreprocessor
        except ImportError as e:
            raise unittest.SkipTest(str(e))
        RowPreprocessor._normalize_tools(row)

    def test_loading_a_jsonl_keeps_every_tool_schema_intact(self):
        rows = make_rows(HETEROGENEOUS_TOOLS)

        dataset = self._load([self.mixed_path])

        self.assertEqual([row['tools'] for row in dataset], [row['tools'] for row in rows])

    def test_the_tools_column_is_stored_as_json(self):
        try:
            from datasets.features import Json, List
        except ImportError as e:
            raise unittest.SkipTest(str(e))

        dataset = self._load([self.mixed_path])

        feature = dataset.features['tools']
        self.assertIsInstance(feature, List)
        self.assertIsInstance(feature.feature, Json)

    def test_concatenating_datasets_keeps_both_tool_schemas(self):
        try:
            from datasets import concatenate_datasets
        except ImportError as e:
            raise unittest.SkipTest(str(e))
        rows = make_rows(HETEROGENEOUS_TOOLS)

        merged = concatenate_datasets([self._load([path]) for path in self.split_paths])

        self.assertEqual([row['tools'] for row in merged], [row['tools'] for row in rows])

    def test_batched_cast_does_not_drop_tool_fields(self):
        # batch_size=1 makes the writer cast every batch to the features inferred from the
        # first one, which silently dropped the parameters only the later tools defined.
        rows = make_rows(HETEROGENEOUS_TOOLS)

        dataset = self._preprocess(rows, batch_size=1)

        for row, expected in zip(dataset, rows):
            self.assertEqual(row['tools'], expected['tools'])

    def test_a_json_string_column_is_parsed(self):
        row = {'tools': json.dumps(HETEROGENEOUS_TOOLS, ensure_ascii=False)}

        self._normalize_tools(row)

        self.assertEqual(row['tools'], HETEROGENEOUS_TOOLS)

    def test_a_single_tool_dict_becomes_a_list(self):
        row = {'tools': deepcopy(SEARCH_WEB)}

        self._normalize_tools(row)

        self.assertEqual(row['tools'], [SEARCH_WEB])

    def test_non_dict_tools_are_left_untouched(self):
        descriptions = ['def get_weather(city): ...', 'def search(query): ...']
        row = {'tools': descriptions}

        self._normalize_tools(row)

        self.assertEqual(row['tools'], descriptions)

    def test_a_bare_string_becomes_a_single_element_list(self):
        row = {'tools': 'not a json schema'}

        self._normalize_tools(row)

        self.assertEqual(row['tools'], ['not a json schema'])

    def test_a_json_scalar_is_not_split_into_characters(self):
        # `json.loads` succeeds but yields no container; left as a bare string the
        # `List(Json())` feature would store it one character per tool.
        for raw, expected in [('"just_a_name"', ['just_a_name']), ('123', [123]), ('null', [None])]:
            with self.subTest(raw=raw):
                row = {'tools': raw}

                self._normalize_tools(row)

                self.assertEqual(row['tools'], expected)

    def test_every_tool_column_is_normalized(self):
        row = {key: [deepcopy(SEARCH_WEB), None] for key in TOOL_KEYS}

        self._normalize_tools(row)

        for key in TOOL_KEYS:
            self.assertEqual(row[key], [SEARCH_WEB, None])

    def test_rows_without_tools_are_untouched(self):
        row = {'messages': [{'role': 'user', 'content': 'q'}], 'tools': None}

        self._normalize_tools(row)

        self.assertIsNone(row['tools'])
        self.assertNotIn('rejected_tools', row)


if __name__ == '__main__':
    unittest.main()
