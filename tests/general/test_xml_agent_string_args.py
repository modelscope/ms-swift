# Copyright (c) ModelScope Contributors. All rights reserved.
"""Round-trip tests for XML agent templates that must honour the tool schema.

hy_v3 / hy_v3_preview / minimax_m2 / minimax_m3 write string arguments unquoted.
Formatting a tool call with the template and parsing it back must keep a
string-typed value exactly (numbers, bools, quotes, ``null``) and still decode
integers, booleans, objects, arrays, and JSON null. With no schema, values keep
the template's existing literal decoding; a bare ``null`` is JSON null.
"""
import inspect
import json
import pytest

from swift.agent_template import agent_template_map

TEMPLATES = ['hy_v3_preview', 'hy_v3', 'minimax_m2', 'minimax_m3']
# minimax_m3 drops None parameters when rendering, so a real null cannot round-trip.
NULL_TEMPLATES = ['hy_v3_preview', 'hy_v3', 'minimax_m2']

# Values whose text is a JSON or Python literal, but the schema says string.
STRING_ARGS = {
    'zip_code': '12345',
    'flag_text': 'true',
    'note': "'hi'",
    'amount': '1_000',
    'literal': '[1, 2]',
    'missing': 'null',
}

TYPED_ARGS = {
    'n': 3,
    'b': True,
    'no': False,
    'zero': 0,
    'items': [1, 2],
    'o': {
        'k': [1]
    },
    's': None,
}

# No schema: hy_v3 / minimax_m2 also accept Python literals via ast.literal_eval.
LEGACY_LITERAL = {
    'zip_code': 12345,
    'flag_text': True,
    'note': 'hi',
    'amount': 1000,
    'literal': [1, 2],
    'missing': None,
}
# minimax_m3 only json.loads, so quotes and `1_000` stay text. null is already None.
LEGACY_M3 = {
    'zip_code': 12345,
    'flag_text': True,
    'note': "'hi'",
    'amount': '1_000',
    'literal': [1, 2],
    'missing': None,
}

# (schema type, argument value, expected). 'MISSING' omits the type key.
# Shared by every template: json.loads agrees, and None is not in the value
# (minimax_m3 would drop it on the way out).
CASES = [
    ('string', '12345', '12345'),
    ('string', 'true', 'true'),
    ('string', 'false', 'false'),
    ('string', "'hi'", "'hi'"),
    ('string', '1_000', '1_000'),
    ('string', '[1, 2]', '[1, 2]'),
    ('string', 'null', 'null'),
    ('string', 'beijing', 'beijing'),
    (['string'], 'false', 'false'),
    (['string', 'null'], 'false', 'false'),
    (['string', 'null'], 'null', None),
    (['null', 'string'], 'null', None),
    (['string', 'integer'], '123', '123'),
    (['integer', 'string'], '123', '123'),
    (['null', 'string', 'number'], 'false', 'false'),
    (['null', 'string', 'number'], 'null', None),
    (['string', 'boolean'], 'true', 'true'),
    ('integer', 123, 123),
    ('integer', 0, 0),
    ('boolean', True, True),
    ('boolean', False, False),
    ('array', [1, 2], [1, 2]),
    ('object', {
        'k': [1]
    }, {
        'k': [1]
    }),
    ('MISSING', 'false', False),
    ('MISSING', '12345', 12345),
    ('MISSING', '[1, 2]', [1, 2]),
    ('MISSING', 'null', None),
    ('MISSING', 'beijing', 'beijing'),
    ('MISSING', 'not-json', 'not-json'),
]


def _tools(properties, name='lookup'):
    return [{
        'type': 'function',
        'function': {
            'name': name,
            'description': 'Look up a record',
            'parameters': {
                'type': 'object',
                'properties': properties
            },
        },
    }]


def _prop(type_decl):
    return {} if type_decl == 'MISSING' else {'type': type_decl}


def _render(agent, arguments, name='lookup'):
    message = {
        'role': 'tool_call',
        'content': json.dumps({
            'name': name,
            'arguments': arguments
        }, ensure_ascii=False),
    }
    rendered = agent._format_tool_calls([message])
    if isinstance(rendered, list):
        rendered = ''.join(part for part in rendered if isinstance(part, str))
    return rendered


def _parsed(agent, rendered, tools=None, name='lookup'):
    kwargs = {} if tools is None else {'tools': tools}
    functions = agent.get_toolcall(rendered, **kwargs)
    assert len(functions) == 1
    assert functions[0].name == name
    return json.loads(functions[0].arguments)


@pytest.mark.parametrize('name', TEMPLATES)
def test_supports_tool_schema(name):
    agent = agent_template_map[name]()
    assert agent.supports_tool_schema is True
    # InferEngine._get_toolcall only forwards tools when this bind succeeds.
    inspect.signature(agent.get_toolcall).bind('response', tools=None)


@pytest.mark.parametrize('name', TEMPLATES)
def test_string_args_round_trip(name):
    agent = agent_template_map[name]()
    tools = _tools({key: {'type': 'string'} for key in STRING_ARGS})
    parsed = _parsed(agent, _render(agent, STRING_ARGS), tools)
    assert parsed == STRING_ARGS


@pytest.mark.parametrize('name,expected', [
    ('hy_v3_preview', LEGACY_LITERAL),
    ('hy_v3', LEGACY_LITERAL),
    ('minimax_m2', LEGACY_LITERAL),
    ('minimax_m3', LEGACY_M3),
])
def test_string_args_without_schema_keep_literal_decoding(name, expected):
    agent = agent_template_map[name]()
    parsed = _parsed(agent, _render(agent, STRING_ARGS))
    assert parsed == expected


@pytest.mark.parametrize('name', NULL_TEMPLATES)
def test_non_string_args_round_trip_with_and_without_schema(name):
    agent = agent_template_map[name]()
    properties = {
        'n': {
            'type': 'integer'
        },
        'b': {
            'type': 'boolean'
        },
        'no': {
            'type': 'boolean'
        },
        'zero': {
            'type': 'integer'
        },
        'items': {
            'type': 'array'
        },
        'o': {
            'type': 'object'
        },
        's': {
            'type': 'null'
        },
    }
    rendered = _render(agent, TYPED_ARGS)
    assert _parsed(agent, rendered, _tools(properties)) == TYPED_ARGS
    assert _parsed(agent, rendered) == TYPED_ARGS


def test_minimax_m3_non_string_args_round_trip():
    # A real null is omitted by the MiniMax-M3 renderer, so it is not part of the round trip.
    arguments = {key: value for key, value in TYPED_ARGS.items() if key != 's'}
    agent = agent_template_map['minimax_m3']()
    properties = {
        'n': {
            'type': 'integer'
        },
        'b': {
            'type': 'boolean'
        },
        'no': {
            'type': 'boolean'
        },
        'zero': {
            'type': 'integer'
        },
        'items': {
            'type': 'array'
        },
        'o': {
            'type': 'object'
        },
    }
    rendered = _render(agent, arguments)
    assert _parsed(agent, rendered, _tools(properties)) == arguments
    assert _parsed(agent, rendered) == arguments


@pytest.mark.parametrize('name', TEMPLATES)
@pytest.mark.parametrize('type_decl,value,expected', CASES)
def test_schema_type_round_trip(name, type_decl, value, expected):
    agent = agent_template_map[name]()
    arguments = {'city': value}
    parsed = _parsed(agent, _render(agent, arguments), _tools({'city': _prop(type_decl)}))
    assert parsed == {'city': expected}
    assert type(parsed['city']) is type(expected)


@pytest.mark.parametrize('name', NULL_TEMPLATES)
def test_python_literals_without_schema(name):
    agent = agent_template_map[name]()
    arguments = {'note': "'hi'", 'amount': '1_000'}
    assert _parsed(agent, _render(agent, arguments)) == {'note': 'hi', 'amount': 1000}


def test_minimax_m3_does_not_use_python_literals_without_schema():
    agent = agent_template_map['minimax_m3']()
    arguments = {'note': "'hi'", 'amount': '1_000'}
    assert _parsed(agent, _render(agent, arguments)) == arguments


@pytest.mark.parametrize('name', TEMPLATES)
def test_schema_is_selected_by_function_name(name):
    agent = agent_template_map[name]()
    tools = _tools({'zip_code': {'type': 'string'}}, name='lookup')
    tools += _tools({'zip_code': {'type': 'integer'}}, name='other')
    rendered = _render(agent, {'zip_code': '12345'}, name='lookup')
    assert _parsed(agent, rendered, tools) == {'zip_code': '12345'}


@pytest.mark.parametrize('name', TEMPLATES)
def test_react_fallback_accepts_tools_kwarg(name):
    agent = agent_template_map[name]()
    response = 'Action: lookup\nAction Input: {"zip_code": "12345"}'
    tools = _tools({'zip_code': {'type': 'integer'}})
    functions = agent.get_toolcall(response, tools=tools)
    assert [function.name for function in functions] == ['lookup']
    assert json.loads(functions[0].arguments) == {'zip_code': '12345'}
