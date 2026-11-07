# Copyright (c) ModelScope Contributors. All rights reserved.
"""Schema-driven decoding contract for agent arguments with untyped wire values."""
import inspect
from pathlib import Path

import json
import pytest

import swift.dev.template.agent
from swift.dev.template.agent import agent_template_map

SCHEMA_TEMPLATES = sorted(name for name, cls in agent_template_map.items()
                          if getattr(cls, 'supports_tool_schema', False))
MIN_SCHEMA_TEMPLATES = 9

# (label, property schema, rendered value, expected value). Compare types too: 1 == True in Python.
CASES = [
    ('string/digits', {
        'type': 'string'
    }, '12345', '12345'),
    ('string/bool', {
        'type': 'string'
    }, 'true', 'true'),
    ('string/leading-zero', {
        'type': 'string'
    }, '01234', '01234'),
    ('string/py-int', {
        'type': 'string'
    }, '1_000', '1_000'),
    ('string/quoted', {
        'type': 'string'
    }, "'hi'", "'hi'"),
    ('string/spaces', {
        'type': 'string'
    }, '  padded  ', '  padded  '),
    ('string/json-like', {
        'type': 'string'
    }, '{"a": 1}', '{"a": 1}'),
    ('integer', {
        'type': 'integer'
    }, 3, 3),
    ('boolean', {
        'type': 'boolean'
    }, False, False),
    ('number', {
        'type': 'number'
    }, 0.5, 0.5),
    ('array', {
        'type': 'array'
    }, [1, 2], [1, 2]),
    ('object', {
        'type': 'object'
    }, {
        'k': 1
    }, {
        'k': 1
    }),
    ('union/null', {
        'type': ['string', 'null']
    }, 'null', None),
    ('union/string-integer', {
        'type': ['string', 'integer']
    }, '123', '123'),
    ('anyOf nullable string', {
        'anyOf': [{
            'type': 'string'
        }, {
            'type': 'null'
        }]
    }, 'false', 'false'),
    ('oneOf string last', {
        'oneOf': [{
            'type': 'integer'
        }, {
            'type': 'string'
        }]
    }, '123', '123'),
    ('undeclared', {}, 'false', False),
]


def _tools(prop):
    return [{
        'type': 'function',
        'function': {
            'name': 'lookup',
            'description': 'Look up a record',
            'parameters': {
                'type': 'object',
                'properties': {
                    'x': prop
                }
            },
        },
    }]


def _render(agent, value):
    content = json.dumps({'name': 'lookup', 'arguments': {'x': value}}, ensure_ascii=False)
    out = agent._format_tool_calls([{'role': 'tool_call', 'content': content}])
    if isinstance(out, list):
        out = ''.join(part for part in out if isinstance(part, str))
    return out


def _parse(agent, text, tools=None):
    kwargs = {} if tools is None else {'tools': tools}
    functions = agent.get_toolcall(text, **kwargs)
    assert len(functions) == 1, text
    return functions[0].name, json.loads(functions[0].arguments)


@pytest.mark.parametrize('name', SCHEMA_TEMPLATES)
@pytest.mark.parametrize('label,prop,value,expected', CASES)
def test_typed_arguments(name, label, prop, value, expected):
    agent = agent_template_map[name]()
    func_name, args = _parse(agent, _render(agent, value), _tools(prop))
    assert func_name == 'lookup'
    assert args['x'] == expected
    assert type(args['x']) is type(expected)


def test_schema_aware_families_are_discovered():
    assert len(SCHEMA_TEMPLATES) >= MIN_SCHEMA_TEMPLATES, SCHEMA_TEMPLATES


@pytest.mark.parametrize('name', SCHEMA_TEMPLATES)
def test_ambiguous_union_is_decoded_deterministically_but_lossy(name):
    agent = agent_template_map[name]()
    prop = {'type': ['string', 'integer']}
    as_str, as_int = _render(agent, '123'), _render(agent, 123)
    assert as_str == as_int
    assert _parse(agent, as_str, _tools(prop))[1]['x'] == '123'
    assert _parse(agent, as_int, _tools(prop))[1]['x'] == '123'


@pytest.mark.parametrize('name', SCHEMA_TEMPLATES)
def test_without_schema_keeps_literal_decoding(name):
    agent = agent_template_map[name]()
    assert _parse(agent, _render(agent, 'false'))[1] == {'x': False}


@pytest.mark.parametrize('name', SCHEMA_TEMPLATES)
def test_supports_tool_schema_and_tools_kwarg_are_paired(name):
    cls = agent_template_map[name]
    assert cls.supports_tool_schema is True
    inspect.signature(cls().get_toolcall).bind('response', tools=None)


@pytest.mark.parametrize('raw_value', ['{1, 2}', "{'a': {1, 2}}", '[{1, 2}]'])
def test_non_json_literal_falls_back_to_raw_text(raw_value):
    raw_text = f'<function name="lookup"><param name="x">{raw_value}</param></function>'
    agent = agent_template_map['minicpm5']()
    func_name, args = _parse(agent, raw_text, _tools({'type': 'object'}))
    assert func_name == 'lookup'
    assert args['x'] == raw_value


@pytest.mark.xfail(strict=True, reason='only top-level properties are decoded; nested items are not wired')
def test_minimax_m3_nested_items_follow_schema():
    from swift.dev.template.agent.minimax_m3 import NS_TOKEN, _to_xml

    agent = agent_template_map['minimax_m3']()
    payload = _to_xml({'x': {'s': 'true', 'a': ['123', 'false']}})
    text = (f'{NS_TOKEN}<tool_call>{NS_TOKEN}<invoke name="lookup">{payload}'
            f'{NS_TOKEN}</invoke>{NS_TOKEN}</tool_call>')
    prop = {
        'type': 'object',
        'properties': {
            's': {
                'type': 'string'
            },
            'a': {
                'type': 'array',
                'items': {
                    'type': 'string'
                }
            }
        }
    }
    _, args = _parse(agent, text, _tools(prop))
    assert args == {'x': {'s': 'true', 'a': ['123', 'false']}}


def test_decoding_rule_is_defined_once():
    package = Path(swift.dev.template.agent.__file__).parent
    duplicated = sorted(
        p.name for p in package.glob('*.py')
        if p.name != 'base.py' and 'def _is_string_param' in p.read_text(encoding='utf-8'))
    assert duplicated == [], f'define the rule in base.py only, found copies in {duplicated}'
