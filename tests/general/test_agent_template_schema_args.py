# Copyright (c) ModelScope Contributors. All rights reserved.
"""Behavioural contract for schema-driven tool arguments.

The unquoted wire formats render string arguments verbatim and everything else with ``tojson``,
so ``"12345"`` and ``12345`` reach the parser as the same bytes. The schema is the only source of
type information, therefore the decode is **deterministic but lossy** for ambiguous unions
(``['string', 'integer']`` with ``123`` always yields the string form).

Every family shares :meth:`BaseAgentTemplate._decode_schema_args`; these tests pin the shared
rules and the boundary conditions that used to regress (AST fallback producing non-JSON values,
``anyOf`` / ``oneOf`` schemas, schema-less fallback, nested items).
"""
import inspect
import json
import pytest
from pathlib import Path
from types import SimpleNamespace

import swift.agent_template
from swift.agent_template import agent_template_map
from swift.infer_engine.infer_engine import InferEngine

# 自动纳入**已声明** supports_tool_schema 的家族；数量下限防止现有接线回退。
# 无法发现“新家族漏设 flag”——那需要 registry 显式分类，不能从 flag 自身推导。
SCHEMA_TEMPLATES = sorted(name for name, cls in agent_template_map.items()
                          if getattr(cls, 'supports_tool_schema', False))
MIN_SCHEMA_TEMPLATES = 13

# (label, property schema, rendered value, expected value).
# Expected types are compared too: `1 == True` and `1 == 1.0` in Python.
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
    """Catches a removed flag (count regression); cannot catch a new family that never set it."""
    assert len(SCHEMA_TEMPLATES) >= MIN_SCHEMA_TEMPLATES, SCHEMA_TEMPLATES


@pytest.mark.parametrize('name', SCHEMA_TEMPLATES)
def test_ambiguous_union_is_decoded_deterministically_but_lossy(name):
    """`'123'` and `123` render to the same bytes, so the union cannot be inverted."""
    agent = agent_template_map[name]()
    prop = {'type': ['string', 'integer']}
    as_str, as_int = _render(agent, '123'), _render(agent, 123)
    assert as_str == as_int
    assert _parse(agent, as_str, _tools(prop))[1]['x'] == '123'
    assert _parse(agent, as_int, _tools(prop))[1]['x'] == '123'


@pytest.mark.parametrize('name', SCHEMA_TEMPLATES)
def test_without_schema_keeps_literal_decoding(name):
    """Explicit contract: `tools=None` still best-effort decodes literals (see DESIGN.md)."""
    agent = agent_template_map[name]()
    text = _render(agent, 'false')
    assert _parse(agent, text)[1] == {'x': False}


@pytest.mark.parametrize('name', SCHEMA_TEMPLATES)
def test_supports_tool_schema_and_tools_kwarg_are_paired(name):
    """The flag alone is not enough: `_get_toolcall` only forwards tools when the signature
    accepts them; a mismatch silently falls back to schema-less decoding."""
    cls = agent_template_map[name]
    assert cls.supports_tool_schema is True
    # bind on the instance so `self` is already taken out of the signature
    inspect.signature(cls().get_toolcall).bind('response', tools=None)


@pytest.mark.parametrize('raw_value', ['{1, 2}', "{'a': {1, 2}}", '[{1, 2}]'])
def test_non_json_literal_falls_back_to_raw_text(raw_value):
    """`ast.literal_eval` can yield non-JSON values, possibly nested.

    `Function` would fail to `json.dumps` them and `InferEngine._get_toolcall` swallows the
    exception, so the tool call silently disappears. The guard must be recursive: a top-level
    `dict`/`list` can still contain a `set`.
    """
    raw_text = f'<tool_call>lookup<param_key>x</param_key><param_value>{raw_value}</param_value></tool_call>'
    agent = agent_template_map['xing4_0']()
    func_name, args = _parse(agent, raw_text, _tools({'type': 'object'}))
    assert func_name == 'lookup'
    assert args['x'] == raw_value

    # Same input through the real engine entry point must not become "no tool call".
    engine = SimpleNamespace(template=SimpleNamespace(agent_template=agent))
    tool_calls = InferEngine._get_toolcall(engine, raw_text, SimpleNamespace(tools=_tools({'type': 'object'})))
    assert tool_calls is not None, raw_value
    assert json.loads(tool_calls[0].function.arguments) == {'x': raw_value}


@pytest.mark.xfail(strict=True, reason='only top-level properties are decoded; nested items are not wired')
def test_minimax_m3_nested_items_follow_schema():
    from swift.agent_template.minimax_m3 import NS_TOKEN, _to_xml

    agent = agent_template_map['minimax_m3']()
    # The nested value must live under the schema-declared parameter `x`, otherwise the fixture
    # would only exercise "parameter without schema" instead of "nested schema not recursive".
    payload = _to_xml({'x': {'s': 'true', 'a': ['123', 'false']}})
    text = (f'{NS_TOKEN}<tool_call>{NS_TOKEN}<invoke name="lookup">{payload}'
            f'{NS_TOKEN}</invoke>{NS_TOKEN}</tool_call>')
    # `_tools(prop)` already wraps `prop` under the parameter `x`; `prop` must therefore describe
    # x's own properties (`s` / `a`), not another nested `x`.
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
    """Auxiliary check: catches the most common copy-paste. Behavioural tests above are the
    actual contract; this one cannot prove that no equivalent logic was inlined elsewhere."""
    package = Path(swift.agent_template.__file__).parent
    duplicated = sorted(
        p.name for p in package.glob('*.py')
        if p.name != 'base.py' and 'def _is_string_param' in p.read_text(encoding='utf-8'))
    assert duplicated == [], f'define the rule in base.py only, found copies in {duplicated}'
