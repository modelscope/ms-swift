# Copyright (c) ModelScope Contributors. All rights reserved.
"""Adversarial tests for tool-argument type discrimination in agent templates.

The wire format renders string arguments unquoted, so the parser relies on the tool schema's
``type`` to decide whether to ``json.loads`` a value. ``type`` may be a bare string, absent, or a
JSON-Schema union list in any order. A value must be kept as a string whenever the union mentions
``'string'``; a bare ``'null'`` value for a nullable string still decodes to ``None``. The cases
below use adversarial values (``'false'``, ``'123'``, ``'{"a":1}'``, ``'  spaced  '``) that
``json.loads`` would silently corrupt if a union containing ``'string'`` were not honoured.

xing4_0 and minicpm5 share the same discrimination logic, so both are exercised over every case.
"""
import json
import pytest

from swift.agent_template import agent_template_map

# (type_decl, value, expected); 'MISSING' means the schema omits the `type` key entirely.
CASES = [
    ('string', 'false', 'false'),
    ('string', '  spaced  ', '  spaced  '),
    ('string', '{"a":1}', '{"a":1}'),
    (['string'], 'false', 'false'),
    (['string', 'null'], 'false', 'false'),
    (['string', 'null'], 'null', None),
    (['null', 'string'], 'null', None),
    (['string', 'integer'], 'false', 'false'),  # union, not the old hard-coded pair
    (['integer', 'string'], '123', '123'),  # union in the other order
    (['null', 'string', 'number'], 'false', 'false'),  # three-element union
    (['null', 'string', 'number'], 'null', None),
    (['string', 'boolean'], 'true', 'true'),
    ('MISSING', 'false', False),  # no type -> best-effort json.loads
    ('MISSING', 'beijing', 'beijing'),
    ('integer', '123', 123),
]


def _make_tools(type_decl):
    prop = {} if type_decl == 'MISSING' else {'type': type_decl}
    return [{
        'type': 'function',
        'function': {
            'name': 'get_weather',
            'parameters': {
                'type': 'object',
                'properties': {
                    'city': prop
                }
            }
        }
    }]


def _parsed_city(functions):
    assert len(functions) == 1
    # Function.__post_init__ json-dumps any non-str arguments, so this is always a JSON string.
    return json.loads(functions[0].arguments)['city']


@pytest.mark.parametrize('type_decl,value,expected', CASES)
def test_xing4_0_tool_arg_types(type_decl, value, expected):
    agent = agent_template_map['xing4_0']()
    content = (f'<tool_call>get_weather<param_key>city</param_key>'
               f'<param_value>{value}</param_value></tool_call>')
    parsed = _parsed_city(agent.get_toolcall(content, _make_tools(type_decl)))
    assert parsed == expected and type(parsed) is type(expected)


@pytest.mark.parametrize('type_decl,value,expected', CASES)
def test_minicpm5_tool_arg_types(type_decl, value, expected):
    agent = agent_template_map['minicpm5']()
    content = f'<function name="get_weather"><param name="city">{value}</param></function>'
    parsed = _parsed_city(agent.get_toolcall(content, _make_tools(type_decl)))
    assert parsed == expected and type(parsed) is type(expected)
