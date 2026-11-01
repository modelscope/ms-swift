# Copyright (c) ModelScope Contributors. All rights reserved.
"""Schema-aware argument typing for templates with an unquoted wire format.

``glm4_5`` / ``glm4_7`` / ``glm5_1`` / ``flint`` / ``ling3`` / ``spark2_5`` /
``seed_oss`` render non-string arguments with ``json.dumps`` (mirroring their
chat templates' ``tojson`` filter), so an integer reaches the wire as ``3`` and
a boolean as ``false``, while strings are written raw. Their parsers must use
the tool schema to tell the two apart: a value whose declared type includes
``string`` is kept verbatim, every other value is decoded as a JSON literal,
and a bare ``null`` on a nullable string is decoded as JSON null. Without a
schema the literal decoding still applies, matching ``xing4_0`` / ``minicpm5``.
"""
import json
import pytest

from swift.agent_template import agent_template_map

TEMPLATES = ['glm4_5', 'glm4_7', 'glm5_1', 'flint', 'ling3', 'spark2_5', 'seed_oss']

# Values that look like JSON literals but are declared (non-nullable) strings.
STRING_CASES = ['false', 'true', '123', '1.5', '{"a": 1}', '[1, 2]', 'null']

TOOLS = [{
    'type': 'function',
    'function': {
        'name': 'run_job',
        'description': 'Run a job',
        'parameters': {
            'type': 'object',
            'properties': {
                'task': {
                    'type': 'string'
                },
                'retries': {
                    'type': 'integer'
                },
                'dry_run': {
                    'type': 'boolean'
                },
                'ratio': {
                    'type': 'number'
                },
                'tags': {
                    'type': 'array',
                    'items': {
                        'type': 'string'
                    }
                },
                'options': {
                    'type': 'object'
                },
                'note': {
                    'type': ['string', 'null']
                },
            }
        }
    }
}]


def _roundtrip(template, arguments, tools=TOOLS):
    agent = agent_template_map[template]()
    content = json.dumps({'name': 'run_job', 'arguments': arguments}, ensure_ascii=False)
    response = agent._format_tool_calls([{'role': 'tool_call', 'content': content}])
    functions = agent.get_toolcall(response, tools)
    assert len(functions) == 1, f'[{template}] response not parsed as one call: {response!r}'
    assert functions[0].name == 'run_job'
    # Function.__post_init__ json-dumps non-str arguments, so this is always a JSON string.
    return json.loads(functions[0].arguments)


@pytest.mark.parametrize('template', TEMPLATES)
def test_typed_arguments_survive_roundtrip(template):
    arguments = {
        'task': 'index_items',
        'retries': 3,
        'dry_run': False,
        'ratio': 0.5,
        'tags': ['a', 'b'],
        'options': {
            'deep': True
        },
        'note': None,
    }
    assert _roundtrip(template, arguments) == arguments


@pytest.mark.parametrize('template', TEMPLATES)
@pytest.mark.parametrize('value', STRING_CASES)
def test_string_arguments_are_kept_verbatim(template, value):
    # The schema declares `task` a plain string, so literal-looking text must survive.
    assert _roundtrip(template, {'task': value}) == {'task': value}


@pytest.mark.parametrize('template', TEMPLATES)
def test_nullable_string_bare_null_decodes_to_none(template):
    # A bare `null` on a ["string", "null"] parameter is ambiguous on the wire;
    # like xing4_0/minicpm5, decode it as JSON null.
    assert _roundtrip(template, {'note': 'null'}) == {'note': None}


@pytest.mark.parametrize('template', TEMPLATES)
def test_no_schema_still_decodes_literals(template):
    arguments = {'task': 'index_items', 'retries': 3, 'dry_run': True, 'note': None}
    assert _roundtrip(template, arguments, tools=None) == arguments


@pytest.mark.parametrize('template', TEMPLATES)
def test_response_without_tool_call_yields_no_functions(template):
    agent = agent_template_map[template]()
    assert agent.get_toolcall('The weather is sunny today.', TOOLS) == []
