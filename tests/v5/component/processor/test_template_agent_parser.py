"""Every agent template parses back the tool calls it renders (CR F3): the parser is chosen by the agent
template's name, never guessed from the response text."""

import os

import json
import pytest

from swift.dev.template.agent import agent_template_map
from swift.dev.template.output import AgentToolParser

# String values only: XML-parameter formats (qwen3_coder, glm4_5, seed_oss, ...) do not carry value types.
CALLS = [{"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}]


def _render(agent, calls):
    return agent._format_tool_calls([{"role": "tool_call", "content": json.dumps(call)} for call in calls])


def _parsed(parser, text):
    return [(c["function"]["name"], c["function"]["arguments"]) for c in parser.parse(text)]


EXPECTED = [(call["name"], call["arguments"]) for call in CALLS]


@pytest.mark.parametrize("name", sorted(agent_template_map))
def test_declared_call_start_opens_the_rendered_call(name):
    """The declared marker is the source of truth; the rendering only cross-checks it."""
    agent = agent_template_map[name]()
    renders = [_render(agent, [{"name": fn, "arguments": {}}]) for fn in ("alpha_probe", "zeta_probe")]
    head = os.path.commonprefix(renders).strip()
    if agent.tool_call_start is None:
        assert not head, f"{name} renders opening markup {head!r}; declare it as tool_call_start"
    else:
        assert head.startswith(agent.tool_call_start)


@pytest.mark.parametrize("name", sorted(agent_template_map))
@pytest.mark.parametrize("separator", ["\n", "\n\n", " \n"])
def test_agent_parser_round_trips_its_own_rendering(name, separator):
    parser = AgentToolParser(agent_template_map[name]())
    text = f"Let me check.{separator}{_render(parser.agent, CALLS)}"

    assert _parsed(parser, text) == EXPECTED
    assert parser.clean(text) == "Let me check."
    assert parser.parse_errors(text) == []


def test_plain_reply_is_not_a_call():
    parser = AgentToolParser(agent_template_map["hermes"]())
    assert parser.parse("Sunny today.") == []
    assert parser.clean("Sunny today.\n") == "Sunny today."
    assert parser.parse_errors("Sunny today.") == []


def test_opened_but_unparseable_call_is_an_error():
    parser = AgentToolParser(agent_template_map["hermes"]())
    text = f"{parser.agent.tool_call_start}\nnot json\n"
    assert parser.parse(text) == []
    assert parser.parse_errors(text)


def test_llama_call_start_ignores_earlier_braces_and_tolerates_spacing():
    parser = AgentToolParser(agent_template_map["llama3"]())
    text = 'The example {x: 1} is not a call.\n{ "name": "get_weather", "parameters": {"city": "Paris"} }'
    assert _parsed(parser, text) == [("get_weather", {"city": "Paris"})]
    assert parser.clean(text) == "The example {x: 1} is not a call."


def test_llama_accepts_legacy_python_literals():
    parser = AgentToolParser(agent_template_map["llama3"]())
    text = "{\"name\": 'get_weather', \"parameters\": {'city': 'Paris'}}"
    assert _parsed(parser, text) == [("get_weather", {"city": "Paris"})]


@pytest.mark.parametrize("name", ["chatglm4", "glm4"])
def test_glm_call_start_ignores_function_name_in_prose(name):
    parser = AgentToolParser(agent_template_map[name]())
    rendered = _render(parser.agent, CALLS)
    text = f"I will use get_weather for this.\n{rendered}"
    assert _parsed(parser, text) == EXPECTED
    assert parser.clean(text) == "I will use get_weather for this."


def test_xml_arguments_are_coerced_by_tool_schema():
    from swift.dev.template_contract import coerce_tool_call_arguments

    parser = AgentToolParser(agent_template_map["qwen3_5"]())
    rendered = _render(parser.agent, [{"name": "search", "arguments": {"limit": 3, "exact": False}}])
    calls = parser.parse(rendered)
    # Booleans render as JSON literals (`false`), not Python's `False` (legacy #9968): the template writes
    # the argument block as JSON, and the schema coercion below is what turns it back into a bool.
    assert calls[0]["function"]["arguments"] == {"limit": "3", "exact": "false"}
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer"},
                        "exact": {"type": "boolean"},
                    },
                },
            },
        }
    ]
    assert coerce_tool_call_arguments(calls, tools)[0]["function"]["arguments"] == {"limit": 3, "exact": False}


MINICPM5_XML = ('<function name="probe">'
                '<param name="count">123</param>'
                '<param name="flag">true</param>'
                '<param name="note">  padded  </param>'
                '<param name="limit">7</param>'
                '</function>')


def _probe_tools(**param_types):
    return [{
        "type": "function",
        "function": {
            "name": "probe",
            "parameters": {
                "type": "object",
                "properties": {name: {"type": kind} for name, kind in param_types.items()},
            },
        },
    }]


def test_declared_string_parameters_keep_their_wire_text():
    """A bare-parameter format must not decode a schema-declared string: `123` stays "123" (not 123),
    `true` stays "true" (not True, which would re-render as "True"), and whitespace survives."""
    parser = AgentToolParser(agent_template_map["minicpm5"]())
    tools = _probe_tools(count="string", flag="string", note="string", limit="integer")

    arguments = parser.parse(MINICPM5_XML, tools=tools)[0]["function"]["arguments"]

    assert arguments == {"count": "123", "flag": "true", "note": "  padded  ", "limit": 7}


def test_without_tools_a_bare_parameter_still_decodes_as_json():
    """No schema means no way to tell a string from a number, so today's JSON decode stays the default."""
    parser = AgentToolParser(agent_template_map["minicpm5"]())

    arguments = parser.parse(MINICPM5_XML)[0]["function"]["arguments"]

    assert arguments["count"] == 123 and arguments["flag"] is True and arguments["limit"] == 7


def test_tool_schemas_accepts_both_tool_shapes_and_json_strings():
    from swift.dev.template.agent.base import BaseAgentTemplate

    flat = {"name": "probe", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}
    wrapped = {"type": "function", "function": flat}

    for tools in ([flat], [wrapped], [json.dumps(wrapped)]):
        assert BaseAgentTemplate._tool_schemas(tools) == {"probe": flat["parameters"]}
    # Unusable entries are skipped rather than raising: tool text arrives from the user's request.
    assert BaseAgentTemplate._tool_schemas([None, {"name": "x"}, 42]) == {}


def test_schema_aware_formats_accept_the_tools_argument():
    """A family that opts into schema-driven parsing must accept ``tools``, otherwise the parser's
    branching call would raise TypeError on every decode."""
    import inspect

    for name, agent_cls in sorted(agent_template_map.items()):
        agent = agent_cls()
        if not getattr(agent, "supports_tool_schema", False):
            continue
        assert "tools" in inspect.signature(agent.get_toolcall).parameters, (
            f"{name} declares supports_tool_schema but get_toolcall takes "
            f"{inspect.signature(agent.get_toolcall).parameters}")


def test_decode_output_threads_tools_into_the_parser():
    """The decode contract carries `tools`; it must reach the parser, not stop at the codec boundary."""
    from swift.dev.template_contract.output import decode_output

    tools = _probe_tools(count="string", flag="string", note="string", limit="integer")
    response = decode_output(
        MINICPM5_XML,
        tool_parser=AgentToolParser(agent_template_map["minicpm5"]()),
        tools=tuple(tools),
    )

    arguments = response.tool_calls[0]["function"]["arguments"]
    assert arguments == {"count": "123", "flag": "true", "note": "  padded  ", "limit": 7}
