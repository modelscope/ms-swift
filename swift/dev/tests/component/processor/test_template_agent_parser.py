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
    assert calls[0]["function"]["arguments"] == {"limit": "3", "exact": "False"}
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
