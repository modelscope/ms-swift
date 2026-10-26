# Copyright (c) ModelScope Contributors. All rights reserved.
"""OpenAI messages at the template boundary: one normaliser, shared by the template and the datasets."""

import json
import pytest

from swift.dev.template.base import Template as FamilyTemplate
from swift.dev.template.inputs import StdTemplateInputs
from swift.dev.template.intent import check_generation_prompt
from swift.dev.template.meta import TemplateMeta
from swift.dev.template.utils import normalize_messages

WEATHER_CALL = {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "SF"}'}}

OPENAI_EPISODE = [
    {"role": "user", "content": "weather?"},
    {"role": "assistant", "content": None, "tool_calls": [WEATHER_CALL]},
    {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
    {"role": "assistant", "content": "It is sunny."},
]

SWIFT_EPISODE = [
    {"role": "user", "content": "weather?"},
    {"role": "tool_call", "content": {"name": "get_weather", "arguments": {"city": "SF"}}},
    {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
    {"role": "assistant", "content": "It is sunny."},
]

# The normalized OpenAI form keeps the spec's str arguments: the model jinja renders a str via
# tojson (an escaped string) and a dict inline, so folding str into dict here would fork training
# and inference tokens (the parity oracle is test_template_parity_tools).
OPENAI_NORMALIZED = [
    {"role": "user", "content": "weather?"},
    {"role": "tool_call", "content": {"name": "get_weather", "arguments": '{"city": "SF"}'}},
    {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
    {"role": "assistant", "content": "It is sunny."},
]

# ---- normalize_messages ------------------------------------------------------------------------


def test_tool_calls_become_tool_call_messages():
    assert normalize_messages(OPENAI_EPISODE) == OPENAI_NORMALIZED


def test_swift_dialect_passes_through_unchanged():
    assert normalize_messages(SWIFT_EPISODE) == SWIFT_EPISODE
    assert normalize_messages(normalize_messages(OPENAI_EPISODE)) == OPENAI_NORMALIZED


def test_turn_content_and_loss_fields_survive_the_split():
    turn = {"role": "assistant", "content": "Checking.", "loss": False, "tool_calls": [WEATHER_CALL, WEATHER_CALL]}
    call = {"role": "tool_call", "content": {"name": "get_weather", "arguments": '{"city": "SF"}'}, "loss": False}
    assert normalize_messages([turn]) == [{"role": "assistant", "content": "Checking.", "loss": False}, call, call]


def test_unparseable_arguments_are_kept_verbatim():
    turn = {"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": "{not json"}}]}
    assert normalize_messages([turn])[0]["content"] == {"name": "f", "arguments": "{not json"}


def test_reasoning_content_survives_normalization():
    turn = {"role": "assistant", "reasoning_content": "\n2 + 2 = 4\n", "content": "\nIt is 4."}
    assert normalize_messages([turn]) == [turn]


def test_reasoning_before_a_tool_call_stays_with_the_turn():
    turn = {"role": "assistant", "reasoning_content": "need data", "content": None, "tool_calls": [WEATHER_CALL]}
    reasoned, call = normalize_messages([turn])
    assert reasoned == {"role": "assistant", "reasoning_content": "need data", "content": None}
    assert call["role"] == "tool_call"


def test_input_messages_are_not_mutated():
    turn = {"role": "assistant", "reasoning_content": "r", "content": "c"}
    normalize_messages([turn])
    assert turn == {"role": "assistant", "reasoning_content": "r", "content": "c"}


# ---- the two entry points share it -------------------------------------------------------------


def test_template_inputs_accept_openai_messages():
    openai = StdTemplateInputs.from_dict({"messages": OPENAI_EPISODE})
    swift = StdTemplateInputs.from_dict({"messages": SWIFT_EPISODE})
    # from_dict str-dumps the payload; the argument form rides inside with tojson semantics.
    assert json.loads(openai.messages[1]["content"]) == {"name": "get_weather", "arguments": '{"city": "SF"}'}
    assert json.loads(swift.messages[1]["content"]) == {"name": "get_weather", "arguments": {"city": "SF"}}


def test_dataset_converter_keeps_reasoning():
    from swift.dev.dataset.format_converter import OpenAIConverter

    row = OpenAIConverter().convert(
        {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "reasoning_content": "r", "content": "a"},
            ]
        }
    )
    assert row["messages"][-1] == {"role": "assistant", "reasoning_content": "r", "content": "a"}


# ---- reasoning folding is per-family, at encode time --------------------------------------------

# Declared thinking marks of the families the matrix covers (copied from their registrations).
QWEN3_META = {"thinking_prefix": "<think>\n", "non_thinking_prefix": "<think>\n\n</think>\n\n"}
DEEPSEEK_V3_1_META = {"thinking_prefix": "<think>", "non_thinking_prefix": "</think>"}
MINIMAX_M3_META = {"thinking_prefix": "<mm:think>", "non_thinking_prefix": "</mm:think>"}
GLM4_5_META = {"non_thinking_prefix": "<think></think>\n"}
YOUTU_META = {"non_thinking_prefix": "<think>\n\n</think>\n\n"}
HUNYUAN_META = {"non_thinking_prefix": "<think>\n\n</think>\n"}
HY_V3_META = {"thinking_prefix": "<think:opensource>", "non_thinking_prefix": "<think:opensource></think:opensource>"}
KIMI_K3_META = {
    "thinking_prefix": "<|open|>think<|sep|>",
    "non_thinking_prefix": "<|open|>think<|sep|><|close|>think<|sep|><|open|>response<|sep|>",
}


def _fold(cls, messages, **meta_kwargs):
    inputs = StdTemplateInputs(messages=messages)
    template = object.__new__(cls)
    template.template_meta = TemplateMeta("dummy", prefix=[], prompt=[], chat_sep=[], **meta_kwargs)
    template._fold_reasoning_content(inputs)
    return inputs.messages


@pytest.mark.parametrize(
    "meta_kwargs, expected",
    [
        ({}, "<think>\n2 + 2 = 4\n</think>\n\nIt is 4."),
        (QWEN3_META, "<think>\n2 + 2 = 4\n</think>\n\nIt is 4."),
        ({"thinking_prefix": "<think>"}, "<think>2 + 2 = 4\n</think>\n\nIt is 4."),
        (DEEPSEEK_V3_1_META, "<think>2 + 2 = 4</think>It is 4."),
        (MINIMAX_M3_META, "<mm:think>2 + 2 = 4</mm:think>It is 4."),
        (GLM4_5_META, "<think>2 + 2 = 4</think>\nIt is 4."),
        (YOUTU_META, "<think>\n2 + 2 = 4\n</think>\n\nIt is 4."),
        (HUNYUAN_META, "<think>\n2 + 2 = 4\n</think>\nIt is 4."),
        (HY_V3_META, "<think:opensource>2 + 2 = 4</think:opensource>It is 4."),
        (KIMI_K3_META, "<|open|>think<|sep|>2 + 2 = 4<|close|>think<|sep|><|open|>response<|sep|>It is 4."),
    ],
)
def test_fold_form_follows_the_declared_marks(meta_kwargs, expected):
    turn = {"role": "assistant", "reasoning_content": "2 + 2 = 4", "content": "It is 4."}
    assert _fold(FamilyTemplate, [turn], **meta_kwargs) == [{"role": "assistant", "content": expected}]


def test_fold_leaves_non_assistant_turns_untouched():
    turn = {"role": "user", "content": "q", "reasoning_content": "r"}
    assert _fold(FamilyTemplate, [turn]) == [turn]


def test_fold_rejects_inline_think_conflicts():
    with pytest.raises(ValueError, match="both reasoning_content and inline"):
        _fold(FamilyTemplate, [{"role": "assistant", "reasoning_content": "a", "content": "<think>b</think>c"}])


def test_fold_requires_text_content():
    with pytest.raises(ValueError, match="must have text content"):
        _fold(FamilyTemplate, [{"role": "assistant", "reasoning_content": "r", "content": {"token_ids": [1]}}])


def test_gemma_folds_into_channel_markers():
    from swift.dev.template.families.gemma import Gemma4Template

    assert _fold(Gemma4Template, [{"role": "assistant", "reasoning_content": "r", "content": "c"}]) == [
        {"role": "assistant", "content": "<|channel>thought\nr<channel|>c"}
    ]


def _template(cls=FamilyTemplate, **meta_kwargs):
    template = object.__new__(cls)
    template.template_meta = TemplateMeta("dummy", prefix=[], prompt=[], chat_sep=[], **meta_kwargs)
    return template


@pytest.mark.parametrize(
    "meta_kwargs",
    [
        QWEN3_META,
        DEEPSEEK_V3_1_META,
        MINIMAX_M3_META,
        GLM4_5_META,
        YOUTU_META,
        HUNYUAN_META,
        HY_V3_META,
        KIMI_K3_META,
        {"is_thinking": True},
    ],
)
def test_split_reasoning_inverts_the_fold(meta_kwargs):
    """A generated reply splits back into what folding it produced, whether or not the open mark was
    generated (thinking prompts end with it, so the model writes only reasoning + close + content)."""
    template = _template(**meta_kwargs)
    wire = template._thinking_turn_text("2 + 2 = 4", "It is 4.")
    open_marker, _ = template._thinking_marks()
    for generated in (wire, wire[len(open_marker) :]):
        assert template._split_reasoning(generated, reasoning_started=True) == ("2 + 2 = 4", "It is 4.")


@pytest.mark.parametrize(
    ("initial", "generated", "expected"),
    [
        (False, "plain", False),
        (False, "<think>partial", True),
        (False, "<think>r</think>answer", False),
        (True, "partial", True),
        (True, "</think>answer", False),
        (True, "</think>answer<think>new", True),
    ],
)
def test_reasoning_state_follows_marker_order(initial, generated, expected):
    from swift.dev.template_contract import split_reasoning_state

    assert (
        split_reasoning_state(generated, open_marker="<think>", close_marker="</think>", reasoning_started=initial)[2]
        is expected
    )


@pytest.mark.parametrize(
    "generated",
    [
        "need weather</think>It is sunny.",
        "need weather\n</think>\nIt is sunny.",
    ],
)
def test_decode_output_uses_same_normalized_markers_for_split_and_state(generated):
    from swift.dev.template_contract import decode_output

    response = decode_output(
        generated,
        thinking_marks=("<think>\n", "\n</think>\n\n"),
        reasoning_started=True,
    )
    assert response.reasoning_content == "need weather"
    assert response.content == "It is sunny."
    assert response.reasoning_started is False


def test_gemma_decode_splits_special_reasoning_close_before_cleanup():
    from swift.dev.template.families.gemma import Gemma4Template
    from swift.dev.template.output import OutputCodec
    from swift.dev.template_contract import DecodeContext

    class Tokenizer:
        all_special_tokens = ["<channel|>"]

        @staticmethod
        def decode(token_ids, *, skip_special_tokens=True, **kwargs):
            text = "need weather<channel|>It is sunny."
            return text.replace("<channel|>", "") if skip_special_tokens else text

    template = object.__new__(Gemma4Template)
    template.processor = Tokenizer()
    template.template_meta = TemplateMeta(
        "gemma", prefix=[], prompt=[], chat_sep=[], is_thinking=True, thinking_prefix="<|channel>thought\n"
    )
    template.enable_thinking = True
    template.decode_generate_ids = lambda token_ids, **kwargs: template.tokenizer.decode(token_ids, **kwargs)
    response = OutputCodec().decode(template, [], DecodeContext(reasoning_started=True))
    assert response.reasoning_content == "need weather"
    assert response.content == "It is sunny."


def test_truncated_reasoning_never_becomes_answer_content():
    from swift.dev.template.output import OutputCodec
    from swift.dev.template_contract import DecodeContext

    class Tokenizer:
        all_special_tokens = []

        @staticmethod
        def decode(token_ids, **kwargs):
            return "unfinished reasoning"

    template = object.__new__(FamilyTemplate)
    template.processor = Tokenizer()
    template.template_meta = TemplateMeta(
        "qwen",
        prefix=[],
        prompt=[],
        chat_sep=[],
        is_thinking=True,
        thinking_prefix="<think>\n",
        non_thinking_prefix="<think>\n\n</think>\n\n",
    )
    template.enable_thinking = True
    template.decode_generate_ids = lambda token_ids, **kwargs: template.tokenizer.decode(token_ids, **kwargs)
    response = OutputCodec().decode(template, [], DecodeContext(reasoning_started=True, is_finished=False))
    assert response.reasoning_content == "unfinished reasoning"
    assert response.content == ""


def test_moonshot_converts_inline_think_content_into_xtml():
    from swift.dev.template.families.moonshot import KimiK3Template

    template = object.__new__(KimiK3Template)
    assert (
        template._thinking_to_xtml("<think>\nr\n</think>\n\nc")
        == "<|open|>think<|sep|>\nr\n<|close|>think<|sep|><|open|>response<|sep|>\n\nc<|close|>response<|sep|>"
    )


# ---- generation-prompt check -------------------------------------------------------------------


def test_assistant_turn_with_tool_calls_is_complete():
    messages = OPENAI_EPISODE[:2]
    check_generation_prompt({"messages": messages}, add_generation_prompt=False)
    with pytest.raises(ValueError, match="already ends with an assistant turn"):
        check_generation_prompt({"messages": messages}, add_generation_prompt=True)


def test_empty_assistant_turn_is_open():
    check_generation_prompt(
        {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": None}]},
        add_generation_prompt=True,
    )


# ---- the real template renders both dialects to the same tokens --------------------------------


@pytest.fixture(scope="module")
def qwen_template():
    from swift.dev.builders.template import build_template
    from swift.dev.config import TemplateConfig

    try:
        from swift.model import get_model_processor

        _, proc = get_model_processor("Qwen/Qwen2.5-0.5B-Instruct", load_model=False)
    except Exception as exc:  # noqa: BLE001 -- an unimportable legacy stack / unfetchable tokenizer is not a template defect
        pytest.skip(f"processor not available ({type(exc).__name__}: {exc})")
    return build_template(TemplateConfig(template="qwen2_5"), proc)


def test_reasoning_folds_before_rendering(qwen_template):
    encoded = qwen_template.encode(
        {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "reasoning_content": "r", "content": "a"},
            ],
        }
    )
    assert "<think>\nr\n</think>\n\na" in qwen_template.tokenizer.decode(encoded["input_ids"])
