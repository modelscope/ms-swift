"""L3 parity for bridge modes and multimodal engine ids."""

import copy

import pytest

from swift.dev.builders.template import build_template
from swift.dev.config import TemplateConfig
from swift.dev.template import ENGINE, INFER, TRAIN
from swift.dev.tests.component.processor.test_chat_template_parity import REPRESENTATIVES
from swift.dev.tests.tiny import TinyModel

FAMILIES = ("qwen2_5", "qwen3", "llama3", "gemma", "deepseek", "glm4")
OBSERVE_FAMILIES = ("qwen2_5", "qwen3", "llama3")
VL_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
PROMPT = {
    "messages": [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "What is 2+2?"},
    ]
}
API_TURN = {"role": "assistant", "content": "5"}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]
TOOL_HISTORY = [
    *PROMPT["messages"],
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": "get_weather", "arguments": {"city": "Paris"}},
            }
        ],
    },
]
TOOL_RESULT = {"role": "tool", "content": "sunny", "name": "get_weather"}


def _processor(family):
    from swift.model import get_model_processor

    model_id = REPRESENTATIVES.get(family)
    if model_id is None:
        pytest.skip(f"{family}: no tokenizer representative")
    try:
        model_dir = TinyModel.tokenizer_dir(model_id)
        return get_model_processor(model_dir, load_model=False)[1]
    except Exception as exc:
        pytest.skip(f"{model_id}: processor not fetchable ({type(exc).__name__})")


def _templates(family):
    from swift.template import get_template

    processor = _processor(family)
    legacy = get_template(processor, template_type=family, max_length=1024)
    dev = build_template(TemplateConfig(template=family, max_length=1024), processor)
    return legacy, dev


def _legacy_ids(template, sample, mode, *, add_eos=False):
    row = copy.deepcopy(sample)
    if add_eos:
        row["add_eos"] = True
    template.set_mode(mode)
    return list(template.encode(row)["input_ids"])


def _delta(before, after):
    assert after[: len(before)] == before
    return after[len(before) :]


@pytest.mark.slow
@pytest.mark.parametrize("family", FAMILIES)
def test_api_turn_bridge_matches_legacy_modes(family):
    legacy, dev = _templates(family)
    complete = {"messages": [*PROMPT["messages"], API_TURN]}

    feature = dev.render_delta(
        copy.deepcopy(PROMPT["messages"]), [copy.deepcopy(API_TURN)], gen_prompt_before=True, gen_prompt_after=False
    )
    expected_feature = _delta(
        _legacy_ids(legacy, PROMPT, "transformers"),
        _legacy_ids(legacy, complete, "train"),
    )
    assert feature == expected_feature

    engine = dev.render_delta(
        copy.deepcopy(PROMPT["messages"]),
        [copy.deepcopy(API_TURN)],
        gen_prompt_before=True,
        gen_prompt_after=False,
        engine=True,
    )
    expected_engine = _delta(
        _legacy_ids(legacy, PROMPT, "vllm"),
        _legacy_ids(legacy, complete, "vllm", add_eos=True),
    )
    assert engine == expected_engine


@pytest.mark.slow
@pytest.mark.parametrize("family", OBSERVE_FAMILIES)
def test_tool_observe_bridge_matches_legacy_modes(family):
    legacy, dev = _templates(family)
    closed = {"messages": TOOL_HISTORY, "tools": TOOLS}
    reopened = {"messages": [*TOOL_HISTORY, TOOL_RESULT], "tools": TOOLS}

    feature = dev.render_delta(
        copy.deepcopy(TOOL_HISTORY),
        [copy.deepcopy(TOOL_RESULT)],
        tools=copy.deepcopy(TOOLS),
        gen_prompt_before=False,
        gen_prompt_after=True,
    )
    expected_feature = _delta(
        _legacy_ids(legacy, closed, "train"),
        _legacy_ids(legacy, reopened, "transformers"),
    )
    assert feature == expected_feature

    engine = dev.render_delta(
        copy.deepcopy(TOOL_HISTORY),
        [copy.deepcopy(TOOL_RESULT)],
        tools=copy.deepcopy(TOOLS),
        gen_prompt_before=False,
        gen_prompt_after=True,
        engine=True,
    )
    expected_engine = _delta(
        _legacy_ids(legacy, closed, "vllm", add_eos=True),
        _legacy_ids(legacy, reopened, "vllm"),
    )
    assert engine == expected_engine


@pytest.mark.slow
@pytest.mark.parametrize("enable_thinking", [False, True])
def test_qwen3_user_followup_fails_on_history_rewrite(enable_thinking):
    dev = build_template(
        TemplateConfig(
            template="qwen3",
            max_length=1024,
            loss_scale="last_round",
            enable_thinking=enable_thinking,
        ),
        _processor("qwen3"),
    )
    history = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "reasoning_content": "Compute it.", "content": "4"},
    ]
    with pytest.raises(RuntimeError, match="non-monotonic"):
        dev.render_delta(
            history, [{"role": "user", "content": "Explain."}], gen_prompt_before=False, gen_prompt_after=True
        )


@pytest.mark.slow
def test_vl_intents_match_legacy_modes():
    import numpy as np
    from PIL import Image

    from swift.model import get_model_processor
    from swift.template import get_template

    try:
        _, processor = get_model_processor(VL_MODEL, load_model=False)
    except Exception as exc:
        pytest.skip(f"{VL_MODEL}: processor not fetchable ({type(exc).__name__})")
    image = Image.fromarray((np.random.RandomState(7).rand(56, 56, 3) * 255).astype("uint8"))
    complete = {
        "messages": [
            {"role": "user", "content": "<image>What is this?"},
            {"role": "assistant", "content": "A cat."},
        ],
        "images": [image],
    }
    prompt = {"messages": complete["messages"][:1], "images": [image]}
    legacy = get_template(processor, template_type="qwen2_5_vl", max_length=1024)
    dev = build_template(TemplateConfig(template="qwen2_5_vl", max_length=1024), processor)

    legacy.set_mode("train")
    legacy_train = legacy.encode(copy.deepcopy(complete))
    dev_train = dev.encode(copy.deepcopy(complete), intent=TRAIN)
    assert list(dev_train["input_ids"]) == list(legacy_train["input_ids"])
    assert list(dev_train["labels"]) == list(legacy_train["labels"][1:]) + [-100]
    for key in ("pixel_values", "image_grid_thw"):
        actual = np.asarray(dev_train[key])
        expected = np.asarray(legacy_train[key])
        assert actual.shape == expected.shape and np.array_equal(actual, expected)
    assert list(dev.encode(copy.deepcopy(prompt), intent=INFER)["input_ids"]) == _legacy_ids(
        legacy, prompt, "transformers"
    )
    assert list(dev.encode(copy.deepcopy(prompt), intent=ENGINE)["input_ids"]) == _legacy_ids(legacy, prompt, "vllm")
