"""L3 bridge parity: the incremental render is what the shipped jinja renders, or its own refusal.

The full render of a conversation is pinned to the checkpoint's chat template by the parity matrix.
What these cases add is the *incremental* path: a rollout appends turns to a conversation it has
already rendered, so the ids it appends must be exactly the tail the full render adds -- and when the
jinja itself rewrites a turn it already wrote (Qwen3 renders its empty think block on the last message
only, so appending a tool response moves it), the incremental path must refuse instead of quietly
disagreeing with the full render.
"""

import copy

import pytest

from swift.dev.builders.template import build_template
from swift.dev.config import TemplateConfig
from swift.dev.template import ENGINE, INFER, TRAIN
from swift.dev.tests.component.processor.test_chat_template_parity import REPRESENTATIVES, _unwrap_official
from swift.dev.tests.tiny import TinyModel

FAMILIES = ("qwen2_5", "qwen3", "llama3", "gemma", "deepseek", "glm4")
OBSERVE_FAMILIES = ("qwen2_5", "qwen3", "llama3")
VL_TEMPLATE = "qwen2_5_vl"
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


def _processor(template):
    from swift.model import get_model_processor

    model_id = REPRESENTATIVES.get(template)
    if model_id is None:
        pytest.skip(f"{template}: no tokenizer representative")
    try:
        model_dir = TinyModel.tokenizer_dir(model_id)
        return get_model_processor(model_dir, load_model=False)[1]
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{model_id}: processor not fetchable ({type(exc).__name__})")


def _dev(template, processor):
    return build_template(TemplateConfig(template=template, max_length=1024), processor)


def _official(renderer, messages, add_generation_prompt, tools=None):
    """(ids, the messages actually rendered) from the checkpoint's own chat template."""
    kwargs = {"tools": copy.deepcopy(tools)} if tools else {}
    try:
        encoded = renderer.apply_chat_template(
            copy.deepcopy(messages), tokenize=True, add_generation_prompt=add_generation_prompt, **kwargs)
        return list(_unwrap_official(encoded)), messages
    except Exception:  # noqa: BLE001 -- the Gemma 2 jinjas refuse the system role
        stripped = [message for message in messages if message["role"] != "system"]
        try:
            encoded = renderer.apply_chat_template(
                copy.deepcopy(stripped), tokenize=True, add_generation_prompt=add_generation_prompt, **kwargs)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"the shipped jinja cannot render this sample ({type(exc).__name__}: {exc})")
        return list(_unwrap_official(encoded)), stripped


@pytest.mark.slow
@pytest.mark.parametrize("family", FAMILIES)
def test_api_turn_bridge_appends_the_jinja_tail(family):
    processor = _processor(family)
    dev = _dev(family, processor)
    renderer = processor if hasattr(processor, "apply_chat_template") else processor.tokenizer
    complete = [*PROMPT["messages"], API_TURN]
    prompt_ids, prompt_messages = _official(renderer, copy.deepcopy(PROMPT["messages"]), True)
    train_ids, _ = _official(renderer, copy.deepcopy(complete), False)
    assert train_ids[: len(prompt_ids)] == prompt_ids, (
        f"{family}: the jinja rewrites the prompt when the turn is appended, so no delta exists")

    expected = train_ids[len(prompt_ids):]
    feature = dev.render_delta(
        copy.deepcopy(prompt_messages), [copy.deepcopy(API_TURN)], gen_prompt_before=True, gen_prompt_after=False)
    assert list(feature) == expected, f"{family}: the feature delta is not the jinja's tail"
    # the jinja has no mode, so the engine form appends the very same tail
    engine = dev.render_delta(
        copy.deepcopy(prompt_messages),
        [copy.deepcopy(API_TURN)],
        gen_prompt_before=True,
        gen_prompt_after=False,
        engine=True,
    )
    assert list(engine) == expected, f"{family}: the engine delta is not the jinja's tail"


@pytest.mark.slow
@pytest.mark.parametrize("family", OBSERVE_FAMILIES)
def test_tool_observe_bridge_matches_the_shipped_jinja(family):
    processor = _processor(family)
    dev = _dev(family, processor)
    renderer = processor if hasattr(processor, "apply_chat_template") else processor.tokenizer
    closed_ids, closed_messages = _official(renderer, copy.deepcopy(TOOL_HISTORY), False, tools=TOOLS)
    reopened_ids, _ = _official(renderer, copy.deepcopy([*TOOL_HISTORY, TOOL_RESULT]), True, tools=TOOLS)
    monotonic = reopened_ids[: len(closed_ids)] == closed_ids

    def observe(engine=False):
        return dev.render_delta(
            copy.deepcopy(closed_messages),
            [copy.deepcopy(TOOL_RESULT)],
            tools=copy.deepcopy(TOOLS),
            gen_prompt_before=False,
            gen_prompt_after=True,
            engine=engine,
        )

    if not monotonic:
        # the jinja rewrites the turn it already rendered (Qwen3's empty think block follows the last
        # message), so the appended turns are not a prefix extension: the incremental path must say so
        with pytest.raises(RuntimeError, match="non-monotonic"):
            observe()
        with pytest.raises(RuntimeError, match="non-monotonic"):
            observe(engine=True)
        return

    # the tool-description layer is dev's own (the parity matrix excludes it), so the tail is checked
    # against dev's own full render -- which the matrix pins to the jinja on every non-tool shape
    sample = {"messages": copy.deepcopy(TOOL_HISTORY), "tools": copy.deepcopy(TOOLS)}
    sample_open = {"messages": copy.deepcopy([*TOOL_HISTORY, TOOL_RESULT]), "tools": copy.deepcopy(TOOLS)}
    feature = list(observe())
    engine = list(observe(engine=True))
    closed = list(dev.encode(sample, intent=TRAIN)["input_ids"])
    for label, delta, intent in (("feature", feature, INFER), ("engine", engine, ENGINE)):
        reopened = list(dev.encode(sample_open, intent=intent)["input_ids"])
        assert reopened[: len(closed)] == closed, f"{family}: the {label} full render is not a prefix extension"
        assert delta == reopened[len(closed):], f"{family}: the {label} delta is not the full render's tail"


@pytest.mark.slow
@pytest.mark.parametrize("enable_thinking", [False, True])
def test_qwen3_user_followup_fails_on_history_rewrite(enable_thinking):
    processor = _processor("qwen3")
    dev = build_template(
        TemplateConfig(
            template="qwen3",
            max_length=1024,
            loss_scale="last_round",
            enable_thinking=enable_thinking,
        ),
        processor,
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
def test_vl_intents_match_the_shipped_jinja():
    import numpy as np
    from PIL import Image

    processor = _processor(VL_TEMPLATE)
    dev = _dev(VL_TEMPLATE, processor)
    renderer = processor if hasattr(processor, "apply_chat_template") else processor.tokenizer

    # text shape: the jinja is the reference for every intent
    complete = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]
    train_ids, train_messages = _official(renderer, copy.deepcopy(complete), False)
    prompt_ids, prompt_messages = _official(renderer, copy.deepcopy(complete[:-1]), True)
    assert list(dev.encode({"messages": copy.deepcopy(train_messages)}, intent=TRAIN)["input_ids"]) == train_ids
    infer = dev.encode({"messages": copy.deepcopy(prompt_messages)}, intent=INFER)
    assert list(infer["input_ids"]) == prompt_ids
    assert list(dev.encode({"messages": copy.deepcopy(prompt_messages)}, intent=ENGINE)["input_ids"]) == prompt_ids

    # media shape: one placeholder per image, and the same media tensors in every intent (an image
    # is processed once; the jinja never renders pixels, so these are cross-intent invariants)
    image = Image.fromarray((np.random.RandomState(7).rand(56, 56, 3) * 255).astype("uint8"))
    media = {
        "messages": [{"role": "user", "content": "<image>What is this?"}, {"role": "assistant", "content": "A cat."}],
        "images": [image],
    }
    prompt_media = {"messages": media["messages"][:1], "images": [image]}
    train_row = dev.encode(copy.deepcopy(media), intent=TRAIN)
    infer_row = dev.encode(copy.deepcopy(prompt_media), intent=INFER)
    engine_row = dev.encode(copy.deepcopy(prompt_media), intent=ENGINE)
    for key in ("pixel_values", "image_grid_thw"):
        assert np.array_equal(np.asarray(train_row[key]), np.asarray(infer_row[key])), f"{key} changed between intents"
    placeholders = sum(1 for token in engine_row["input_ids"] if token == dev.image_token_id)
    assert placeholders == len(prompt_media["images"]), f"expected one image placeholder per image, got {placeholders}"
