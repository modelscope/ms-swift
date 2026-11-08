# Copyright (c) ModelScope Contributors. All rights reserved.
"""The representative text families encode exactly like the chat template their checkpoint ships.

Legacy is not the oracle: dev follows the model author's inference path, so the reference here is the
jinja the checkpoint itself renders. The parity matrix pins the ids for its shared case shapes; this
file re-checks them per intent and adds the field-level invariants one render can get wrong -- labels
stay target-aligned, loss_scale covers every position, and the scoring / engine forms only truncate or
repeat the very same render.

A jinja that rejects the ``system`` role (the Gemma 2 templates do) is compared on the system-less
shape, because that is the only shape its own inference path uses.
"""

import copy

import pytest

from swift.dev.builders.template import build_template
from swift.dev.config import TemplateConfig
from swift.dev.template import ENGINE, INFER, SCORE, TRAIN
from swift.dev.tests.component.processor.test_chat_template_parity import (
    REPRESENTATIVES,
    _CaseSkip,
    _render_official,
)
from swift.dev.tests.tiny import TinyModel

FAMILIES = ("qwen2_5", "qwen3", "llama3", "gemma", "deepseek", "glm4")
MESSAGES = [
    {"role": "system", "content": "Be concise."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
]


def _family(family):
    """The dev template plus the renderer that owns the checkpoint's chat template."""
    from swift.model import get_model_processor

    model_id = REPRESENTATIVES.get(family)
    if model_id is None:
        pytest.skip(f"{family}: no tokenizer representative")
    model_dir = TinyModel.tokenizer_dir(model_id)
    _, processor = get_model_processor(model_dir, load_model=False)
    dev = build_template(
        TemplateConfig(template=family, max_length=512, is_binary_loss_scale=False), processor)
    renderer = processor if hasattr(processor, "apply_chat_template") else processor.tokenizer
    return dev, renderer


def _official(renderer, messages, add_generation_prompt):
    """(ids, the messages actually rendered) from the checkpoint's own chat template."""
    try:
        ids, _ = _render_official(renderer, messages, add_generation_prompt, prefer_parts=False)
        return list(ids), messages
    except _CaseSkip:
        # the Gemma 2 jinjas refuse the system role: their inference path carries no system turn
        stripped = [message for message in messages if message["role"] != "system"]
        try:
            ids, _ = _render_official(renderer, stripped, add_generation_prompt, prefer_parts=False)
        except _CaseSkip as exc:
            pytest.skip(f"the shipped jinja cannot render this sample ({exc})")
        return list(ids), stripped


def _assert_labels_predict_the_next_token(row):
    """labels[i] is input_ids[i + 1]; the last position and the prompt are masked."""
    ids, labels = list(row["input_ids"]), list(row["labels"])
    assert len(ids) == len(labels) == len(row["loss_scale"])
    assert labels[-1] == -100
    for index, label in enumerate(labels[:-1]):
        if label >= 0:
            assert label == ids[index + 1]
    assert any(label >= 0 for label in labels), "no supervised position"


@pytest.mark.parametrize("family", FAMILIES)
def test_encoding_matches_the_shipped_chat_template(family):
    dev, renderer = _family(family)
    train_ids, train_messages = _official(renderer, copy.deepcopy(MESSAGES), False)
    prompt_ids, prompt_messages = _official(renderer, copy.deepcopy(MESSAGES[:-1]), True)

    train = dev.encode({"messages": copy.deepcopy(train_messages)}, intent=TRAIN)
    assert list(train["input_ids"]) == train_ids, f"{family}: training render diverged from the jinja"
    _assert_labels_predict_the_next_token(train)

    infer = dev.encode({"messages": copy.deepcopy(prompt_messages)}, intent=INFER)
    assert list(infer["input_ids"]) == prompt_ids, f"{family}: generation prompt diverged from the jinja"

    # scoring renders the same conversation and only drops the separator after the response
    score = list(dev.encode({"messages": copy.deepcopy(train_messages)}, intent=SCORE)["input_ids"])
    assert train_ids[: len(score)] == score, f"{family}: scoring is not a truncated training render"

    # the jinja has no mode: the engine form must be the very prompt the inference intent renders
    engine = list(dev.encode({"messages": copy.deepcopy(prompt_messages)}, intent=ENGINE)["input_ids"])
    assert engine == prompt_ids, f"{family}: engine form diverged from the inference prompt"


@pytest.mark.parametrize("family", FAMILIES)
def test_preference_sides_match_the_shipped_chat_template(family):
    dev, renderer = _family(family)
    chosen_ids, chosen_messages = _official(renderer, copy.deepcopy(MESSAGES), False)
    rejected_messages = copy.deepcopy(chosen_messages)
    rejected_messages[-1]["content"] = "5"
    rejected_ids, _ = _official(renderer, rejected_messages, False)

    row = dev.encode_preference({"messages": copy.deepcopy(chosen_messages), "rejected_response": "5"})
    for side, expected in (("chosen", chosen_ids), ("rejected", rejected_ids)):
        assert list(row[side]["input_ids"]) == expected, f"{family}: {side} diverged from the jinja"
        _assert_labels_predict_the_next_token(row[side])
