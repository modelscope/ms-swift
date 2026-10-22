# Copyright (c) ModelScope Contributors. All rights reserved.
"""The template's rollout contract and the field registry it splices by.

Labels / masks are asserted in INPUT order (unrolled), where each appended role's effect is a plain
slice; the stored next-token order is that rolled left by one.
"""

from dataclasses import fields
from pathlib import Path

import pytest

from swift.dev.template.base import MaxLengthError, Template
from swift.dev.template.conventions import (
    APPEND_ROLES,
    FIELDS,
    SPLICE_FILL,
    TOKEN,
    is_bookkeeping_key,
    model_payload_keys,
)
from swift.dev.template.meta import TemplateMeta
from swift.dev.template.output import OutputCodec
from swift.dev.template_contract import CompletionEncoding, MediaBundle, PromptEncoding, RolloutState

IGNORE = -100


class _Tokenizer:
    @staticmethod
    def encode(text, **kwargs):
        return [ord(char) for char in text]

    @staticmethod
    def decode(ids, **kwargs):
        return " ".join(map(str, ids))


class _Contract(Template):
    tokenizer = _Tokenizer()
    placeholder_tokens = []
    template_meta = TemplateMeta("dummy", prefix=[], prompt=[], chat_sep=[])
    enable_thinking = False
    media_types = frozenset()

    def __init__(self, max_length=None, truncation_strategy="raise"):
        self.max_length = max_length
        self.truncation_strategy = truncation_strategy
        self.output_codec = OutputCodec()

    def decode_generate_ids(self, generate_ids, **kwargs):
        return self.tokenizer.decode(generate_ids, **kwargs)


def _unroll(values):
    return values[-1:] + values[:-1]


def _prompt():
    """An INFER-encoded prompt: ids only, no labels."""
    return {"input_ids": [1, 2, 3], "messages": [{"role": "user", "content": "q"}]}


def test_tools_require_an_agent_parser():
    prompt = _prompt()
    prompt["tools"] = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
    with pytest.raises(RuntimeError, match="no tool-call parser"):
        _Contract().append_completion(prompt, [10])


# ---- append roles ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "appended_as, labels_tail, mask_tail",
    [
        ("completion", [10, 11], [1, 1]),
        ("demonstration", [10, 11], [0, 0]),
        ("context", [IGNORE, IGNORE], [0, 0]),
    ],
)
def test_append_role_sets_labels_and_completion_mask(appended_as, labels_tail, mask_tail):
    result = _Contract().append_completion(_prompt(), [10, 11], appended_as=appended_as).state.feature
    assert _unroll(result["labels"]) == [IGNORE] * 3 + labels_tail
    assert _unroll(result["completion_mask"]) == [0] * 3 + mask_tail


def test_append_completion_advances_rollout_state():
    prompt = _prompt()
    state = RolloutState(
        feature=prompt,
        engine_input_ids=(1, 2, 3),
        messages=tuple(prompt["messages"]),
        tools=(),
        reasoning_started=False,
        media=MediaBundle(),
    )
    result = _Contract().append_completion(state, [10, 11])
    assert result.state.feature["input_ids"] == [1, 2, 3, 10, 11]
    assert result.state.engine_input_ids == (1, 2, 3, 10, 11)
    assert result.state.reasoning_started is False


def test_append_roles_match_twinkle():
    """demonstration is scored (SFT imitates it) but carries no log-prob (RL skips it)."""
    assert {name: tuple(role) for name, role in APPEND_ROLES.items()} == {
        "completion": (True, True),
        "demonstration": (True, False),
        "context": (False, False),
    }


def test_unknown_append_role_raises():
    with pytest.raises(ValueError, match="appended_as"):
        _Contract().append_completion(_prompt(), [10], appended_as="observation")


def test_concat_rejects_a_train_prefix_with_loss_scale():
    """The splicer extends only the role-owned fields; a TRAIN prefix's loss_scale would silently
    fall behind the ids."""
    train_prefix = dict(_prompt(), labels=[IGNORE] * 3, loss_scale=[0.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="loss_scale"):
        _Contract().append_completion(train_prefix, [10])


def test_turns_stay_aligned_across_appends():
    contract = _Contract()
    first = contract.append_completion(_prompt(), [10, 11]).state
    second = contract.append_completion(first, [20], appended_as="context").state
    third = contract.append_completion(second, [30]).state.feature
    assert third["input_ids"] == [1, 2, 3, 10, 11, 20, 30]
    assert _unroll(third["labels"]) == [IGNORE] * 3 + [10, 11, IGNORE, 30]
    assert _unroll(third["completion_mask"]) == [0, 0, 0, 1, 1, 0, 1]
    assert third["attention_mask"] == [1] * 7


def test_multimodal_concat_requires_state_and_preserves_payload():
    prompt = _prompt()
    payload = object()
    prompt["pixel_values"] = payload
    with pytest.raises(NotImplementedError, match="RolloutState"):
        _Contract().append_completion(prompt, [10, 11])
    state = RolloutState(
        feature={key: value for key, value in prompt.items() if key != "messages"},
        engine_input_ids=(1, 2, 3),
        messages=tuple(prompt["messages"]),
        tools=(),
        reasoning_started=False,
        media=MediaBundle(images=("image",)),
    )
    result = _Contract().append_completion(state, [10, 11]).state
    assert result.messages[-1] == {"role": "assistant", "content": "10 11"}
    assert prompt["messages"] == [{"role": "user", "content": "q"}]
    assert result.feature["pixel_values"] is payload


def test_concat_extends_token_types_with_text():
    prompt = dict(_prompt(), mm_token_type_ids=[0, 1, 1])
    result = _Contract().append_completion(prompt, [10, 11]).state.feature
    assert result["mm_token_type_ids"] == [0, 1, 1, 0, 0]


# ---- max_length --------------------------------------------------------------------------------


def test_sampled_turn_obeys_raise_max_length_strategy():
    with pytest.raises(MaxLengthError):
        _Contract(max_length=4).append_completion(_prompt(), [10, 11, 12])


@pytest.mark.parametrize(
    ("strategy", "expected_ids", "expected_labels"),
    [
        ("right", [1, 2, 3, 4], [IGNORE, 3, 4, IGNORE]),
        ("left", [2, 3, 4, 5], [3, 4, 5, IGNORE]),
    ],
)
def test_post_pipeline_truncates_an_append(strategy, expected_ids, expected_labels):
    contract = _Contract(max_length=4, truncation_strategy=strategy)
    result = contract.finalize_appended_features(
        [
            {
                "input_ids": [1, 2, 3, 4, 5],
                "labels": [IGNORE, IGNORE, 3, 4, 5],
            }
        ]
    )[0]
    assert result["input_ids"] == expected_ids
    assert result["labels"] == expected_labels
    assert result["attention_mask"] == [1, 1, 1, 1]


def test_post_pipeline_raise_and_delete_match_strategy():
    feature = {"input_ids": [1, 2, 3], "labels": [IGNORE, 2, 3]}
    with pytest.raises(MaxLengthError):
        _Contract(max_length=2, truncation_strategy="raise").finalize_appended_features([feature])
    assert _Contract(max_length=2, truncation_strategy="delete").finalize_appended_features([feature]) == []


def test_prompt_and_completion_encodings_are_state_only():
    assert [field.name for field in fields(PromptEncoding)] == ["state"]
    assert [field.name for field in fields(CompletionEncoding)] == ["state", "response"]


def test_runtime_order_marker_is_absent_from_production_code():
    dev = Path(__file__).parents[3]
    paths = (
        *dev.joinpath("template").rglob("*.py"),
        *dev.joinpath("template_contract").rglob("*.py"),
        *dev.joinpath("rollout").rglob("*.py"),
    )
    assert not [str(path.relative_to(dev)) for path in paths if "_labels_shifted" in path.read_text()]


def test_unsupported_capabilities_fail_before_silent_fallback():
    template = object.__new__(Template)
    template.task_type = "embedding"
    template.supported_task_types = frozenset({"causal_lm"})
    template.padding_free = False
    template.media_types = frozenset()
    with pytest.raises(NotImplementedError, match="task_type"):
        template._validate_declared_capabilities()

    template.task_type = "causal_lm"
    template.supported_task_types = None
    template.padding_free = True
    template.media_types = frozenset({"image"})
    template.support_padding_free = None
    with pytest.raises(NotImplementedError, match="padding_free"):
        template._validate_declared_capabilities()

    template.template_meta = TemplateMeta("single", prefix=[], prompt=[], chat_sep=None)
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    with pytest.raises(NotImplementedError, match="multi-round"):
        template._validate_message_capabilities(messages)


def test_unsupported_system_prompt_fails_explicitly():
    meta = TemplateMeta("no-system", prefix=[], prompt=[], chat_sep=[])
    with pytest.raises(NotImplementedError, match="does not support `system`"):
        meta.check_system("custom")


# ---- field registry ----------------------------------------------------------------------------


def test_registered_fields_are_not_bookkeeping():
    assert not [name for name in FIELDS if is_bookkeeping_key(name)]


def test_only_token_fields_follow_a_splice():
    assert set(SPLICE_FILL) == {"token_type_ids", "mm_token_type_ids"}
    assert all(FIELDS[name].kind == TOKEN for name in SPLICE_FILL)


def test_payload_is_everything_unregistered():
    feature = {
        "input_ids": [1, 2],
        "labels": [IGNORE, 2],
        "messages": [],
        "tools": [{"name": "f"}],
        "images": ["a.png"],  # a raw prompt reference, not a model input
        "pixel_values": [[0.1]],
        "image_grid_thw": [[1, 2, 2]],
        "lengths": [2],
    }
    assert model_payload_keys(feature) == {"pixel_values", "image_grid_thw"}
