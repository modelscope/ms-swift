# Copyright (c) ModelScope Contributors. All rights reserved.
"""Direct dev-to-legacy parity for representative text template families."""

import copy

import pytest

from swift.dev.builders.template import build_template
from swift.dev.config import TemplateConfig
from swift.dev.template import ENGINE, INFER, SCORE, TRAIN
from swift.dev.template_contract import shift
from swift.dev.tests.component.processor.test_template_meta_model import _processor

FAMILIES = ("qwen2_5", "qwen3", "llama3", "gemma", "deepseek", "glm4")
COMPLETE = {
    "messages": [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]
}
PROMPT = {"messages": COMPLETE["messages"][:-1]}


def _plain(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _legacy(template, sample, mode, *, return_length=True):
    template.set_mode(mode)
    return _plain(template.encode(copy.deepcopy(sample), return_length=return_length))


def _expected_row(row, *, with_labels):
    expected = dict(row)
    if with_labels:
        expected = _plain(shift(expected))
    else:
        expected.pop("labels", None)
        expected.pop("loss_scale", None)
    return expected


@pytest.mark.parametrize("family", FAMILIES)
def test_template_encoding_matches_legacy(family):
    from swift.template import get_template

    processor = _processor(model_type=family)
    processor.tokenizer.bos_token_id = 1
    dev = build_template(
        TemplateConfig(template=family, max_length=512, is_binary_loss_scale=False),
        processor,
    )
    legacy = get_template(
        processor, template_type=family, max_length=512, is_binary_loss_scale=False, loss_scale=dev._loss_scale
    )

    cases = (
        (TRAIN, COMPLETE, "train", True),
        (SCORE, COMPLETE, "transformers", False),
        (INFER, PROMPT, "transformers", False),
        (ENGINE, PROMPT, "vllm", False),
    )
    for intent, sample, mode, with_labels in cases:
        actual = _plain(dev.encode(copy.deepcopy(sample), intent=intent, return_length=True))
        expected = _expected_row(_legacy(legacy, sample, mode), with_labels=with_labels)
        assert actual == expected, f"{family} {mode} diverged from legacy"


@pytest.mark.parametrize("family", FAMILIES)
def test_preference_encoding_matches_legacy(family):
    from swift.template import get_template

    processor = _processor(model_type=family)
    processor.tokenizer.bos_token_id = 1
    dev = build_template(
        TemplateConfig(template=family, max_length=512, is_binary_loss_scale=False),
        processor,
    )
    legacy = get_template(
        processor, template_type=family, max_length=512, is_binary_loss_scale=False, loss_scale=dev._loss_scale
    )
    sample = {**copy.deepcopy(COMPLETE), "rejected_response": "5"}
    actual = _plain(dev.encode_preference(copy.deepcopy(sample)))
    expected_flat = _legacy(legacy, sample, "rlhf", return_length=False)

    for side in ("chosen", "rejected"):
        expected = {key[len(side) + 1 :]: value for key, value in expected_flat.items() if key.startswith(side + "_")}
        assert actual[side] == _plain(shift(expected)), f"{family} {side} diverged from legacy"
