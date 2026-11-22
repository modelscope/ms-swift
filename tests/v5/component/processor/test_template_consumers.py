"""Contracts shared by the native template refactor and the new v5 consumers."""

from types import SimpleNamespace

import pytest
import torch
from twinkle import Platform

from swift.dev.processor import InputProcessor, processor_kwargs
from swift.dev.template import Template
from swift.dev.template.conventions import model_payload_keys


@pytest.fixture(autouse=True)
def cpu_processor(monkeypatch):
    monkeypatch.setattr(Platform, "get_local_device", lambda: torch.device("cpu"))


def _template(task_type="causal_lm"):
    template = object.__new__(Template)
    template.__dict__.update(task_type=task_type, processor=SimpleNamespace(pad_token_id=9))
    return template


@pytest.mark.parametrize("padding_free", [False, True])
@pytest.mark.parametrize("micro_batch_size", [None, 1])
def test_router_replay_is_a_sequence_field(padding_free, micro_batch_size):
    template = _template()
    processor = InputProcessor(padding_free=padding_free, **processor_kwargs(template))
    rows = [
        {"input_ids": [1, 2, 3], "labels": [-100, 3, -100], "replay_mask": [0, 1, 0]},
        {"input_ids": [4, 5], "labels": [5, -100]},
    ]
    assert model_payload_keys(rows[0]) == frozenset()
    prepared = processor.prepare_inputs(rows)
    assert all("_media" not in row for row in prepared)
    outputs = processor.to_transformers_dict(processor.collate_fn(prepared, micro_batch_size=micro_batch_size))
    if micro_batch_size == 1:
        second = [[0, 0]] if padding_free else [[0, 0, 0]]
        assert [row["replay_mask"].tolist() for row in outputs] == [[[0, 1, 0]], second]
    elif padding_free:
        assert outputs[0]["replay_mask"].tolist() == [[0, 1, 0, 0, 0]]
    else:
        assert outputs[0]["replay_mask"].tolist() == [[0, 1, 0], [0, 0, 0]]
        assert outputs[0]["input_ids"].tolist() == [[1, 2, 3], [4, 5, 9]]


@pytest.mark.parametrize("rank", [0, 1])
def test_router_replay_cp_padding_and_split(rank):
    mesh = SimpleNamespace(cp_world_size=2, cp_rank=rank, tp_world_size=1, sequence_parallel=False)
    processor = InputProcessor(framework="megatron", padding_free=True, device_mesh=mesh)
    masks = [[0, 1, 0, 1, 0], [1, 0, 1]]
    rows = [
        {
            "input_ids": list(range(1, len(mask) + 1)),
            "labels": [-100] * len(mask),
            "completion_mask": mask,
            "replay_mask": mask,
        }
        for mask in masks
    ]
    prepared = processor.prepare_inputs([rows])
    padded = processor.pad_cp(prepared)
    assert [row["replay_mask"].tolist() for row in padded] == [[[0, 1, 0, 1, 0, 0, 0, 0]], [[1, 0, 1, 0]]]
    collated = processor.collate_fn(padded)
    processor.add_extra_padding_free_args(collated)
    outputs = processor.split_cp(collated)
    expected = [[0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1]][rank]
    assert outputs[0]["replay_mask"].flatten().tolist() == expected
    assert outputs[0]["replay_mask"].shape == outputs[0]["input_ids"].shape


def test_native_embedding_groups_reach_processor():
    template = _template("embedding")
    assert processor_kwargs(template)["task_type"] == "embedding"
    processor = InputProcessor(**processor_kwargs(template))
    rows = [{"anchor_input_ids": [1, 2], "positive_input_ids": [3], "negative_input_ids": [[4, 5]]}]
    prepared = processor.prepare_inputs(rows)
    assert [row["input_ids"].flatten().tolist() for row in prepared] == [[1, 2], [3], [4, 5]]
    assert [row["labels"] for row in prepared] == [1.0, 0.0, 0.0]


@pytest.mark.parametrize("nested", [False, True])
def test_native_qwen_uses_upstream_hf_layout(nested):
    from swift.dev.template.families.qwen import Qwen2VLTemplate

    visual = object()
    text = SimpleNamespace(embed_tokens=lambda ids: ids)
    body = SimpleNamespace(language_model=text, visual=visual) if nested else text
    model = SimpleNamespace(model=body, config=object())
    if not nested:
        model.visual = visual
    family = object.__new__(Qwen2VLTemplate)
    family.processor = object()
    family.get_base_model = lambda model: model

    def merge(embeds, inputs, tower, processor, config):
        assert tower is visual
        return embeds

    family._get_inputs_embeds_hf = merge
    ids = torch.tensor([[1, 2]])
    assert torch.equal(family._post_encode(model, {"input_ids": ids})["inputs_embeds"], ids)
