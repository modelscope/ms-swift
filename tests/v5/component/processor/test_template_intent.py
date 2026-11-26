"""Real native encoders for split rows and explicit per-call intent; no model downloads."""

import copy
import threading
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from datasets import Dataset

from swift.dev.builders.template import build_template
from swift.dev.config import TemplateConfig
from swift.dev.dataset import EncodePreprocessor
from swift.dev.template import ENGINE, INFER, SCORE, TRAIN
from swift.dev.template.base import _rendering
from swift.dev.template.inputs import TemplateInputs
from tests.v5.component.processor.test_template_meta_model import _processor


def _template(*, max_length=1024, split=False, task_type="causal_lm"):
    config = SimpleNamespace(problem_type="single_label_classification") if task_type == "seq_cls" else None
    return build_template(
        TemplateConfig(
            template="qwen2_5",
            max_length=max_length,
            truncation_strategy="split" if split else "delete",
            is_binary_loss_scale=False,
        ),
        _processor(config),
        task_type=task_type,
    )


def _conversation(answer="answer"):
    return {"messages": [{"role": "user", "content": "question"}, {"role": "assistant", "content": answer}]}


def test_encode_prompt_uses_native_engine_intent():
    template = _template()
    sample = {"messages": [{"role": "user", "content": "question"}]}
    expected_feature = template.encode(copy.deepcopy(sample), intent=INFER)
    expected_engine = template.encode(copy.deepcopy(sample), intent=ENGINE)

    assert not hasattr(template, "engine_input_ids")
    prompt = template.encode_prompt(sample)
    from twinkle.template.contract import engine_input_ids

    assert prompt["input_ids"] == expected_feature["input_ids"]
    assert engine_input_ids(prompt) == expected_engine["input_ids"]
    assert prompt["messages"] == sample["messages"]


@pytest.mark.parametrize("backend", ["transformers", "vllm", "sglang", "lmdeploy"])
def test_encode_prompt_selects_images_for_backend(backend, monkeypatch):
    import torch

    template = _template()
    sample = {"messages": [{"role": "user", "content": "question"}], "images": ["original.png"]}
    tensor = torch.ones(1, 3, 2, 2)
    engine_image = tensor if backend == "transformers" else "resized.png"

    def encode(_sample, *, intent, engine):
        assert engine == backend
        image = tensor if intent.expand_media else engine_image
        feature = {"input_ids": [1, 2], "images": [image]}
        if not intent.expand_media and backend == "vllm":
            feature["mm_processor_kwargs"] = {"do_resize": False}
        return feature

    monkeypatch.setattr(template, "encode", encode)
    state = template.encode_prompt(sample, backend=backend).state

    assert state.media.images == (("original.png" if backend == "transformers" else "resized.png"),)
    assert state.feature["images"][0] is tensor
    assert state.mm_processor_kwargs == ({"do_resize": False} if backend == "vllm" else {})
    assert "mm_processor_kwargs" not in state.feature


@pytest.mark.parametrize("with_image", [False, True])
def test_vllm_encode_prompt_keeps_resize_override_only_for_processed_images(with_image, monkeypatch):
    template = _template()
    sample = {"messages": [{"role": "user", "content": "question"}], "videos": ["original.mp4"]}
    engine_kwargs = {"do_resize": False, "do_sample_frames": False, "video_metadata": [], "fps": 2}

    def encode(_sample, *, intent, engine):
        if intent.expand_media:
            return {"input_ids": [1, 2]}
        return {
            "input_ids": [1, 2],
            "images": ["resized.png"] if with_image else [],
            "videos": ["sampled_video"],
            "mm_processor_kwargs": engine_kwargs,
        }

    monkeypatch.setattr(template, "encode", encode)
    state = template.encode_prompt(sample, backend="vllm").state
    assert state.media.videos == ("original.mp4",)
    assert state.media.images == (("resized.png",) if with_image else ())
    expected = {"fps": 2}
    if with_image:
        expected["images_kwargs"] = {"do_resize": False}
    assert state.mm_processor_kwargs == expected
    assert engine_kwargs["do_resize"] is False and engine_kwargs["do_sample_frames"] is False


@pytest.mark.parametrize("intent", [TRAIN, SCORE])
@pytest.mark.parametrize("return_length", [False, True])
def test_split_finalizes_every_real_encoder_chunk_once(intent, return_length):
    template = _template(max_length=24, split=True)
    sample = _conversation("abcdefghijklmnopqrstuvwxyz" * 3)
    with _rendering(intent.render_mode()):
        raw = template._encode_sample(copy.deepcopy(sample), return_length=return_length)
    output = template.encode(copy.deepcopy(sample), intent=intent, return_length=return_length)
    assert isinstance(output, list) and len(output) == len(raw) > 2
    unbounded = template.replace(max_length=None).encode(copy.deepcopy(sample), intent=intent)
    assert sum((chunk["input_ids"] for chunk in output), []) == unbounded["input_ids"]
    for actual, original in zip(output, raw):
        assert 0 < len(actual["input_ids"]) <= 24
        if return_length:
            assert actual["lengths"] == [len(actual["input_ids"])]
        if intent.with_labels:
            assert actual["labels"] == original["labels"][1:] + [-100]
            assert actual["loss_scale"] == original["loss_scale"][1:] + [0]
            assert len(actual["labels"]) == len(actual["loss_scale"]) == len(actual["input_ids"])
            assert '_original_input_ids' not in actual and '_kept_token_indices' not in actual
        else:
            assert "labels" not in actual and "loss_scale" not in actual


def test_split_keeps_single_row_return_for_short_sample():
    template = _template(split=True)
    output = template.encode(_conversation())
    assert isinstance(output, dict)
    assert output["labels"][-1] == -100


def test_split_dataset_map_materializes_all_chunks():
    template = _template(max_length=24, split=True)
    samples = [_conversation("first answer" * 8), _conversation("second answer" * 9)]
    expected = [chunk for sample in samples for chunk in template.encode(sample, return_length=True)]
    output = EncodePreprocessor(template)(
        Dataset.from_list(samples), num_proc=1, strict=True, load_from_cache_file=False
    )
    assert len(output) == len(expected) > len(samples)
    for actual, original in zip(output, expected):
        assert actual["input_ids"] == original["input_ids"]
        assert actual["labels"] == original["labels"]
        assert actual["loss_scale"] == original["loss_scale"]


@pytest.mark.parametrize("kto", [False, True])
def test_preference_split_is_rejected_before_pair_encoding(kto, monkeypatch):
    template = _template(max_length=24, split=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("pair encoder must not run when paired chunk semantics are undefined")

    monkeypatch.setattr(template, "_encode_sample", forbidden)
    with pytest.raises(ValueError, match="split"):
        template.encode_preference({**_conversation(), "rejected_response": "other", "label": True}, kto=kto)


@pytest.mark.parametrize("intent", [INFER, ENGINE, TRAIN, SCORE])
@pytest.mark.parametrize("form", ["dict", "template_inputs", "dataclass"])
def test_explicit_intent_rejects_role_conflict_without_alias(intent, form):
    template = _template()
    sample = _conversation() if intent.add_generation_prompt else {"messages": _conversation()["messages"][:1]}
    if form == "template_inputs":
        sample = TemplateInputs.from_dict(sample)
    elif form == "dataclass":

        @dataclass
        class Row:
            messages: list

        sample = Row(sample["messages"])
    before = copy.deepcopy(sample)
    with pytest.raises(ValueError, match="add_generation_prompt"):
        template.encode(sample, intent=intent)
    assert sample == before


@pytest.mark.parametrize("task_type", ["embedding", "reranker", "generative_reranker", "seq_cls"])
def test_scoring_tasks_accept_user_queries_without_emitting_labels(task_type):
    template = _template(task_type=task_type)
    sample = {"messages": [{"role": "user", "content": "query"}], "label": 1}
    encoded = template.encode(copy.deepcopy(sample), intent=SCORE)
    assert encoded == template.encode(copy.deepcopy(sample), intent=SCORE, add_generation_prompt=False)
    assert encoded["input_ids"]
    assert "labels" not in encoded and "loss_scale" not in encoded


@pytest.mark.parametrize("long_first", [False, True])
def test_split_batch_encode_mixes_long_and_short_rows(long_first, monkeypatch):
    import torch
    from twinkle import Platform

    from swift.dev.processor import InputProcessor

    monkeypatch.setattr(Platform, "get_local_device", lambda: torch.device("cpu"))
    template = _template(max_length=256, split=True)
    samples = [_conversation("short"), _conversation("long answer" * 100)]
    if long_first:
        samples.reverse()
    individual = [template.encode(sample) for sample in samples]
    expected = [row for item in individual for row in (item if isinstance(item, list) else [item])]
    encoded = template.batch_encode(samples)
    assert all(isinstance(row, dict) for row in encoded)
    assert len(encoded) == len(expected)
    assert [row['input_ids'] for row in encoded] == [row['input_ids'] for row in expected]
    processor = InputProcessor(template=template, padding_free=True)
    prepared = processor.prepare_inputs(encoded)
    assert len(prepared) == len(expected)
    for actual, original in zip(prepared, expected):
        assert actual["labels"].flatten().tolist() == original["labels"]
        assert actual["position_ids"].flatten().tolist() == list(range(len(original["input_ids"])))
    output = processor.to_transformers_dict(processor.collate_fn(prepared))[0]
    assert output["input_ids"].flatten().tolist() == sum((row["input_ids"] for row in expected), [])
    assert output["labels"].flatten().tolist() == sum((row["labels"] for row in expected), [])


def test_per_call_intent_does_not_change_later_train_encoding():
    template = _template()
    before = template.encode(_conversation(), intent=TRAIN)
    template.encode({"messages": _conversation()["messages"][:1]}, intent=INFER)
    template.encode(_conversation(), intent=SCORE)
    assert template.encode(_conversation(), intent=TRAIN) == before
    assert template.mode == "train"


def test_render_mode_is_scoped_per_call_and_thread(monkeypatch):
    template = _template()
    prompt = {"messages": _conversation()["messages"][:1]}
    expected = {
        "train": template.encode(_conversation()),
        "engine": template.encode(copy.deepcopy(prompt), intent=ENGINE),
    }
    both_inside = threading.Barrier(2)
    seen, results = {}, {}
    encode_truncated = template._encode_truncated

    def spy(inputs):
        # Both threads read while both are inside their render scopes (the barrier is cyclic).
        both_inside.wait(timeout=30)
        seen[threading.current_thread().name] = template.mode
        both_inside.wait(timeout=30)
        return encode_truncated(inputs)

    monkeypatch.setattr(template, "_encode_truncated", spy)
    calls = {
        "train": lambda: template.encode(_conversation()),
        "engine": lambda: template.encode(copy.deepcopy(prompt), intent=ENGINE),
    }
    threads = [threading.Thread(target=lambda n=name: results.__setitem__(n, calls[n]()), name=name) for name in calls]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert seen == {"train": "train", "engine": "vllm"}
    assert results == expected
    assert template.mode == "train"
