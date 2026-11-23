"""Real-tensor regressions for native MiniCPM media and optional token-type sequences."""

import copy

import pytest
import torch
from twinkle import DeviceMesh, Platform
from twinkle.processor import InputProcessor as TwinkleInputProcessor

from swift.dev.processor import InputProcessor
from swift.dev.template.conventions import model_payload_keys
from swift.dev.template.families.gemma import Gemma3VisionTemplate
from swift.dev.template.families.minicpm import (
    MiniCPMO4_5Template,
    MiniCPMV2_5Template,
    MiniCPMV2_6Template,
    MiniCPMV4_5Template,
    MiniCPMVTemplate,
)


@pytest.fixture(autouse=True)
def cpu_processor(monkeypatch):
    monkeypatch.setattr(Platform, "get_local_device", lambda: torch.device("cpu"))


def _template(family_cls):
    # The collator consumes encoded rows, so no tokenizer/model download or encode stub is needed.
    template = object.__new__(family_cls)
    template.task_type = "causal_lm"
    return template


def _pipeline(processor, rows, **kwargs):
    batches = copy.deepcopy(rows)
    for stage in processor.process_pipeline[:-1]:
        batches = stage(batches, **kwargs)
    final = processor.process_pipeline[-1](batches, **kwargs)
    # HF's inherited return contract selects the first batch; inspect ALL converted micro-batches
    # as well, whereas Megatron returns the entire list. No padding/CP stage is replaced here.
    assert final is (batches[0] if processor.framework == "transformers" else batches)
    return batches


def _mesh(processor, *, cp_size, tp_size, rank=0):
    # Place this process at the requested CP coordinate in a real CPU DeviceMesh. Assign after
    # construction because twinkle's remote wrapper drops meshes before runtime initialization.
    ranks = torch.arange(cp_size * tp_size).reshape(cp_size, tp_size).roll(rank, dims=0)
    processor.device_mesh = DeviceMesh(
        mesh=ranks.numpy() + Platform.get_rank(),
        mesh_dim_names=("cp", "tp"),
        device_type="cpu",
        sequence_parallel=tp_size > 1,
    )
    assert processor.device_mesh.cp_rank == rank
    assert processor.device_mesh.cp_world_size == cp_size
    assert processor.device_mesh.tp_world_size == tp_size


def _same(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _same(a, b)
    else:
        assert actual == expected


@pytest.mark.parametrize(
    "family_cls", [MiniCPMVTemplate, MiniCPMV2_5Template, MiniCPMV2_6Template, MiniCPMV4_5Template, MiniCPMO4_5Template]
)
@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("framework", ["transformers", "megatron"])
def test_minicpm_nested_pixels_keep_sample_and_slice_axes(family_cls, single, framework):
    pixels = [[torch.arange(24).reshape(3, 2, 4).float(), torch.ones(3, 4, 2)], [], [torch.full((3, 2, 6), 2.0)]]
    bounds = [torch.tensor([[1, 2], [3, 4]]), [], torch.tensor([[1, 3]])]
    sizes = [torch.tensor([[1, 2], [2, 1]]), [], torch.tensor([[1, 3]])]
    rows = [
        {
            "input_ids": torch.arange(1, 6),
            "labels": torch.full((5,), -100),
            "pixel_values": [p],
            "image_bound": [b],
            "tgt_sizes": [s],
        }
        for p, b, s in zip(pixels, bounds, sizes)
    ]
    expected = {"pixel_values": pixels, "image_bound": bounds, "tgt_sizes": sizes}
    if family_cls is MiniCPMV4_5Template:
        temporal = [torch.tensor([0, 1]), [], torch.tensor([2])]
        for row, ids in zip(rows, temporal):
            row["temporal_ids"] = [ids]
        expected["temporal_ids"] = temporal
    if family_cls is MiniCPMO4_5Template:
        features = [torch.ones(2, 80, 3), None, torch.full((1, 80, 5), 2.0)]
        lengths = [torch.tensor([3, 2]), [], torch.tensor([5])]
        audio_bounds = [torch.tensor([[1, 2], [3, 4]]), [], torch.tensor([[1, 4]])]
        for row, feat, length, bound in zip(rows, features, lengths, audio_bounds):
            row.update(audio_features=feat, audio_feature_lens=[length], audio_bounds=[bound])
        expected.update(audio_feature_lens=lengths, audio_bounds=audio_bounds)
        expected["audio_features"] = (
            features[0] if single else torch.cat([torch.nn.functional.pad(features[0], (0, 2)), features[2]])
        )
    if single:
        rows = rows[:1]
        expected = {key: value[:1] if isinstance(value, list) else value for key, value in expected.items()}
    # A generic payload proves super still runs, after consuming the nested MiniCPM fields.
    rows[0]["video_grid_thw"] = torch.tensor([[1, 2, 3]])
    expected["video_grid_thw"] = rows[0]["video_grid_thw"]
    processor = InputProcessor(template=_template(family_cls), framework=framework)
    output = _pipeline(processor, rows)[0]
    for key, value in expected.items():
        _same(output[key], value)
    assert "_media" not in output
    _same(rows[0]["pixel_values"], [pixels[0]])


def _gemma_rows():
    return [
        {
            "input_ids": [1, 2, 3, 4, 5],
            "labels": [-100, 3, 4, 5, -100],
            "token_type_ids": [0, 1, 1, 0, 0],
            "pixel_values": torch.ones(1, 3, 2, 2),
        },
        {"input_ids": [6, 7, 8], "labels": [7, 8, -100]},
        {
            "input_ids": [9, 10, 11, 12],
            "labels": [10, 11, 12, -100],
            "token_type_ids": torch.tensor([0, 1, 0, 1]),
            "pixel_values": torch.full((1, 3, 2, 2), 2.0),
        },
        {"input_ids": [13, 14], "labels": [14, -100]},
    ]


@pytest.mark.parametrize("framework", ["transformers", "megatron"])
@pytest.mark.parametrize("padding_side", ["left", "right"])
@pytest.mark.parametrize("padding_free", [False, True])
@pytest.mark.parametrize("image", [False, True])
def test_gemma_single_row_final_output(framework, padding_side, padding_free, image):
    processor = InputProcessor(
        template=_template(Gemma3VisionTemplate),
        framework=framework,
        padding_side=padding_side,
        padding_free=padding_free,
    )
    row = _gemma_rows()[0 if image else 1]
    output = _pipeline(processor, row)[0]
    if image:
        assert output["token_type_ids"].tolist() == [[0, 1, 1, 0, 0]]
        assert output["token_type_ids"].shape == output["input_ids"].shape
        _same(output["pixel_values"], row["pixel_values"])
    else:
        assert "token_type_ids" not in output
        assert "pixel_values" not in output


@pytest.mark.parametrize("framework", ["transformers", "megatron"])
@pytest.mark.parametrize("padding_side", ["left", "right"])
@pytest.mark.parametrize("padding_free", [False, True])
@pytest.mark.parametrize("variable_seq_lengths", [False, True])
@pytest.mark.parametrize("micro_batch_size", [None, 1, 2])
def test_gemma_mixed_token_types_follow_text_microbatches(
    framework, padding_side, padding_free, variable_seq_lengths, micro_batch_size
):
    processor = InputProcessor(
        template=_template(Gemma3VisionTemplate),
        framework=framework,
        pad_token_id=99,
        padding_side=padding_side,
        padding_free=padding_free,
    )
    # Text comes first, so a first-row-only key check cannot accidentally pass this test.
    rows = list(reversed(_gemma_rows()))
    for row in rows:
        assert "token_type_ids" not in model_payload_keys(row)
    prepared = processor.prepare_inputs(copy.deepcopy(rows))
    for row in prepared:
        assert "token_type_ids" not in row.get("_media", {})
    outputs = _pipeline(processor, rows, micro_batch_size=micro_batch_size, variable_seq_lengths=variable_seq_lengths)
    size = len(rows) if micro_batch_size is None else micro_batch_size
    groups = [rows[i : i + size] for i in range(0, len(rows), size)]
    assert len(outputs) == len(groups)
    for output, group in zip(outputs, groups):
        types = [list(row.get("token_type_ids", [0] * len(row["input_ids"]))) for row in group]
        if padding_free:
            expected = [sum(types, [])]
        else:
            max_len = max(len(row["input_ids"]) for row in (group if variable_seq_lengths else rows))
            expected = [
                (
                    values + [0] * (max_len - len(values))
                    if padding_side == "right"
                    else [0] * (max_len - len(values)) + values
                )
                for values in types
            ]
        _same(output["token_type_ids"], torch.tensor(expected, dtype=torch.long))
        assert output["token_type_ids"].shape == output["input_ids"].shape
        assert output["token_type_ids"].dtype == torch.long
        images = [row["pixel_values"] for row in group if "pixel_values" in row]
        if images:
            _same(output["pixel_values"], torch.cat(images))
        else:
            assert "pixel_values" not in output


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("micro_batch_size", [None, 1])
def test_token_types_packed_cp_resets_two_ranks(rank, micro_batch_size):
    processor = InputProcessor(framework="megatron", padding_free=True, pad_token_id=99)
    _mesh(processor, cp_size=2, tp_size=2, rank=rank)
    rows = [
        [
            {
                "input_ids": list(range(11, 19)),
                "labels": [-100] * 8,
                "position_ids": [0, 1, 2, 3, 4, 0, 1, 2],
                "token_type_ids": list(range(1, 9)),
            },
            {"input_ids": [21, 22, 23], "labels": [-100] * 3},
        ]
    ]
    outputs = _pipeline(processor, rows, micro_batch_size=micro_batch_size)
    # Segments 5 + 3 pad to 8 + 4 BEFORE concatenating the final text segment (3 -> 4).
    typed = [[1, 2, 0, 0, 6, 0], [3, 4, 5, 0, 7, 8]][rank]
    ids = [[11, 12, 99, 99, 16, 99], [13, 14, 15, 99, 17, 18]][rank]
    text_ids = [[21, 99], [22, 23]][rank]
    expected_types = [[typed + [0, 0]]] if micro_batch_size is None else [[typed], [[0, 0]]]
    expected_ids = [[ids + text_ids]] if micro_batch_size is None else [[ids], [text_ids]]
    expected_cu = [[0, 8, 12, 16]] if micro_batch_size is None else [[0, 8, 12], [0, 4]]
    for output, types, tokens, cu in zip(outputs, expected_types, expected_ids, expected_cu):
        assert output["token_type_ids"].tolist() == types
        assert output["input_ids"].tolist() == tokens
        assert output["token_type_ids"].shape == output["input_ids"].shape
        assert output["packed_seq_params"].cu_seqlens_q.tolist() == cu
    assert len(outputs) == len(expected_types)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("padding_side", ["left", "right"])
@pytest.mark.parametrize("variable_seq_lengths", [False, True])
def test_token_types_unpacked_cp_microbatches(rank, padding_side, variable_seq_lengths):
    processor = InputProcessor(framework="megatron", padding_side=padding_side)
    _mesh(processor, cp_size=2, tp_size=2, rank=rank)
    rows = [{"input_ids": [11, 12, 13, 14, 15], "token_type_ids": [1, 2, 3, 4, 5]}, {"input_ids": [21, 22, 23]}]
    outputs = _pipeline(processor, rows, micro_batch_size=1, variable_seq_lengths=variable_seq_lengths)
    assert outputs[0]["token_type_ids"].tolist() == [[[1, 2, 0, 0]], [[3, 4, 5, 0]]][rank]
    assert outputs[1]["token_type_ids"].tolist() == [[0] * (2 if variable_seq_lengths else 4)]
    assert len(outputs) == 2
    for output in outputs:
        assert output["token_type_ids"].shape == output["input_ids"].shape
        assert "packed_seq_params" not in output


@pytest.mark.parametrize("padding_free", [False, True])
@pytest.mark.parametrize("micro_batch_size", [None, 1])
def test_token_types_tp_only_padding(padding_free, micro_batch_size):
    processor = InputProcessor(framework="megatron", padding_free=padding_free)
    _mesh(processor, cp_size=1, tp_size=4)
    rows = [{"input_ids": [11, 12, 13, 14, 15], "token_type_ids": [0, 1, 1, 0, 1]}, {"input_ids": [21, 22]}]
    outputs = _pipeline(processor, rows, micro_batch_size=micro_batch_size, variable_seq_lengths=True)
    typed = [0, 1, 1, 0, 1, 0, 0, 0]
    expected = (
        [[typed], [[0] * 4]] if micro_batch_size == 1 else [[typed + [0] * 4]] if padding_free else [[typed, [0] * 8]]
    )
    assert [output["token_type_ids"].tolist() for output in outputs] == expected
    assert all(output["token_type_ids"].shape == output["input_ids"].shape for output in outputs)


@pytest.mark.parametrize("rank", [0, 1])
def test_replay_and_token_types_share_cp_without_second_transform(rank):
    processor = InputProcessor(framework="megatron", padding_free=True)
    _mesh(processor, cp_size=2, tp_size=1, rank=rank)
    masks = [[0, 1, 0, 1, 0], [1, 0, 1]]
    rows = [
        {
            "input_ids": list(range(1, len(mask) + 1)),
            "labels": [-100] * len(mask),
            "completion_mask": mask,
            "replay_mask": mask,
            "token_type_ids": mask,
        }
        for mask in masks
    ]
    # Isolate the CP contract just as the pre-existing replay regression does: without routing data,
    # align_routed_experts deliberately removes replay_mask, unrelated to its pad/split semantics.
    padded = processor.pad_cp(processor.prepare_inputs([rows]))
    for key in ("completion_mask", "replay_mask", "token_type_ids"):
        assert [row[key].tolist() for row in padded] == [[[0, 1, 0, 1, 0, 0, 0, 0]], [[1, 0, 1, 0]]]
    collated = processor.to_transformers_dict(processor.collate_fn(padded))
    processor.add_extra_padding_free_args(collated)
    output = processor.prepare_outputs(processor.split_cp(collated))[0]
    expected = [[0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1]][rank]
    for key in ("completion_mask", "replay_mask", "token_type_ids"):
        assert output[key].flatten().tolist() == expected
        assert output[key].shape == output["input_ids"].shape


@pytest.mark.parametrize("task_type", ["embedding", "reranker", "generative_reranker"])
def test_optional_token_types_do_not_expand_scalar_group_labels(task_type):
    if task_type == "embedding":
        rows = [
            {
                "anchor_input_ids": [1, 2, 3],
                "anchor_token_type_ids": [0, 1, 1],
                "positive_input_ids": [4, 5],
                "negative_input_ids": [[6, 7, 8, 9]],
            }
        ]
        expected = [1.0, 0.0, 0.0]
    else:
        rows = [{"input_ids": [[1, 2, 3], [4, 5]], "token_type_ids": [[0, 1, 1], None], "labels": [1, 0]}]
        expected = [1, 0]
    processor = InputProcessor(task_type=task_type)
    prepared = processor.prepare_inputs(copy.deepcopy(rows))
    assert all(row["labels"].numel() == 1 for row in prepared)
    output = _pipeline(processor, rows)[0]
    assert output["labels"].flatten().tolist() == expected
    assert output["labels"].numel() == len(expected)
    assert output["token_type_ids"].shape == output["input_ids"].shape
    assert not output["token_type_ids"][1:].any()


def test_upstream_optional_defaults_and_padding_map_are_reused():
    upstream = dict(TwinkleInputProcessor.padding_map)
    processor = InputProcessor(pad_token_id=99)
    assert processor.padding_map["token_type_ids"] == upstream.get("token_type_ids", upstream["completion_mask"])
    assert TwinkleInputProcessor.padding_map == upstream
    assert processor.padding_map is not TwinkleInputProcessor.padding_map
    rows = [
        {"input_ids": [1, 2, 3], "labels": [-100, 3, -100], "completion_mask": [0, 1, 0]},
        {"input_ids": [4, 5], "labels": [5, -100]},
    ]
    prepared = processor.prepare_inputs(rows)
    # Latest upstream derives absent completion_mask from token labels, rather than zeroing loss.
    assert prepared[1]["completion_mask"].flatten().tolist() == [1, 0]
