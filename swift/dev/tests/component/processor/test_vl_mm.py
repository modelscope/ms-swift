"""Multimodal (VL) encode + collate regression tests.

Scope: multimodal is a sample->batch data-format concern. The dev Template (TRAIN) must produce
input_ids / pixel_values / image_grid_thw bit-identical to legacy swift (the ground truth the model was
trained on); the InputProcessor must batch media through the family's `_data_collator_mm_data`
(concatenated, not padded) and derive mRoPE position ids per row -- matching legacy's batch computation on
every real token, for padded AND packed batches. CP/SP splitting is the Model's job (out of scope).

Marked slow: needs the Qwen2.5-VL processor/config (no weights) + image processing. Run with -m slow.
"""

import copy

import numpy as np
import pytest
import torch

VL_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
_ENGINE_MODE_CASES = {
    "llava1_5_hf": "llava-hf/llava-1.5-7b-hf",
    "llava_onevision1_5": "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct",
    "llava_onevision_hf": "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    "qwen2_5_omni": "Qwen/Qwen2.5-Omni-3B",
    "qwen2_5_vl": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2_vl": "Qwen/Qwen2-VL-2B-Instruct",
    "qwen3_5": "Qwen/Qwen3.5-0.8B",
}


def _rand_image(sz=56):
    from PIL import Image

    return Image.fromarray((np.random.RandomState(sz).rand(sz, sz, 3) * 255).astype("uint8"))


def _sample(n_img, answer="A cat."):
    return {
        "messages": [
            {"role": "user", "content": "<image>" * n_img + "What is this?"},
            {"role": "assistant", "content": answer},
        ],
        "images": [_rand_image(56 + 28 * i) for i in range(n_img)],
    }


@pytest.fixture(scope="module")
def vl():
    """(dev Template, legacy train-mode template) for the SAME VL family, independent instances."""
    from swift.dev.builders import build_template
    from swift.dev.config import TemplateConfig
    from swift.model import get_model_processor
    from swift.template import get_template

    _, proc = get_model_processor(VL_MODEL, load_model=False)
    dev = build_template(TemplateConfig(template="qwen2_5_vl", max_length=1024), proc)
    legacy = get_template(proc, template_type="qwen2_5_vl", max_length=1024)
    legacy.set_mode("train")
    return dev, legacy


@pytest.mark.slow
def test_vl_train_encode_matches_legacy_bit_exact(vl):
    """input_ids token-strict, pixel_values / image_grid_thw maxdiff 0; labels = legacy shifted."""
    dev, legacy = vl
    le = legacy.encode(_sample(1))
    de = dev.encode(_sample(1))
    assert list(le["input_ids"]) == list(de["input_ids"]), "VL input_ids diverged from legacy"
    for k in ("pixel_values", "image_grid_thw"):
        a, b = torch.as_tensor(le[k]).float(), torch.as_tensor(de[k]).float()
        assert a.shape == b.shape and (a - b).abs().max().item() == 0.0, f"{k} not bit-exact vs legacy"
    assert list(de["labels"]) == list(le["labels"][1:]) + [-100]


@pytest.mark.slow
def test_vl_template_prepares_positions_and_discovers_media(vl):
    dev, _ = vl
    row = dev.encode(_sample(1))
    assert {"pixel_values", "image_grid_thw"} <= dev.media_keys(row)
    dev.prepare_row(row)
    assert row["position_ids"].dim() == 3 and row["position_ids"].shape[1] == 1


def _collate(dev, samples, padding_free):
    from swift.dev.processor import InputProcessor, processor_kwargs

    ip = InputProcessor(padding_free=padding_free, **processor_kwargs(dev))
    return ip.collate_fn(ip.prepare_inputs([copy.deepcopy(s) for s in samples]))[0]


@pytest.mark.slow
@pytest.mark.parametrize("padding_free", [False, True])
def test_vl_batch_matches_legacy_collator(vl, padding_free):
    """dev processor == legacy _data_collator on media (concat) and mRoPE positions (real tokens).

    Regression: dev used to inject arange position_ids for VL rows, which HF expands into three
    IDENTICAL rope axes -- mRoPE silently degraded to 1-D RoPE with every test green.
    """
    dev, legacy = vl
    encoded = [dev.encode(_sample(1)), dev.encode(_sample(2, answer="Two pictures of noise."))]
    # return_length: legacy's packing_row reads each row's `length`.
    rows = [
        legacy.encode(_sample(1), return_length=True),
        legacy.encode(_sample(2, answer="Two pictures of noise."), return_length=True),
    ]
    legacy.padding_free = padding_free
    try:
        ref = legacy.data_collator(copy.deepcopy(rows))
    finally:
        legacy.padding_free = False
    out = _collate(dev, encoded, padding_free)

    for k in ("pixel_values", "image_grid_thw"):
        a, b = torch.as_tensor(ref[k]).float(), out[k].float().cpu()
        assert a.shape == b.shape and (a - b).abs().max().item() == 0.0, f"{k} differs from legacy collate"

    pos = out["position_ids"].cpu()  # [4, B, L]: row 0 text positions, rows 1..3 the mRoPE axes
    assert pos.dim() == 3 and pos.shape[0] == 4, f"expected [4, B, L] mRoPE positions, got {tuple(pos.shape)}"
    ref_rope = ref["position_ids"].cpu()  # legacy hands transformers position_ids[1:] (+ text_position_ids)
    if padding_free:
        assert torch.equal(pos[1:], ref_rope)
        assert torch.equal(pos[0], ref["text_position_ids"].cpu())
    else:
        lengths = [len(r["input_ids"]) for r in rows]
        for i, n in enumerate(lengths):
            assert torch.equal(pos[1:, i, :n], ref_rope[:, i, :n]), f"row {i} mRoPE differs from legacy"
            assert torch.equal(pos[0, i, :n], torch.arange(n))
        assert not torch.equal(pos[1], pos[2]), "rope axes identical: mRoPE degraded to 1-D"


@pytest.mark.slow
@pytest.mark.parametrize("template_type,model_id", sorted(_ENGINE_MODE_CASES.items()))
def test_native_engine_encode_matches_legacy(template_type, model_id):
    from swift.dev.builders import build_template
    from swift.dev.config import TemplateConfig
    from swift.dev.template import ENGINE, INFER
    from swift.model import get_model_processor
    from swift.template import get_template

    if template_type == "qwen3_5":
        import inspect

        from qwen_vl_utils.vision_process import fetch_image

        if "image_patch_size" not in inspect.signature(fetch_image).parameters:
            pytest.skip("installed qwen-vl-utils lacks Qwen3.5 image_patch_size support")
    _, processor = get_model_processor(model_id, load_model=False)
    dev = build_template(TemplateConfig(template=template_type, max_length=8192), processor)
    legacy = get_template(processor, template_type=template_type, max_length=8192)
    prompt = {"messages": _sample(2)["messages"][:1], "images": _sample(2)["images"]}
    encoded = dev.encode_prompt(copy.deepcopy(prompt))
    expanded = dev.encode(copy.deepcopy(prompt), intent=INFER)["input_ids"]
    engine = dev.encode(copy.deepcopy(prompt), intent=ENGINE)["input_ids"]
    legacy.set_mode("vllm")
    legacy_engine = legacy.encode(copy.deepcopy(prompt))["input_ids"]
    assert encoded.state.feature["input_ids"] == expanded
    assert list(encoded.state.engine_input_ids) == list(engine) == list(legacy_engine)


@pytest.mark.slow
def test_vl_encode_prompt_checks_one_placeholder_per_image(vl):
    from swift.dev.template import MediaBundle

    dev, _ = vl
    prompt = dev.encode_prompt({"messages": _sample(2)["messages"][:1], "images": _sample(2)["images"]})
    assert len(prompt.state.media.images) == 2
    with pytest.raises(ValueError, match="image placeholder"):
        dev._check_engine_placeholders(
            list(prompt.state.engine_input_ids), MediaBundle(images=prompt.state.media.images[:1])
        )


@pytest.mark.slow
def test_vl_single_row_keeps_the_batch_axis(vl):
    """twinkle hands a lone row to the model uncollated; HF reads a 2-D ``[4, L]`` as 4 rows of text
    positions, so the per-row mRoPE positions must already be ``[4, 1, L]``."""
    dev, _ = vl
    encoded = dev.encode(_sample(1))
    out = _collate(dev, [encoded], padding_free=False)
    assert tuple(out["position_ids"].shape) == (4, 1, len(encoded["input_ids"]))


@pytest.mark.slow
def test_vl_rollout_encode_loads_image_refs_in_place(vl, tmp_path):
    """twinkle's samplers read multi_modal_data off the trajectory list they encode; an engine needs PIL
    images there, not the dataset's path / ``{'bytes', 'path'}`` refs."""
    from PIL import Image

    dev, _ = vl
    path = str(tmp_path / "a.png")
    _rand_image().save(path)
    images = [path, {"bytes": None, "path": path}]
    prompt = {"messages": [{"role": "user", "content": "<image><image>What is this?"}], "images": images}
    dev.encode(prompt, add_generation_prompt=True)
    assert prompt["images"] is images and all(isinstance(image, Image.Image) for image in images)
    # a training encode leaves the row alone
    row = dict(_sample(1), images=[path])
    dev.encode(row)
    assert row["images"] == [path]


def test_mixed_image_text_batch_fills_missing_mm_token_type_ids():
    """A mixed image+text batch must collate without KeyError, matching legacy.

    Only multimodal samples carry sequence-level mm fields like mm_token_type_ids (dev Template
    emits it under `requires_mm_token_type_ids and any(mm_mask)`, exactly like legacy qwen.py). But
    twinkle's _collate_macro_batch unions all sample keys and then indexes every sample with
    `item[key]`, so a text row with no mm_token_type_ids raised KeyError. legacy pads such fields up
    to the full batch with the padding value (a 2-image/2-text batch -> shape (4, L), text rows
    all-zero); dev's prepare_inputs reproduces that fill. Pure-CPU: builds the ragged dicts directly,
    no model needed.
    """
    from swift.dev.processor import InputProcessor

    ip = InputProcessor()
    # image rows carry mm_token_type_ids (non-zero where the image token sits); text rows do not.
    img_a = {"input_ids": [10, 151655, 12], "labels": [-100, -100, 12], "mm_token_type_ids": torch.tensor([0, 1, 0])}
    txt = {"input_ids": [10, 11], "labels": [-100, 11]}
    img_b = {
        "input_ids": [10, 151655, 151655, 12],
        "labels": [-100, -100, -100, 12],
        "mm_token_type_ids": torch.tensor([0, 1, 1, 0]),
    }
    prepared = ip.prepare_inputs([dict(img_a), dict(txt), dict(img_b), dict(txt)])

    # every row now carries mm_token_type_ids, length == its own input_ids (legacy fill semantics)
    for row in prepared:
        assert row.get("mm_token_type_ids") is not None
        assert len(row["mm_token_type_ids"]) == len(row["input_ids"])
    # the filled text rows are all-zero (padding value 0 == "not a multimodal token")
    assert int(prepared[1]["mm_token_type_ids"].sum()) == 0
    assert int(prepared[3]["mm_token_type_ids"].sum()) == 0
    # the real image rows are untouched
    assert int(prepared[0]["mm_token_type_ids"].sum()) == 1

    # and the collate stage no longer KeyErrors on the union index
    collated = ip.collate_fn(prepared)
    mm = collated[0]["mm_token_type_ids"] if isinstance(collated, list) else collated["mm_token_type_ids"]
    assert torch.as_tensor(mm).shape[0] == 4, "mm_token_type_ids should pad to the full batch of 4"


def test_all_text_batch_leaves_no_mm_token_type_ids():
    """A pure-text batch must NOT gain a phantom mm_token_type_ids: the fill only fires when the
    field is present on some-but-not-all rows, never when it is absent everywhere (legacy emits it
    for no text row either)."""
    from swift.dev.processor import InputProcessor

    ip = InputProcessor()
    txt = {"input_ids": [10, 11], "labels": [-100, 11]}
    prepared = ip.prepare_inputs([dict(txt), dict(txt)])
    assert all(row.get("mm_token_type_ids") is None for row in prepared)
