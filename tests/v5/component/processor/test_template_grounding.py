# Copyright (c) ModelScope Contributors. All rights reserved.
"""Grounding rendering parity for the Qwen2.5-VL family: structured ``objects`` in, official
grounding text out, token for token.

The dialogue matrix in test_chat_template_parity.py never carries ``objects``, so the whole
grounding path -- ``<ref-object>``/``<bbox>`` marker consumption, replace_ref/replace_bbox,
normalize_bbox coordinate math -- is untested there. This file pins it against the model
author's own rendering: the official jinja fed the pre-baked grounding text that Qwen2.5-VL
documents (``<|object_ref_start|>ref<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>``).

Coordinate contract (Qwen2.5-VL cookbook): with norm_bbox='none' the coordinates are ABSOLUTE
pixels of the image the model actually sees, i.e. AFTER the processor's smart_resize. The user
supplies original-image pixels; dev must rescale them. The oracle therefore bakes coordinates
computed from the official image_processor's own grid (independent of dev's resize code), not
the raw input pixels. norm1000 is resize-invariant: x/orig_w*1000.

PLAN (case | input | expected | failure it kills):
  1 single_object_train | one ref + one real-pixel bbox on a 96x64 image, TRAIN
      | dev ids == official ids of the baked text with smart_resize-scaled coordinates
      | marker left literal / ref-box pairing off / bbox string format drift / missing rescale
  2 multi_object_train  | two refs + two bboxes, one image, TRAIN
      | dev ids == official ids
      | ref_idx/bbox_idx desync (second object rendered with the first's payload)
  3 norm1000_train      | TemplateConfig(norm_bbox='norm1000'), bbox [10,20,60,50] on 96x64, TRAIN
      | dev ids == official ids of '(104,312),(625,781)' (int(round(x/orig*1000)))
      | normalization denominator/rounding drift, norm1000 wrongly rescaled by the resize
  4 oracle_sensitivity  | case 1 with one coordinate shifted by 1
      | ids differ from case 1's official ids
      | a vacuous comparison that would pass even if dev ignored the bbox payload
"""
import pytest

from tests.v5.tiny import TinyModel

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
WIDTH, HEIGHT = 96, 64


def _grounding_text(ref, bbox):
    x1, y1, x2, y2 = bbox
    return f"<|object_ref_start|>{ref}<|object_ref_end|><|box_start|>({x1},{y1}),({x2},{y2})<|box_end|>"


@pytest.fixture(scope="module")
def rig():
    from PIL import Image

    from swift.dev.builder import build_template
    from swift.dev.config import TemplateConfig
    from swift.model import get_model_processor

    model_dir = TinyModel.tokenizer_dir(MODEL_ID)
    _, processor = get_model_processor(model_dir, load_model=False)
    image = Image.new("RGB", (WIDTH, HEIGHT), (120, 40, 200))
    # The dimensions the model actually sees, straight from the official preprocessor:
    # image_grid_thw is in patch units (patch_size=14), so grid * 14 = resized pixels.
    grid = processor.image_processor(images=[image], return_tensors="pt")["image_grid_thw"][0]
    patch = processor.image_processor.patch_size
    resized_w, resized_h = int(grid[2]) * patch, int(grid[1]) * patch
    return processor, build_template(TemplateConfig(template="qwen2_5_vl", max_length=4096), processor), image, (resized_w, resized_h)


def _rescaled(bbox, resized):
    """Original-image pixels -> resized-image pixels, the documented Qwen2.5-VL contract."""
    rw, rh = resized
    x1, y1, x2, y2 = bbox
    return [int(round(x1 / WIDTH * rw)), int(round(y1 / HEIGHT * rh)), int(round(x2 / WIDTH * rw)), int(round(y2 / HEIGHT * rh))]


def _official_ids(processor, answer, image):
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "Locate the objects."}]},
        {"role": "assistant", "content": answer},
    ]
    encoded = processor.apply_chat_template(messages, tokenize=True, return_dict=True)
    # return_dict=True yields a batch dimension: input_ids is [[ids]], not [ids]. The image lives
    # in the content part (transformers 5.x rejects a separate images= kwarg) so the processor
    # expands <|image_pad|> to the real grid, matching dev's encode.
    return list(encoded["input_ids"][0])


def _sample(answer_markers, refs, bboxes, image):
    return {
        "messages": [
            {"role": "user", "content": "<image>Locate the objects."},
            {"role": "assistant", "content": answer_markers},
        ],
        "images": [image],
        "objects": {"ref": refs, "bbox": bboxes},
    }


def test_single_object_grounding_matches_official(rig):
    from swift.dev.template import TRAIN

    processor, template, image, resized = rig
    answer = _grounding_text("the cat", _rescaled([10, 20, 60, 50], resized))
    official = _official_ids(processor, answer, image)
    sample = _sample("<ref-object><bbox>", ["the cat"], [[10, 20, 60, 50]], image)
    rendered = list(template.encode(sample, intent=TRAIN)["input_ids"])
    assert rendered == official, "single-object grounding render diverges from the official jinja"


def test_multi_object_grounding_keeps_ref_bbox_pairing(rig):
    from swift.dev.template import TRAIN

    processor, template, image, resized = rig
    answer = _grounding_text("the cat", _rescaled([10, 20, 60, 50], resized)) + " and " + _grounding_text("a dog", _rescaled([0, 0, 95, 63], resized))
    official = _official_ids(processor, answer, image)
    sample = _sample(
        "<ref-object><bbox> and <ref-object><bbox>",
        ["the cat", "a dog"],
        [[10, 20, 60, 50], [0, 0, 95, 63]],
        image,
    )
    rendered = list(template.encode(sample, intent=TRAIN)["input_ids"])
    assert rendered == official, "second object's ref/bbox pairing drifted"


def test_norm1000_coordinates_match_official(rig):
    from swift.dev.builder import build_template
    from swift.dev.config import TemplateConfig
    from swift.dev.template import TRAIN

    processor, _, image, _ = rig
    template = build_template(TemplateConfig(template="qwen2_5_vl", max_length=4096, norm_bbox="norm1000"), processor)
    # int(round(x / 96 * 1000)), int(round(y / 64 * 1000)) for [10, 20, 60, 50]; resize-invariant.
    answer = _grounding_text("the cat", [104, 312, 625, 781])
    official = _official_ids(processor, answer, image)
    sample = _sample("<ref-object><bbox>", ["the cat"], [[10, 20, 60, 50]], image)
    rendered = list(template.encode(sample, intent=TRAIN)["input_ids"])
    assert rendered == official, "norm1000 coordinate math drifted"


def test_oracle_is_sensitive_to_the_bbox_payload(rig):
    processor, _, image, resized = rig
    base = _official_ids(processor, _grounding_text("the cat", _rescaled([10, 20, 60, 50], resized)), image)
    shifted = _official_ids(processor, _grounding_text("the cat", _rescaled([11, 20, 60, 50], resized)), image)
    assert base != shifted, "the oracle cannot see bbox coordinates; the parity cases would be vacuous"