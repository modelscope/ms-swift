# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end multimodal tests: a local image on disk all the way to vision tensors in a batch.

``test_vl_mm.py`` covers the VL encode/collate contract, but against a *downloaded* Qwen2.5-VL (marked
slow, CUDA-gated) and from samples built in memory -- it never runs the loader, so the seam where a
dataset's ``images`` column of file paths becomes media the encoder can read is unproven there. This
file drives that whole chain against the *local* model with a generated image (no download, no CUDA):

    a jsonl referencing an image path -> load_dataset (cast_mm_data normalises ``images``)
        -> template.encode (reads the path, produces pixel_values / image_grid_thw)
        -> data_collator (concatenates the vision tensors across a ragged batch).

The collator's Qwen mrope path resolves position ids through a base model, so a meta-device dummy is
attached to the template -- it carries the architecture and nothing else, keeping the test on CPU.
"""
import json
import os

import numpy as np
import pytest
import torch
from PIL import Image

MODEL = os.environ.get('SWIFT_TEST_MODEL', '/mnt/workspace/yzhao/tastelikefeet/Qwen3.5-4B-CM-v2')
MODEL_TYPE = os.environ.get('SWIFT_TEST_MODEL_TYPE', 'qwen3_5')

needs_model = pytest.mark.skipif(not os.path.isdir(MODEL), reason=f'no local model at {MODEL}')


def make_image(path, size=56):
    Image.fromarray((np.random.RandomState(0).rand(size, size, 3) * 255).astype('uint8')).save(str(path))
    return str(path)


def write_mm_jsonl(path, rows):
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    return str(path)


def mm_row(text, image_paths):
    """A multimodal row: one ``<image>`` placeholder per image, plus the images by path."""
    content = '<image>' * len(image_paths) + text
    return {
        'messages': [{'role': 'user', 'content': content}, {'role': 'assistant', 'content': 'A cat.'}],
        'images': list(image_paths),
    }


@pytest.fixture(scope='module')
def vl_template():
    """A train-mode qwen3_5 template with a meta-device dummy model attached for the collator."""
    from swift.model import get_model_processor
    from swift.template import get_template
    processor = get_model_processor(MODEL, load_model=False, model_type=MODEL_TYPE)[1]
    template = get_template(processor, template_type='qwen3_5', max_length=1024)
    template.set_mode('train')
    with torch.device('meta'):
        template.model = get_model_processor(MODEL, model_type=MODEL_TYPE, return_dummy_model=True)[0]
    return template


# ---- the loader normalises a local image path (no model needed) ------------------------------


def test_load_normalises_local_image_paths(tmp_path):
    from swift.dev.dataset import load_dataset
    image = make_image(tmp_path / 'cat.png')
    path = write_mm_jsonl(tmp_path / 'a4_mm.jsonl', [mm_row('What is this?', [image])])
    train, _ = load_dataset([path])
    # cast_mm_data turns the list of path strings into the {bytes, path} layout the encoder reads.
    assert train[0]['images'] == [{'bytes': None, 'path': image}]
    assert train[0]['messages'][0]['content'] == '<image>What is this?'


# ---- the encode chain produces the vision tensors (needs the local model) ---------------------


@needs_model
def test_image_row_encodes_to_vision_tensors(vl_template, tmp_path):
    from swift.dev.dataset import SwiftDataset, load_dataset
    image = make_image(tmp_path / 'cat.png')
    path = write_mm_jsonl(tmp_path / 'a4_encode.jsonl', [mm_row('What is this?', [image])])
    train, _ = load_dataset([path])
    row = SwiftDataset(train, vl_template, load_from_cache_file=False)[0]

    assert 'pixel_values' in row and 'image_grid_thw' in row, f'vision tensors missing: {sorted(row)}'
    pixel_values = torch.as_tensor(row['pixel_values'])
    grid = torch.as_tensor(row['image_grid_thw'])
    assert pixel_values.dim() == 2 and pixel_values.shape[0] > 0
    assert grid.shape == (1, 3), 'one image -> one (t, h, w) grid row'
    # The patch count is the product of the grid dims -- the invariant Qwen-VL vision encoding keeps.
    assert pixel_values.shape[0] == grid.prod(dim=1).sum().item()
    assert len(row['input_ids']) == len(row['labels'])
    # The image tokens are marked so the model knows which positions carry media.
    assert int(torch.as_tensor(row['mm_token_type_ids']).sum()) > 0


@needs_model
def test_ragged_multimodal_batch_concats_vision_tensors(vl_template, tmp_path):
    """One-image and two-image rows in one batch: text pads, vision tensors concatenate."""
    from swift.dev.dataset import SwiftDataset, load_dataset
    image = make_image(tmp_path / 'cat.png')
    rows = [mm_row('What is this?', [image]), mm_row('And these?', [image, image])]
    path = write_mm_jsonl(tmp_path / 'a4_batch.jsonl', rows)
    train, _ = load_dataset([path])
    ds = SwiftDataset(train, vl_template, load_from_cache_file=False)

    batch = vl_template.data_collator([ds[0], ds[1]])

    grid = torch.as_tensor(batch['image_grid_thw'])
    assert grid.shape[0] == 3, 'three images across the batch -> three grid rows (concat, not pad)'
    pixel_values = torch.as_tensor(batch['pixel_values'])
    expected_rows = sum(torch.as_tensor(ds[i]['image_grid_thw']).prod(dim=1).sum().item() for i in (0, 1))
    assert pixel_values.shape[0] == expected_rows, 'pixel_values dim0 is the sum over every image'
    assert batch['input_ids'].shape[0] == 2, 'text keys stay batched [B, T]'
    assert batch['input_ids'].shape == batch['labels'].shape == batch['attention_mask'].shape
