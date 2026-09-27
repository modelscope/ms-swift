# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end multimodal (vision-language) coverage for ``run_infer`` on the transformers backend.

The generative transformers backend used to load every checkpoint through ``AutoModelForCausalLM``,
which rejects a VL config outright (``Unrecognized configuration class ... ForCausalLM``), and the
sampler dropped the image tensors the template encoded, keeping only ``input_ids``. Both gaps are now
closed: ``build_sampler`` hands the engine the family loader's declared ``model_cls``
(``Qwen2_5_VLForConditionalGeneration``), and ``TransformersSampler._run_chunk`` collates the encoded
``pixel_values`` / ``image_grid_thw`` into the engine's ``extra_model_inputs``.

These tests are the proof of that wiring. The weights are random, so nothing here asserts *what* the
model said -- only that a VL checkpoint loads under its own class and that generation succeeds with an
image attached. That success is itself the assertion: a VL model given ``<image>`` placeholder tokens
in ``input_ids`` but no ``pixel_values`` raises on the token/patch-count mismatch, so a clean
completion means the image tensors really reached ``generate``.

All tests are ``@pytest.mark.slow`` + ``@pytest.mark.accel(1)``; run with ``-m slow``.
"""
import json

import pytest

from swift.dev.tests.tiny_loader import build_tiny_multimodal

pytestmark = [pytest.mark.slow, pytest.mark.accel(1)]

MODEL_TYPE = 'qwen2_5_vl'
TEMPLATE = 'qwen2_5_vl'


@pytest.fixture(scope='module')
def vl_model(tmp_path_factory):
    """One tiny VL checkpoint for the whole module -- the snapshot download and random init are shared."""
    dest = tmp_path_factory.mktemp('vl') / 'model'
    return build_tiny_multimodal(str(dest))


def _write_image(path, size=56):
    """A tiny random RGB image on disk, big enough for one vision patch grid."""
    import numpy as np
    from PIL import Image
    Image.fromarray((np.random.rand(size, size, 3) * 255).astype('uint8')).save(path)
    return str(path)


def _write_vl_data(path, image_paths):
    """A jsonl dataset whose rows carry an ``<image>`` placeholder and a parallel ``images`` list."""
    rows = [{
        'messages': [{
            'role': 'user',
            'content': '<image>What is in this image?'
        }],
        'images': [img],
    } for img in image_paths]
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    return str(path)


def _read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def _image_paths(row):
    """The image paths on a row. run_infer round-trips media through HF datasets, which normalises a
    bare path into ``{'bytes': None, 'path': ...}``; accept either shape so the assertion is about the
    reference surviving, not its serialisation."""
    return [im['path'] if isinstance(im, dict) else im for im in row['images']]


def _run_vl(model_dir, data_path, *, num_return_sequences=1, batch_size=1, out_path=None):
    """Drive ``run_infer``'s generative path over VL rows on the in-process transformers backend.

    ``distributed_config=None`` keeps it single-process and ``temperature>0`` is required because HF
    greedy decoding rejects ``num_return_sequences>1`` -- the same constraints the text backend tests
    run under.
    """
    from swift.dev.config import DatasetConfig, GenerationConfig, InferConfig, ModelConfig, TemplateConfig
    from swift.dev.recipe.run_infer import run_infer

    return run_infer(
        ModelConfig(model=model_dir, model_type=MODEL_TYPE, task_type='causal_lm', torch_dtype='bfloat16'),
        TemplateConfig(template=TEMPLATE, max_length=512),
        DatasetConfig(dataset=[data_path]),
        InferConfig(num_return_sequences=num_return_sequences, batch_size=batch_size, output_format='all'),
        GenerationConfig(max_new_tokens=8, temperature=0.8),
        backend='transformers',
        distributed_config=None,
        split_dataset_ratio=0.0,
        output_path=out_path)


def test_vl_transformers_generation_consumes_image(vl_model, tmp_path):
    """A single image-bearing row generates on the transformers backend: the VL class loaded and the
    encoded ``pixel_values`` reached ``generate`` (a text-only path would raise on the image tokens)."""
    img = _write_image(tmp_path / 'img.png')
    data = _write_vl_data(tmp_path / 'vl.jsonl', [img])
    out = str(tmp_path / 'vl_out.jsonl')

    rows = _run_vl(vl_model, data, out_path=out)

    assert len(rows) == 1
    row = rows[0]
    assert isinstance(row['response'], str)
    assert row['response'] == row['responses'][0]
    # the image reference survives on the row, and the completion is appended as an assistant turn
    assert _image_paths(row) == [img]
    assert row['messages'][-1] == {'role': 'assistant', 'content': row['response']}

    written = _read_jsonl(out)
    assert len(written) == 1
    assert written[0]['response'] == row['response']


def test_vl_transformers_generation_batches_image_rows(vl_model, tmp_path):
    """Two image rows in one padded batch: ``_extra_model_inputs`` concatenates each row's
    ``pixel_values`` / ``image_grid_thw`` along dim 0, so a batched VL forward gets every image.

    Row order out of ``run_infer`` is not the input order (the text-backend batch tests make the same
    allowance), so the assertion is that both images were carried through -- one per row -- and both
    rows generated, not that they line up positionally.
    """
    imgs = [_write_image(tmp_path / f'img{i}.png') for i in range(2)]
    data = _write_vl_data(tmp_path / 'vl.jsonl', imgs)

    rows = _run_vl(vl_model, data, batch_size=2)

    assert len(rows) == 2
    for row in rows:
        assert isinstance(row['response'], str)
    assert sorted(_image_paths(row)[0] for row in rows) == sorted(imgs)
