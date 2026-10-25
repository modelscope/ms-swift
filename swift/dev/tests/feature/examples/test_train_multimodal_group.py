# Copyright (c) ModelScope Contributors. All rights reserved.
"""S16-03 (L2): smoke-run the train/multimodal-group v5 examples.

The plan row's long list (caption/ocr/audio/video/grounding/omni/vit_gradient_checkpointing/
llm_full_vit) predates the tree's final layout: those cases live under ``examples/models/``
(S16-14's per-model group) or ``examples/notebook/`` now. The v5 multimodal training scripts that
actually sit under ``examples/train/`` are exactly two -- ``multi-modal/train.sh`` (VL SFT) and
``multimodal_embedding/train.sh`` (VL embedding with InfoNCE) -- and they are what this file
smokes, on the real cached ``Qwen/Qwen2.5-VL-3B-Instruct``.

Dataset substitution detail: the scripts' own ``examples/train/data/vl*.jsonl`` corpora reference
REMOTE image URLs; the smoke writes the same row SHAPES (``messages`` with ``<image>`` tags plus
the ``images`` key; the embedding twin adds ``positive_*``/``negative_*``) pointing at two tiny
locally generated PNGs, so the run stays offline. ``MAX_PIXELS`` -- an env var the scripts export
for the VL processor -- is set the same way.
"""
import json

import pytest

from swift.dev.tests.feature.examples.conftest import assert_smoke

MODEL_VL = 'Qwen/Qwen2.5-VL-3B-Instruct'
MAX_PIXELS = '1003520'

pytestmark = [pytest.mark.slow, pytest.mark.accel(1)]


def _png(path, color):
    from PIL import Image
    Image.new('RGB', (64, 64), color).save(path, format='PNG')
    return str(path)


@pytest.fixture
def vl_images(tmp_path):
    """Two tiny local PNGs standing in for the scripts' remote OSS images."""
    return _png(tmp_path / 'animal.png', (200, 40, 40)), _png(tmp_path / 'cat.png', (40, 40, 200))


@pytest.fixture
def vl_sft_rows(tmp_path, vl_images):
    """Chat rows in the ``examples/train/data/vl.jsonl`` shape (``<image>`` tag + ``images``)."""
    rows = []
    for i, img in enumerate(vl_images * 2):
        rows.append({
            'messages': [
                {'role': 'user', 'content': '<image>这张图里有什么？'},
                {'role': 'assistant', 'content': f'图里是第 {i} 号示例动物。'},
            ],
            'images': [img],
        })
    path = tmp_path / 'vl_rows.jsonl'
    with open(path, 'w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    return str(path)


@pytest.fixture
def vl_embedding_rows(tmp_path, vl_images):
    """Rows in the ``examples/train/data/vl_embedding.jsonl`` shape: anchor messages+images with
    positive/negative counterparts (lists of conversations / lists of image-lists)."""
    animal, cat = vl_images
    rows = [{
        'messages': [{'role': 'user', 'content': '<image>这张图的表示是什么？'}],
        'images': [animal],
        'positive_messages': [[{'role': 'user', 'content': '<image>同一张图。'}]],
        'positive_images': [[animal]],
        'negative_messages': [[{'role': 'user', 'content': '<image>另一张不同的图。'}]],
        'negative_images': [[cat]],
    }, {
        'messages': [{'role': 'user', 'content': '<image>这张图的表示是什么？'}],
        'images': [cat],
        'positive_messages': [[{'role': 'user', 'content': '<image>同一张图。'}]],
        'positive_images': [[cat]],
        'negative_messages': [[{'role': 'user', 'content': '<image>另一张不同的图。'}]],
        'negative_images': [[animal]],
    }]
    path = tmp_path / 'vl_emb_rows.jsonl'
    with open(path, 'w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    return str(path)


def test_multi_modal_example_smokes(tmp_path, monkeypatch, vl_sft_rows):
    """examples/train/multi-modal/train.sh: VL LoRA SFT on the cached 3B VL checkpoint, verbatim
    flags (MAX_PIXELS env included)."""
    monkeypatch.setenv('MAX_PIXELS', MAX_PIXELS)
    out_dir = str(tmp_path / 'mm_out')
    from swift.dev.cli.sft import sft_main
    history = sft_main([
        '--model', MODEL_VL,
        '--task_type', 'causal_lm',
        '--torch_dtype', 'bfloat16',
        '--dataset', vl_sft_rows,
        '--tuner', 'lora',
        '--lora_rank', '8',
        '--lora_alpha', '32',
        '--target_modules', 'all-linear',
        '--max_steps', '3',
        '--per_device_train_batch_size', '1',
        '--gradient_accumulation_steps', '4',
        '--learning_rate', '1e-4',
        '--lr_scheduler', 'cosine',
        '--warmup_ratio', '0.05',
        '--max_length', '4096',
        '--logging_steps', '5',
        '--save_steps', '100',
        '--save_total_limit', '2',
        '--split_dataset_ratio', '0',
        '--dataset_num_proc', '4',
        '--output_dir', out_dir,
        '--report_to', 'none',
    ])
    assert_smoke(history, out_dir, 'causal_lm')


def test_multimodal_embedding_example_smokes(tmp_path, monkeypatch, vl_embedding_rows):
    """examples/train/multimodal_embedding/train.sh: VL embedding task with InfoNCE -- anchor /
    positive / negative image-message pairs through the real contrastive path."""
    monkeypatch.setenv('MAX_PIXELS', MAX_PIXELS)
    out_dir = str(tmp_path / 'mme_out')
    from swift.dev.cli.sft import sft_main
    history = sft_main([
        '--model', MODEL_VL,
        '--task_type', 'embedding',
        '--loss', 'infonce',
        '--torch_dtype', 'bfloat16',
        '--dataset', vl_embedding_rows,
        '--tuner', 'lora',
        '--lora_rank', '8',
        '--lora_alpha', '32',
        '--target_modules', 'all-linear',
        '--max_steps', '3',
        '--per_device_train_batch_size', '2',
        '--gradient_accumulation_steps', '1',
        '--learning_rate', '1e-4',
        '--lr_scheduler', 'cosine',
        '--warmup_ratio', '0.05',
        '--max_length', '2048',
        '--logging_steps', '1',
        '--save_steps', '100',
        '--save_total_limit', '2',
        '--split_dataset_ratio', '0',
        '--dataset_num_proc', '4',
        '--output_dir', out_dir,
        '--report_to', 'none',
    ])
    ckpt = assert_smoke(history, out_dir, 'embedding')
    assert ckpt  # the task_type=embedding pin lives in assert_smoke; loss rides the verbatim argv
