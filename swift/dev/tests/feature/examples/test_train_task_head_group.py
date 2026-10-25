# Copyright (c) ModelScope Contributors. All rights reserved.
"""S16-04 (L2): smoke-run the train/task-head-group v5 examples (seq_cls / reranker / embedding).

The plan row's per-family sub-lists (bert / qwen2_5 / multi_label / regression / the eight
reranker variants / qwen3-embedding / gme ...) are LEGACY scripts (no ``USE_SWIFT_V5`` flag,
``--tuner_type`` old syntax, ``--eval_steps`` the offline path no longer takes) kept for docs
coordination; the v5 entries under ``examples/train/{seq_cls,reranker,embedding}/`` are exactly
one ``train.sh`` each, and they are what this file smokes.

Dataset substitution: the scripts' hub datasets become the repo's own tiny self-contained
corpora (``examples/train/data/{seq_cls,reranker,embedding}.jsonl``, 8 rows each) so the smoke
stays offline; every other flag is verbatim.
"""
import pytest

from swift.dev.tests.feature.examples.conftest import MODEL_INSTRUCT, assert_smoke

EXAMPLES_DATA = __import__('pathlib').Path(__file__).parents[5] / 'examples' / 'train' / 'data'

pytestmark = [pytest.mark.slow, pytest.mark.accel(1)]


def _data(name):
    return str(EXAMPLES_DATA / name)


def _run(argv, out_dir, task_type):
    from swift.dev.cli.sft import sft_main
    history = sft_main(argv)
    return assert_smoke(history, out_dir, task_type)


def test_seq_cls_example_smokes(tmp_path):
    """examples/train/seq_cls/train.sh: ``--task_type seq_cls`` head (num_labels 2, single-label
    classification problem type) on full-param, verbatim."""
    out_dir = str(tmp_path / 'seq_cls_out')
    ckpt = _run([
        '--model', MODEL_INSTRUCT,
        '--task_type', 'seq_cls',
        '--num_labels', '2',
        '--problem_type', 'single_label_classification',
        '--torch_dtype', 'bfloat16',
        '--dataset', _data('seq_cls.jsonl'),
        '--tuner', 'full',
        '--max_steps', '3',
        '--per_device_train_batch_size', '2',
        '--gradient_accumulation_steps', '1',
        '--learning_rate', '1e-5',
        '--lr_scheduler', 'cosine',
        '--warmup_ratio', '0.05',
        '--max_length', '512',
        '--logging_steps', '1',
        '--save_steps', '100',
        '--save_total_limit', '2',
        '--output_dir', out_dir,
        '--report_to', 'none',
    ], out_dir, 'seq_cls')
    # The head contract: a seq_cls checkpoint must record num_labels so infer rebuilds the same
    # head shape instead of silently defaulting to a different one.
    import json as _json
    with open(f'{ckpt}/args.json') as f:
        args = _json.load(f)
    assert args.get('num_labels') == 2, f'args.json num_labels={args.get("num_labels")!r}'


def test_reranker_example_smokes(tmp_path):
    """examples/train/reranker/train.sh: ``--task_type reranker`` with the pointwise loss,
    verbatim."""
    out_dir = str(tmp_path / 'reranker_out')
    _run([
        '--model', MODEL_INSTRUCT,
        '--task_type', 'reranker',
        '--loss', 'pointwise_reranker',
        '--torch_dtype', 'bfloat16',
        '--dataset', _data('reranker.jsonl'),
        '--tuner', 'full',
        '--max_steps', '3',
        '--per_device_train_batch_size', '2',
        '--gradient_accumulation_steps', '1',
        '--learning_rate', '1e-5',
        '--lr_scheduler', 'cosine',
        '--warmup_ratio', '0.05',
        '--max_length', '512',
        '--logging_steps', '1',
        '--save_steps', '100',
        '--save_total_limit', '2',
        '--output_dir', out_dir,
        '--report_to', 'none',
    ], out_dir, 'reranker')


def test_embedding_example_smokes(tmp_path):
    """examples/train/embedding/train.sh: ``--task_type embedding`` with infonce on LoRA; the
    script pins ``--attn_impl flash_attention_2`` (the cuDNN SDPA bf16 backward NaN workaround)
    and so does this smoke, verbatim."""
    out_dir = str(tmp_path / 'embedding_out')
    _run([
        '--model', MODEL_INSTRUCT,
        '--task_type', 'embedding',
        '--loss', 'infonce',
        '--attn_impl', 'flash_attention_2',
        '--torch_dtype', 'bfloat16',
        '--dataset', _data('embedding.jsonl'),
        '--tuner', 'lora',
        '--lora_rank', '8',
        '--lora_alpha', '32',
        '--target_modules', 'all-linear',
        '--max_steps', '3',
        '--per_device_train_batch_size', '8',
        '--gradient_accumulation_steps', '1',
        '--learning_rate', '5e-5',
        '--lr_scheduler', 'cosine',
        '--warmup_ratio', '0.05',
        '--max_length', '512',
        '--logging_steps', '1',
        '--save_steps', '100',
        '--save_total_limit', '2',
        '--output_dir', out_dir,
        '--report_to', 'none',
    ], out_dir, 'embedding')
