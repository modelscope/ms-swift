# Copyright (c) ModelScope Contributors. All rights reserved.
"""S16-01 (L2): smoke-run the train/basic-group examples on the real CLI-to-recipe path.

Each test replays ONE ``examples/train/<name>/train.sh`` with its argv taken VERBATIM from the
script (every semantic / legality flag -- tuner knobs, schedulers, precision, lengths, cadence --
must survive, because the example is the contract under test), with exactly three smoke
substitutions:

- ``--model``  -> the locally cached 0.5B checkpoint (same family as the script pins),
- ``--dataset`` -> a tiny local ``.jsonl`` (or, for pretrain, the script's own bundled corpus),
- training volume (``--num_train_epochs``) -> ``--max_steps 3`` and ``--report_to`` -> ``none``,
  with ``--output_dir`` under pytest's tmp dir.

The entry point is ``sft_main`` / ``pt_main`` -- the same functions ``USE_SWIFT_V5=1`` routes the
``swift`` executable to -- so a green test means the example runs past argv parsing, config
validation and the full training loop, and lands a self-describing ``checkpoint-final``. The
routing itself (flag vs subcommand) is pinned statically by ``test_examples_static.py``.

One caveat worth stating: LoRA/full/full-with-lr-1e-5 run 3 optimizer steps here, so these tests
assert finiteness + artifact shape, NOT convergence (the examples' multi-epoch convergence story
is covered by the SFT feature suite's parity/optimizer tests).
"""
import json
import os

import pytest

from swift.dev.tests.feature.examples.conftest import (
    MODEL_BASE,
    MODEL_INSTRUCT,
    assert_smoke,
)

#: Path of the pretrain example's own bundled corpus (used verbatim by the pt smoke).
from pathlib import Path
EXAMPLES = Path(__file__).parents[5] / 'examples'
PRETRAIN_CORPUS = str(EXAMPLES / 'train' / 'data' / 'pretrain.jsonl')

pytestmark = [pytest.mark.slow, pytest.mark.accel(1)]

#: The ids are passed to the recipe VERBATIM (as the scripts do): resolve_loader_cls matches the
#: family by id first (both resolve to Qwen2_5Loader), and safe_snapshot_download serves them from
#: the local MODELSCOPE_CACHE. A local snapshot PATH would break the id match and fall to the
#: qwen2/qwen2_gte architectures ambiguity -- which is precisely not the example's situation.


def _run_sft_example(argv, out_dir, task_type='causal_lm'):
    """Run one example's argv through ``sft_main``; assert the smoke post-condition and return
    the checkpoint dir (for per-example extra assertions)."""
    from swift.dev.cli.sft import sft_main
    history = sft_main(argv)
    return assert_smoke(history, out_dir, task_type)


def test_lora_train_example_smokes(tmp_path, sft_rows):
    """examples/train/lora/train.sh: ``--tuner lora`` + all-linear + cosine, verbatim."""
    out_dir = str(tmp_path / 'lora_out')
    _run_sft_example([
        '--model', MODEL_INSTRUCT,
        '--task_type', 'causal_lm',
        '--torch_dtype', 'bfloat16',
        '--dataset', sft_rows,
        '--tuner', 'lora',
        '--lora_rank', '8',
        '--lora_alpha', '32',
        '--target_modules', 'all-linear',
        '--max_steps', '3',
        '--per_device_train_batch_size', '1',
        '--gradient_accumulation_steps', '8',
        '--learning_rate', '1e-4',
        '--lr_scheduler', 'cosine',
        '--warmup_ratio', '0.05',
        '--max_length', '2048',
        '--logging_steps', '5',
        '--save_steps', '100',
        '--save_total_limit', '2',
        '--split_dataset_ratio', '0.01',
        '--output_dir', out_dir,
        '--report_to', 'none',
    ], out_dir)


def test_full_train_example_smokes(tmp_path, sft_rows):
    """examples/train/full/train.sh: ``--tuner full`` at the example's small 1e-5 learning rate."""
    out_dir = str(tmp_path / 'full_out')
    _run_sft_example([
        '--model', MODEL_INSTRUCT,
        '--task_type', 'causal_lm',
        '--torch_dtype', 'bfloat16',
        '--dataset', sft_rows,
        '--tuner', 'full',
        '--max_steps', '3',
        '--per_device_train_batch_size', '1',
        '--gradient_accumulation_steps', '8',
        '--learning_rate', '1e-5',
        '--lr_scheduler', 'cosine',
        '--warmup_ratio', '0.05',
        '--max_length', '2048',
        '--logging_steps', '5',
        '--save_steps', '100',
        '--save_total_limit', '2',
        '--split_dataset_ratio', '0.01',
        '--output_dir', out_dir,
        '--report_to', 'none',
    ], out_dir)


def test_trainable_tokens_train_example_smokes(tmp_path, sft_rows):
    """examples/train/trainable_tokens/train.sh: ``--tuner trainable_tokens`` with the script's
    verbatim indices (existing frequent token ids of Qwen2.5-0.5B-Instruct)."""
    out_dir = str(tmp_path / 'tt_out')
    # The tuner contract the example documents: ONLY those embedding rows train. The saved
    # checkpoint says so via args.json, and the adapter dir is the trainable-tokens one.
    ckpt = _run_sft_example([
        '--model', MODEL_INSTRUCT,
        '--task_type', 'causal_lm',
        '--torch_dtype', 'bfloat16',
        '--dataset', sft_rows,
        '--tuner', 'trainable_tokens',
        '--trainable_token_indices', '9330', '546', '1782', '77091', '3838', '40', '64', '983',
        '--max_steps', '3',
        '--per_device_train_batch_size', '1',
        '--gradient_accumulation_steps', '8',
        '--learning_rate', '1e-3',
        '--lr_scheduler', 'cosine',
        '--warmup_ratio', '0.05',
        '--max_length', '2048',
        '--logging_steps', '5',
        '--save_steps', '100',
        '--save_total_limit', '2',
        '--split_dataset_ratio', '0.01',
        '--output_dir', out_dir,
        '--report_to', 'none',
    ], out_dir)
    # The tuner contract the example documents: ONLY those embedding rows train, and the saved
    # checkpoint records the tuner it was trained with.
    with open(os.path.join(ckpt, 'args.json')) as f:
        ckpt_args = json.load(f)
    assert ckpt_args.get('tuner_type') == 'trainable_tokens'


def test_pretrain_train_example_smokes(tmp_path):
    """examples/train/pretrain/train.sh: ``swift pt`` on the script's OWN bundled corpus
    (examples/train/data/pretrain.jsonl, 24 raw-text rows), verbatim flags."""
    out_dir = str(tmp_path / 'pt_out')
    from swift.dev.cli.pt import pt_main
    history = pt_main([
        '--model', MODEL_BASE,
        '--torch_dtype', 'bfloat16',
        '--dataset', PRETRAIN_CORPUS,
        '--tuner', 'full',
        '--max_steps', '3',
        '--per_device_train_batch_size', '1',
        '--gradient_accumulation_steps', '1',
        '--learning_rate', '2e-5',
        '--lr_scheduler', 'cosine',
        '--warmup_ratio', '0.05',
        '--max_length', '1024',
        '--logging_steps', '1',
        '--save_steps', '100',
        '--save_total_limit', '2',
        '--split_dataset_ratio', '0',
        '--dataset_num_proc', '4',
        '--output_dir', out_dir,
        '--report_to', 'none',
    ])
    assert_smoke(history, out_dir, 'causal_lm')


def test_base_to_chat_train_example_smokes(tmp_path, sft_rows):
    """examples/train/base_to_chat/train.sh: the BASE (non-instruct) checkpoint with
    ``--use_chat_template true`` -- the chat format is imposed by the template, not the weights."""
    out_dir = str(tmp_path / 'b2c_out')
    _run_sft_example([
        '--model', MODEL_BASE,
        '--task_type', 'causal_lm',
        '--use_chat_template', 'true',
        '--torch_dtype', 'bfloat16',
        '--dataset', sft_rows,
        '--tuner', 'full',
        '--max_steps', '3',
        '--per_device_train_batch_size', '1',
        '--gradient_accumulation_steps', '8',
        '--learning_rate', '1e-5',
        '--lr_scheduler', 'cosine',
        '--warmup_ratio', '0.05',
        '--max_length', '2048',
        '--logging_steps', '5',
        '--save_steps', '100',
        '--save_total_limit', '2',
        '--split_dataset_ratio', '0.01',
        '--output_dir', out_dir,
        '--report_to', 'none',
    ], out_dir)
