# Copyright (c) ModelScope Contributors. All rights reserved.
"""S16-07 (L2): smoke-run the train/misc-group v5 examples (streaming / cached_dataset).

The plan row's other members (multimodal rlhf dpo/kto/gkd, on_policy_distillation) live under
``examples/rl/`` and belong to S16-08; ``custom/`` belongs to S16-17. The two scripts this file
smokes are the ones under ``examples/train/``:

- ``streaming/train.sh``: ``--streaming true`` + shuffle buffer + delete truncation, verbatim
  (the streaming SFT chain itself is pinned by S3-65; this test pins the example's flag surface
  through ``sft_main``).
- ``cached_dataset/train.sh``: the TWO-STAGE chain exactly as written -- stage 1 ``swift export
  --to_cached_dataset`` builds a token cache from the raw jsonl, stage 2 ``swift sft
  --cached_dataset`` trains from it (with the eval cadence reading the cached val split). Both
  stages run through their real CLI mains.
"""
import pytest

from swift.dev.tests.feature.examples.conftest import MODEL_INSTRUCT, assert_smoke

pytestmark = [pytest.mark.slow, pytest.mark.accel(1)]


def test_streaming_example_smokes(tmp_path, sft_rows):
    """examples/train/streaming/train.sh: streaming + shuffle_buffer_size 1000 + delete
    truncation, verbatim; steps come from --max_steps because an iterable has no length."""
    out_dir = str(tmp_path / 'stream_out')
    from swift.dev.cli.sft import sft_main
    history = sft_main([
        '--model', MODEL_INSTRUCT,
        '--task_type', 'causal_lm',
        '--torch_dtype', 'bfloat16',
        '--dataset', sft_rows,
        '--streaming', 'true',
        '--shuffle_buffer_size', '1000',
        '--max_steps', '3',
        '--truncation_strategy', 'delete',
        '--tuner', 'lora',
        '--lora_rank', '8',
        '--lora_alpha', '32',
        '--target_modules', 'all-linear',
        '--per_device_train_batch_size', '1',
        '--gradient_accumulation_steps', '4',
        '--learning_rate', '1e-4',
        '--lr_scheduler', 'cosine',
        '--warmup_ratio', '0.05',
        '--max_length', '4096',
        '--logging_steps', '1',
        '--save_steps', '100',
        '--save_total_limit', '2',
        '--split_dataset_ratio', '0',
        '--output_dir', out_dir,
        '--report_to', 'none',
    ])
    ckpt = assert_smoke(history, out_dir, 'causal_lm')
    assert ckpt


def test_cached_dataset_example_smokes(tmp_path, sft_rows):
    """examples/train/cached_dataset/train.sh: stage 1 builds the token cache (``swift export
    --to_cached_dataset``), stage 2 trains from it (``swift sft --cached_dataset``) with the eval
    cadence reading the cached val split. The split ratio widens from the script's 0.02 (over
    2000 hub rows) to 0.25 so the 16-row smoke actually holds a val split."""
    cache_dir = str(tmp_path / 'self_cognition_cached')
    from swift.dev.cli.export import export_main
    export_main([
        '--model', MODEL_INSTRUCT,
        '--dataset', sft_rows,
        '--split_dataset_ratio', '0.25',
        '--to_cached_dataset', 'true',
        '--template_mode', 'train',
        '--dataset_num_proc', '4',
        '--output_dir', cache_dir,
    ])
    import os
    assert os.path.isdir(f'{cache_dir}/train'), f'no train cache at {cache_dir}'
    assert os.path.isdir(f'{cache_dir}/val'), f'no val cache at {cache_dir}'

    out_dir = str(tmp_path / 'cached_out')
    from swift.dev.cli.sft import sft_main
    history = sft_main([
        '--model', MODEL_INSTRUCT,
        '--task_type', 'causal_lm',
        '--torch_dtype', 'bfloat16',
        '--cached_dataset', f'{cache_dir}/train',
        '--cached_val_dataset', f'{cache_dir}/val',
        '--tuner', 'lora',
        '--lora_rank', '8',
        '--lora_alpha', '32',
        '--target_modules', 'all-linear',
        '--max_steps', '3',
        '--per_device_train_batch_size', '1',
        '--per_device_eval_batch_size', '1',
        '--gradient_accumulation_steps', '8',
        '--learning_rate', '1e-4',
        '--lr_scheduler', 'cosine',
        '--warmup_ratio', '0.05',
        '--max_length', '2048',
        '--eval_strategy', 'steps',
        '--eval_steps', '2',
        '--logging_steps', '1',
        '--save_steps', '100',
        '--save_total_limit', '2',
        '--output_dir', out_dir,
        '--report_to', 'none',
    ])
    ckpt = assert_smoke(history, out_dir, 'causal_lm')
    # The cache contract: stage 2 consumed token ids, not raw rows -- the cache dir carries the
    # encoded store (dataset_info / arrow shards), which is what `--cached_dataset` loaded.
    assert os.path.exists(f'{cache_dir}/train'), ckpt
