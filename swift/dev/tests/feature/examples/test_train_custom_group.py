# Copyright (c) ModelScope Contributors. All rights reserved.
"""S16-17 (L2): smoke-run the custom-extension examples end to end.

Three extension surfaces, one test each, all through ``sft_main`` with the scripts' argv
verbatim under the standard smoke substitutions:

- ``custom/dataset``: a local jsonl registered through ``--custom_dataset_info`` (column mapping
  prompt/completion -> query/response, dataset selected BY NAME ``--dataset my_qa``). The smoke
  writes a twin dataset_info pointing at a local data file, because the shipped one uses a
  relative ``data.jsonl`` resolved against the info file's own directory.
- ``custom/model_template``: a custom model family + chat template registered as an
  ``--external_plugins`` file and selected by ``--model_type custom_im`` / ``--template
  custom_im`` (both registries the header documents).
- ``plugins``: the ``progress_printer`` train-callback plugin (``--external_plugins`` +
  ``--callbacks``), with the eval cadence scaled to the smoke volume so the callback's
  on-evaluate path also fires.

``rft/rft.py`` is a LEGACY multi-process driver (drives ``swift sample``/legacy PRM in loops);
the v5 RFT path is ``examples/rl/rft`` (S16-08), and rft.py itself was already smoke-driven in
the earlier examples batches -- registered, not re-run here.
"""
import json
import os
import shutil
from pathlib import Path

import pytest

from swift.dev.tests.feature.examples.conftest import MODEL_INSTRUCT, assert_smoke

EXAMPLES = Path(__file__).parents[5] / 'examples'

pytestmark = [pytest.mark.slow, pytest.mark.accel(1)]


def test_custom_dataset_example_smokes(tmp_path):
    """examples/train/custom/dataset/train.sh: the custom dataset registry. A twin
    ``dataset_info.json`` (same mapping contract, absolute data path) is written under tmp so
    the smoke does not depend on the shipped file's relative path resolution."""
    src_dir = EXAMPLES / 'train' / 'custom' / 'dataset'
    shutil.copy(src_dir / 'data.jsonl', tmp_path / 'data.jsonl')
    info = [{
        'dataset_type': 'my_qa',
        'dataset_path': str(tmp_path / 'data.jsonl'),
        'columns': {'prompt': 'query', 'completion': 'response'},
    }]
    info_path = tmp_path / 'dataset_info.json'
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=4), encoding='utf-8')

    out_dir = str(tmp_path / 'custom_ds_out')
    from swift.dev.cli.sft import sft_main
    history = sft_main([
        '--model', MODEL_INSTRUCT,
        '--task_type', 'causal_lm',
        '--torch_dtype', 'bfloat16',
        '--custom_dataset_info', str(info_path),
        '--dataset', 'my_qa',
        '--tuner', 'lora',
        '--lora_rank', '8',
        '--lora_alpha', '32',
        '--target_modules', 'all-linear',
        '--max_steps', '3',
        '--per_device_train_batch_size', '1',
        '--gradient_accumulation_steps', '2',
        '--learning_rate', '1e-4',
        '--lr_scheduler', 'cosine',
        '--warmup_ratio', '0.05',
        '--max_length', '1024',
        '--logging_steps', '1',
        '--save_steps', '100',
        '--save_total_limit', '2',
        '--split_dataset_ratio', '0.1',
        '--output_dir', out_dir,
        '--report_to', 'none',
    ])
    assert_smoke(history, out_dir, 'causal_lm')


def test_custom_model_template_example_smokes(tmp_path, sft_rows):
    """examples/train/custom/model_template/train.sh: custom family + custom template via
    ``--external_plugins``; ``--model_type custom_im`` and ``--template custom_im`` must both
    resolve from the plugin's registrations."""
    out_dir = str(tmp_path / 'custom_mt_out')
    from swift.dev.cli.sft import sft_main
    history = sft_main([
        '--external_plugins', str(EXAMPLES / 'train' / 'custom' / 'model_template' / 'model.py'),
        '--model', MODEL_INSTRUCT,
        '--model_type', 'custom_im',
        '--template', 'custom_im',
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
    ])
    assert_smoke(history, out_dir, 'causal_lm')


def test_plugins_example_smokes(tmp_path, sft_rows):
    """examples/train/plugins/train.sh: the progress_printer callback plugin. The eval cadence
    scales with the smoke volume (eval every 2 of 6 steps) so the callback's on-evaluate path
    fires too, not just on_step_end."""
    out_dir = str(tmp_path / 'plugins_out')
    from swift.dev.cli.sft import sft_main
    history = sft_main([
        '--model', MODEL_INSTRUCT,
        '--task_type', 'causal_lm',
        '--torch_dtype', 'bfloat16',
        '--external_plugins', str(EXAMPLES / 'train' / 'plugins' / 'my_plugin.py'),
        '--callbacks', 'progress_printer',
        '--dataset', sft_rows,
        '--tuner', 'lora',
        '--lora_rank', '8',
        '--lora_alpha', '32',
        '--target_modules', 'all-linear',
        '--max_steps', '6',
        '--per_device_train_batch_size', '1',
        '--per_device_eval_batch_size', '1',
        '--gradient_accumulation_steps', '4',
        '--learning_rate', '1e-4',
        '--lr_scheduler', 'cosine',
        '--warmup_ratio', '0.05',
        '--max_length', '2048',
        '--eval_strategy', 'steps',
        '--eval_steps', '2',
        '--logging_steps', '1',
        '--save_steps', '100',
        '--save_total_limit', '2',
        '--split_dataset_ratio', '0.25',
        '--output_dir', out_dir,
        '--report_to', 'none',
    ])
    ckpt = assert_smoke(history, out_dir, 'causal_lm')
    assert ckpt
