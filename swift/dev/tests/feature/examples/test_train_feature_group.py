# Copyright (c) ModelScope Contributors. All rights reserved.
"""S16-02 (L2): smoke-run the train/feature-group examples on the real CLI-to-recipe path.

Same smoke contract as ``test_train_basic_group.py`` (see ``conftest.py``): one test per v5
example script, argv verbatim, three smoke substitutions (local-cache model id, tiny local
dataset, ``--max_steps`` volume). Group members with a special twist:

- ``early_stop`` needs its eval cadence to actually fire inside the smoke budget, so its cadence
  values are scaled down with the volume (eval/save every 2 steps, patience 2) -- the semantics
  (eval strategy, best-model tracking, early stop) all stay on.
- ``yaml`` replays the script's yaml twin through the REAL expansion path
  (``swift.cli.main.parse_yaml_args`` -> ``sft_main``), the same chain ``cli_main`` uses, with the
  smoke substitutions written into a copy of the yaml.
- ``flash_attention_3`` is REFERENCE-ONLY on this box (FA3 needs a Hopper GPU; the authoring box
  is an L20X/Ada -- the script header says so). The runtime half cannot be smoked here; its parse
  surface is pinned instead (the dev CLI accepts ``--attn_impl flash_attention_3``), and the
  hardware gap is registered in the plan notes.
- ``new_special_tokens`` and ``predict_with_generate`` are legacy remnants (``--tuner_type`` and
  no ``USE_SWIFT_V5``): outside the v5 smoke surface, registered in the plan notes.
"""
import pytest

from swift.dev.tests.feature.examples.conftest import MODEL_INSTRUCT, MODEL_THINK, assert_smoke

pytestmark = [pytest.mark.slow, pytest.mark.accel(1)]


def _run(argv, out_dir, task_type='causal_lm'):
    from swift.dev.cli.sft import sft_main
    history = sft_main(argv)
    return assert_smoke(history, out_dir, task_type)


def test_liger_example_smokes(tmp_path, sft_rows):
    """examples/train/liger/train.sh: ``--use_liger_kernel true`` + the fused-linear-CE config,
    verbatim (kernel substitution is exactly what a broken flag would break)."""
    out_dir = str(tmp_path / 'liger_out')
    _run([
        '--model', MODEL_INSTRUCT,
        '--task_type', 'causal_lm',
        '--torch_dtype', 'bfloat16',
        '--dataset', sft_rows,
        '--tuner', 'lora',
        '--lora_rank', '8',
        '--lora_alpha', '32',
        '--target_modules', 'all-linear',
        '--use_liger_kernel', 'true',
        '--liger_kernel_config', '{"fused_linear_cross_entropy": true}',
        '--max_steps', '3',
        '--per_device_train_batch_size', '1',
        '--gradient_accumulation_steps', '8',
        '--learning_rate', '1e-4',
        '--lr_scheduler', 'cosine',
        '--warmup_ratio', '0.05',
        '--max_length', '2048',
        '--logging_steps', '5',
        '--save_steps', '100',
        '--output_dir', out_dir,
        '--report_to', 'none',
    ], out_dir)


def test_galore_example_smokes(tmp_path, sft_rows):
    """examples/train/galore/train.sh: full-param + adamw_torch with the GaLore rank/schedule
    knobs verbatim (the GaLore optimizer group split is the path under test)."""
    out_dir = str(tmp_path / 'galore_out')
    _run([
        '--model', MODEL_INSTRUCT,
        '--task_type', 'causal_lm',
        '--torch_dtype', 'bfloat16',
        '--dataset', sft_rows,
        '--tuner', 'full',
        '--optim', 'adamw_torch',
        '--use_galore', 'true',
        '--galore_rank', '128',
        '--galore_update_proj_gap', '50',
        '--galore_scale', '1.0',
        '--max_steps', '3',
        '--per_device_train_batch_size', '1',
        '--gradient_accumulation_steps', '8',
        '--learning_rate', '1e-4',
        '--lr_scheduler', 'cosine',
        '--warmup_ratio', '0.05',
        '--max_length', '2048',
        '--logging_steps', '5',
        '--save_steps', '100',
        '--output_dir', out_dir,
        '--report_to', 'none',
    ], out_dir)


def test_muon_example_smokes(tmp_path, sft_rows):
    """examples/train/muon/train.sh: ``--optim muon`` full-param, verbatim."""
    out_dir = str(tmp_path / 'muon_out')
    _run([
        '--model', MODEL_INSTRUCT,
        '--task_type', 'causal_lm',
        '--torch_dtype', 'bfloat16',
        '--dataset', sft_rows,
        '--tuner', 'full',
        '--optim', 'muon',
        '--max_steps', '3',
        '--per_device_train_batch_size', '1',
        '--gradient_accumulation_steps', '8',
        '--learning_rate', '1e-4',
        '--lr_scheduler', 'cosine',
        '--warmup_ratio', '0.05',
        '--max_length', '2048',
        '--logging_steps', '5',
        '--save_steps', '100',
        '--output_dir', out_dir,
        '--report_to', 'none',
    ], out_dir)


def test_packing_example_smokes(tmp_path, sft_rows):
    """examples/train/packing/train.sh: ``--packing true`` binpack at 1024 on FA2, verbatim."""
    out_dir = str(tmp_path / 'packing_out')
    _run([
        '--model', MODEL_INSTRUCT,
        '--task_type', 'causal_lm',
        '--torch_dtype', 'bfloat16',
        '--dataset', sft_rows,
        '--tuner', 'lora',
        '--lora_rank', '8',
        '--lora_alpha', '32',
        '--target_modules', 'all-linear',
        '--packing', 'true',
        '--packing_length', '1024',
        '--packing_strategy', 'binpack',
        '--attn_impl', 'flash_attention_2',
        '--max_steps', '3',
        '--per_device_train_batch_size', '2',
        '--gradient_accumulation_steps', '4',
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


def test_padding_free_example_smokes(tmp_path, sft_rows):
    """examples/train/padding_free/train.sh: ``--padding_free true`` on FA2 with the script's
    wider batch (4 x 4), verbatim."""
    out_dir = str(tmp_path / 'pf_out')
    _run([
        '--model', MODEL_INSTRUCT,
        '--task_type', 'causal_lm',
        '--torch_dtype', 'bfloat16',
        '--dataset', sft_rows,
        '--tuner', 'lora',
        '--lora_rank', '8',
        '--lora_alpha', '32',
        '--target_modules', 'all-linear',
        '--padding_free', 'true',
        '--attn_impl', 'flash_attention_2',
        '--max_steps', '3',
        '--per_device_train_batch_size', '4',
        '--gradient_accumulation_steps', '4',
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


def test_early_stop_example_smokes(tmp_path, sft_rows):
    """examples/train/early_stop/train.sh: eval + best-model tracking + early stop. The cadence
    values scale down with the smoke volume (eval/save every 2 of 6 steps, patience 2) so the
    eval loop and the early-stop checker actually run; the split widens to 0.25 so the eval set
    is non-empty on 16 rows."""
    out_dir = str(tmp_path / 'es_out')
    _run([
        '--model', MODEL_INSTRUCT,
        '--task_type', 'causal_lm',
        '--torch_dtype', 'bfloat16',
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
        '--metric_for_best_model', 'eval_loss',
        '--greater_is_better', 'false',
        '--load_best_model_at_end', 'true',
        '--early_stop_interval', '2',
        '--logging_steps', '1',
        '--save_strategy', 'steps',
        '--save_steps', '2',
        '--save_total_limit', '10',
        '--split_dataset_ratio', '0.25',
        '--output_dir', out_dir,
        '--report_to', 'none',
    ], out_dir)


def test_think_model_example_smokes(tmp_path, sft_rows):
    """examples/train/think_model/train.sh: Qwen3-0.6B with ``--enable_thinking true`` (the
    thinking-block template contract), verbatim including the 4096 max_length."""
    out_dir = str(tmp_path / 'think_out')
    _run([
        '--model', MODEL_THINK,
        '--task_type', 'causal_lm',
        '--enable_thinking', 'true',
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
        '--max_length', '4096',
        '--logging_steps', '5',
        '--save_steps', '100',
        '--save_total_limit', '2',
        '--split_dataset_ratio', '0.01',
        '--output_dir', out_dir,
        '--report_to', 'none',
    ], out_dir)


def test_yaml_example_smokes(tmp_path, sft_rows):
    """examples/train/yaml/: replay the script's sft.yaml through the REAL expansion path --
    ``parse_yaml_args`` rewrites the yaml into ``--key value`` argv exactly as ``cli_main`` does
    -- with the smoke substitutions written into a copy of the yaml."""
    import shutil
    from pathlib import Path

    from swift.cli.main import parse_yaml_args

    src_yaml = Path(__file__).parents[5] / 'examples' / 'train' / 'yaml' / 'sft.yaml'
    out_dir = str(tmp_path / 'yaml_out')
    text = src_yaml.read_text(encoding='utf-8')
    text = text.replace("- 'swift/self-cognition#600'", f"- '{sft_rows}'")
    text = text.replace('num_train_epochs: 3', 'max_steps: 3')
    text = text.replace('output_dir: output', f'output_dir: {out_dir}')
    text = text.replace('report_to: tensorboard', "report_to:\n  - 'none'")
    smoke_yaml = tmp_path / 'sft.yaml'
    smoke_yaml.write_text(text, encoding='utf-8')

    argv = [str(smoke_yaml)]
    parse_yaml_args(argv)
    assert argv[0] == '--model', f'yaml expansion did not rewrite argv: {argv[:4]}'
    from swift.dev.cli.sft import sft_main
    history = sft_main(argv)
    assert_smoke(history, out_dir, 'causal_lm')


def test_flash_attention_3_example_parses():
    """examples/train/flash_attention_3/train.sh is REFERENCE-ONLY hardware-wise (FA3 needs
    Hopper; this box is Ada/L20X). Its PARSE surface is still pinned: ``--attn_impl
    flash_attention_3`` is a legal dev-CLI value, so the example survives argv parsing and fails
    only at engine bring-up on non-Hopper hardware -- not silently mis-parsed into something
    else."""
    from swift.dev.cli.sft import parse_sft_configs
    configs = parse_sft_configs([
        '--model', MODEL_INSTRUCT,
        '--task_type', 'causal_lm',
        '--tuner', 'lora',
        '--attn_impl', 'flash_attention_3',
        '--max_steps', '1',
        '--output_dir', '/tmp/fa3_parse_only',
    ])
    assert configs['model_config'].attn_impl == 'flash_attention_3'
