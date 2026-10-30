# Copyright (c) ModelScope Contributors. All rights reserved.
"""Opt-in GPU export regression with a local Qwen3 model and calibration dataset."""
import json
import os
import pytest
import subprocess
import sys
import torch
from pathlib import Path


@pytest.mark.parametrize('method', ['awq', 'gptq', 'gptq_v2'])
def test_qwen3_quantization_export(tmp_path, method):
    model = os.environ.get('SWIFT_TEST_QUANT_MODEL')
    dataset = os.environ.get('SWIFT_TEST_QUANT_DATASET')
    if not model or not dataset or not torch.cuda.is_available():
        pytest.skip('Set SWIFT_TEST_QUANT_MODEL and SWIFT_TEST_QUANT_DATASET and provide a CUDA GPU')

    output_dir = tmp_path / method
    script = Path(__file__).resolve().parents[2] / 'swift' / 'cli' / 'export.py'
    command = [
        sys.executable,
        str(script), '--model', model, '--model_type', 'qwen3', '--template', 'qwen3', '--dataset', dataset,
        '--output_dir',
        str(output_dir), '--quant_method', method, '--quant_bits', '4', '--group_size', '128', '--quant_n_samples',
        '32', '--quant_batch_size', '1', '--max_length', '256'
    ]
    subprocess.run(command, check=True, timeout=1800)
    config = json.loads((output_dir / 'config.json').read_text())['quantization_config']
    assert config['bits'] == 4
    assert config['group_size'] == 128
    assert config['quant_method'] == ('awq' if method == 'awq' else 'gptq')
    if method == 'awq':
        assert config['zero_point'] is True
        assert config['version'].lower() == 'gemm'
    assert list(output_dir.glob('*.safetensors'))
