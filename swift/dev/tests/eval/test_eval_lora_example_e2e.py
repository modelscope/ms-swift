# Copyright (c) ModelScope Contributors. All rights reserved.
"""Slow tier: the real ``swift eval --adapters`` command, mirroring ``examples/v5/eval/eval_lora.sh``.

``eval_lora.sh`` scores a LoRA adapter *loaded live* into the engine: the base model comes from ``--model``,
``--adapters`` points at a trained checkpoint, the engine is built with LoRA enabled and the adapter is
selected for every request. There is no merge step, so no merged weights ever land on disk -- that is the
distinction from ``swift deploy --merge_lora`` and the thing this test pins hardest.

The adapter here is a real (randomly-initialised) LoRA over ``Qwen/Qwen2.5-0.5B-Instruct``: the weights are
untrained because what matters is that the adapter targets real modules of the base architecture, loads into
the engine and is selected per request, not what it learned. A green module means ``eval_lora.sh`` runs as
written -- a real vLLM engine, a real EvalScope benchmark, a live adapter -- not that a stub wrote a report.

All tests are ``@pytest.mark.slow`` + ``@pytest.mark.accel(1)``; run with ``-m slow`` and a free card in
``CUDA_VISIBLE_DEVICES`` (inherited by the subprocess).
"""
import gc
import os

import pytest

from swift.dev.tests.eval.conftest import MODEL, read_jsonl, run_eval_cli

pytestmark = [pytest.mark.slow, pytest.mark.accel(1)]


@pytest.fixture(scope='module')
def lora_adapter(tmp_path_factory):
    """A real (randomly-initialised) LoRA over ``MODEL``, saved as a PEFT adapter directory."""
    import torch
    from peft import LoraConfig, get_peft_model

    from swift.dev.builders import load_model_processor
    from swift.dev.config import ModelConfig

    dest = str(tmp_path_factory.mktemp('lora') / 'checkpoint-100')
    # No model_type: swift resolves it from --model (Qwen2.5 text -> qwen2), exactly as the eval CLI does.
    model_config = ModelConfig(model=MODEL, torch_dtype='bfloat16', device_map='cpu')
    model, _ = load_model_processor(model_config, load_model=True)
    lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=['q_proj', 'v_proj'], bias='none')
    peft_model = get_peft_model(model, lora_config)
    peft_model.save_pretrained(dest)
    # Release the CPU copy of the base before the subprocess loads its own for the engine.
    del peft_model
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    assert os.path.isfile(os.path.join(dest, 'adapter_config.json'))
    return dest


def test_eval_lora_example_scores_live_adapter_end_to_end(tmp_path, lora_adapter):
    """``swift eval --adapters``: exit 0, the adapter recorded, scored, and never merged to disk."""
    out_dir = tmp_path / 'eval_output'
    jsonl = out_dir / 'results.jsonl'
    proc = run_eval_cli([
        '--model', MODEL,
        '--adapters', lora_adapter,
        '--infer_backend', 'vllm',
        '--eval_dataset', 'gsm8k',
        '--eval_limit', '5',
        '--eval_num_proc', '8',
        '--eval_generation_config', '{"max_tokens": 256, "temperature": 0.0}',
        '--vllm_gpu_memory_utilization', '0.5',
        '--eval_output_dir', str(out_dir),
        '--result_jsonl', str(jsonl),
    ])
    # A non-zero exit here would mean the engine rejected the live adapter (LoRA not enabled, or adapter_path
    # not honoured per request) -- the exact failure eval_lora.sh's live-load path must never hit.
    assert proc.returncode == 0, f'swift eval --adapters failed:\n--- stdout ---\n{proc.stdout}\n' \
                                 f'--- stderr ---\n{proc.stderr}'

    recorded = read_jsonl(str(jsonl))
    assert len(recorded) == 1, recorded
    row = recorded[0]
    # run_eval records the adapters it threaded to the engine; the base model is still --model (resolved to
    # its local cache snapshot by config validation, so compare the checkpoint's leaf name, not the raw id).
    assert MODEL.split('/')[-1] in row['model'], row['model']
    assert [os.path.realpath(a) for a in row['adapters']] == [os.path.realpath(lora_adapter)], row['adapters']
    assert [r for r in row['Native'] if r.get('dataset_name') == 'gsm8k'], row['Native']

    # Live-load, not merge: nothing was folded into a merged checkpoint beside the adapter (deploy's
    # merge_lora writes ``{adapter}-merged``), and the adapter dir is left exactly as it was.
    assert not os.path.exists(f'{lora_adapter}-merged'), 'eval must load the adapter live, not merge it'
    assert not os.path.isfile(os.path.join(lora_adapter, 'model.safetensors')), 'adapter gained merged weights'
    assert os.path.isfile(os.path.join(lora_adapter, 'adapter_config.json'))
