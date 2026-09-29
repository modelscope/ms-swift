# Copyright (c) ModelScope Contributors. All rights reserved.
"""Slow end-to-end HF <-> mcore weight-conversion round-trip.

``test_convert.py`` pins ``run_convert``'s control flow (guards, dispatch, output_dir) with the weight
workers stubbed; this file drives the REAL thing so the seams a stub cannot see are proven -- the ones
that only exist when ``swift.megatron`` + mcore-bridge actually migrate tensors:

* the dev ``swift export --backend megatron`` CLI is launched as a real single-rank ``torch.distributed.run``
  subprocess (the exact launch ``examples/v5/export/mcore_convert.sh`` uses), so parsing, the Megatron
  backend shim, ``run_convert`` and the legacy converters all run for real;
* HF safetensors --``--to_mcore``--> mcore ``torch_dist`` --``--to_hf``--> HF safetensors on a tiny model
  at TP=PP=1, asserting each step writes its artifact (``iter_0000001`` + ``args.json``, then
  ``config.json`` + ``*.safetensors``);
* the round-tripped HF model is numerically equivalent to the original -- same greedy tokens, logits
  within the precision bar the codebase itself uses for ``--test_convert_precision`` -- so a tensor-name
  mapping or sharding regression shows up here rather than only at megatron train time.

``@pytest.mark.slow`` + ``@pytest.mark.accel(1)``; run with ``-m slow``.
"""
import os
import socket
import subprocess
import sys

import pytest

MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'

pytestmark = [pytest.mark.slow, pytest.mark.accel(1)]


def _free_port():
    with socket.socket() as sock:
        sock.bind(('', 0))
        return sock.getsockname()[1]


def _run_export(args, log_path):
    """Launch the dev ``swift export`` CLI as a real single-rank torchrun subprocess.

    This mirrors what ``swift/cli/main.py`` does for a Megatron export (``python -m torch.distributed.run
    --nproc_per_node=1 --module swift.dev.cli.export ...``), so the conversion runs under the same
    distributed context the real command uses. ``CUDA_VISIBLE_DEVICES`` is inherited so the harness's
    GPU assignment (``accel(1)``) is respected. stdout/stderr are captured to a file the assertion can
    point at on failure.
    """
    env = dict(os.environ)
    env['USE_SWIFT_V5'] = '1'
    env['NPROC_PER_NODE'] = '1'
    cmd = [
        sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node=1', '--master_port',
        str(_free_port()), '--module', 'swift.dev.cli.export', *args
    ]
    with open(log_path, 'w') as log:
        return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=os.getcwd()).returncode


def test_hf_to_mcore_to_hf_roundtrip(tmp_path):
    """HF -> mcore -> HF really migrates the weights both ways and lands back on an equivalent model."""
    import torch
    from modelscope import snapshot_download
    from transformers import AutoModelForCausalLM

    model_path = snapshot_download(MODEL)
    mcore_dir = str(tmp_path / 'mcore')
    hf_out = str(tmp_path / 'hf')

    # --- HF safetensors -> mcore torch_dist -------------------------------------------------
    rc = _run_export(
        ['--backend', 'megatron', '--model', model_path, '--to_mcore', 'true', '--output_dir', mcore_dir],
        str(tmp_path / 'to_mcore.log'))
    assert rc == 0, f'--to_mcore exited {rc}; see {tmp_path / "to_mcore.log"}'
    assert os.path.isfile(os.path.join(mcore_dir, 'args.json')), 'to_mcore did not write args.json'
    assert os.path.isdir(os.path.join(mcore_dir, 'iter_0000001')), 'to_mcore did not write the dist checkpoint'

    # --- mcore torch_dist -> HF safetensors -------------------------------------------------
    # --model is passed explicitly (the source architecture); it also rides in mcore_dir/args.json, which
    # the to_hf step would otherwise restore, so the two agree.
    rc = _run_export([
        '--backend', 'megatron', '--model', model_path, '--mcore_model', mcore_dir, '--to_hf', 'true',
        '--safe_serialization', 'true', '--output_dir', hf_out
    ], str(tmp_path / 'to_hf.log'))
    assert rc == 0, f'--to_hf exited {rc}; see {tmp_path / "to_hf.log"}'
    assert os.path.isfile(os.path.join(hf_out, 'config.json')), 'to_hf did not write config.json'
    assert any(f.endswith('.safetensors') for f in os.listdir(hf_out)), 'to_hf did not write safetensors'

    # --- numerical equivalence: original vs round-tripped (CPU, fp32 upcast of the bf16 weights) ---
    input_ids = torch.tensor([[10, 20, 30, 40, 50, 60]])

    def _logits(path):
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).eval()
        with torch.no_grad():
            return model(input_ids).logits

    original = _logits(model_path)
    roundtripped = _logits(hf_out)
    assert original.shape == roundtripped.shape
    # Greedy next-token prediction must survive the round-trip exactly ...
    assert (original.argmax(-1) == roundtripped.argmax(-1)).all(), 'round-trip changed the greedy tokens'
    # ... and the logits stay within the bar the codebase uses for --test_convert_precision (mean < 0.1).
    mean_diff = (original - roundtripped).abs().mean().item()
    assert mean_diff < 0.1, f'round-trip logit mean abs diff {mean_diff} >= 0.1'
