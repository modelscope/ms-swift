# Copyright (c) ModelScope Contributors. All rights reserved.
"""S16-05 (L2): smoke-run the train/distributed-group v5 examples.

Two v5 launch topologies under ``examples/train/`` are the group's real members:

- ``multi-gpu/torchrun/train.sh`` -- NPROC_PER_NODE=2 makes ``swift/cli/main.py`` relaunch the
  dev CLI under ``torch.distributed.run`` (``--parallel_spec dp2``). Smoked here as a REAL
  subprocess of the full chain (``python -m swift.cli.main sft ...`` with the script's env), so
  the relaunch itself is under test, not just the training it lands in.
- ``multi-gpu/ray/train.sh`` -- ``--mode ray --nproc_per_node 2`` in-process Ray driver; smoked
  through ``sft_main`` with the script's argv verbatim (the ray launch mode is already driven
  end-to-end by the SFT suite's Config-level test; this pins the example's flag surface).

``multi-node/*`` needs two nodes by construction (the scripts demand ``NNODES``/``NODE_RANK``/
``MASTER_ADDR``); its single-box dual-torchrun simulation was hand-verified during the examples
refactor (world=4, both processes' losses bit-identical, NCCL ``nranks 4`` in the log) and is
registered in the plan notes rather than re-enacted per test run. ``sequence_parallel/*`` and
``all_to_all/*`` are legacy scripts (no USE_SWIFT_V5); the SP capability itself is covered by the
RL suite's wired-types tests. ``megatron/{tp_pp,moe}`` belong to S16-13.
"""
import os
import signal
import subprocess
import sys

import pytest

from swift.dev.tests.feature.examples.conftest import MODEL_INSTRUCT, assert_smoke

pytestmark = [pytest.mark.slow, pytest.mark.accel(2)]


def _launch(cmd, env, *, timeout=1200):
    """A bounded subprocess launch (process-group kill on timeout -- torchrun's worker
    grandchildren inherit the pipes, so a plain timeout never fires; same reasoning as
    ``Runners.launch``), with the caller's env."""
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as expired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        out, err = proc.communicate()
        raise AssertionError(f'subprocess exceeded {timeout}s and was killed. '
                             f'stderr tail:\n{err[-3000:]}') from expired
    return proc.returncode, out, err


def test_multi_gpu_torchrun_example_smokes(tmp_path, sft_rows):
    """examples/train/multi-gpu/torchrun/train.sh: the FULL chain a user runs -- ``swift sft``
    with NPROC_PER_NODE=2 relaunches under torchrun with ``--parallel_spec dp2``. Asserting on
    the relaunch's exit code and the landed checkpoint covers routing, rendezvous, both ranks'
    training and the merged final save."""
    out_dir = str(tmp_path / 'mg_torchrun_out')
    argv = [
        '--model', MODEL_INSTRUCT,
        '--task_type', 'causal_lm',
        '--torch_dtype', 'bfloat16',
        '--attn_impl', 'flash_attention_2',
        '--dataset', sft_rows,
        '--tuner', 'lora',
        '--lora_rank', '8',
        '--lora_alpha', '32',
        '--target_modules', 'all-linear',
        '--parallel_spec', 'dp2',
        '--max_steps', '3',
        '--per_device_train_batch_size', '2',
        '--gradient_accumulation_steps', '4',
        '--learning_rate', '1e-4',
        '--lr_scheduler', 'cosine',
        '--warmup_ratio', '0.05',
        '--max_length', '2048',
        '--logging_steps', '1',
        '--save_steps', '100',
        '--output_dir', out_dir,
        '--report_to', 'none',
    ]
    env = dict(os.environ)
    env.update({
        'USE_SWIFT_V5': '1',
        'NPROC_PER_NODE': '2',
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', '0,1'),
    })
    returncode, out, err = _launch([sys.executable, '-m', 'swift.cli.main', 'sft', *argv], env)
    assert returncode == 0, f'swift sft relaunch failed. stdout tail:\n{out[-2000:]}\nstderr tail:\n{err[-3000:]}'
    # The relaunch really happened (the printed run-sh line is torch.distributed.run), and the
    # final checkpoint landed with weights.
    assert 'torch.distributed.run' in out, f'no torchrun relaunch in output: {out[-500:]}'
    import glob as _glob
    finals = sorted(_glob.glob(os.path.join(out_dir, 'v*', 'checkpoint-final')))
    assert finals, f'no final checkpoint under {out_dir}'


def test_multi_gpu_ray_example_smokes(tmp_path, sft_rows):
    """examples/train/multi-gpu/ray/train.sh: ``--mode ray --nproc_per_node 2`` in-process Ray
    driver, argv verbatim (per_device_train_batch_size 2 >= dp 2, as the script pins)."""
    out_dir = str(tmp_path / 'mg_ray_out')
    from swift.dev.cli.sft import sft_main
    history = sft_main([
        '--model', MODEL_INSTRUCT,
        '--task_type', 'causal_lm',
        '--torch_dtype', 'bfloat16',
        '--attn_impl', 'flash_attention_2',
        '--dataset', sft_rows,
        '--tuner', 'lora',
        '--lora_rank', '8',
        '--lora_alpha', '32',
        '--target_modules', 'all-linear',
        '--mode', 'ray',
        '--nproc_per_node', '2',
        '--max_steps', '3',
        '--per_device_train_batch_size', '2',
        '--gradient_accumulation_steps', '4',
        '--learning_rate', '1e-4',
        '--lr_scheduler', 'cosine',
        '--warmup_ratio', '0.05',
        '--max_length', '2048',
        '--logging_steps', '1',
        '--save_steps', '100',
        '--output_dir', out_dir,
        '--report_to', 'none',
    ])
    assert_smoke(history, out_dir, 'causal_lm')
    # Tear the ray session down so later modules in the same pytest process start clean.
    try:
        import ray
        ray.shutdown()
    except Exception:  # noqa: BLE001 -- best-effort cleanup only
        pass
