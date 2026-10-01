# Copyright (c) ModelScope Contributors. All rights reserved.
"""切分方式 (维度 5) x 启动方式 (维度 8) -- coverage map plus the one launch mode nothing else drives.

Where each parallel / launch combination is ALREADY gated end to end on real weights (this file adds
to that matrix, it never re-runs it):

  - transformers dp2 -- the reported loss is the GLOBAL token-weighted mean, not an avg-of-avg:
    ``test_e2e.py::test_run_sft_hf_dp_loss_is_globally_token_weighted``.
  - transformers sp2 (Ulysses) and the hybrid world=4/sp=2 layout -- loss == the single-process loss
    on the same global batch (SP is a pure re-partition):
    ``test_e2e.py::test_run_sft_hf_sp_matches_single`` / ``::test_run_sft_hf_sp_hybrid_dp``.
  - megatron tp2 / pp2 / cp2 / ep2 / sp -- each completes its steps with finite loss and writes a
    checkpoint (tiny weights, so availability not numerics):
    ``capability/test_capability.py::test_combination_trains_and_saves``.
  - megatron tp2+sp real-weight legacy parity -- 50-step lr bit-exact + banded loss vs legacy:
    ``test_e2e.py::test_megatron_dense_cli_vs_legacy_50_steps``.
  - single card (a plain process, no launcher, twinkle default dp=1 mesh):
    ``test_e2e.py::test_run_sft_end_to_end_happy_path``.

The launch mode NONE of the above exercises is **Ray on the transformers backend**. Every Ray run in
``test_e2e.py`` is megatron (``DistributedConfig(backend='megatron', mode='ray')``), and every
transformers multi-GPU run there is torchrun (``mode='local'``). The transformers Ray assembly is
distinct on both ends: ``initialize_twinkle`` builds a 'model' DeviceGroup the workers join while the
driver only orchestrates (``assembly.py``), ``build_model`` places the model there through
``_apply_ray_placement`` (a pure-DP mesh, since the transformers backend has no TP/PP weight
sharding), and the DP scatter moves OUT of the dataloader's ``DeviceMeshSampler`` INTO
``forward_backward``'s ``dispatch='slice_dp'`` on the driver (``train_loop.py``). A torchrun-local
test cannot reach any of that, so it is driven here.
"""
import os

import json
import pytest
import torch

MODEL_TEXT = 'Qwen/Qwen2.5-0.5B-Instruct'

#: causal_lm rows, distinct enough to be a real (non-degenerate) batch but short so a 0.5B run over
#: two Ray workers stays light. Four rows == two optimizer steps at per_device_train_batch_size=2.
_TOY_ROWS = [
    ('What is 2+2?', '2 + 2 equals 4.'),
    ('Say hello.', 'Hello! How can I help you today?'),
    ('Capital of France?', 'The capital of France is Paris.'),
    ('Name a color.', 'Blue is a color.'),
]


def _write_toy_dataset(path):
    with open(path, 'w') as f:
        for user, assistant in _TOY_ROWS:
            f.write(
                json.dumps({
                    'messages': [{
                        'role': 'user',
                        'content': user
                    }, {
                        'role': 'assistant',
                        'content': assistant
                    }]
                }) + '\n')


@pytest.mark.slow
@pytest.mark.accel(2)
def test_run_sft_hf_ray_launch_end_to_end(tmp_path):
    """transformers-backend SFT launched through Ray (mode='ray', nproc_per_node=2) must train.

    This is the launch-mode gate for the non-megatron Ray path. The driver runs the loop once and
    scatters each batch across the two DP workers via ``forward_backward(dispatch='slice_dp')``; the
    workers hold the model in the 'model' DeviceGroup built by ``initialize_twinkle``. A green run
    means that whole driver/worker assembly wired up and stepped the optimizer -- which the torchrun
    (local-mode) dp/sp tests in ``test_e2e.py`` structurally cannot exercise.

    per_device_train_batch_size must be >= the DP size (2): slice_dp splits each driver batch across
    the two workers, so a batch of 1 would starve one (same constraint the megatron Ray runs note).
    LoRA keeps the checkpoint small and additionally exercises the adapter's own OptimizerGroup under
    the Ray DP mesh, mirroring the torchrun dp runner.
    """
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    from modelscope import snapshot_download

    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        ModelConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )
    from swift.dev.recipe import run_sft

    model_path = snapshot_download(MODEL_TEXT)
    data_path = str(tmp_path / 'toy_sft.jsonl')
    out_dir = str(tmp_path / 'out')
    _write_toy_dataset(data_path)

    history = run_sft(
        ModelConfig(model=model_path, torch_dtype='bfloat16'),
        TemplateConfig(template='qwen2_5', max_length=256),
        DatasetConfig(dataset=[data_path], dataset_shuffle=False),
        TrainConfig(
            learning_rate=1e-4,
            lr_scheduler='constant',
            warmup_ratio=0.0,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=2,
            max_grad_norm=1.0),
        # mode='ray' + nproc_per_node: initialize_twinkle builds the 'model' DeviceGroup the two
        # workers join; there is no torchrun, the driver orchestrates in-process.
        DistributedConfig(mode='ray', nproc_per_node=2),
        CheckpointConfig(),
        tuner_config=TunerConfig(tuner='lora'),
        output_dir=out_dir,
    )

    assert history, 'run_sft (hf/ray) produced no optimizer steps -- the driver never stepped the workers'
    losses = [r['loss'] for r in history]
    assert all(loss == loss and abs(loss) != float('inf') for loss in losses), f'non-finite loss: {losses}'
    assert losses[0] < 20, f'first loss {losses[0]:.2f} too large -> not normalized (raw sum?)'

    ckpt = os.path.join(out_dir, 'checkpoint-final')
    assert os.path.isdir(ckpt), f'no checkpoint dir at {ckpt}'
    files = set(os.listdir(ckpt))
    assert any(f.endswith('.safetensors') for f in files), f'no model weights in checkpoint: {sorted(files)}'
    assert 'args.json' in files, f'no args.json (checkpoint not self-describing): {sorted(files)}'
    print(f'\nrun_sft hf/ray: steps={len(history)} losses={losses}')
