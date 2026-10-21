# Copyright (c) ModelScope Contributors. All rights reserved.
"""unsloth backend e2e, isolated from the rest of test_frameworks.py on purpose.

Importing unsloth closes the process-level stderr (an fd-level replace, not a ``sys.stderr``
rebinding -- holding the Python object aside does not survive it), which destroys pytest's capture
stream and takes the whole process down at teardown with ``ValueError: I/O operation on closed
file / lost sys.stderr``. Isolated here, the blast radius is this file alone; run it WITHOUT
capture::

    CUDA_VISIBLE_DEVICES=<card> pytest swift/dev/tests/feature/sft/test_frameworks_unsloth.py -s

Verified green that way on a real 3-step LoRA run (unsloth installed, transformers 5.x).
"""
import pytest
import torch

pytestmark = [pytest.mark.slow, pytest.mark.accel(1)]

#: causal_lm cross-entropy over the ~151k Qwen vocab settles near ln(vocab) ~= 11.9; 20.0 sits above
#: a correctly mean-reduced run yet far below a sum-reduced or diverged one (conftest.assert_trained).
_CAUSAL_LM_MAX_LOSS = 20.0


def _tiny_sft_rows():
    """Four plain causal_lm rows -- the framework is the variable under test, not the data."""
    return [{
        'messages': [{
            'role': 'user',
            'content': q
        }, {
            'role': 'assistant',
            'content': a
        }]
    } for q, a in (('What is 2+2?', '2 + 2 equals 4.'), ('Say hello.', 'Hello! How can I help you?'),
                    ('Capital of France?', 'The capital of France is Paris.'), ('Name a color.', 'Blue is a color.'))]


@pytest.mark.slow
def test_run_sft_unsloth_tuner_end_to_end(tmp_path, text_model_path, assert_trained, write_jsonl):
    """unsloth backend: a real LoRA run when the package is importable, documented skip otherwise.

    dev routes unsloth through ``DistributedConfig(backend='unsloth')`` (the old
    ``TunerConfig.tuner_backend`` knob is gone); the sentence-transformers builder rejects that
    combination explicitly (builders/model.py).
    """
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    try:
        import unsloth  # noqa: F401
    except ImportError:
        pytest.skip('unsloth not installed; dev routes it via DistributedConfig(backend="unsloth")')

    from swift.dev.config import (CheckpointConfig, DatasetConfig, DistributedConfig, ModelConfig, TemplateConfig,
                                  TrainConfig, TunerConfig)
    from swift.dev.recipe.sft.run_sft import run_sft

    data = write_jsonl(tmp_path / 'unsloth_sft.jsonl', _tiny_sft_rows())
    out_dir = str(tmp_path / 'out_unsloth')
    history = run_sft(
        ModelConfig(model=text_model_path, task_type='causal_lm', torch_dtype='bfloat16'),
        TemplateConfig(template='qwen2_5', max_length=256),
        DatasetConfig(dataset=[data], dataset_shuffle=False),
        TrainConfig(learning_rate=1e-4, per_device_train_batch_size=2, gradient_accumulation_steps=1, max_steps=3),
        DistributedConfig(backend='unsloth'),
        CheckpointConfig(),
        TunerConfig(tuner='lora'),
        output_dir=out_dir,
    )
    assert_trained(history, out_dir, 'causal_lm', max_loss=_CAUSAL_LM_MAX_LOSS)
