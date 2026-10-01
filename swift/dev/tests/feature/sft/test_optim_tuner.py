# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end SFT training across the optimizer x tuner matrix dev supports (dimension 9).

The other files in this suite hold the optimizer fixed (the ``adamw_torch_fused`` default) and vary the
task_type / framework / distribution. This one holds the task fixed -- a plain causal_lm run on the
shared 0.5B -- and varies how the weights are updated and which of them:

  - **tuner axis** -- ``full`` (``tuner_config=None``: every parameter trains) vs ``lora``
    (``TunerConfig(tuner='lora')``: only the low-rank adapters train, and twinkle's set_optimizer
    filters the weight-decay groups by ``adapter_name``). Both are driven under the AdamW default.
  - **optimizer axis** -- ``adafactor`` (resolved to HF's own ``transformers.optimization.Adafactor``
    with ``scale_parameter``/``relative_step`` forced off, NOT ``torch.optim.Adafactor``), ``galore``
    (``use_galore`` upgrades the AdamW base to ``GaLoreAdamW`` and projects the full-parameter gradient
    into a low-rank subspace), and ``muon`` (``optim='muon'`` -> twinkle ``MuonClip`` + a ``muon_config``,
    without which MuonClip is plain momentum SGD). GaLore is a full-parameter technique, so it is driven
    with ``tuner_config=None`` -- ``validate._check_galore`` refuses it alongside an adapter.
  - **QGaLore fail-loudly** -- ``galore_quantization`` resolves AdamW to ``QGaLoreAdamW8bit``, which
    delegates to the external ``q_galore_torch`` package. That package is NOT installed here, so the run
    must raise an actionable ``ImportError`` at optimizer construction rather than silently training an
    unquantized projection. Asserted as the real fail-loudly path (skipped, not stubbed, if the package
    is ever installed).

These are the end-to-end counterparts of ``tests/component/optimizer/test_optimizer_config.py``, which
asserts the config->kwargs mapping against a recording model without loading weights; here the mapping
is proven by a real optimizer step on real weights with a finite, normalized loss.

No mocks: every run loads real weights and drives the real recipe. All tests are ``@pytest.mark.slow``
(+ ``@pytest.mark.accel(1)``); run with
``CUDA_VISIBLE_DEVICES=<card> pytest swift/dev/tests/feature/sft/test_optim_tuner.py -m slow``.
"""
import pytest
import torch

pytestmark = [pytest.mark.slow, pytest.mark.accel(1)]

#: causal_lm cross-entropy over the ~151k Qwen vocab settles near ln(vocab) ~= 11.9, so 20.0 sits above
#: a correctly mean-reduced run yet far below a sum-reduced (scales with sequence length) or diverged one.
#: Shared with test_frameworks/test_multimodal -- the ceiling is a property of the task, not the optimizer.
_CAUSAL_LM_MAX_LOSS = 20.0


def _tiny_sft_rows():
    """Four plain causal_lm rows -- the optimizer/tuner is the variable under test, not the data."""
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


def _run_causal_lm(model_path, data_path, out_dir, train_config, tuner_config=None):
    """Drive one real causal_lm run_sft on the shared 0.5B with a caller-supplied TrainConfig/tuner."""
    from swift.dev.config import (CheckpointConfig, DatasetConfig, DistributedConfig, ModelConfig, TemplateConfig)
    from swift.dev.recipe import run_sft
    return run_sft(
        ModelConfig(model=model_path, task_type='causal_lm', torch_dtype='bfloat16'),
        TemplateConfig(template='qwen2_5', max_length=256),
        DatasetConfig(dataset=[data_path], dataset_shuffle=False),
        train_config,
        DistributedConfig(),
        CheckpointConfig(),
        tuner_config,
        output_dir=out_dir,
    )


def _qgalore_available():
    """True when the external ``q_galore_torch`` package (QGaLore's quantized projection) is importable."""
    try:
        import q_galore_torch  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.parametrize('tuner', ['full', 'lora'])
def test_adamw_trains_full_and_lora(tmp_path, text_model_path, assert_trained, write_jsonl, tuner):
    """The AdamW default trains on both ends of the tuner axis: full-param and LoRA.

    ``full`` passes ``tuner_config=None`` so every parameter trains; ``lora`` passes
    ``TunerConfig(tuner='lora')`` so only the adapters train and twinkle's set_optimizer filters the
    weight-decay groups by ``adapter_name``. A finite, normalized loss plus a self-describing
    ``task_type='causal_lm'`` checkpoint on both means the same optimizer path drives either parameter
    set -- the LoRA branch additionally proves the checkpoint records ``tuner_type`` (asserted via
    args.json in conftest.assert_trained's sibling checks) so ``swift infer`` re-loads it as an adapter.
    """
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    from swift.dev.config import TrainConfig, TunerConfig

    data = write_jsonl(tmp_path / f'adamw_{tuner}_sft.jsonl', _tiny_sft_rows())
    out_dir = str(tmp_path / f'out_adamw_{tuner}')
    tuner_config = None if tuner == 'full' else TunerConfig(tuner='lora')
    history = _run_causal_lm(
        text_model_path, data, out_dir,
        TrainConfig(
            learning_rate=1e-4, per_device_train_batch_size=2, gradient_accumulation_steps=1, max_steps=3),
        tuner_config)
    assert_trained(history, out_dir, 'causal_lm', max_loss=_CAUSAL_LM_MAX_LOSS)


def test_adafactor_full_trains(tmp_path, text_model_path, assert_trained, write_jsonl):
    """optim='adafactor' resolves to HF's Adafactor (scale_parameter/relative_step off) and trains.

    naming._OPTIM_EXTRA_KWARGS forces the two flags HF's trainer does, so dev's Adafactor matches legacy
    swift instead of torch.optim.Adafactor's differing defaults -- the one silent training-effect
    divergence in the optim map. A higher lr than AdamW is Adafactor's normal regime (it scales updates
    by the parameter RMS); a finite, normalized loss means the resolved class + flags really trained.
    """
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    from swift.dev.config import TrainConfig

    data = write_jsonl(tmp_path / 'adafactor_sft.jsonl', _tiny_sft_rows())
    out_dir = str(tmp_path / 'out_adafactor')
    history = _run_causal_lm(
        text_model_path, data, out_dir,
        TrainConfig(
            optim='adafactor', learning_rate=1e-3, per_device_train_batch_size=2,
            gradient_accumulation_steps=1, max_steps=3))
    assert_trained(history, out_dir, 'causal_lm', max_loss=_CAUSAL_LM_MAX_LOSS)


def test_galore_full_trains(tmp_path, text_model_path, assert_trained, write_jsonl):
    """use_galore upgrades AdamW to GaLoreAdamW and trains the full-parameter gradient via projection.

    ``galore_target_modules`` is left unset so twinkle's GaLoreConfig projects the attn/mlp Linear +
    Embedding weights it defaults to. ``galore_update_proj_gap=2`` is smaller than ``max_steps=3`` so the
    low-rank projection is recomputed at least once during the run -- exercising the update path, not just
    the initial projection. GaLore is full-parameter, so ``tuner_config=None`` (validate._check_galore
    refuses it alongside an adapter). A finite, normalized loss means the projected optimizer really stepped.
    """
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    from swift.dev.config import TrainConfig

    data = write_jsonl(tmp_path / 'galore_sft.jsonl', _tiny_sft_rows())
    out_dir = str(tmp_path / 'out_galore')
    history = _run_causal_lm(
        text_model_path, data, out_dir,
        TrainConfig(
            optim='adamw_torch', use_galore=True, galore_rank=8, galore_update_proj_gap=2, learning_rate=1e-4,
            per_device_train_batch_size=2, gradient_accumulation_steps=1, max_steps=3))
    assert_trained(history, out_dir, 'causal_lm', max_loss=_CAUSAL_LM_MAX_LOSS)


def test_muon_full_trains(tmp_path, text_model_path, assert_trained, write_jsonl):
    """optim='muon' resolves to twinkle MuonClip + a muon_config and trains.

    MuonClip orthogonalises the update for the parameter groups it selects and runs an AdamW step for the
    rest; without the muon_config configure_optimizer builds, it would degrade to plain momentum SGD. The
    config is assembled from TrainConfig's dual-backend muon_* / adam_* fields (optimizer._muon_config),
    all left at their defaults here so the run exercises the stock MuonClip grouping. A finite, normalized
    loss means the orthogonalised update really stepped the full-parameter model.
    """
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    from swift.dev.config import TrainConfig

    data = write_jsonl(tmp_path / 'muon_sft.jsonl', _tiny_sft_rows())
    out_dir = str(tmp_path / 'out_muon')
    history = _run_causal_lm(
        text_model_path, data, out_dir,
        TrainConfig(
            optim='muon', learning_rate=1e-4, per_device_train_batch_size=2, gradient_accumulation_steps=1,
            max_steps=3))
    assert_trained(history, out_dir, 'causal_lm', max_loss=_CAUSAL_LM_MAX_LOSS)


def _read_safetensor(ckpt_dir, name):
    """Load one tensor by name from a checkpoint dir, single-file or sharded."""
    import glob

    from safetensors import safe_open

    for path in glob.glob(glob.escape(ckpt_dir) + '/*.safetensors'):
        with safe_open(path, framework='pt') as f:
            if name in f.keys():
                return f.get_tensor(name)
    raise KeyError(f'{name} not found under {ckpt_dir}')


def test_full_param_freeze_holds_frozen_weights(tmp_path, text_model_path, assert_trained, write_jsonl):
    """freeze_parameters really freezes a full-param run: the frozen weights are bit-identical to the
    original checkpoint after training, while an unfrozen layer's weights moved.

    A loss-only assertion would go green whether or not the knob does anything -- exactly why the old
    gap stub refused to fake one -- so this compares weights. ``model.layers.0.`` is frozen by name
    prefix; layer 1 is left trainable. Proves the twinkle remote freeze seam reaches the real model on
    real weights (not just the recording fake in tests/component/optimizer/test_freeze_parameters.py)
    and that configure_optimizer built its param groups AFTER the freeze, so the frozen layer was never
    stepped. lr is raised above the other tests' 1e-4 so the trainable layer's movement is unambiguous
    in bf16.
    """
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    import os

    from swift.dev.config import TrainConfig

    data = write_jsonl(tmp_path / 'freeze_sft.jsonl', _tiny_sft_rows())
    out_dir = str(tmp_path / 'out_freeze')
    history = _run_causal_lm(
        text_model_path, data, out_dir,
        TrainConfig(
            learning_rate=1e-3, per_device_train_batch_size=2, gradient_accumulation_steps=1, max_steps=3,
            freeze_parameters=['model.layers.0.']))
    assert_trained(history, out_dir, 'causal_lm', max_loss=_CAUSAL_LM_MAX_LOSS)

    ckpt = os.path.join(out_dir, 'checkpoint-final')
    frozen_name = 'model.layers.0.mlp.gate_proj.weight'
    trained_name = 'model.layers.1.mlp.gate_proj.weight'
    assert torch.equal(
        _read_safetensor(ckpt, frozen_name), _read_safetensor(text_model_path, frozen_name)
    ), 'frozen layer 0 weight changed during training -- freeze_parameters had no effect'
    assert not torch.equal(
        _read_safetensor(ckpt, trained_name), _read_safetensor(text_model_path, trained_name)
    ), 'unfrozen layer 1 weight did not move -- the run trained nothing, so the freeze assertion is vacuous'


def test_qgalore_missing_package_fails_loudly(tmp_path, text_model_path, write_jsonl):
    """galore_quantization without q_galore_torch raises an actionable ImportError, not a silent fallback.

    QGaLoreAdamW8bit lazily subclasses ``q_galore_torch.QGaLoreAdamW8bit``; when the package is absent the
    base collapses to ``object`` and its ``__init__`` raises, pointing at ``pip install q_galore_torch``.
    validate._check_galore only resolves the NAME (AdamW + quantize -> QGaLoreAdamW8bit is a legal target),
    so the failure surfaces at optimizer construction -- after build_model, hence the real-weights setup.
    This is the fail-loudly contract: a quantized projection the environment cannot honour must never train
    an unquantized one quietly. Skipped (not stubbed) if the package is ever installed, since then the real
    quantized step is what should be exercised instead.
    """
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    if _qgalore_available():
        pytest.skip('q_galore_torch is installed; the missing-package fail-loudly path does not apply '
                    '(a real quantized GaLore step would be the honest test here).')
    from swift.dev.config import TrainConfig

    data = write_jsonl(tmp_path / 'qgalore_sft.jsonl', _tiny_sft_rows())
    out_dir = str(tmp_path / 'out_qgalore')
    with pytest.raises(ImportError, match='q_galore_torch'):
        _run_causal_lm(
            text_model_path, data, out_dir,
            TrainConfig(
                optim='adamw_torch', use_galore=True, galore_quantization=True, galore_rank=8, learning_rate=1e-4,
                per_device_train_batch_size=2, gradient_accumulation_steps=1, max_steps=3))
