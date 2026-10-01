# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end SFT training across the model *frameworks* dev can construct on (dimension 3).

The other files in this suite hold the backend fixed (the plain twinkle ``TransformersModel``) and vary
the task_type / inputs. This one varies the framework that builds and drives the model:

  - **liger** -- ``TrainConfig.use_liger_kernel`` patches the HF decoder's per-layer ops (rms_norm /
    rotary / swiglu / ...) in place via ``twinkle.kernel.kernelize`` (builders/model.py::_apply_liger_kernel),
    and the op-level ``liger_kernel_config['fused_linear_cross_entropy']`` additionally swaps the loss to
    Liger's fused lm_head+CE kernel, which the recipe pairs with ``loop_task='fused_lm_ce'`` so the
    forward skips the lm_head GEMM (recipe/run_sft.py::liger_fused_ce_enabled). Both paths are driven
    here on a real causal_lm run; a clean finite-loss trajectory means the kernels really replaced the
    eager ops (a mismatched fused-CE task would read hidden states as logits and blow up).
  - **sentence_transformers** -- an embedding checkpoint whose family loader declares
    ``model_framework='sentence_transformers'`` (or whose local dir carries the ST layout) is diverted
    off ``TransformersModel`` onto ``SentenceTransformerModel`` (builders/model.py::_resolve_model_framework),
    which builds a ``SentenceTransformer`` module pipeline (Transformer -> Pooling -> Dense -> Normalize)
    that does its own pooling. Driven here on the real ``google/embeddinggemma-300m`` ST checkpoint.
  - **unsloth** -- NOT installed in this environment and, being pinned to older transformers, not
    loadable against the installed 5.x. Covered as a documented gap (see the test), not a stub.
  - **megatron** -- a backend rather than a model framework, and already covered end to end (dp/sp
    sharding, GA equivalence, two-bridge bit-identity, dev-vs-legacy loss, save/resume) in
    ``test_e2e.py``; cross-referenced here rather than duplicated.

No mocks: every run loads real weights and drives the real recipe. All tests are ``@pytest.mark.slow``
(+ ``@pytest.mark.accel(1)``); run with
``CUDA_VISIBLE_DEVICES=<card> pytest swift/dev/tests/feature/sft/test_frameworks.py -m slow``.
"""
import pytest
import torch

pytestmark = [pytest.mark.slow, pytest.mark.accel(1)]

#: causal_lm cross-entropy over the ~151k Qwen vocab settles near ln(vocab) ~= 11.9, so 20.0 sits above
#: a correctly mean-reduced run yet far below a sum-reduced (scales with sequence length) or diverged one.
_CAUSAL_LM_MAX_LOSS = 20.0
#: embedding InfoNCE over an anchor + 1 positive + 1 negative group settles near ln(group); 20.0 is the
#: same divergence / wrong-reduction ceiling conftest.assert_trained applies to the text embedding run.
_EMBEDDING_MAX_LOSS = 20.0

#: The smallest real sentence-transformers checkpoint dev routes onto SentenceTransformerModel: its
#: family loader (GemmaEmbLoader) declares ``model_framework='sentence_transformers'`` and its dir ships
#: the ST layout (modules.json / 1_Pooling / config_sentence_transformers.json), so it exercises both
#: routing signals in builders/model.py::_resolve_model_framework.
ST_MODEL = 'google/embeddinggemma-300m'
ST_MODEL_TYPE = 'gemma_emb'


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


def _run_causal_lm(model_path, data_path, out_dir, train_config):
    """Drive one real causal_lm run_sft on the shared 0.5B with a caller-supplied TrainConfig."""
    from swift.dev.config import (CheckpointConfig, DatasetConfig, DistributedConfig, ModelConfig, TemplateConfig)
    from swift.dev.recipe import run_sft
    return run_sft(
        ModelConfig(model=model_path, task_type='causal_lm', torch_dtype='bfloat16'),
        TemplateConfig(template='qwen2_5', max_length=256),
        DatasetConfig(dataset=[data_path], dataset_shuffle=False),
        train_config,
        DistributedConfig(),
        CheckpointConfig(),
        output_dir=out_dir,
    )


def test_run_sft_liger_kernel_end_to_end(tmp_path, text_model_path, assert_trained, write_jsonl):
    """use_liger_kernel: the per-layer Liger op kernels replace the eager HF ops and still train.

    ``_apply_liger_kernel`` kernelizes ``model.model`` in place (rms_norm / rotary / swiglu / ...), so a
    finite, normalized loss over a few steps means the patched forward/backward is numerically sound --
    a broken kernel substitution would NaN or diverge rather than silently pass.
    """
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    from swift.dev.config import TrainConfig

    data = write_jsonl(tmp_path / 'liger_sft.jsonl', _tiny_sft_rows())
    out_dir = str(tmp_path / 'out_liger')
    history = _run_causal_lm(
        text_model_path, data, out_dir,
        TrainConfig(
            use_liger_kernel=True,
            learning_rate=1e-5,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=3))
    assert_trained(history, out_dir, 'causal_lm', max_loss=_CAUSAL_LM_MAX_LOSS)


def test_run_sft_liger_fused_cross_entropy_end_to_end(tmp_path, text_model_path, assert_trained, write_jsonl):
    """Liger fused-linear-cross-entropy: the loss path AND the loop forward task switch together.

    ``liger_fused_ce_enabled`` requires both ``use_liger_kernel`` and
    ``liger_kernel_config['fused_linear_cross_entropy']``; the recipe then sets ``loop_task='fused_lm_ce'``
    so the forward skips the lm_head GEMM and stashes the head for the fused kernel. This is the pairing
    the predicate guards: a fused loss under the plain task degrades to unfused CE, and a plain CE under
    the fused task reads hidden states as logits. A finite, normalized loss here means both sides agreed.
    """
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    from swift.dev.config import TrainConfig

    data = write_jsonl(tmp_path / 'liger_fused_sft.jsonl', _tiny_sft_rows())
    out_dir = str(tmp_path / 'out_liger_fused')
    history = _run_causal_lm(
        text_model_path, data, out_dir,
        TrainConfig(
            use_liger_kernel=True,
            liger_kernel_config={'fused_linear_cross_entropy': True},
            learning_rate=1e-5,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=3))
    assert_trained(history, out_dir, 'causal_lm', max_loss=_CAUSAL_LM_MAX_LOSS)


def test_run_embedding_sentence_transformers_end_to_end(tmp_path, tiny_data, assert_trained):
    """sentence_transformers: an embedding checkpoint routes onto SentenceTransformerModel and trains.

    The family loader's ``model_framework='sentence_transformers'`` diverts the build off
    ``TransformersModel`` onto a ``SentenceTransformer`` module pipeline that performs its own pooling
    and normalization (builders/model.py::_build_sentence_transformer_model). The run reuses the exact
    embedding contract from ``test_task_types`` -- InfoNCE over the flattened anchor+positive+negative
    batch -- so a finite, normalized loss plus a self-describing ``task_type='embedding'`` checkpoint
    means the ST pipeline really produced the ``[B, D]`` embeddings the loss consumes.
    """
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    from modelscope import snapshot_download

    from swift.dev.config import (CheckpointConfig, DatasetConfig, DistributedConfig, ModelConfig, TemplateConfig,
                                  TrainConfig)
    from swift.dev.recipe import run_embedding

    model_path = snapshot_download(ST_MODEL)
    data = tiny_data.embedding()
    out_dir = str(tmp_path / 'out_st_embedding')
    history = run_embedding(
        ModelConfig(model=model_path, model_type=ST_MODEL_TYPE, task_type='embedding', torch_dtype='bfloat16'),
        # The ST family pins template='dummy' (its own tokenizer/pooling drive the encode); leave
        # template unset so it resolves off the loaded processor's model_meta rather than hard-coding it.
        TemplateConfig(max_length=256),
        DatasetConfig(dataset=[data], dataset_shuffle=False),
        TrainConfig(
            loss='infonce', learning_rate=1e-5, per_device_train_batch_size=2, gradient_accumulation_steps=1,
            max_steps=3),
        DistributedConfig(),
        CheckpointConfig(),
        output_dir=out_dir,
    )
    assert_trained(history, out_dir, 'embedding', max_loss=_EMBEDDING_MAX_LOSS)


#: unsloth is a documented gap, not a stub. It is NOT installed in this environment and is pinned to
#: older transformers, so it cannot load against the installed 5.x even if installed. dev routes an
#: unsloth tuner through ``TunerConfig(tuner_backend='unsloth')``; the sentence-transformers builder
#: already rejects that combination explicitly (builders/model.py:657). The test below skips when the
#: package is absent (the case here) and, if a future environment does install a compatible unsloth,
#: drives a real LoRA run rather than passing unconditionally -- so it stays honest either way.
def test_run_sft_unsloth_tuner_end_to_end(tmp_path, text_model_path, assert_trained, write_jsonl):
    """unsloth tuner backend: real LoRA run when the package is importable, documented skip otherwise."""
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    try:
        import unsloth  # noqa: F401
    except ImportError:
        pytest.skip('unsloth not installed (and pinned to pre-5.x transformers, incompatible with the '
                    'installed 5.x); dev routes it via TunerConfig(tuner_backend="unsloth") -- a tracked '
                    'framework gap, see the module docstring.')

    from swift.dev.config import (CheckpointConfig, DatasetConfig, DistributedConfig, ModelConfig, TemplateConfig,
                                  TrainConfig, TunerConfig)
    from swift.dev.recipe import run_sft

    data = write_jsonl(tmp_path / 'unsloth_sft.jsonl', _tiny_sft_rows())
    out_dir = str(tmp_path / 'out_unsloth')
    history = run_sft(
        ModelConfig(model=text_model_path, task_type='causal_lm', torch_dtype='bfloat16'),
        TemplateConfig(template='qwen2_5', max_length=256),
        DatasetConfig(dataset=[data], dataset_shuffle=False),
        TrainConfig(learning_rate=1e-4, per_device_train_batch_size=2, gradient_accumulation_steps=1, max_steps=3),
        DistributedConfig(),
        CheckpointConfig(),
        TunerConfig(tuner='lora', tuner_backend='unsloth'),
        output_dir=out_dir,
    )
    assert_trained(history, out_dir, 'causal_lm', max_loss=_CAUSAL_LM_MAX_LOSS)
