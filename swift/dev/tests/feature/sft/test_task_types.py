"""End-to-end SFT training for the non-causal-lm task_types.

``test_e2e.py`` covers ``causal_lm`` (transformers + Megatron, dp/sp sharding, GA, resume). This file
covers the other task_types the dev SFT CLI dispatches to -- seq_cls / embedding / reranker /
generative_reranker -- each driven through its real recipe (``run_seq_cls`` / ``run_embedding`` /
``run_reranker``) on real weights and a tiny local dataset in the documented row format, asserting the
same contract ``test_e2e`` does: optimizer steps ran, loss is finite and normalized, and a
self-describing checkpoint (weights + args.json carrying ``task_type``) is written.

One cached 0.5B serves all four: a SequenceClassification head is built for seq_cls / reranker,
embedding pools the last token off the causal LM, and generative_reranker scores off the vocab head
(see conftest.MODEL_TEXT). No mocks: every test loads real weights and runs the full assembly. Marked
slow; run on a free GPU with
``CUDA_VISIBLE_DEVICES=<card> pytest swift/dev/tests/feature/sft/test_task_types.py -m slow``.
"""
import os

import pytest
import torch

#: Per-task ceiling for assert_trained's divergence / wrong-reduction guard, applied to the converged
#: (final) loss. Each sits above the scale a correctly mean-reduced loss settles at after a few steps
#: (seq_cls CE ~ ln(num_labels), embedding InfoNCE ~ ln(group), reranker/generative BCE ~ ln(2)) yet
#: far below what a sum-reduced (scales with sequence count) or diverging run produces, so a wrong
#: reduction or a blown-up run trips it while a fresh head's large init transient does not.
_MAX_LOSS = {
    'seq_cls': 20.0,
    'embedding': 20.0,
    'reranker': 20.0,
    'generative_reranker': 20.0,
}


def _safetensor_keys(ckpt_dir):
    """Every tensor key across the ``*.safetensors`` shards in a checkpoint directory."""
    from safetensors import safe_open

    keys = []
    for name in sorted(os.listdir(ckpt_dir)):
        if not name.endswith('.safetensors'):
            continue
        with safe_open(os.path.join(ckpt_dir, name), framework='pt') as f:
            keys.extend(f.keys())
    return keys


def _read_safetensor(ckpt_dir, key):
    """Read one tensor by key from whichever ``*.safetensors`` shard in the directory holds it."""
    from safetensors import safe_open

    for name in sorted(os.listdir(ckpt_dir)):
        if not name.endswith('.safetensors'):
            continue
        with safe_open(os.path.join(ckpt_dir, name), framework='pt') as f:
            if key in f.keys():
                return f.get_tensor(key)
    raise KeyError(f'{key} not found in {ckpt_dir}')


def _train(recipe, task_type, assert_trained, *, model_path, data_path, out_dir, model_kwargs=None, loss=None):
    """Shared driver: run a task_type's real recipe on a tiny dataset and assert the trained contract.

    Every recipe takes the same six positional Configs; only ModelConfig's head fields (num_labels /
    problem_type) and the loss name differ per task_type, so those are the only things callers vary.
    lr=1e-5 is a stable full-parameter rate on 8 samples (a LoRA-scale 1e-4 diverges under full FT).
    """
    from swift.dev.config import (CheckpointConfig, DatasetConfig, DistributedConfig, ModelConfig, TemplateConfig,
                                  TrainConfig)

    train_kwargs = {
        'learning_rate': 1e-5,
        'per_device_train_batch_size': 2,
        'gradient_accumulation_steps': 1,
        'max_steps': 3,
    }
    if loss is not None:
        train_kwargs['loss'] = loss

    history = recipe(
        ModelConfig(model=model_path, task_type=task_type, torch_dtype='bfloat16', **(model_kwargs or {})),
        TemplateConfig(template='qwen2_5', max_length=256),
        DatasetConfig(dataset=[data_path], dataset_shuffle=False),
        TrainConfig(**train_kwargs),
        DistributedConfig(),
        CheckpointConfig(),
        output_dir=out_dir,
    )
    assert_trained(history, out_dir, task_type, max_loss=_MAX_LOSS[task_type])
    return history


@pytest.mark.slow
def test_run_seq_cls_single_label_end_to_end(tmp_path, text_model_path, tiny_data, assert_trained):
    """seq_cls / single_label_classification: a num_labels-wide CE head trains to a usable checkpoint."""
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    from swift.dev.recipe import run_seq_cls
    _train(
        run_seq_cls,
        'seq_cls',
        assert_trained,
        model_path=text_model_path,
        data_path=tiny_data.seq_cls('single_label', 'single'),
        out_dir=str(tmp_path / 'out_seq_cls_single'),
        model_kwargs={'num_labels': 2, 'problem_type': 'single_label_classification'})


@pytest.mark.slow
def test_run_seq_cls_regression_end_to_end(tmp_path, text_model_path, tiny_data, assert_trained):
    """seq_cls / regression: a num_labels=1 head fits a scalar target with the MSE objective."""
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    from swift.dev.recipe import run_seq_cls
    _train(
        run_seq_cls,
        'seq_cls',
        assert_trained,
        model_path=text_model_path,
        data_path=tiny_data.seq_cls('regression', 'regression'),
        out_dir=str(tmp_path / 'out_seq_cls_regression'),
        model_kwargs={'num_labels': 1, 'problem_type': 'regression'})


@pytest.mark.slow
def test_run_embedding_end_to_end(tmp_path, text_model_path, tiny_data, assert_trained):
    """embedding / InfoNCE: last-token pooling over anchor+positive+negative trains to a checkpoint."""
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    from swift.dev.recipe import run_embedding
    _train(
        run_embedding,
        'embedding',
        assert_trained,
        model_path=text_model_path,
        data_path=tiny_data.embedding(),
        out_dir=str(tmp_path / 'out_embedding'),
        loss='infonce')


@pytest.mark.slow
def test_run_reranker_end_to_end(tmp_path, text_model_path, tiny_data, assert_trained):
    """reranker / pointwise: a num_labels=1 cross-encoder head scores query-doc pairs to a checkpoint."""
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    from swift.dev.recipe import run_reranker
    _train(
        run_reranker,
        'reranker',
        assert_trained,
        model_path=text_model_path,
        data_path=tiny_data.reranker(),
        out_dir=str(tmp_path / 'out_reranker'),
        loss='pointwise_reranker')


@pytest.mark.slow
def test_run_generative_reranker_end_to_end(tmp_path, text_model_path, tiny_data, assert_trained):
    """generative_reranker: the CausalLM stays intact and scores off logit('yes')-logit('no')."""
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    from swift.dev.recipe import run_reranker
    _train(
        run_reranker,
        'generative_reranker',
        assert_trained,
        model_path=text_model_path,
        data_path=tiny_data.reranker(),
        out_dir=str(tmp_path / 'out_generative_reranker'),
        loss='pointwise_reranker')


@pytest.mark.slow
def test_run_seq_cls_lora_saves_classification_head(tmp_path, text_model_path, tiny_data, assert_trained):
    """LoRA + seq_cls: the fresh classification head must ride in modules_to_save so it TRAINS and is
    SAVED.

    LoRA wraps only target_modules; the seq_cls head is a new module built on top of the base model, so
    without it in modules_to_save peft neither trains nor persists it and the checkpoint reloads with a
    randomly-initialized head (predictions meaningless). dev cannot probe the head's real name from the
    built model under Ray, so it lists both HF candidates ('score'/'classifier') and peft keeps the one
    this family has. The decisive post-condition: the checkpoint carries the head as a full tensor
    (``...score.weight``), which only happens when the head is trainable -- empirically, disabling the
    head's modules_to_save treatment drops this key entirely and leaves only the lora_A/lora_B shards.
    """
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    from swift.dev.config import (CheckpointConfig, DatasetConfig, DistributedConfig, ModelConfig, TemplateConfig,
                                  TrainConfig, TunerConfig)
    from swift.dev.recipe import run_seq_cls

    out_dir = str(tmp_path / 'out_seq_cls_lora')
    history = run_seq_cls(
        ModelConfig(
            model=text_model_path,
            task_type='seq_cls',
            torch_dtype='bfloat16',
            num_labels=2,
            problem_type='single_label_classification'),
        TemplateConfig(template='qwen2_5', max_length=256),
        DatasetConfig(dataset=[tiny_data.seq_cls('single_label', 'single')], dataset_shuffle=False),
        TrainConfig(learning_rate=1e-4, per_device_train_batch_size=2, gradient_accumulation_steps=1, max_steps=3),
        DistributedConfig(),
        CheckpointConfig(),
        TunerConfig(tuner='lora'),
        output_dir=out_dir,
    )
    assert_trained(history, out_dir, 'seq_cls', max_loss=_MAX_LOSS['seq_cls'])

    ckpt = os.path.join(out_dir, 'checkpoint-final')
    keys = _safetensor_keys(ckpt)
    # endswith('score.weight') matches the full head tensor but not any lora_A/lora_B shard.
    head_saved = [k for k in keys if k.endswith(('score.weight', 'classifier.weight'))]
    assert head_saved, f'LoRA seq_cls checkpoint saved no trainable classification head; keys={sorted(keys)}'
    head = _read_safetensor(ckpt, head_saved[0])
    assert torch.isfinite(head).all() and head.abs().sum() > 0, f'saved head is degenerate: {head_saved[0]}'


