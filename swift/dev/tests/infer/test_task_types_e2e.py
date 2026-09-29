# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end task-type coverage for ``run_infer``'s forward-pass paths (real TinyModel + GPU).

The fast tier drives the generative pipeline through ``sampler='no'`` (cache-only, no weights).
This tier loads a real 4-layer ``TinyModel`` and runs the OTHER ``run_infer`` branch: the pooling /
forward path that ``task_type in {seq_cls, embedding, reranker}`` takes through ``_run_pooling`` ->
``_forward_via_transformers`` (a plain HF ``forward_only``). It asserts the wiring and the shape of the
one-value-per-row output, not numeric quality -- the weights are random.

These three task types are exactly the ones a regression in the inference template mode breaks: a
train-mode ``_embedding_encode`` / ``_reranker_encode`` demands ``positive_messages`` /
``negative_messages`` that an inference row (anchor only) does not carry, so ``_forward_via_transformers``
must build the template in an inference mode. ``test_embedding_*`` / ``test_reranker_*`` below are the
guard that keeps it there.

All tests are ``@pytest.mark.slow`` + ``@pytest.mark.accel(1)`` (real weights + one GPU); run with
``-m slow``. ``DistributedConfig()`` mirrors ``feature/test_recipes.py``'s seq_cls / embedding rows.
"""
import json

import pytest

from swift.dev.tests.tiny import TinyData, TinyModel

pytestmark = [pytest.mark.slow, pytest.mark.accel(1)]


def _run_pooling(model_dir, data_path, task_type, out_path=None, **model_kwargs):
    """Drive ``run_infer``'s forward-pass path on the transformers backend and return the rows."""
    from swift.dev.config import (DatasetConfig, DistributedConfig, GenerationConfig, InferConfig, ModelConfig,
                                  TemplateConfig)
    from swift.dev.recipe.run_infer import run_infer

    return run_infer(
        ModelConfig(
            model=model_dir,
            model_type=TinyModel.MODEL_TYPE,
            task_type=task_type,
            torch_dtype='bfloat16',
            **model_kwargs),
        TemplateConfig(template=TinyModel.TEMPLATE, max_length=128),
        DatasetConfig(dataset=[data_path]),
        InferConfig(),
        GenerationConfig(),
        backend='transformers',
        distributed_config=DistributedConfig(),
        split_dataset_ratio=0.0,
        output_path=out_path)


# ---------------------------------------------------------------------
# seq_cls: one row of class logits per input, label carried through
# ---------------------------------------------------------------------


def test_seq_cls_forward_returns_per_class_logits(tmp_path):
    """seq_cls forwards to a num_labels-wide head: each row gets ``num_labels`` logits, label passed through."""
    model_dir = TinyModel.build(tmp_path / 'model')
    data = TinyData.seq_cls(tmp_path / 'seq_cls.jsonl', n=3, num_labels=4)

    rows = _run_pooling(model_dir, data, 'seq_cls', num_labels=4)

    assert len(rows) == 3
    for row in rows:
        logits = row['response']
        assert isinstance(logits, list) and len(logits) == 4, f'expected 4 class logits, got {logits}'
        assert all(isinstance(v, float) for v in logits), f'logits are not plain floats: {logits}'
        assert row['responses'] == [logits]  # forward paths wrap the single value in a list
        assert row['labels'] == row['label']  # the row's own label column, not a stripped assistant turn
        assert row['messages'], 'messages passthrough is empty'


# ---------------------------------------------------------------------
# embedding: one pooled vector per input (guards the inference template mode)
# ---------------------------------------------------------------------


def test_embedding_forward_returns_a_plain_vector(tmp_path):
    """embedding forwards to a pooled vector per row -- a plain list of floats, jsonl-serialisable.

    Two regressions live here. (1) A train-mode template would raise IndexError on the missing
    ``positive_messages``; the inference mode encodes the anchor only. (2) ``forward_only(task='embedding')``
    returns the vectors under the ``'embeddings'`` (plural) key, which ``_per_row_outputs`` must recognise
    -- otherwise the row value is the raw output dict (tensors and all) rather than a vector.
    """
    model_dir = TinyModel.build(tmp_path / 'model')
    data = TinyData.embedding(tmp_path / 'emb.jsonl', n=3)
    out = str(tmp_path / 'emb_out.jsonl')

    rows = _run_pooling(model_dir, data, 'embedding', out_path=out)

    assert len(rows) == 3
    hidden = None
    for row in rows:
        vector = row['response']
        assert isinstance(vector, list) and vector, f'embedding row is not a non-empty list: {type(vector)}'
        assert all(isinstance(v, float) for v in vector), f'embedding is not plain floats: {vector[:4]}'
        hidden = hidden or len(vector)
        assert len(vector) == hidden, 'embedding width differs across rows'
    assert hidden == TinyModel.DIMS['hidden_size'], f'pooled width {hidden} != hidden_size'

    # the whole point of the plain-list conversion: the file round-trips through json
    with open(out) as f:
        written = [json.loads(line) for line in f if line.strip()]
    assert len(written) == 3
    assert isinstance(written[0]['response'], list) and len(written[0]['response']) == hidden


# ---------------------------------------------------------------------
# reranker: one sigmoid relevance in [0, 1] per input
# ---------------------------------------------------------------------


def test_reranker_forward_returns_bounded_relevance(tmp_path):
    """reranker (num_labels=1 cross-encoder) forwards to one relevance, sigmoid-ed into [0, 1].

    ``reranker_use_activation`` defaults True, so ``_forward_via_transformers`` maps the raw score through
    ``_sigmoid`` -- the same [0, 1] convention the vLLM/SGLang pooling paths use. A train-mode template
    would instead iterate the (empty) positive/negative pairs and crash, so this also guards the mode.
    """
    model_dir = TinyModel.build(tmp_path / 'model')
    data = TinyData.embedding(tmp_path / 'rerank.jsonl', n=3)

    rows = _run_pooling(model_dir, data, 'reranker')

    assert len(rows) == 3
    for row in rows:
        value = row['response']
        scalar = value[0] if isinstance(value, (list, tuple)) else value
        assert isinstance(scalar, float), f'reranker score is not a float: {value!r}'
        assert 0.0 <= scalar <= 1.0, f'reranker relevance not sigmoid-bounded to [0,1]: {scalar}'
