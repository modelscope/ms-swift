# Copyright (c) ModelScope Contributors. All rights reserved.
"""S16-09 (L2): smoke-run the infer-group v5 examples through ``infer_main``.

The plan row's ``demo_*.py`` thirteen are LEGACY demo scripts (no USE_SWIFT_V5; kept for docs
coordination); the v5 infer surface is the eleven ``infer.sh`` scripts. Coverage split, so no
suite duplicates another:

- transformers/embedding/reranker/reward_model: pooled / scalar-head inference whose forward
  path is ALREADY driven end-to-end by ``swift/dev/tests/infer`` (task-types e2e on real weights
  + the examples batch's own real-run smoke). This file pins the EXAMPLE surface -- the scripts'
  exact argv through the real CLI main -- on the tiny ``examples/infer/data/*.jsonl`` sets, and
  asserts the result files land with one row per input.
- vllm/sglang: engine bring-up is the deploy suite's slow tier (S10) and the sampler e2e suites
  already exercise the vllm engine; sglang is not installed on this box (registered in the plan
  notes, not silently skipped).
- agentic/code/dataset/grounding/sample: each needs live tool loops / vllm / prm models or
  multi-gpu ray; they were real-run smoked during the examples batches and are registered here,
  not re-enacted per test run.
"""
import json
import os
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parents[5] / 'examples'
pytestmark = [pytest.mark.slow, pytest.mark.accel(1)]

MODEL_TEXT = 'Qwen/Qwen2.5-0.5B-Instruct'
MODEL_EMB = 'google/embeddinggemma-300m'
MODEL_RERANK = 'BAAI/bge-reranker-base'


def _run_infer(argv, result_path):
    from swift.dev.cli.infer import infer_main
    infer_main(argv)
    assert os.path.exists(result_path), f'no result file at {result_path}'
    with open(result_path, encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows


def _input_rows(name):
    with open(EXAMPLES / 'infer' / 'data' / name, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def test_transformers_infer_example_smokes(tmp_path):
    """examples/infer/transformers/infer.sh: causal_lm batch inference. The model shrinks to the
    cached 0.5B (the script's 7B is a size choice, not semantics); every other flag verbatim."""
    result = str(tmp_path / 'transformers_infer.jsonl')
    rows = _run_infer([
        '--model', MODEL_TEXT,
        '--sampler', 'transformers',
        '--dataset', str(EXAMPLES / 'infer' / 'data' / 'embedding.jsonl'),
        '--max_batch_size', '8',
        '--max_new_tokens', '16',
        '--temperature', '0.7',
        '--output_format', 'all',
        '--result_path', result,
    ], result)
    inputs = _input_rows('embedding.jsonl')
    assert len(rows) == len(inputs), f'expected {len(inputs)} rows, got {len(rows)}'
    assert all('response' in row or 'messages' in row for row in rows), rows[0].keys()


def test_embedding_infer_example_smokes(tmp_path):
    """examples/infer/embedding/infer.sh: embeddinggemma pooling inference, verbatim model and
    data (the script's own tiny anchor-only set)."""
    result = str(tmp_path / 'embedding_infer.jsonl')
    rows = _run_infer([
        '--model', MODEL_EMB,
        '--task_type', 'embedding',
        '--sampler', 'transformers',
        '--dataset', str(EXAMPLES / 'infer' / 'data' / 'embedding.jsonl'),
        '--max_batch_size', '8',
        '--output_format', 'all',
        '--result_path', result,
    ], result)
    inputs = _input_rows('embedding.jsonl')
    assert len(rows) == len(inputs)
    # Each row carries the pooled embedding as its response (the ST pipeline's contract, read
    # back from a real run: 768 floats for embeddinggemma, one length shared by all rows).
    vectors = [row['response'] for row in rows]
    assert all(isinstance(v, list) and v and isinstance(v[0], float) for v in vectors)
    assert all(len(v) == len(vectors[0]) for v in vectors)


def test_reranker_infer_example_smokes(tmp_path):
    """examples/infer/reranker/infer.sh: bge-reranker scoring, verbatim model and data (the
    script's own query-passage pairs, no system role -- the bge_reranker template refuses one)."""
    result = str(tmp_path / 'reranker_infer.jsonl')
    rows = _run_infer([
        '--model', MODEL_RERANK,
        '--task_type', 'reranker',
        '--sampler', 'transformers',
        '--dataset', str(EXAMPLES / 'infer' / 'data' / 'reranker.jsonl'),
        '--max_batch_size', '8',
        '--output_format', 'all',
        '--result_path', result,
    ], result)
    inputs = _input_rows('reranker.jsonl')
    assert len(rows) == len(inputs)
    # Each row's response is the sigmoid relevance score in [0, 1] (the activation contract the
    # example documents).
    scores = [row['response'] for row in rows]
    assert all(isinstance(s, list) and len(s) == 1 and 0.0 <= s[0] <= 1.0 for s in scores), scores


def test_reward_model_infer_example_smokes(tmp_path):
    """examples/infer/reward_model/infer.sh: the seq_cls num_labels=1 scalar-head form, verbatim
    (the header warns the fresh head's scores are untrained -- shape, not value, is the point)."""
    result = str(tmp_path / 'reward_model_infer.jsonl')
    rows = _run_infer([
        '--model', MODEL_TEXT,
        '--task_type', 'seq_cls',
        '--num_labels', '1',
        '--sampler', 'transformers',
        '--dataset', str(EXAMPLES / 'infer' / 'data' / 'reward_model.jsonl'),
        '--max_batch_size', '8',
        '--output_format', 'all',
        '--result_path', result,
    ], result)
    inputs = _input_rows('reward_model.jsonl')
    assert len(rows) == len(inputs)
