# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end generative-backend coverage for ``run_infer`` (real TinyModel + GPU).

The fast tier (``test_pipeline_fast.py``) drives the generative pipeline through ``infer_backend='no'``:
candidates come from a cache, so sampling itself is never exercised. This tier loads a real 4-layer
``TinyModel`` and runs an actual backend, so the sampler -> score -> shape -> write path is covered with
real generation rather than canned text. The weights are random, so the assertions are about SHAPE and
WIRING (how many candidates, which columns, does the file round-trip), never about what the model said.

Two backends live here:
  - ``transformers``: the reach backend, an in-process HF generate. ``distributed_config=None`` keeps it
    single-process (a ``DistributedConfig()`` would ask ``build_device_mesh`` for ``nproc_per_node``), and
    ``temperature>0`` is required because HF greedy decoding rejects ``num_return_sequences>1``.
  - ``vllm``: the throughput backend, gated behind ``importorskip`` + a real engine build.

All tests are ``@pytest.mark.slow`` + ``@pytest.mark.accel(1)``; run with ``-m slow``.
"""
import json

import pytest

from swift.dev.tests.tiny import TinyData, TinyModel

pytestmark = [pytest.mark.slow, pytest.mark.accel(1)]


def _read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def _length_reward(completions, **kwargs):
    """A deterministic callable ORM that varies across candidates (longer -> higher score).

    A real reward model needs weights and a GPU-resident seq_cls engine; scoring is orthogonal to the
    backend under test, so a callable stands in. ``compute_rewards_per_func`` calls it as
    ``func(completions_list, **columns)`` and expects one float per completion. Length gives distinct
    values across a random-generation group, so best-of-n ranking is actually exercised.
    """
    return [float(len(str(c))) for c in completions]


def _run_generative(model_dir, data_path, infer_config, *, backend='transformers', reward=None, out_path=None,
                    gen_kwargs=None, engine_args=None):
    """Drive ``run_infer``'s generative path on a real backend and return the rows."""
    from swift.dev.config import (DatasetConfig, GenerationConfig, InferConfig, ModelConfig, RLHFConfig,
                                  TemplateConfig)
    from swift.dev.recipe.run_infer import run_infer

    gen = GenerationConfig(max_new_tokens=8, temperature=0.8, **(gen_kwargs or {}))
    return run_infer(
        ModelConfig(model=model_dir, model_type=TinyModel.MODEL_TYPE, task_type='causal_lm',
                    torch_dtype='bfloat16'),
        TemplateConfig(template=TinyModel.TEMPLATE, max_length=256),
        DatasetConfig(dataset=[data_path]),
        infer_config,
        gen,
        rlhf_config=RLHFConfig(reward_funcs=[reward]) if reward else None,
        backend=backend,
        engine_args=engine_args,
        distributed_config=None,
        split_dataset_ratio=0.0,
        output_path=out_path)


# ---------------------------------------------------------------------
# transformers backend
# ---------------------------------------------------------------------


def test_transformers_all_format_samples_n_candidates(tmp_path):
    """'all' format on the transformers backend: one row per prompt carrying ``num_return_sequences``
    real sampled candidates (``response`` = first, ``responses`` = the whole group), written to jsonl."""
    from swift.dev.config import InferConfig

    model_dir = TinyModel.build(tmp_path / 'model')
    data = TinyData.prompt_only(tmp_path / 'prompts.jsonl', n=2)
    out = str(tmp_path / 'all.jsonl')
    infer_config = InferConfig(num_return_sequences=2, batch_size=2, output_format='all')

    rows = _run_generative(model_dir, data, infer_config, out_path=out)

    assert len(rows) == 2
    for row in rows:
        assert isinstance(row['response'], str) and row['response'] == row['responses'][0]
        assert isinstance(row['responses'], list) and len(row['responses']) == 2
        assert all(isinstance(r, str) for r in row['responses'])
        # the prompt is preserved and the first candidate is appended as an assistant turn
        assert row['messages'][-1] == {'role': 'assistant', 'content': row['response']}
        assert 'scores' not in row  # no reward func -> no scores column

    written = _read_jsonl(out)
    assert len(written) == 2
    assert written[0]['responses'] == rows[0]['responses']  # the incremental writer flushed the group


def test_transformers_all_format_scores_with_callable_reward(tmp_path):
    """A reward func scores every sampled candidate in 'all' format; scoring rides alongside generation."""
    from swift.dev.config import InferConfig

    model_dir = TinyModel.build(tmp_path / 'model')
    data = TinyData.prompt_only(tmp_path / 'prompts.jsonl', n=2)
    infer_config = InferConfig(num_return_sequences=2, batch_size=2, output_format='all')

    rows = _run_generative(model_dir, data, infer_config, reward=_length_reward)

    assert len(rows) == 2
    for row in rows:
        scores = row['scores']
        assert isinstance(scores, list) and len(scores) == 2
        assert all(isinstance(s, float) for s in scores)
        # the reward is exactly len(candidate), so scores must line up with the sampled texts
        assert scores == [float(len(r)) for r in row['responses']]


def test_transformers_dpo_format_pairs_best_against_worst(tmp_path):
    """'dpo' format on a real backend: best-of-n picks a chosen (assistant turn) and a rejected_response,
    written through the checkpointed writer."""
    from swift.dev.config import InferConfig

    model_dir = TinyModel.build(tmp_path / 'model')
    data = TinyData.prompt_only(tmp_path / 'prompts.jsonl', n=2)
    out = str(tmp_path / 'dpo.jsonl')
    infer_config = InferConfig(num_return_sequences=3, n_best_to_keep=1, batch_size=2, output_format='dpo')

    rows = _run_generative(model_dir, data, infer_config, reward=_length_reward, out_path=out)

    assert len(rows) == 2  # one chosen/rejected pair per prompt (n_best_to_keep=1)
    for row in rows:
        assert row['messages'][-1]['role'] == 'assistant'  # the positive is an assistant turn
        assert 'rejected_response' in row and isinstance(row['rejected_response'], str)
        assert row['id']  # each pair carries a group id
    assert rows[0]['id'] != rows[1]['id']

    written = _read_jsonl(out)
    assert len(written) == 2
    assert all('messages' in row and 'rejected_response' in row for row in written)


# ---------------------------------------------------------------------
# vllm backend (gated: a real engine build)
# ---------------------------------------------------------------------


def test_vllm_all_format_samples_n_candidates(tmp_path):
    """The vLLM backend serves the same generative pipeline: ``num_return_sequences`` real candidates per
    prompt through the throughput engine rather than the in-process HF generate."""
    pytest.importorskip('vllm', reason='vLLM backend not installed')
    from swift.dev.config import InferConfig

    model_dir = TinyModel.build(tmp_path / 'model')
    data = TinyData.prompt_only(tmp_path / 'prompts.jsonl', n=2)
    infer_config = InferConfig(num_return_sequences=2, batch_size=2, output_format='all')

    rows = _run_generative(
        model_dir,
        data,
        infer_config,
        backend='vllm',
        # A real engine reserves gpu_memory_utilization * total VRAM up front; on a shared box the
        # default 0.7 can exceed what is free. Cap it, run eager (no CUDA-graph capture), and bound
        # max_model_len so the KV cache stays tiny -- this is a wiring test, not a throughput one.
        engine_args={'gpu_memory_utilization': 0.25, 'enforce_eager': True, 'max_model_len': 512})

    assert len(rows) == 2
    for row in rows:
        assert isinstance(row['responses'], list) and len(row['responses']) == 2
        assert all(isinstance(r, str) for r in row['responses'])
        assert row['response'] == row['responses'][0]
