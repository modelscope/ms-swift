# Copyright (c) ModelScope Contributors. All rights reserved.
"""Fast tier, end-to-end: the REAL ``run_eval`` chain on CPU, no GPU and no download.

These drive ``swift.dev.recipe.run_eval`` for real -- the twinkle ``Evaluator``, EvalScope's Native
``run_task``, ``SamplerModelAPI`` and ``Summarizer`` all execute -- over a hermetic local ``general_qa``
benchmark, replacing only the three model-construction calls (``build_sampler`` / ``load_model_processor`` /
``build_template``) with a scripted sampler. That is the seam the retired stub test faked away: here a break
anywhere in config -> Evaluator -> adapter -> evalscope -> summarize -> result_jsonl fails the test.

The scored answers match the references verbatim, so BLEU/Rouge yield ``score == 1.0`` over ``num == 3`` rows
deterministically; the assertions check the report's fields and side effects, not merely that a call returned.
"""
from swift.dev.config import EvalConfig, ModelConfig, TemplateConfig
from swift.dev.recipe import run_eval

from .conftest import make_scripted_sampler, read_jsonl

#: The report row ``result_jsonl`` records -- pinned so a silently dropped field is caught.
REPORT_KEYS = {'Native', 'time', 'model', 'adapters', 'eval_output_dir', 'eval_limit'}


def _drive(monkeypatch, tmp_path, hermetic_qa_dataset, *, continuous, adapters=None):
    """Run the real ``run_eval`` with a scripted sampler; return ``(report, sampler, out_dir, jsonl)``."""
    dataset_name, dataset_args, _qa = hermetic_qa_dataset
    sampler = make_scripted_sampler(continuous=continuous)
    # Only the model-construction seam is replaced; Evaluator / evalscope / summarize all run for real.
    monkeypatch.setattr('swift.dev.builders.build_sampler', lambda *_a, **_k: sampler)
    monkeypatch.setattr('swift.dev.builders.load_model_processor', lambda *_a, **_k: (None, None))
    monkeypatch.setattr('swift.dev.builders.build_template', lambda *_a, **_k: None)
    out_dir = tmp_path / 'out'
    jsonl = tmp_path / 'results.jsonl'
    eval_config = EvalConfig(
        eval_dataset=[dataset_name],
        eval_dataset_args=dataset_args,
        eval_num_proc=2,
        eval_generation_config={'max_tokens': 16, 'temperature': 0.0},
        eval_output_dir=str(out_dir),
        result_jsonl=str(jsonl),
    )
    report = run_eval(
        ModelConfig(model='scripted-model'),
        TemplateConfig(),
        eval_config,
        backend='vllm',
        engine_args={},
        adapters=adapters,
    )
    return report, sampler, out_dir, jsonl


def _native_row(report):
    rows = report['Native']
    assert isinstance(rows, list) and len(rows) == 1, rows
    return rows[0]


def test_continuous_evaluator_runs_end_to_end(monkeypatch, tmp_path, hermetic_qa_dataset):
    report, sampler, out_dir, jsonl = _drive(monkeypatch, tmp_path, hermetic_qa_dataset, continuous=True)

    # The real EvalScope Native runner scored the hermetic benchmark through the real SamplerModelAPI.
    row = _native_row(report)
    assert row['dataset_name'] == 'general_qa'
    assert row['num'] == 3
    assert row['score'] == 1.0

    # Report side effects: the exact shape result_jsonl records, and the on-disk EvalScope work dir.
    assert set(report) == REPORT_KEYS
    assert report['model'] == 'scripted-model'
    assert report['adapters'] == []
    assert report['eval_limit'] is None
    assert out_dir.is_dir() and any(out_dir.rglob('*'))
    recorded = read_jsonl(str(jsonl))
    assert len(recorded) == 1
    assert set(recorded[0]) == REPORT_KEYS
    assert recorded[0]['Native'] == report['Native']

    # A continuous-work engine is driven one trajectory at a time (the adapter bypasses the batcher), which is
    # the whole point of the optimization: no trajectory waits on the longest one in a coalesced batch.
    assert sampler.calls and all(size == 1 for size in sampler.calls), sampler.calls
    assert sampler.shutdown_called is True


def test_non_continuous_evaluator_runs_end_to_end(monkeypatch, tmp_path, hermetic_qa_dataset):
    """A non-continuous engine (transformers) goes through SamplerBatcher and still scores correctly."""
    report, sampler, _out_dir, jsonl = _drive(monkeypatch, tmp_path, hermetic_qa_dataset, continuous=False)
    row = _native_row(report)
    assert row['dataset_name'] == 'general_qa'
    assert row['num'] == 3
    assert row['score'] == 1.0
    assert len(read_jsonl(str(jsonl))) == 1
    assert sampler.calls and sampler.shutdown_called is True


def test_adapters_select_the_first_and_warn_on_extras(monkeypatch, tmp_path, hermetic_qa_dataset, caplog):
    """eval scores one model: ``adapters[0]`` is selected per request as ``adapter_path``, extras warn."""
    import logging
    with caplog.at_level(logging.WARNING):
        report, sampler, _out_dir, _jsonl = _drive(
            monkeypatch, tmp_path, hermetic_qa_dataset, continuous=True, adapters=['/ckpt/first', '/ckpt/second'])
    assert report['adapters'] == ['/ckpt/first', '/ckpt/second']
    # the LoRA the engine is asked to select for every request is adapters[0], threaded as adapter_path
    assert sampler.seen_kwargs and all(kw.get('adapter_path') == '/ckpt/first' for kw in sampler.seen_kwargs)
    assert any('adapters[0]' in record.message for record in caplog.records)
