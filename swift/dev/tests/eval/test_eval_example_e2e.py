# Copyright (c) ModelScope Contributors. All rights reserved.
"""Slow tier: the real ``swift eval`` command, mirroring ``examples/v5/eval/eval.sh``.

This spawns ``swift eval`` as a subprocess with the arguments ``eval.sh`` uses -- a real vLLM engine on
``Qwen/Qwen2.5-0.5B-Instruct`` scoring a real EvalScope Native benchmark (``gsm8k``) -- and asserts the
observable contract: the command exits 0, EvalScope writes reports under ``--eval_output_dir``, and the one
row appended to ``--result_jsonl`` carries a ``gsm8k`` score for the model under test. Nothing here is
stubbed, so a green module means ``eval.sh`` runs as written rather than that a fake wrote a file.

The generation/limit magnitudes are trimmed from the example (``eval_limit 5``, ``eval_num_proc 8``,
``max_tokens 256``, ``gpu_memory_utilization 0.5``) so the run stays quick and fits comfortably on one card;
none of these change the code path the example exercises.

All tests are ``@pytest.mark.slow`` + ``@pytest.mark.accel(1)``; run with ``-m slow`` and a free card in
``CUDA_VISIBLE_DEVICES`` (inherited by the subprocess).
"""
import pytest

from swift.dev.tests.eval.conftest import MODEL, read_jsonl, run_eval_cli

pytestmark = [pytest.mark.slow, pytest.mark.accel(1)]


def _native_rows(report_row):
    """The EvalScope report rows recorded for this run (one per benchmark)."""
    rows = report_row.get('Native')
    assert isinstance(rows, list) and rows, report_row
    return rows


def test_eval_example_scores_gsm8k_end_to_end(tmp_path):
    """``swift eval`` on gsm8k: exit 0, an on-disk EvalScope report, and a scored ``result_jsonl`` row."""
    out_dir = tmp_path / 'eval_output'
    jsonl = out_dir / 'results.jsonl'
    proc = run_eval_cli([
        '--model', MODEL,
        '--infer_backend', 'vllm',
        '--eval_dataset', 'gsm8k',
        '--eval_limit', '5',
        '--eval_num_proc', '8',
        '--eval_generation_config', '{"max_tokens": 256, "temperature": 0.0}',
        '--vllm_gpu_memory_utilization', '0.5',
        '--eval_output_dir', str(out_dir),
        '--result_jsonl', str(jsonl),
    ])
    assert proc.returncode == 0, f'swift eval failed:\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}'

    # EvalScope wrote its report tree under the work dir (per-benchmark predictions/reviews/configs).
    assert out_dir.is_dir() and any(out_dir.rglob('*')), f'no EvalScope output under {out_dir}'

    # Exactly one summary row was appended, and it records the model under test plus a real gsm8k score.
    recorded = read_jsonl(str(jsonl))
    assert len(recorded) == 1, recorded
    row = recorded[0]
    # config validation resolves --model to its local cache snapshot, so report['model'] is that path; the
    # checkpoint scored is still the model under test (its hub leaf name is a substring of the resolved path).
    assert MODEL.split('/')[-1] in row['model'], row['model']
    assert row['eval_limit'] == 5
    assert row['adapters'] == []
    assert row['eval_output_dir'] == str(out_dir)

    gsm8k = [r for r in _native_rows(row) if r.get('dataset_name') == 'gsm8k']
    assert len(gsm8k) == 1, _native_rows(row)
    score_row = gsm8k[0]
    # The model was really run over the (limited) benchmark: some samples scored, a numeric score reported.
    assert isinstance(score_row.get('score'), (int, float))
    assert score_row.get('num', 0) > 0
