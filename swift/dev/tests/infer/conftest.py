# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared fixtures for the swift infer suite.

Two tiers live here. The fast tier (no GPU, no model download) drives the real generative pipeline
through its cleanest seam: ``infer_backend='no'`` builds no sampler and loads no weights, so every
prompt must be served from ``cache_files``. That exercises sampling -> scoring -> emit -> writer end
to end while staying offline. The slow tier (``@pytest.mark.slow``) builds a real ``TinyModel`` and
runs an actual backend.

Nothing here imports a model at module scope: the helpers build plain ``_Candidate`` / dict objects so
the emit / writer / cache tests stay pure, and the pipeline fixture monkeypatches ``load_prompt_rows``
so no dataset loader (and thus no hub access) is touched.
"""
from typing import Any, Dict, List, Optional

import json
import pytest


def make_candidate(text: str, *, multi_turn: bool = False, messages: Optional[List[Dict[str,
                                                                                   Any]]] = None,
                   rollout_infos: Optional[Dict[str, Any]] = None, truncated: bool = False,
                   response_token_ids: Optional[List[int]] = None,
                   rollout_logprobs: Optional[List[float]] = None) -> Any:
    """Build a real ``run_infer._Candidate`` so emit tests exercise the production dataclass."""
    from swift.dev.recipe.run_infer import _Candidate
    return _Candidate(
        text,
        messages=messages,
        rollout_infos=rollout_infos,
        truncated=truncated,
        multi_turn=multi_turn,
        response_token_ids=list(response_token_ids or []),
        rollout_logprobs=list(rollout_logprobs or []))


def write_cache_file(path: str, prompt_responses: List[tuple]) -> str:
    """Write a ``cache_files`` jsonl: one ``(prompt_messages, [response texts])`` pair per line.

    Each response becomes its own cached candidate row (``messages`` = prompt + one assistant turn),
    which is exactly the shape ``_CandidateCache._load`` keys on (``_prompt_key(messages[:-1])``).
    """
    with open(path, 'w', encoding='utf-8') as f:
        for prompt_messages, responses in prompt_responses:
            for response in responses:
                row = {'messages': list(prompt_messages) + [{'role': 'assistant', 'content': response}]}
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
    return path


class ScriptedSampler:
    """A sampler stub returning canned text, shaped like twinkle's sample() output.

    Mirrors ``feature/grpo/test_rollout.py``'s ``_ScriptedSampler``: ``sample`` takes a list of
    trajectories and returns one response object per (trajectory, sample) with a ``sequences[0]``
    carrying ``tokens`` / ``logprobs`` / ``decoded``. Used by the slow TUI / backend tests that inject
    a sampler instead of building an engine.
    """

    def __init__(self, replies: Optional[List[str]] = None):
        self.replies = replies or ['answer']
        self.calls: List[Any] = []

    def sample(self, trajectories, params=None, *, sampling_params=None, **kwargs):
        from types import SimpleNamespace
        self.calls.append((trajectories, params or sampling_params))
        responses = []
        for index in range(len(trajectories)):
            text = self.replies[index % len(self.replies)]
            sequence = SimpleNamespace(
                tokens=[1, 2, 3], logprobs=[[(1, -0.1), (2, -0.2), (3, -0.3)]], decoded=text, stop_reason='stop')
            responses.append(SimpleNamespace(sequences=[sequence], prompt_token_ids=[9, 9]))
        return responses

    def sample_stream(self, trajectories, params=None, **kwargs):
        yield from self.sample(trajectories, params)

    def shutdown(self):
        pass

    def close(self):
        pass


@pytest.fixture
def patch_prompt_rows(monkeypatch):
    """Monkeypatch ``swift.dev.builders.load_prompt_rows`` to return a caller-supplied batch.

    ``_run_generative`` / ``run_infer`` import ``load_prompt_rows`` from ``swift.dev.builders`` at call
    time, so patching the attribute there is picked up. This keeps the fast pipeline test off the real
    dataset loader (and thus off the hub) while still driving the production code path.
    """

    def _patch(rows: List[Dict[str, Any]]):
        import swift.dev.builders as builders
        monkeypatch.setattr(builders, 'load_prompt_rows', lambda *a, **k: list(rows))
        return rows

    return _patch
