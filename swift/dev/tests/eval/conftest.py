# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared fixtures for the ``swift eval`` suite.

Two tiers, mirroring ``swift/dev/tests/deploy``:

- **fast** (no GPU, no download): drive ``run_eval``'s real evaluation chain -- the twinkle ``Evaluator``
  and EvalScope's Native ``run_task`` -- against a *scripted* sampler and a *hermetic* local benchmark, so
  the whole seam (config -> Evaluator -> SamplerModelAPI -> evalscope -> summarize -> result_jsonl) runs for
  real on CPU. Only the three model-construction calls (``build_sampler`` / ``load_model_processor`` /
  ``build_template``) are replaced; nothing stubs the Evaluator or EvalScope itself. ``test_config_build``
  and ``test_cli_parse`` cover the pure helpers and every CLI flag.
- **slow** (``@pytest.mark.slow`` + ``@pytest.mark.accel(1)``): spawn the real ``swift eval`` command with
  exactly the arguments ``examples/v5/eval/{eval,eval_lora}.sh`` use -- a real vLLM engine on
  ``Qwen/Qwen2.5-0.5B-Instruct`` scoring a real EvalScope benchmark (and a live-loaded LoRA adapter) -- so a
  green suite means the examples run as written.

The hermetic benchmark is ``general_qa`` pointed at a local ``.jsonl`` (via ``dataset_args``' ``local_path``),
which EvalScope loads from disk with no network. Its rows are Chinese so the BLEU/Rouge scorers take the
``jieba`` path and never reach for NLTK's ``punkt_tab`` download -- keeping the fast tier truly offline.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

#: One real checkpoint for the slow tier; small enough to bring up quickly on a single card. ``model_type`` /
#: ``template`` are never hardcoded: swift auto-resolves both from the model id exactly as ``swift eval`` does,
#: which is the behaviour the examples rely on (and avoids pinning a wrong registry key).
MODEL = os.environ.get('EVAL_TEST_MODEL', 'Qwen/Qwen2.5-0.5B-Instruct')
#: vLLM engine args shared by every slow run: eager mode skips CUDA-graph capture (faster bring-up) and a
#: modest utilisation keeps the engine well inside one card's free memory (0.5, not the 0.9 default, so a
#: previous module's engine that has not fully released the GPU cannot OOM this one).
ENGINE_ARGS: Dict[str, Any] = {
    'gpu_memory_utilization': 0.5,
    'enforce_eager': True,
    'max_model_len': 4096,
}
#: Bringing up vLLM + downloading a benchmark is minutes, not seconds; give the subprocess room.
EVAL_TIMEOUT = float(os.environ.get('EVAL_TEST_TIMEOUT', '1800'))

#: Chinese question/answer rows for the hermetic ``general_qa`` benchmark. The scripted sampler returns the
#: matching answer verbatim, so BLEU/Rouge score 1.0 deterministically regardless of eval concurrency/order.
QA_PAIRS: Tuple[Tuple[str, str], ...] = (
    ('二加二等于几？', '四'),
    ('法国的首都是哪里？', '巴黎'),
    ('天空是什么颜色？', '蓝色'),
)


def build_configs(*, eval_overrides: Optional[Dict[str, Any]] = None, infer_backend: str = 'vllm',
                  model: str = MODEL):
    """The dev Configs ``run_eval`` needs, with ``EvalConfig`` overridden per test.

    Returns ``(model_config, template_config, eval_config, infer_config)``. ``infer_config`` is included so a
    test can drive ``build_engine_args`` off the same parsed backend the recipe would see.
    """
    from swift.dev.config import EvalConfig, InferConfig, ModelConfig, TemplateConfig
    model_config = ModelConfig(model=model, torch_dtype='bfloat16')
    template_config = TemplateConfig()
    eval_config = EvalConfig(**(eval_overrides or {}))
    infer_config = InferConfig(infer_backend=infer_backend)
    return model_config, template_config, eval_config, infer_config


def write_qa_jsonl(path: str, pairs: Sequence[Tuple[str, str]] = QA_PAIRS) -> str:
    """Write a hermetic ``general_qa`` dataset (``{'question','answer'}`` per line) and return its path."""
    with open(path, 'w', encoding='utf-8') as handle:
        for question, answer in pairs:
            handle.write(json.dumps({'question': question, 'answer': answer}, ensure_ascii=False) + '\n')
    return path


@pytest.fixture
def hermetic_qa_dataset(tmp_path):
    """``(dataset_name, dataset_args, qa_pairs)`` for an offline ``general_qa`` run over a local jsonl."""
    jsonl = write_qa_jsonl(str(tmp_path / 'qa.jsonl'))
    dataset_args = {'general_qa': {'local_path': jsonl}}
    return 'general_qa', dataset_args, QA_PAIRS


class ScriptedSampler:
    """A sampler that returns canned answers without touching a model -- the fast tier's model stand-in.

    It answers each trajectory by matching the question embedded in the prompt against ``QA_PAIRS``, so the
    decoded text equals the reference answer and EvalScope's scorers yield 1.0 no matter what order or how
    many trajectories EvalScope submits concurrently. ``sample`` records the size of every call it receives:
    a continuous-work sampler is driven one trajectory at a time (``SamplerModelAPI`` bypasses the batcher),
    while a non-continuous one goes through ``SamplerBatcher``, which may coalesce several into one call.
    """

    #: ``model_id`` / ``template`` / ``device_mesh`` are read by the Evaluator and the batcher.
    model_id = 'scripted'
    template = None
    device_mesh = None

    def __init__(self, *, pairs: Sequence[Tuple[str, str]] = QA_PAIRS):
        self._pairs = list(pairs)
        #: number of trajectories each ``sample`` call received, in call order.
        self.calls: List[int] = []
        #: the per-request kwargs each ``sample`` call received (e.g. ``adapter_path`` for a live LoRA).
        self.seen_kwargs: List[Dict[str, Any]] = []
        self.shutdown_called = False

    def _answer_for(self, trajectory: Any) -> str:
        content = _last_user_content(trajectory)
        for question, answer in self._pairs:
            if question in content:
                return answer
        return self._pairs[0][1]

    def sample(self, trajectories, *, sampling_params=None, **kwargs):
        from twinkle.data_format import SampleResponse, SampledSequence
        self.calls.append(len(trajectories))
        self.seen_kwargs.append(dict(kwargs))
        responses = []
        for trajectory in trajectories:
            answer = self._answer_for(trajectory)
            responses.append(
                SampleResponse(
                    sequences=[SampledSequence(stop_reason='stop', tokens=[1, 2, 3], decoded=answer)],
                    prompt_token_ids=[9, 9]))
        return responses

    def shutdown(self):
        self.shutdown_called = True


def _set_continuous(continuous: bool) -> None:
    """Set/clear the class-level ``_enable_continous_work`` flag ``SamplerModelAPI`` reads.

    The flag lives on the ``sample`` function (twinkle's remote_function sets it there) and
    ``SamplerModelAPI`` reads it from ``type(sampler).sample``, so it is process-global: a non-continuous
    test must clear what a previous continuous test set, hence delete (not set False) to restore the
    ``getattr`` default regardless of run order.
    """
    if continuous:
        ScriptedSampler.sample._enable_continous_work = True
    elif getattr(ScriptedSampler.sample, '_enable_continous_work', False):
        del ScriptedSampler.sample._enable_continous_work


def make_scripted_sampler(*, continuous: bool) -> ScriptedSampler:
    """Build a :class:`ScriptedSampler` with the class-level continuous flag matching ``continuous``."""
    _set_continuous(continuous)
    return ScriptedSampler()


def _last_user_content(trajectory: Any) -> str:
    """The text of the last user message in a twinkle trajectory dict (what ``to_twinkle_trajectory`` builds)."""
    messages = trajectory.get('messages', []) if isinstance(trajectory, dict) else []
    for message in reversed(messages):
        if isinstance(message, dict) and message.get('role') == 'user':
            content = message.get('content')
            return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    return ''


def real_template():
    """The model's real twinkle chat template (tokenizer only, no weights) -- for a test that needs decoding.

    The hermetic fast tier does not need it (the QA scoring path never decodes tokens), so it is only used
    where a real template is wanted; it downloads/caches just the tokenizer.
    """
    from swift.dev.builders import build_template, load_model_processor
    from swift.dev.config import ModelConfig, TemplateConfig
    _, processor = load_model_processor(ModelConfig(model=MODEL), load_model=False)
    return build_template(TemplateConfig(), processor)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Parse a ``.jsonl`` file into a list of dicts (empty list if the file is absent)."""
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_eval_cli(args: Sequence[str], *, timeout: float = EVAL_TIMEOUT,
                 env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    """Run the real ``swift eval`` command in a subprocess, exactly as ``examples/v5/eval`` does.

    ``swift eval`` with ``USE_SWIFT_V5=1`` routes to ``python -m swift.dev.cli.eval``; prefer the installed
    ``swift`` console script (the true example entry point) and fall back to the module form when it is not on
    ``PATH``. Returns the completed process with combined stdout/stderr captured as text.
    """
    run_env = dict(os.environ)
    run_env['USE_SWIFT_V5'] = '1'
    run_env.update(env or {})
    swift_bin = shutil.which('swift')
    if swift_bin:
        cmd = [swift_bin, 'eval', *args]
    else:
        cmd = [sys.executable, '-m', 'swift.dev.cli.eval', *args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=run_env)
