# Copyright (c) ModelScope Contributors. All rights reserved.
"""H-UT3 ⑤ — closing a stream mid-generation leaves the sampler reusable
(twinkle 4b0013f9 + 45cefb4e, runs/H07 MIDCLOSE_ADJUDICATION; the committed
tests/sampler/test_vllm_stream_lifetime.py pins the close-joins-finalisation invariant on
the Sampler base, this case pins the ENGINE-side abort+join and post-close reuse).

Contract: when the consumer abandons ``generate_stream`` early, the engine's finally sets the
cooperative abort event and JOINS the worker thread before releasing ``_generate_lock`` --
``model.generate`` has really stopped, so the next generate / weight sync never runs
concurrently with an abandoned one on the same nn.Module.

Oracle (independent):
* after ``aclose()`` returns, no live ``TransformersEngine-stream`` worker thread exists
  (the join guarantee, observed via ``threading.enumerate``);
* a subsequent greedy ``generate_batch`` on the SAME engine returns tokens identical to a
  plain-HF greedy decode of the same prompt (the engine is reusable and uncorrupted).

Reverse (single point, in-process, test-boundary source injection -- the E04
reverse_autocast.py pattern): re-exec ``generate_stream`` from ``inspect.getsource`` with the
fix's ``aborted.set()`` / ``thread.join()`` lines removed (the pre-4b0013f9 finally) and
rebind it on the class. With a long ``max_new_tokens`` the abandoned worker is still running
when aclose returns -> the thread-liveness assertion goes RED. Shared source is never edited.

Placement: tests/v5/feature/rl/test_stream_midclose_reuse.py (uses ``tiny_qwen2_5``).
Markers: slow + accel(1). Estimated runtime: ~45-90 s.
"""
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.accel(1)]


def _engine(model_path):
    import torch
    from twinkle.sampler.transformers_sampler.transformers_engine import TransformersEngine

    return TransformersEngine(model_id=model_path, dtype=torch.float32)


def _prompt_ids(model_path):
    from swift.dev.builders import build_template
    from swift.dev.config import TemplateConfig
    from swift.model import get_model_processor

    _, proc = get_model_processor(model_path, model_type='qwen2', load_model=False)
    template = build_template(TemplateConfig(template='qwen2_5', max_length=256), proc)
    return list(template.encode({'messages': [{'role': 'user', 'content': 'Say hello.'}]})['input_ids'])


def _stream_workers_alive():
    import threading

    return [t for t in threading.enumerate() if t.name == 'TransformersEngine-stream' and t.is_alive()]


def _hf_greedy(model_path, prompt_ids, max_new_tokens):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_path)
    ids = torch.tensor([prompt_ids], dtype=torch.long)
    with torch.inference_mode():
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    tokens = out[0, ids.shape[-1]:].tolist()
    del model
    return tokens


def _close_mid_stream(engine, prompt_ids, max_new_tokens):
    """Consume one delta, then close the stream early (the abandoned-consumer path).

    EOS stopping is disabled for the call so the generation is guaranteed to still be running
    at close time regardless of the tiny fixture's random weights (a greedy EOS-first checkpoint
    would otherwise finish before the consumer closes -- the scenario under test is an abandoned
    IN-PROGRESS generation). The engine tokenizer is restored right after.
    """
    import asyncio

    from twinkle.data_format.sampling import SamplingParams

    async def _run():
        gen = engine.generate_stream(prompt_ids, SamplingParams(max_tokens=max_new_tokens, temperature=0.0))
        first = None
        async for delta, reason in gen:
            if delta:
                first = delta
                break
        await gen.aclose()
        return first

    eos = engine.tokenizer.eos_token_id
    engine.tokenizer.eos_token_id = None
    try:
        return asyncio.run(_run())
    finally:
        engine.tokenizer.eos_token_id = eos


def test_midclose_joins_worker_and_engine_still_generates(tiny_qwen2_5):
    import asyncio

    from twinkle.data_format.sampling import SamplingParams

    engine = _engine(tiny_qwen2_5)
    try:
        prompt_ids = _prompt_ids(tiny_qwen2_5)
        first = _close_mid_stream(engine, prompt_ids, max_new_tokens=256)
        assert first, 'stream produced no delta before close'
        assert not _stream_workers_alive(), \
            'aborted stream worker outlived aclose() -- generate still runs after lock release'

        # The engine is reusable: greedy batch generate equals the independent plain-HF oracle.
        resp = engine.generate_batch([prompt_ids], SamplingParams(max_tokens=12, temperature=0.0),
                                     logprobs=False)
        got = [int(t) for t in resp[0].sequences[0].tokens]
        oracle = _hf_greedy(tiny_qwen2_5, prompt_ids, 12)
        assert got == oracle, f'post-close generate diverges from plain-HF greedy: {got} vs {oracle}'
        assert not _stream_workers_alive()
    finally:
        asyncio.run(engine.shutdown())


def test_reverse_no_abort_join_leaves_worker_running(tiny_qwen2_5, monkeypatch):
    """Single-point reverse: strip the fix's ``aborted.set()`` + ``thread.join()`` from the
    finally (test-boundary source re-exec). The abandoned worker must then outlive aclose()."""
    import asyncio
    import inspect
    import textwrap

    from twinkle.sampler.transformers_sampler import transformers_engine as mod
    from twinkle.sampler.transformers_sampler.transformers_engine import TransformersEngine

    src = inspect.getsource(TransformersEngine.generate_stream)
    src = textwrap.dedent(src)
    lines = [ln for ln in src.splitlines()
             if ln.strip() not in ('aborted.set()', 'thread.join()')]
    assert len(lines) < len(src.splitlines()), 'reverse injection must remove the abort+join lines'
    namespace = dict(vars(mod))
    exec(compile('\n'.join(lines), '<reverse_generate_stream>', 'exec'), namespace)
    monkeypatch.setattr(TransformersEngine, 'generate_stream', namespace['generate_stream'])

    engine = _engine(tiny_qwen2_5)
    try:
        prompt_ids = _prompt_ids(tiny_qwen2_5)
        _close_mid_stream(engine, prompt_ids, max_new_tokens=512)
        alive = _stream_workers_alive()
        assert alive, 'reverse control failed: without abort+join the worker must still be generating'
        for t in alive:
            t.join(timeout=60)
    finally:
        asyncio.run(engine.shutdown())