# Copyright (c) ModelScope Contributors. All rights reserved.
"""H-UT3 ③ — a seeded streaming generation leaves the caller's training RNG untouched
(twinkle 45cefb4e, runs/H07/g47e_stream_seed_restore, G47-e).

Contract: ``SamplingParams.seed`` reseeds torch for the generate call only (process-global
``torch.manual_seed`` inside ``to_transformers``). ``generate_stream`` snapshots CPU + CUDA
RNG before building the gen kwargs and restores both in the finally AFTER the worker join,
so a seeded stream stays reproducible without clobbering the training process's randomness
(dropout / data order share this process).

Oracle (independent): the RNG state bytes themselves -- ``torch.get_rng_state()`` /
``torch.cuda.get_rng_state_all()`` captured before the stream must be byte-identical after
it, while two seeded streams still produce identical text (the seed DID reach the sampler).
A control draw of ``torch.rand`` after the stream must equal the pre-computed next draw from
the untouched state.

Reverse (single point, in-process): monkeypatch ``torch.set_rng_state`` and
``torch.cuda.set_rng_state_all`` to no-ops for the duration of the stream call -- this
disables exactly the fix's restore seam. The post-stream state then differs (the reseed
leaks) -> red.

Placement: tests/v5/feature/rl/test_stream_seed_restore.py (uses ``tiny_qwen2_5``).
Markers: slow + accel(1). Estimated runtime: ~45-75 s.
"""
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.accel(1)]


def _engine(model_path):
    import torch
    from twinkle.sampler.transformers_sampler.transformers_engine import TransformersEngine

    return TransformersEngine(model_id=model_path, dtype=torch.float32)


def _prompt_ids(model_path):
    from swift.dev.builder import build_template
    from swift.dev.config import TemplateConfig
    from swift.model import get_model_processor

    _, proc = get_model_processor(model_path, model_type='qwen2', load_model=False)
    template = build_template(TemplateConfig(template='qwen2_5', max_length=256), proc)
    return list(template.encode({'messages': [{'role': 'user', 'content': 'Count to three.'}]})['input_ids'])


def _drain_stream(engine, prompt_ids, seed):
    import asyncio

    from twinkle.data_format.sampling import SamplingParams

    async def _run():
        parts = []
        async for delta, reason in engine.generate_stream(
                prompt_ids, SamplingParams(max_tokens=16, temperature=0.8, seed=seed)):
            parts.append(delta)
        return ''.join(parts)

    return asyncio.run(_run())


def _rng_snapshot():
    import torch

    cpu = torch.get_rng_state().clone()
    cuda = [s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None
    return cpu, cuda


def test_seeded_stream_restores_cpu_and_cuda_rng(tiny_qwen2_5):
    import asyncio

    import torch

    engine = _engine(tiny_qwen2_5)
    try:
        prompt_ids = _prompt_ids(tiny_qwen2_5)
        torch.manual_seed(1234)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1234)
        # Advance the streams a little so a leaked reseed is observable in the next draw.
        torch.rand(8)
        before_cpu, before_cuda = _rng_snapshot()
        expected_next = torch.rand(4)  # what the UNTOUCHED stream must produce next
        # Rewind to the captured state (the draw above consumed it).
        torch.set_rng_state(before_cpu)
        if before_cuda is not None:
            torch.cuda.set_rng_state_all(before_cuda)

        text_a = _drain_stream(engine, prompt_ids, seed=42)
        after_cpu, after_cuda = _rng_snapshot()
        assert torch.equal(after_cpu, before_cpu), 'seeded stream clobbered the CPU training RNG'
        if before_cuda is not None:
            assert all(torch.equal(a, b) for a, b in zip(after_cuda, before_cuda)), \
                'seeded stream clobbered the CUDA training RNG'
        assert torch.equal(torch.rand(4), expected_next), 'post-stream random draw diverged'
        assert text_a, 'seeded stream produced no text'

        # The seed still reaches the sampler: a second seeded stream is reproducible.
        text_b = _drain_stream(engine, prompt_ids, seed=42)
        assert text_b == text_a, 'seeded stream is not reproducible (seed never reached generate)'
    finally:
        asyncio.run(engine.shutdown())


def test_reverse_restore_disabled_leaks_the_reseed(tiny_qwen2_5, monkeypatch):
    """Single-point reverse: no-op the restore calls (the fix's seam). The seeded stream's
    ``manual_seed(42)`` then leaks into the caller's RNG -> the state contract goes red."""
    import asyncio

    import torch

    engine = _engine(tiny_qwen2_5)
    try:
        prompt_ids = _prompt_ids(tiny_qwen2_5)
        torch.manual_seed(1234)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1234)
        torch.rand(8)
        before_cpu, before_cuda = _rng_snapshot()

        monkeypatch.setattr(torch, 'set_rng_state', lambda *a, **k: None)
        if torch.cuda.is_available():
            monkeypatch.setattr(torch.cuda, 'set_rng_state_all', lambda *a, **k: None)
        _drain_stream(engine, prompt_ids, seed=42)

        after_cpu, _ = _rng_snapshot()
        assert not torch.equal(after_cpu, before_cpu), \
            'reverse control failed: with the restore disabled the reseed must leak into the CPU RNG'
    finally:
        asyncio.run(engine.shutdown())