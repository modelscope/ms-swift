# Copyright (c) ModelScope Contributors. All rights reserved.
"""H-UT3 ① — HF prompt scoring: a variable-length batch scores each row exactly as an
independent per-row NO-PADDING forward (twinkle 45cefb4e, runs/H07/g41, G41).

Oracle (independent of the code under test): a plain ``AutoModelForCausalLM`` loaded from the
same tiny checkpoint dir, one forward per prompt with NO padding and NO batch neighbours,
fp32, ``log_softmax`` over the real tokens only. The engine's ``prompt_logprobs`` /
``topk_prompt_logprobs`` must match that oracle position-for-position (tiny dense fp32 model:
bit-exact up to fp32 reduction noise, band 1e-6), and a prompt's score must not depend on
which OTHER prompts share the batch (score-alone == score-in-mixed-batch).

Reverse (single point, in-process): rebind ``TransformersEngine._prompt_logprobs`` to the
verbatim pre-a4be8355 batched body (``git show a4be8355^:.../transformers_engine.py``). The
engine LEFT-pads, so the old ``attention_mask[row, 1:]`` mask scores ONE EXTRA position on
every short row (pad->first-real-token "prediction"): lengths go red deterministically on
any model, independent of kernel numerics (G41 old_mask RED signature, len L+1 vs L).

Placement: tests/v5/feature/rl/test_prompt_scoring_per_row.py (uses ``tiny_qwen2_5``).
Markers: slow + accel(1). Estimated runtime: ~60-90 s (one tiny model load + a few forwards).
"""
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.accel(1)]


def _engine_and_template(model_path):
    import torch
    from swift.dev.builder import build_template
    from swift.dev.config import TemplateConfig
    from swift.model import get_model_processor
    from twinkle.sampler.transformers_sampler.transformers_engine import TransformersEngine

    _, proc = get_model_processor(model_path, model_type='qwen2', load_model=False)
    template = build_template(TemplateConfig(template='qwen2_5', max_length=256), proc)
    engine = TransformersEngine(model_id=model_path, dtype=torch.float32)
    return engine, template


def _prompt_ids(template, text):
    feature = template.encode({'messages': [{'role': 'user', 'content': text}]})
    return list(feature['input_ids'])


def _hf_oracle(model_path, prompt_ids):
    """Independent plain-HF per-row no-padding forward: chosen logprob per real position."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    # Device parity with the engine (CUDA fp32 vs CPU fp32 reduction order drifts ~2e-6 nats,
    # above the 1e-6 band): the oracle stays an independent plain-HF per-row no-padding forward.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(input_ids=ids, use_cache=False).logits
    logprobs = torch.log_softmax(logits[0, :-1].float(), dim=-1)
    targets = ids[0, 1:]
    picked = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    values, indices = logprobs.topk(5, dim=-1)
    del model
    return ([None] + picked.tolist(),
            [None] + [list(zip(i, v)) for i, v in zip(indices.tolist(), values.tolist())])


def _assert_matches_oracle(resp, oracle_chosen, oracle_topk, label):
    got = resp.prompt_logprobs
    assert got is not None and len(got) == len(oracle_chosen), \
        f'{label}: scored length {None if got is None else len(got)} != real {len(oracle_chosen)}'
    for pos, (g, o) in enumerate(zip(got, oracle_chosen)):
        if o is None:
            assert g is None, f'{label}: position {pos} must carry the leading None'
        else:
            assert g is not None and abs(g - o) <= 1e-6, f'{label}: pos {pos} logprob {g} vs oracle {o}'
    for pos, (g, o) in enumerate(zip(resp.topk_prompt_logprobs, oracle_topk)):
        if o is None:
            assert g is None
            continue
        assert [tid for tid, _ in g] == [tid for tid, _ in o], f'{label}: pos {pos} top-k ids differ'


def test_variable_length_batch_matches_per_row_no_padding_oracle(tiny_qwen2_5):
    from twinkle.data_format.sampling import SamplingParams

    engine, template = _engine_and_template(tiny_qwen2_5)
    try:
        # Distinct real lengths (the G41 contract: short rows must not inherit the long row's width).
        texts = ['Hi.', 'Count to three please.', 'What is the capital of France? Answer briefly.']
        prompts = [_prompt_ids(template, t) for t in texts]
        assert len({len(p) for p in prompts}) == 3, 'fixture must have three distinct lengths'
        oracles = [_hf_oracle(tiny_qwen2_5, p) for p in prompts]

        params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=5)
        responses = engine.generate_batch(prompts, params, logprobs=False,
                                          include_prompt_logprobs=True, topk_prompt_logprobs=5)
        for i, resp in enumerate(responses):
            _assert_matches_oracle(resp, *oracles[i], label=f'batched row {i}')

        # Batch-composition invariance: the same prompt scored ALONE must be identical.
        for i, p in enumerate(prompts):
            alone = engine.generate_batch([p], params, logprobs=False,
                                          include_prompt_logprobs=True, topk_prompt_logprobs=5)[0]
            assert alone.prompt_logprobs == responses[i].prompt_logprobs, \
                f'row {i}: score depends on batch neighbours (per-row independence violated)'
    finally:
        import asyncio
        asyncio.run(engine.shutdown())


def test_reverse_old_batched_mask_scores_an_extra_position(tiny_qwen2_5, monkeypatch):
    """Single-point reverse: the pre-a4be8355 batched body (left-pad + ``attention[row, 1:]``)
    must be DETECTED -- every short row gains one bogus scored position. Red without the fix."""
    import torch
    from twinkle.data_format.sampling import SamplingParams
    from twinkle.sampler.transformers_sampler.transformers_engine import TransformersEngine

    def _old_prompt_logprobs(self, inputs, topk):
        # verbatim pre-a4be8355 body (git show a4be8355^), test-boundary injection only
        with torch.inference_mode():
            logits = self.model(input_ids=inputs['input_ids'],
                                attention_mask=inputs['attention_mask']).logits
        logprobs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        targets = inputs['input_ids'][:, 1:]
        picked = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        out, alternatives = [], [] if topk else None
        for row in range(picked.shape[0]):
            mask = inputs['attention_mask'][row, 1:].bool()
            out.append([None] + picked[row][mask].tolist())
            if topk:
                values, indices = logprobs[row][mask].topk(topk, dim=-1)
                alternatives.append([None] + [list(zip(i, v)) for i, v in
                                              zip(indices.tolist(), values.tolist())])
        return out, alternatives

    engine, template = _engine_and_template(tiny_qwen2_5)
    try:
        prompts = [_prompt_ids(template, t) for t in
                   ['Hi.', 'What is the capital of France? Answer briefly.']]
        params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=5)
        monkeypatch.setattr(TransformersEngine, '_prompt_logprobs', _old_prompt_logprobs)
        responses = engine.generate_batch(prompts, params, logprobs=False,
                                          include_prompt_logprobs=True, topk_prompt_logprobs=5)
        short = responses[0]
        # The old mask scores len(real)+1 positions on the short (left-padded) row: the oracle
        # length contract is violated -- this is the RED the forward test above would report.
        assert len(short.prompt_logprobs) == len(prompts[0]) + 1, \
            'reverse control failed to reproduce the old off-by-one scoring'
        oracle_chosen, _ = _hf_oracle(tiny_qwen2_5, prompts[0])
        assert len(short.prompt_logprobs) != len(oracle_chosen)
    finally:
        import asyncio
        asyncio.run(engine.shutdown())