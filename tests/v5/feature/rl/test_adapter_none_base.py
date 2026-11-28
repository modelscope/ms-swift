# Copyright (c) ModelScope Contributors. All rights reserved.
"""H-UT3 ② — adapter=None routes to the BASE model (twinkle 45cefb4e, runs/H07,
h07_adapter_none_base{,_sampler}).

Contract: a sampler request with NO adapter must sample the base weights even when a peft
adapter is loaded (``adapter_uri=None`` alone means "keep the current adapter" for in-place
generation, so the facade passes ``use_base_model=adapter_uri is None``). An illegal adapter
request (uri set, nothing loaded) must raise loudly.

Oracle (independent): a plain ``AutoModelForCausalLM`` from the same tiny dir (NO peft) greedy
decodes the prompt; the base-path tokens must equal it exactly. The adapter carries a REAL
non-zero lora_B (peft-injected, scaled so greedy diverges), so adapter-path tokens must DIFFER
from the base oracle -- proving the switch actually changes the weights used.

Reverse (single point, in-process): monkeypatch ``TransformersEngine._base_model_context`` to
always return ``contextlib.nullcontext()`` (the pre-fix behavior: no adapter disable). The
"base" request then rides the adapter and diverges from the independent base oracle -> red.

Placement: tests/v5/feature/rl/test_adapter_none_base.py (uses ``tiny_qwen2_5``).
Markers: slow + accel(1). Estimated runtime: ~60-90 s.
"""
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.accel(1)]


def _make_nonzero_lora_adapter(model_path, dest):
    """A real peft LoRA adapter whose B is non-zero (random init scaled up so greedy diverges)."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    cfg = LoraConfig(r=8, lora_alpha=32, target_modules=['q_proj', 'v_proj'], lora_dropout=0.0)
    peft_model = get_peft_model(model, cfg)
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if 'lora_B' in name:
                param.normal_(0, 0.5)
    peft_model.save_pretrained(dest)
    del peft_model, model
    return str(dest)


def _greedy_tokens_from_ids(engine, prompt_ids, **kwargs):
    from twinkle.data_format.sampling import SamplingParams

    params = SamplingParams(max_tokens=12, temperature=0.0)
    resp = engine.generate_batch([prompt_ids], params, logprobs=False, **kwargs)
    return [int(t) for t in resp[0].sequences[0].tokens]


def _hf_base_greedy(model_path, prompt_ids):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_path)
    ids = torch.tensor([prompt_ids], dtype=torch.long)
    with torch.inference_mode():
        out = model.generate(ids, max_new_tokens=12, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    tokens = out[0, ids.shape[-1]:].tolist()
    del model
    return tokens


def _prompt_ids(model_path):
    from swift.dev.builder import build_template
    from swift.dev.config import TemplateConfig
    from swift.model import get_model_processor

    _, proc = get_model_processor(model_path, model_type='qwen2', load_model=False)
    template = build_template(TemplateConfig(template='qwen2_5', max_length=256), proc)
    return list(template.encode({'messages': [{'role': 'user', 'content': 'Say hello.'}]})['input_ids'])


def test_adapter_none_samples_base_and_illegal_adapter_raises(tiny_qwen2_5, tmp_path):
    import asyncio

    import torch
    from twinkle.sampler.transformers_sampler.transformers_engine import TransformersEngine

    adapter_dir = _make_nonzero_lora_adapter(tiny_qwen2_5, tmp_path / 'adapter')
    prompt_ids = _prompt_ids(tiny_qwen2_5)
    base_oracle = _hf_base_greedy(tiny_qwen2_5, prompt_ids)

    engine = TransformersEngine(model_id=tiny_qwen2_5, dtype=torch.float32)
    try:
        # Illegal request BEFORE any load: loud ValueError, not silent base fallback.
        with pytest.raises(ValueError, match='no LoRA is loaded'):
            engine.generate_batch([prompt_ids], None, logprobs=False, adapter_uri='ghost')

        name = engine.load_lora(adapter_dir)
        adapter_tokens = _greedy_tokens_from_ids(engine, prompt_ids, adapter_uri=name)
        base_tokens = _greedy_tokens_from_ids(engine, prompt_ids, adapter_uri=None, use_base_model=True)

        assert base_tokens == base_oracle, \
            f'base-path tokens diverge from independent plain-HF base greedy: {base_tokens} vs {base_oracle}'
        assert adapter_tokens != base_oracle, \
            'non-zero lora_B adapter did not change greedy output -- the fixture cannot prove the switch'
        assert base_tokens != adapter_tokens, 'adapter=None still rode the adapter weights'
        # The adapter stays usable after a base request (disable is a context, not an unload).
        again = _greedy_tokens_from_ids(engine, prompt_ids, adapter_uri=name)
        assert again == adapter_tokens
    finally:
        asyncio.run(engine.shutdown())


def test_reverse_base_context_disabled_rides_adapter(tiny_qwen2_5, tmp_path, monkeypatch):
    """Single-point reverse: pre-fix ``_base_model_context`` (no disable) must be DETECTED --
    the 'base' request diverges from the independent base oracle. Red without the fix."""
    import asyncio
    import contextlib

    import torch
    from twinkle.sampler.transformers_sampler.transformers_engine import TransformersEngine

    adapter_dir = _make_nonzero_lora_adapter(tiny_qwen2_5, tmp_path / 'adapter')
    prompt_ids = _prompt_ids(tiny_qwen2_5)
    base_oracle = _hf_base_greedy(tiny_qwen2_5, prompt_ids)

    engine = TransformersEngine(model_id=tiny_qwen2_5, dtype=torch.float32)
    try:
        name = engine.load_lora(adapter_dir)
        monkeypatch.setattr(TransformersEngine, '_base_model_context',
                            lambda self, use_base_model: contextlib.nullcontext())
        fake_base = _greedy_tokens_from_ids(engine, prompt_ids, adapter_uri=None, use_base_model=True)
        assert fake_base != base_oracle, \
            'reverse control failed: disabling the base switch must divert the request onto the adapter'
        assert fake_base == _greedy_tokens_from_ids(engine, prompt_ids, adapter_uri=name)
    finally:
        asyncio.run(engine.shutdown())