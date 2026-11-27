# Copyright (c) ModelScope Contributors. All rights reserved.
"""H-UT3 ⑥ — an excluded stop string trims ONLY the decoded text, never the sampled tokens
(twinkle 45cefb4e vllm_engine ``decoded=output.text``, runs/H06/fixes/d2_vllm_decoded_stop_trim.diff;
ms-swift 5560fc49d builders/streaming.py stop_words->stop, d1_rollout_stop_words.diff).

Contract: vLLM's detokenizer applies the stop-string trim to ``output.text`` while
``output.token_ids`` keeps every sampled id. The engine carries that trimmed text as
``decoded`` so the complete trajectory's token ids stay intact for RL training. The dev
rollout forwards the user's ``stop_words`` as ``stop`` (excluded by default).

Oracle (independent): the raw tokenizer decode of the returned ``tokens`` -- it MUST contain
the stop string (complete trajectory), while ``seq.decoded`` must NOT (decoded-only trim),
and ``stop_reason == 'stop'``. The stop string is taken from the model's OWN greedy output
(first pass without stop), so the second greedy pass provably reaches it -- no degenerate
"stop never matched" input can substitute.

Reverse (single point, in-process): wrap ``VLLMEngine.sample`` to null out ``decoded`` on
every returned sequence (the pre-fix output: no ``decoded=`` carried). The sampler then
decodes the complete tokens downstream and the stop string reappears in ``decoded`` -> red.

Placement: tests/v5/feature/rl/test_stop_decoded_only.py (uses ``tiny_qwen2_5``).
Markers: slow + accel(1) + requires_vllm. Estimated runtime: ~90-150 s (vLLM engine init).
"""
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.accel(1), pytest.mark.requires_vllm]


def _sampler(model_path):
    from swift.dev.builders import build_template
    from swift.dev.config import TemplateConfig
    from swift.model import get_model_processor
    from twinkle.sampler.vllm_sampler.vllm_sampler import vLLMSampler

    _, proc = get_model_processor(model_path, model_type='qwen2', load_model=False)
    sampler = vLLMSampler(model_id=model_path,
                          engine_args={'tensor_parallel_size': 1, 'gpu_memory_utilization': 0.3,
                                       'max_model_len': 512, 'enforce_eager': True})
    sampler.set_template(build_template(TemplateConfig(template='qwen2_5', max_length=256), proc))
    return sampler


def _rollout_stop_mapping():
    """CPU half of the contract: the dev rollout maps stop_words -> stop (d1)."""
    from swift.dev.builders.streaming import rollout_sampling_params
    from swift.dev.config import GenerationConfig

    params = rollout_sampling_params(64, GenerationConfig(stop_words=['<|end|>']))
    assert params['stop'] == ['<|end|>'], f'rollout dropped stop_words: {params}'
    assert rollout_sampling_params(64, GenerationConfig()) .get('stop') is None


def test_excluded_stop_trims_decoded_not_tokens(tiny_qwen2_5):
    from transformers import AutoTokenizer

    from twinkle.data_format.sampling import SamplingParams

    _rollout_stop_mapping()
    sampler = _sampler(tiny_qwen2_5)
    tok = AutoTokenizer.from_pretrained(tiny_qwen2_5)
    try:
        traj = [{'messages': [{'role': 'user', 'content': 'Count to three.'}]}]
        # Pass 1: greedy WITHOUT stop -- take the stop string from the model's own output.
        plain = sampler.sample(traj, sampling_params=SamplingParams(max_tokens=32, temperature=0.0))
        full_text = tok.decode(plain[0].sequences[0].tokens, skip_special_tokens=True)
        assert len(full_text) >= 6, f'greedy output too short to carve a stop: {full_text!r}'
        stop_str = full_text[3:6]

        # Pass 2: same greedy prompt WITH the excluded stop -- deterministic replay reaches it.
        resp = sampler.sample(traj, sampling_params=SamplingParams(
            max_tokens=32, temperature=0.0, stop=[stop_str]))
        seq = resp[0].sequences[0]
        assert seq.stop_reason == 'stop', f'stop string never matched: {seq.stop_reason}'
        assert seq.decoded is not None and stop_str not in seq.decoded, \
            f'excluded stop leaked into decoded: {seq.decoded!r}'
        raw = tok.decode(seq.tokens, skip_special_tokens=True)
        assert stop_str in raw, \
            f'sampled tokens were trimmed too (trajectory must stay complete): {raw!r}'
        assert raw.startswith(seq.decoded), 'decoded must be a prefix of the full token decode'
    finally:
        sampler.shutdown()


def test_reverse_decoded_dropped_reintroduces_stop(tiny_qwen2_5, monkeypatch):
    """Single-point reverse: pre-fix output carried no trimmed ``decoded`` (None). The
    downstream decode of the complete tokens then contains the excluded stop -> red."""
    from transformers import AutoTokenizer

    from twinkle.data_format.sampling import SamplingParams
    from twinkle.sampler.vllm_sampler.vllm_engine import VLLMEngine

    sampler = _sampler(tiny_qwen2_5)
    tok = AutoTokenizer.from_pretrained(tiny_qwen2_5)
    try:
        traj = [{'messages': [{'role': 'user', 'content': 'Count to three.'}]}]
        plain = sampler.sample(traj, sampling_params=SamplingParams(max_tokens=32, temperature=0.0))
        stop_str = tok.decode(plain[0].sequences[0].tokens, skip_special_tokens=True)[3:6]

        original = VLLMEngine.sample

        async def _old_sample(self, *args, **kwargs):
            response = await original(self, *args, **kwargs)
            for seq in response.sequences:
                seq.decoded = None  # pre-fix: trimmed text never carried
            return response

        monkeypatch.setattr(VLLMEngine, 'sample', _old_sample)
        resp = sampler.sample(traj, sampling_params=SamplingParams(
            max_tokens=32, temperature=0.0, stop=[stop_str]))
        seq = resp[0].sequences[0]
        assert seq.decoded is None or stop_str in seq.decoded, \
            'reverse control failed: without the trimmed decoded the stop must reappear downstream'
    finally:
        sampler.shutdown()