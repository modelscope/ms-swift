# Copyright (c) ModelScope Contributors. All rights reserved.
"""Sampler dimension (RL_PLAN §四/③): the rollout engine honours sampling params and generates in process.

The components under test are twinkle's real ``vLLMSampler`` (the engine the dev rollout builds via
``swift.dev.builders.build_sampler``) and the one TransformersModel-only verb, in-place ``generate``. The
external contract is basic principle 1: on-policy generation always rides an independent sampler + per-step
weight sync (a route both backends share), and ``generate`` is the SINGLE capability that is TransformersModel-
only -- so it must work in process, and it must never be mistaken for the RL rollout main path (that mistaken
wiring is refused loudly; see ``test_infeasible_combos.py`` case 2).

PLAN table (finalized against the real code -- ``vLLMSampler.sample`` returns one ``SampleResponse`` per input,
each carrying ``sequences[*].tokens``):

| # | case | input | expected (contract) | failure it kills |
|---|------|-------|---------------------|------------------|
| 1 | vLLM greedy determinism + temperature reaches the engine | one prompt; greedy ``temperature=0`` called twice vs hot ``temperature=1.5`` ``num_samples=4`` | greedy: the two calls return IDENTICAL token sequences; hot: the 4 draws are NOT all identical | sampling_params swallowed by the engine (a fixed default temperature would make greedy non-reproducible or hot collapse); degenerate/empty generation. (greedy uses two ``num_samples=1`` calls, not ``n=4`` in one: vLLM forbids ``n>1`` under greedy) |
| 2 | transformers in-process generate | a Trajectory row, ``max_tokens>0`` | one ``SampleResponse`` per input, each with a non-empty generated token sequence | the TransformersModel-only in-place generate verb broken/removed (basic principle 1's one allowed backend difference) |

Case 3 (the former "weight publication fires every window" on a colocate GRPO run) is GONE: a colocate run is
now a driver-less plain loop (``run_grpo._colocate_loop``) that calls ``sync_weights(merge_and_sync=True)``
UNCONDITIONALLY every window -- there is no StreamingDriver, so no ``stream_publishes`` counter to read, and
no skip branch that could silently regress. The per-window publication is therefore structural on colocate;
the MEASURABLE publish cadence (``stream_publishes >= 2``) is asserted on the disaggregated driver path in
``test_placement_e2e.py``, the colocate on-policy multi-window cycle in ``test_async_e2e.py`` row 1, and the
colocate weight DELIVERY (synced bytes arriving) in ``test_backends_e2e.py``'s megatron GRPO IPC hand-over.

N/A -> skip-guarded (S7-21): ``sglang`` is not installed in this environment; case 3 below carries the full
peer contract behind ``pytest.importorskip('sglang')`` so it SKIPS here (never errors) and runs the real
``SGLangSampler`` oracles the moment the dependency appears. Until then the sampler dimension is exercised
with vLLM (the real engine) and transformers (in-process generate) only. The weight DELIVERY itself (the
synced bytes arriving in the sampler) is anchored end to end by ``test_backends_e2e.py``'s megatron GRPO run,
whose colocate CUDA-IPC hand-over logs "Updated N base weights via IPC".

Reverse-verification: case 1 goes red if the engine ignores ``temperature`` (greedy stops being deterministic
or hot stops varying); case 2 goes red if ``generate`` is removed/raises. Run with ``CUDA_VISIBLE_DEVICES=<card>
pytest swift/dev/tests/feature/rl/test_samplers_e2e.py -m slow``.
"""
import pytest


def _tiny_template(model_path, model_type='qwen2'):
    """Build the dev Template for a tiny checkpoint (real tokenizer), the way the rollout sampler needs it.

    ``model_type`` is passed explicitly: the shrunk ``tiny_qwen2_5`` config matches BOTH ``qwen2`` and
    ``qwen2_gte``, so ``get_model_processor`` refuses to auto-pick and raises without it.
    """
    from swift.dev.builders import build_template
    from swift.dev.config import TemplateConfig
    from swift.model import get_model_processor
    _, proc = get_model_processor(model_path, model_type=model_type, load_model=False)
    return build_template(TemplateConfig(template='qwen2_5', max_length=256), proc)


@pytest.mark.slow
@pytest.mark.accel(1)
def test_vllm_greedy_deterministic_and_temperature_reaches_engine(tiny_qwen2_5):
    """Case 1: greedy vLLM rollout is token-for-token deterministic; a hot rollout is not.

    Drives the REAL ``vLLMSampler`` (the engine the dev rollout builds via ``build_sampler``) on a tiny
    checkpoint -- no stub. Two independent oracles, both anchored to the sampling parameters rather than to a
    self-consistent round trip:

    * greedy (``temperature=0``) called TWICE must return the SAME token sequence each time -- argmax decoding
      is a deterministic function of the weights, so any variation between two calls means the engine is not
      actually decoding greedily (a swallowed ``temperature`` leaving it at a sampling default), and an empty
      sequence means generation degenerated. Greedy is drawn with ``num_samples=1`` per call, NOT ``n=4`` in one
      call: vLLM rejects ``n > 1`` under greedy sampling (``n must be 1 when using greedy sampling``) because
      ``n`` identical argmax draws are redundant -- so determinism is asserted across two separate calls.
    * hot (``temperature=1.5``, ``num_samples=4``) must return four NON-identical sequences -- a stochastic
      sampler diverges across draws, so four identical hot samples would mean the engine ignored ``temperature``
      and collapsed, i.e. the sampling params never reached it.

    Together these kill "sampling_params 被吞 / 生成退化": the engine provably distinguishes temp=0 (repeatable)
    from temp=1.5 (varied), which is only possible if the parameter is threaded all the way into vLLM's sampler.
    Mirrors twinkle's own ``test_vllm_sampler_e2e`` build pattern (the dev ``RolloutEngine`` wrapper is gone;
    dev builds this same sampler through ``build_sampler``).
    """
    from twinkle.data_format.sampling import SamplingParams
    from twinkle.sampler.vllm_sampler.vllm_sampler import vLLMSampler

    sampler = vLLMSampler(
        model_id=tiny_qwen2_5,
        engine_args={'tensor_parallel_size': 1, 'gpu_memory_utilization': 0.3, 'max_model_len': 512,
                     'enforce_eager': True})
    # The dev qwen2_5 template (the one the recipe's build_sampler sets), not twinkle's generic 'Template':
    # it resolves the shrunk checkpoint's qwen2/qwen2_gte ambiguity and encodes the prompt the way the online
    # rollout does, so the engine receives the same feature shape it gets in a real run.
    sampler.set_template(_tiny_template(tiny_qwen2_5))
    try:
        traj = [{'messages': [{'role': 'user', 'content': 'Count to three.'}]}]

        def _draw(params):
            # one SampleResponse per input; flatten every sequence's tokens so num_samples expansion is
            # shape-agnostic (4 draws may arrive as 4 sequences of one response or 4 responses). Cast to
            # plain ints: the engine may hand back numpy scalars, whose list == list is element-wise ambiguous.
            return [[int(t) for t in seq.tokens]
                    for resp in sampler.sample(traj, sampling_params=params) for seq in resp.sequences]

        greedy = SamplingParams(max_tokens=16, temperature=0.0)
        first = _draw(greedy)
        assert len(first) == 1 and len(first[0]) > 0, f'greedy generation degenerated to empty: {first}'
        second = _draw(greedy)
        assert second == first, (
            f'greedy (temperature=0) rollout not reproducible across two calls -- the engine is not decoding '
            f'greedily (a swallowed temperature would sample stochastically): {second} != {first}')

        hot = _draw(SamplingParams(max_tokens=16, temperature=1.5, top_p=1.0, num_samples=4))
        assert len(hot) == 4, f'hot num_samples=4 returned {len(hot)} sequences, expected one per draw'
        assert not all(t == hot[0] for t in hot), \
            f'hot (temperature=1.5) draws are all identical -- temperature never reached the engine: {hot}'
    finally:
        sampler.shutdown()


@pytest.mark.slow
@pytest.mark.accel(1)
def test_transformers_in_process_generate_runs(tiny_qwen2_5):
    """Case 2: ``TransformersModel.generate`` produces completions in process (the one backend-only verb).

    Basic principle 1 makes ``generate`` the SINGLE TransformersModel-only capability (``MegatronModel.generate``
    refuses by design -- ``test_infeasible_combos.py`` case 1). This asserts the positive half: a transformers
    model generates on the weights it already holds, inside its own worker, with no second model copy and no
    weight sync (a ``TransformersEngine`` is wrapped around the live weights per call). It is NOT the RL rollout
    main path -- an in-process TransformersSampler cannot be weight-synced, so ``run_grpo`` refuses it as an
    online sampler (``test_infeasible_combos.py`` case 2); this test pins that the verb itself works, so the
    refusal is a policy choice about the rollout path, not a broken capability.
    """
    import torch

    from swift.dev.model import TransformersModel
    from swift.dev.processor import InputProcessor

    model = TransformersModel(model_id=tiny_qwen2_5, mixed_precision='no', strategy='accelerate',
                              dtype=torch.float32)
    model.set_processor(InputProcessor())
    model.set_template(_tiny_template(tiny_qwen2_5))
    trajectories = [
        {'messages': [{'role': 'user', 'content': 'Say hello.'}]},
        {'messages': [{'role': 'user', 'content': 'Count to three.'}]},
    ]
    responses = model.generate(trajectories, sampling_params={'max_tokens': 8, 'temperature': 0.0})
    assert len(responses) == len(trajectories), \
        f'in-process generate returned {len(responses)} responses for {len(trajectories)} inputs'
    for i, resp in enumerate(responses):
        assert resp.sequences, f'input {i}: no sequence generated in process'
        assert len(resp.sequences[0].tokens) > 0, f'input {i}: generated an empty token sequence'


@pytest.mark.slow
@pytest.mark.accel(1)
def test_sglang_greedy_deterministic_and_temperature_reaches_engine(tiny_qwen2_5):
    """Case 3: the sglang sampler is the vLLM sampler's peer (S7-21) -- skipped while sglang is absent.

    ``SGLangSampler.__init__`` itself calls ``requires('sglang')``, so the engine cannot even be built
    without the dependency; ``importorskip`` turns that into a collect-time SKIP instead of an error, per
    the suite's environment rule. Where sglang IS installed this runs the SAME two-oracle contract as the
    vLLM case 1 (greedy ``temperature=0`` twice -> identical tokens; hot ``temperature=1.5`` ``num_samples=4``
    -> four non-identical draws), against the REAL ``SGLangSampler`` on the same tiny checkpoint and the
    same dev template -- the peer guarantee is that the sampling parameters thread into sglang's engine
    exactly as they do into vLLM's, so a swallowed temperature trips the same two oracles.
    """
    pytest.importorskip('sglang')

    from twinkle.data_format.sampling import SamplingParams
    from twinkle.sampler.sglang_sampler import SGLangSampler

    sampler = SGLangSampler(
        model_id=tiny_qwen2_5,
        engine_args={'tensor_parallel_size': 1, 'mem_fraction_static': 0.3, 'context_length': 512})
    sampler.set_template(_tiny_template(tiny_qwen2_5))
    try:
        traj = [{'messages': [{'role': 'user', 'content': 'Count to three.'}]}]

        def _draw(params):
            return [[int(t) for t in seq.tokens]
                    for resp in sampler.sample(traj, sampling_params=params) for seq in resp.sequences]

        greedy = SamplingParams(max_tokens=16, temperature=0.0)
        first = _draw(greedy)
        assert len(first) == 1 and len(first[0]) > 0, f'greedy generation degenerated to empty: {first}'
        second = _draw(greedy)
        assert second == first, (
            f'greedy (temperature=0) sglang rollout not reproducible across two calls -- the engine is not '
            f'decoding greedily (a swallowed temperature would sample stochastically): {second} != {first}')

        hot = _draw(SamplingParams(max_tokens=16, temperature=1.5, top_p=1.0, num_samples=4))
        assert len(hot) == 4, f'hot num_samples=4 returned {len(hot)} sequences, expected one per draw'
        assert not all(t == hot[0] for t in hot), \
            f'hot (temperature=1.5) sglang draws are all identical -- temperature never reached the engine: {hot}'
    finally:
        sampler.shutdown()
