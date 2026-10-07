# Copyright (c) ModelScope Contributors. All rights reserved.
"""Sampler dimension (RL_PLAN §四/③): the rollout engine honours sampling params, generates in process,
and publishes the trained policy back into the sampler.

The component under test is the ROLLOUT ENGINE (twinkle's vLLM sampler behind ``swift.dev.rollout.
RolloutEngine``) and the one TransformersModel-only verb, in-place ``generate``. The external contract is
basic principle 1: on-policy generation always rides an independent sampler + per-step weight sync (a route
both backends share), and ``generate`` is the SINGLE capability that is TransformersModel-only -- so it must
work in process, and it must never be mistaken for the RL rollout main path (that mistaken wiring is refused
loudly; see ``test_infeasible_combos.py`` case 2).

PLAN table (finalized against the real code -- ``RolloutEngine.generate`` returns a FLAT list of
``RolloutSample`` grouped by prompt, ``num_samples`` per prompt, each carrying ``response_token_ids``):

| # | case | input | expected (contract) | failure it kills |
|---|------|-------|---------------------|------------------|
| 1 | vLLM greedy determinism + temperature reaches the engine | one prompt; greedy ``temperature=0`` ``num_samples=1`` called twice vs hot ``temperature=1.5`` ``num_samples=4`` | greedy: the two calls return IDENTICAL ``response_token_ids``; hot: the 4 draws are NOT all identical | sampling_params swallowed by the engine (a fixed default temperature would make greedy non-reproducible or hot collapse); degenerate/empty generation. (greedy uses two ``num_samples=1`` calls, not ``n=4`` in one: vLLM forbids ``n>1`` under greedy) |
| 2 | transformers in-process generate | a Trajectory row, ``max_tokens>0`` | one ``SampleResponse`` per input, each with a non-empty generated token sequence | the TransformersModel-only in-place generate verb broken/removed (basic principle 1's one allowed backend difference) |
| 3 | weight publication fires every window | real colocate GRPO via ``run_rl``, ``num_generations=2`` / ``max_steps=6`` (spans 3 windows) | ``stream_publishes`` present, monotone, reaching ``>= 2`` (a re-publish per generation window), policy trained (``lora_B`` moved) | the per-window policy->sampler publication silently skipped (W14) -- the loop would roll out on stale weights forever; or an initial-only load (``<= 1``) mistaken for a cadence |

N/A (RL_PLAN §九): ``sglang`` -- not installed in this environment, so the sampler dimension is exercised
with vLLM (the real engine) and transformers (in-process generate) only. The weight DELIVERY itself (the
synced bytes arriving in the sampler) is anchored end to end by ``test_backends_e2e.py``'s megatron GRPO run,
whose colocate CUDA-IPC hand-over logs "Updated N base weights via IPC"; case 3 here asserts the publication
CADENCE on the transformers colocate path, which that megatron run does not isolate.

Reverse-verification: case 1 goes red if the engine ignores ``temperature`` (greedy stops being deterministic
or hot stops varying); case 2 goes red if ``generate`` is removed/raises; case 3 goes red if the loop stops
publishing per step (``stream_publishes`` flat). Run with ``CUDA_VISIBLE_DEVICES=<card> pytest
swift/dev/tests/feature/rl/test_samplers_e2e.py -m slow``.
"""
import pytest

from swift.dev.tests.feature.rl.conftest import rl_configs, run_rl


def _tiny_template(model_path, model_type='qwen2'):
    """Build the dev Template for a tiny checkpoint (real tokenizer), the way ``RolloutEngine`` needs it.

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

    Drives the REAL ``RolloutEngine`` (twinkle's vLLM sampler) on a tiny checkpoint -- no stub. Two independent
    oracles, both anchored to the sampling parameters rather than to a self-consistent round trip:

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
    """
    from swift.dev.rollout import RolloutEngine

    engine = RolloutEngine(
        tiny_qwen2_5, _tiny_template(tiny_qwen2_5),
        engine_args={'gpu_memory_utilization': 0.3, 'max_model_len': 512, 'enforce_eager': True})
    try:
        prompt = [[{'role': 'user', 'content': 'Count to three.'}]]
        greedy = {'temperature': 0.0, 'max_tokens': 16}
        first = engine.generate(prompt, num_samples=1, sampling_params=greedy)
        second = engine.generate(prompt, num_samples=1, sampling_params=greedy)
        assert len(first) == 1, f'greedy num_samples=1 returned {len(first)} samples, expected exactly one'
        tokens = first[0].response_token_ids
        assert len(tokens) > 0, f'greedy generation degenerated to empty: {tokens}'
        repeat = second[0].response_token_ids
        assert repeat == tokens, (
            f'greedy (temperature=0) rollout not reproducible across two calls -- the engine is not decoding '
            f'greedily (a swallowed temperature would sample stochastically): {repeat} != {tokens}')

        hot = [s.response_token_ids for s in
               engine.generate(prompt, num_samples=4, sampling_params={'temperature': 1.5, 'top_p': 1.0,
                                                                       'max_tokens': 16})]
        assert len(hot) == 4, f'hot num_samples=4 returned {len(hot)} samples, expected one per draw'
        assert not all(t == hot[0] for t in hot), \
            f'hot (temperature=1.5) draws are all identical -- temperature never reached the engine: {hot}'
    finally:
        engine.shutdown()


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
def test_colocate_grpo_publishes_weights_every_step(tiny_qwen2_5, rl_data, tmp_path, assert_rl_trained):
    """Case 3: a real colocate GRPO run re-publishes the trained policy into the sampler EVERY window.

    The on-policy loop hands the device (and the freshly trained policy) to the sampler once per generation
    window: ``StreamingLoopMixin._enter_generation`` -> ``SyncableRollout.sync_weights`` -> the colocate
    CUDA-IPC hand-over. The streaming driver counts each publication in ``stream_publishes`` (a lifetime,
    monotone counter folded into every optimizer step's record). A loop that silently skipped the sync (W14 --
    the base ``RolloutEngine.sync_weights`` is a warn-once no-op) would still roll out, but on the sampler's
    initial weights forever, and ``stream_publishes`` would sit at 0.

    WHY the geometry is ``num_generations=2`` / ``max_steps=6`` (this is the part a naive run gets wrong): at
    ``async_mode='none'`` the driver publishes only when a NEXT window exists to publish TO -- it admits ONE
    window, drains it, trains it, publishes, then opens the next (``streaming_driver.run`` phase (1); no
    trailing publish after the last step, since training is done). One GRPO window is exactly one group
    (``groups_per_partition = ceil(parameter_sync_step * num_generations / num_generations) = 1``), and one
    group of ``num_generations`` rows trains ``num_generations / (per_device_train_batch_size * ga * dp)``
    optimizer steps. With the harness default ``num_generations=4`` / ``bs=ga=1`` a SINGLE window trains 4
    steps, so a ``max_steps<=4`` run reaches its budget inside window 0 and breaks at ``_reached_max`` before
    ANY publish -- ``stream_publishes`` stays 0 by correct design, not by a skipped sync. Shrinking the group
    to 2 (one window = 2 steps) and asking for 6 steps spans THREE windows, so the driver must re-sync twice
    (``[0,0,1,1,2,2]``): the advancing count is then a real observation of per-window publication, and
    ``>= 2`` rules out a one-shot initial-only load. ``lora_B`` moving (``assert_rl_trained``) confirms the
    steps trained.
    """
    configs = rl_configs(
        rlhf_type='grpo',
        model=tiny_qwen2_5,
        model_type='qwen2',
        template='qwen2_5',
        dataset=rl_data.prompt_only(n=6),
        out_dir=str(tmp_path / 'out_sampler_sync'),
        max_steps=6,
        rlhf_over={'num_generations': 2, 'orm': [_varied_content_reward]},
        generation_over={'temperature': 1.0, 'top_p': 1.0, 'max_new_tokens': 16},
    )
    history = run_rl(configs)
    assert_rl_trained(history, configs, 'grpo', max_loss=20.0, require_keys=('stream_publishes', ))
    publishes = [r['stream_publishes'] for r in history]
    assert all(b >= a for a, b in zip(publishes, publishes[1:])), \
        f'stream_publishes is a monotone lifetime counter but went backwards: {publishes}'
    assert publishes[-1] >= 2, (
        f'stream_publishes only reached {publishes[-1]} across a 3-window run -- the policy was not '
        f're-synced per generation window (a silently skipped sync, W14, leaves it at 0; an initial-only '
        f'load leaves it at <=1): {publishes}')
    assert publishes[-1] > publishes[0], \
        f'stream_publishes never advanced across the run -- the policy was not re-synced per step: {publishes}'


def _varied_content_reward(completions, **kwargs):
    """A deterministic ORM reward that VARIES with completion CONTENT (char-sum mod 11, scaled to [0,1)).

    GRPO's advantage is ``(r - group_mean) / group_std``, so a constant reward across a prompt's group gives
    std=0 -> advantage 0 -> zero gradient -> ``lora_B`` never moves. Keying the reward on content gives most
    groups a non-zero std so the run really trains (mirrors ``test_algorithms_e2e._varied_content_reward``).
    """
    return [float(sum(ord(ch) for ch in c) % 11) / 11.0 for c in completions]
