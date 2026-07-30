"""GRPO rollout regression tests.

Two tiers:
  - cheap (no vLLM, runs in the normal suite): the vLLM-mode no-shift guard and the
    group-advantage wiring.
  - slow (@pytest.mark.slow, real vLLM engine, skipped in normal runs): end-to-end GRPO
    pipeline smoke — rollout -> advantages -> GRPOLoss forward_backward -> param update, plus
    the vLLM-logprobs vs train-forward-logps comparison.

Discipline: a green slow test means "pipeline works + params update". It is NOT a claim that
GRPO is algorithmically correct — weight-sync is delayed (vLLM keeps initial weights => stale
behavior policy). Tests assert this intermediate-state fact explicitly.
"""
import os
import pytest
import torch

MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'

# ----------------------------------------------------------------------
# cheap tier (no vLLM) — runs in the normal suite
# ----------------------------------------------------------------------


def test_vllm_mode_encode_has_no_labels_no_shift():
    """Under set_mode('vllm') the dev Template must NOT emit labels and must NOT apply the
    next-token shift (rollout inputs are inference, not training)."""
    from swift.dev.template import Template as DevTemplate
    from swift.model import get_model_processor
    from swift.template import get_template

    _, proc = get_model_processor(MODEL, load_model=False)
    tpl = DevTemplate.from_template(get_template(proc, template_type='qwen2_5', max_length=256))
    msgs = [{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'hello there'}]

    # training mode: labels present AND shifted
    tpl.set_mode('train')
    enc_train = tpl.encode({'messages': msgs})
    assert enc_train.get('labels') is not None
    assert enc_train.get(DevTemplate.SHIFTED_KEY) is True  # shift fired

    # vllm mode: no labels, shift guard skipped
    tpl.set_mode('vllm')
    enc_vllm = tpl.encode({'messages': msgs})
    assert enc_vllm.get('labels') is None, 'vLLM-mode encode must not produce labels'
    assert not enc_vllm.get(DevTemplate.SHIFTED_KEY), 'vLLM-mode must not apply next-token shift'


def test_group_advantages_reuse_twinkle():
    """compute_group_advantages reuses twinkle GRPOAdvantage: group-mean subtracted, std=0 -> 0."""
    from swift.dev.recipes.grpo import compute_group_advantages
    adv = compute_group_advantages([1., 2., 3., 4., 5., 5., 5., 5.], num_generations=4, scale='group')
    assert abs(sum(adv[:4]) / 4) < 1e-6  # group1 mean ~ 0
    assert all(abs(a) < 1e-6 for a in adv[4:])  # group2 constant -> zero advantage


def test_rollout_sample_shape_and_shift_marker():
    """RolloutSample is the RL-sample layer: `encoded` carries next-token-shifted labels + the
    SHIFTED_KEY marker (contract 14 recorded, not commented); per-turn lists are 2D; and the
    back-compat aliases (input_feature/old_logps/prompt_index) still resolve."""
    from swift.dev.recipes.grpo import toy_length_reward
    from swift.dev.rollout import SHIFTED_KEY, RolloutSample

    s = RolloutSample(
        encoded={
            'input_ids': [1, 2, 3, 4],
            'labels': [-100, 3, 4, -100],
            SHIFTED_KEY: True
        },
        response_token_ids=[[3, 4]],  # 2D per-turn (single turn here)
        rollout_logprobs=[[-0.1, -0.2]],  # 2D mirrors response_token_ids
        prompt_id='0',
        extra={'solution': '42'},
    )
    # SHIFTED_KEY marks the feature as already shifted (RL path bypasses Template.encode).
    assert s.encoded[SHIFTED_KEY] is True
    # 2D per-turn contract.
    assert s.response_token_ids == [[3, 4]] and s.rollout_logprobs == [[-0.1, -0.2]]
    # back-compat aliases (GRPOLoop still reads these).
    assert s.input_feature is s.encoded
    assert s.old_logps == [-0.1, -0.2]  # flattened across turns
    assert s.prompt_index == '0'
    # toy reward sums inner-turn lengths (not turn count).
    assert toy_length_reward(s) == 2.0


def test_get_reward_funcs_resolves_orms_and_callables():
    """L1 reward API: registered `orms` names resolve to ORM instances (args=config), callables pass
    through; an unknown name fails fast."""
    from swift.dev.reward import get_reward_funcs

    def my_reward(completions, **kw):
        return [1.0] * len(completions)

    funcs, names = get_reward_funcs(['format', my_reward])
    from swift.rewards.orm import Format
    assert isinstance(funcs[0], Format) and names[0] == 'Format'  # name -> orms instance
    assert funcs[1] is my_reward and names[1] == 'my_reward'  # callable passthrough
    with pytest.raises(ValueError, match='not registered'):
        get_reward_funcs(['no_such_reward'])


def test_compute_rewards_per_func_takes_completions_and_columns():
    """L1 reward API is general: it takes plain completions + batched columns (no sample class), and
    fails fast on misaligned columns / wrong-length reward output."""
    import torch

    from swift.dev.reward import compute_rewards_per_func, get_reward_funcs, weight_rewards

    def solved_reward(completions, solution=None, **kw):
        return [1.0 if c.strip() == s else 0.0 for c, s in zip(completions, solution)]

    funcs, _ = get_reward_funcs([solved_reward])
    r = compute_rewards_per_func(['4', '5'], funcs, {'solution': ['4', '4']})
    assert r.shape == (2, 1)
    assert torch.allclose(r[:, 0], torch.tensor([1.0, 0.0]))
    # weighted combination -> [N]
    assert torch.allclose(weight_rewards(r), torch.tensor([1.0, 0.0]))
    # column length must align with completions
    with pytest.raises(ValueError, match='has length'):
        compute_rewards_per_func(['4', '5'], funcs, {'solution': ['4']})
    # a reward returning the wrong number of scores is fatal, not silently padded
    with pytest.raises(ValueError, match='returned 1 scores'):
        compute_rewards_per_func(['4', '5'], [lambda c, **kw: [0.0]])


def test_advantage_api_delegates_to_rl_core():
    """L1 advantage API: group scale zero-centers a group; accepts a flat [N] list or [N,n_funcs]
    matrix; validates estimator/scaling/group divisibility/weights."""
    import torch

    from swift.dev.advantage import compute_advantages

    # flat list input (single reward)
    adv = compute_advantages([1.0, 2.0, 3.0, 4.0], num_generations=4, scale_rewards='group')
    assert abs(sum(adv)) < 1e-5
    # matrix input (multi reward)
    r = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    assert abs(sum(compute_advantages(r, num_generations=4))) < 1e-5
    # validation
    with pytest.raises(ValueError, match='reward_weights length'):
        compute_advantages(r, num_generations=4, reward_weights=[1.0, 2.0])
    with pytest.raises(ValueError, match='not in'):
        compute_advantages(r, num_generations=4, advantage_estimator='nope')
    with pytest.raises(ValueError, match='not divisible'):
        compute_advantages(r, num_generations=3)
    with pytest.raises(ValueError, match='requires kl_values'):
        compute_advantages(r, num_generations=4, kl_in_reward=True, beta=0.1)


def test_rollout_shifted_key_matches_template():
    """rollout.py re-declares the marker literal instead of importing the (heavy) dev Template.
    Gate the divergence: if Template.SHIFTED_KEY is ever renamed, the RL feature would stop being
    recognized as already-shifted and the processor would no longer drop it."""
    from swift.dev.rollout import SHIFTED_KEY
    from swift.dev.template import Template as DevTemplate
    assert SHIFTED_KEY == DevTemplate.SHIFTED_KEY


def test_extract_chosen_logps_raises_on_missing_logprobs():
    """old_logps must never be silently padded: 0.0 is a legal logprob (p=1.0), so padding a
    missing/short list would corrupt the importance ratio instead of failing."""
    from swift.dev.rollout import RolloutEngine

    extract = RolloutEngine._extract_chosen_logps
    # exact match is the only accepted shape
    assert extract({'content': [{'logprob': -0.5}, {'logprob': -1.5}]}, 2) == [-0.5, -1.5]
    assert extract(None, 0) == []
    # logprobs not requested at all -> empty content while tokens exist
    with pytest.raises(RuntimeError, match='logprobs misaligned'):
        extract(None, 3)
    # short list (partial logprobs) is equally fatal
    with pytest.raises(RuntimeError, match='logprobs misaligned'):
        extract({'content': [{'logprob': -0.5}]}, 2)


# ----------------------------------------------------------------------
# slow tier (real vLLM engine) — skipped in normal runs
# ----------------------------------------------------------------------


@pytest.mark.slow
def test_grpo_rollout_e2e_updates_params_intermediate_state():
    """End-to-end GRPO smoke: rollout(real vLLM) -> advantages -> GRPOLoss -> param update.

    Asserts:
      1. pipeline runs (history recorded).
      2. params actually change (step-before vs after diff > threshold; reject noise-region).
      3. INTERMEDIATE STATE: vLLM keeps initial weights (no weight-sync) -> this is NOT an
         algorithmically-correct GRPO run. Asserted explicitly so a green here is never
         mistaken for "GRPO correct".
    """
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    from swift.dev.builders import build_template
    from swift.dev.configs import TemplateConfig, TunerConfig
    from swift.dev.model import TransformersModel
    from swift.dev.processor import InputProcessor
    from swift.dev.recipes.grpo import GRPOLoop
    from swift.dev.rollout import RolloutEngine
    from swift.dev.tuner import apply_tuner
    from swift.model import get_model_processor

    _, proc = get_model_processor(MODEL, load_model=False)
    template = build_template(TemplateConfig(template='qwen2_5', max_length=256), proc)

    model = TransformersModel(model_id=MODEL, mixed_precision='no', strategy='accelerate', dtype=torch.float32)
    torch.manual_seed(0)
    apply_tuner(
        model,
        TunerConfig(tuner_type='lora', lora_rank=8, lora_alpha=16, target_modules=['q_proj', 'v_proj']),
        gradient_accumulation_steps=1)
    model.set_processor(InputProcessor())
    from twinkle.loss import GRPOLoss
    model.set_loss(GRPOLoss(epsilon=0.2, beta=0.0))  # beta=0: no KL/ref
    model.set_optimizer('AdamW', lr=5e-4)

    def _lora_snapshot():
        raw = model.strategy.unwrap_model(model.model)
        return {
            n: p.detach().float().cpu().clone()
            for n, p in raw.named_parameters() if 'lora_' in n and p.requires_grad
        }

    engine = RolloutEngine(
        MODEL, template, engine_args={
            'gpu_memory_utilization': 0.3,
            'max_model_len': 512,
            'enforce_eager': True
        })
    try:
        prompts = [[{'role': 'user', 'content': 'Count to three.'}], [{'role': 'user', 'content': 'Say hello.'}]]
        before = _lora_snapshot()
        loop = GRPOLoop(
            model,
            engine,
            prompts,
            num_generations=4,
            max_steps=2,
            gradient_accumulation_steps=4,  # 1 seq/micro, 4 seqs/window
            sampling_params={
                'temperature': 1.0,
                'max_tokens': 16
            })
        history = loop.fit()
        after = _lora_snapshot()

        # 1. pipeline ran
        assert history, 'GRPO loop produced no optimizer steps'
        # 2. params actually updated (strong-signal, reject noise region)
        max_delta = max((after[k] - before[k]).abs().max().item() for k in before if k in after)
        assert max_delta > 1e-5, f'params did not update (max_delta={max_delta:.2e})'
        # 3. intermediate-state fact: rollout policy never synced (weight-sync not wired yet)
        assert not hasattr(engine, '_weight_sync_done'), (
            'must NOT do weight-sync yet; vLLM stays on initial weights -> NOT correct GRPO')
    finally:
        engine.shutdown()


@pytest.mark.slow
def test_vllm_logprobs_match_train_forward_logps():
    """vLLM sampling logprobs must match a train-forward logps pass at the same
    temperature=1.0, else ratio != 1 silently pollutes the importance-sampling weight."""
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')

    from swift.dev.builders import build_template
    from swift.dev.configs import TemplateConfig
    from swift.dev.model import TransformersModel
    from swift.dev.processor import InputProcessor
    from swift.dev.rollout import RolloutEngine
    from swift.model import get_model_processor

    _, proc = get_model_processor(MODEL, load_model=False)
    template = build_template(TemplateConfig(template='qwen2_5', max_length=256), proc)
    engine = RolloutEngine(
        MODEL, template, engine_args={
            'gpu_memory_utilization': 0.3,
            'max_model_len': 512,
            'enforce_eager': True
        })
    try:
        prompts = [[{'role': 'user', 'content': 'Say hi.'}]]
        samples = engine.generate(prompts, num_samples=1, sampling_params={'temperature': 1.0, 'max_tokens': 8})
        assert samples, 'no rollout samples'
        s = samples[0]
        # train-forward logps on the same full sequence, at temperature=1.0 (twinkle forward does
        # logits.div_(temperature)) so it matches vLLM's sampling temperature.
        model = TransformersModel(model_id=MODEL, mixed_precision='no', strategy='accelerate', dtype=torch.float32)
        model.set_processor(InputProcessor())
        out = model.forward_only(inputs=[s.input_feature], temperature=1.0)
        train_logps = out.get('logps')
        assert train_logps is not None
        # Align via the shifted labels' loss_mask: response-token logps live exactly at the
        # positions where labels != -100 (the next-token shift already lines logits[i] up with
        # the response token it predicts). Compare those to vLLM old_logps token-for-token.
        labels = torch.tensor(s.input_feature['labels'])
        mask = (labels != -100)
        tl = train_logps.flatten().float().cpu()
        train_resp = tl[mask.nonzero(as_tuple=True)[0]] if tl.numel() == mask.numel() else tl[-int(mask.sum()):]
        vllm_lp = torch.tensor(s.old_logps, dtype=torch.float32)
        n = min(len(vllm_lp), len(train_resp))
        assert n > 0, 'no response tokens to compare'
        diff = (vllm_lp[:n] - train_resp[:n]).abs().max().item()
        # No weight-sync yet: vLLM and the training model are the SAME initial weights, so at
        # temperature=1.0 the logps must match to fp/backend noise (ratio~1). A shift off-by-one
        # in the RL feature would blow this up to ~20 -> product bug.
        assert diff < 0.05, (
            f'vLLM vs train logps diverge (max|diff|={diff:.4f}); expected ~fp noise at step0 '
            f'same-weights same-temperature. >0.05 => shift/temperature/BPE misalignment (product bug).')
    finally:
        engine.shutdown()
