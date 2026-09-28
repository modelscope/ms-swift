# Copyright (c) ModelScope Contributors. All rights reserved.
"""Reward-channel resolution, judging and device placement for ``run_infer`` (no GPU, no reward download).

``--reward_funcs`` (a callable ORM) is already covered end to end by ``test_pipeline_fast`` /
``test_backends_e2e``. This file covers the OTHER reward surface the ``math_prm_orm_ray.sh`` example and
its cousins ride: a MODEL reward channel (``--orm_model`` / ``--prm_model``), which ``run_infer`` resolves
into one of four kinds before it builds anything --

* ``api``: an http(s) endpoint (checked first, so a URL is never fed to ``get_model_info_meta``);
* ``scalar``: a seq_cls/reranker RM scored by a forward -- GPU-resident, so it needs its own twinkle
  DeviceGroup, which only ``mode='ray'`` can place (the guard ``test_scalar_reward_needs_ray`` pins);
* ``generative_reuse``: a causal_lm judge whose base IS the sampler's model, scored THROUGH that sampler;
* ``generative_independent``: a causal_lm judge on its own model/engine (also GPU-resident -> ray).

The decision is driven by the reward model's real ``task_type`` from ``get_model_info_meta``, so the
pure-function tests monkeypatch it (a seq_cls/causal_lm stand-in) rather than downloading a 7B PRM. The
headline ``test_generative_judge_scores_candidates_through_run_infer`` then drives the WHOLE chain with no
GPU: resolve -> ``_build_model_reward`` -> a real ``_GenerativeJudgeReward`` reusing a scripted sampler ->
``_judge_query`` render -> ``_parse_judge_score`` -> ``_RewardChannels.score`` -> ``_emit_all`` scores.
Only the sampler is scripted (it plays both the candidate generator and the reused judge, told apart by
the judge template's marker), so the verdicts -- and thus the emitted scores -- are exact.
"""
import pytest
from twinkle.data_format.sampling import SampledSequence, SampleResponse

from swift.dev.recipe.run_infer import (_exclusive_reward_groups, _judge_query, _parse_judge_score,
                                        _render_conversation, _resolve_reward_model,
                                        plan_sampling_device_groups)

MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
#: The judge template's candidate marker -- only a judge query carries it, so the reused sampler tells a
#: scoring call from a generation call by its presence.
_JUDGE_MARKER = '=== Candidate response ==='


def _patch_task_type(monkeypatch, task_type):
    """Stand in for ``swift.model.get_model_info_meta``: only ``task_type`` drives the reward-kind choice,
    so a seq_cls/causal_lm stand-in avoids resolving (and downloading) a real reward model."""
    from types import SimpleNamespace

    import swift.model
    monkeypatch.setattr(swift.model, 'get_model_info_meta', lambda value, *a, **k: (SimpleNamespace(
        task_type=task_type), None))


# --- _resolve_reward_model: the scalar / generative / api decision -----------------------


def test_resolve_url_is_an_api_judge_without_touching_model_meta():
    """An http(s) value is an API judge, decided BEFORE get_model_info_meta (which would try to download
    a URL as a model id). No meta patch here on purpose: calling it would prove the ordering wrong."""
    spec = _resolve_reward_model('https://judge.example/v1', None, MODEL, 'transformers')
    assert spec.kind == 'api' and spec.model_id == 'https://judge.example/v1'


@pytest.mark.parametrize('task_type', ['seq_cls', 'reranker'])
def test_resolve_scalar_task_types(monkeypatch, task_type):
    """A seq_cls/reranker RM scores by a forward, so it is a scalar channel (and GPU-resident)."""
    _patch_task_type(monkeypatch, task_type)
    spec = _resolve_reward_model('some-prm', None, MODEL, 'transformers')
    assert spec.kind == 'scalar' and spec.task_type == task_type


def test_resolve_generative_reuse_when_judge_is_the_samplers_own_local_model(monkeypatch):
    """A causal_lm judge whose base equals the sampler's model, on a local backend, REUSES that sampler
    (no second engine, no extra device) -- the only model reward a single-card run can serve."""
    _patch_task_type(monkeypatch, 'causal_lm')
    spec = _resolve_reward_model(MODEL, None, MODEL, 'transformers')
    assert spec.kind == 'generative_reuse' and spec.model_id == MODEL


def test_resolve_generative_independent_for_a_different_model(monkeypatch):
    """A causal_lm judge on its OWN model cannot reuse the sampler, so it builds a standalone engine."""
    _patch_task_type(monkeypatch, 'causal_lm')
    spec = _resolve_reward_model('other-judge', None, MODEL, 'transformers')
    assert spec.kind == 'generative_independent' and spec.model_id == 'other-judge'


def test_resolve_client_backend_never_reuses_the_sampler(monkeypatch):
    """With backend='client' there is no local sampler to reuse, so even the same model id builds an
    independent judge rather than claiming a reuse that has no engine behind it."""
    _patch_task_type(monkeypatch, 'causal_lm')
    spec = _resolve_reward_model(MODEL, None, MODEL, 'client')
    assert spec.kind == 'generative_independent'


def test_resolve_adapter_only_reuses_the_samplers_model(monkeypatch):
    """``*_adapter`` with no ``*_model`` means "the sampler's own model as a generative judge, plus this
    reward LoRA"; on a client backend there is no local sampler to hang the LoRA on, so it is rejected."""
    spec = _resolve_reward_model(None, 'lora-x', MODEL, 'transformers')
    assert spec.kind == 'generative_reuse' and spec.model_id == MODEL and spec.adapter == 'lora-x'
    with pytest.raises(ValueError, match='loads no local'):
        _resolve_reward_model(None, 'lora-x', MODEL, 'client')


def test_resolve_no_model_no_adapter_is_no_channel():
    assert _resolve_reward_model(None, None, MODEL, 'transformers') is None


def test_resolve_unknown_task_type_raises(monkeypatch):
    """A task_type that is neither scalar nor generative cannot be served as a reward -- say so rather
    than silently picking a kind."""
    _patch_task_type(monkeypatch, 'embedding')
    with pytest.raises(ValueError, match='neither a scalar'):
        _resolve_reward_model('some-embed-model', None, MODEL, 'transformers')


# --- judge prompt rendering + score parsing ---------------------------------------------


def test_render_conversation_flattens_role_content_lines():
    rendered = _render_conversation([{
        'role': 'user',
        'content': 'hi'
    }, {
        'role': 'assistant',
        'content': 'hello'
    }])
    assert rendered == 'User: hi\nAssistant: hello'


def test_judge_query_strips_the_candidate_turn_and_fills_the_template():
    """The candidate arrives as the trailing assistant turn; ``_judge_query`` strips it so ``{prompt}`` is
    the conversation alone and ``{completion}`` is the candidate verbatim."""
    query = _judge_query('P=[{prompt}] C=[{completion}]', [{
        'role': 'user',
        'content': 'Q'
    }, {
        'role': 'assistant',
        'content': 'A'
    }], 'A')
    assert query == 'P=[User: Q] C=[A]'


@pytest.mark.parametrize(
    'text,expected',
    [('Reward: 0.85', 0.85), ('reward: 1', 1.0), ('analysis...\nReward: 0.0', 0.0), ('bare 0.7 here', 0.7),
     ('no verdict at all', None), ('', None), (None, None)])
def test_parse_judge_score(text, expected):
    """Prefers the ``Reward: <score>`` convention, falls back to the first bare number, and returns None
    (recorded as nan, never a guessed 0) when nothing numeric is present."""
    assert _parse_judge_score(text) == expected


# --- ray device placement for a GPU-resident reward -------------------------------------


def test_exclusive_reward_groups_names_only_gpu_resident_channels():
    """A scalar PRM needs its own DeviceGroup; a generative_reuse ORM and an API judge do not."""
    from swift.dev.recipe.run_infer import _RewardModelSpec
    scalar_prm = _RewardModelSpec('scalar', model_id='prm', task_type='seq_cls')
    reuse_orm = _RewardModelSpec('generative_reuse', model_id=MODEL, task_type='causal_lm')
    assert _exclusive_reward_groups(reuse_orm, scalar_prm) == ['prm']
    assert _exclusive_reward_groups(reuse_orm, None) == []


def test_plan_sampling_device_groups_gives_each_role_a_disjoint_block():
    """``nproc_per_node`` sizes ONE role; the sampler and each GPU-resident reward get their own block, so
    a run with R reward groups needs ``ranks * (1 + R)`` GPUs (the ``math_prm_orm_ray.sh`` placement)."""
    groups, total = plan_sampling_device_groups(2, ['prm'])
    assert total == 4
    assert groups[0][1] == [0, 1]  # the sampler role
    assert groups[1] == ('prm', [2, 3])  # the scalar PRM's own disjoint group

    groups, total = plan_sampling_device_groups(2, ['orm', 'prm'])
    assert total == 6
    assert groups[1] == ('orm', [2, 3]) and groups[2] == ('prm', [4, 5])

    with pytest.raises(ValueError, match='ranks_per_group'):
        plan_sampling_device_groups(None, ['prm'])


def test_scalar_reward_needs_ray(monkeypatch):
    """The guard ``math_prm_orm_ray.sh`` satisfies with ``--mode ray``: a scalar PRM keeps its own model
    resident on GPU, which only a ray DeviceGroup can place, so a non-ray run is rejected up front --
    before any dataset load, sampler build or model download."""
    from swift.dev.config import (DatasetConfig, GenerationConfig, InferConfig, ModelConfig,
                                  TemplateConfig)
    from swift.dev.recipe.run_infer import run_infer

    _patch_task_type(monkeypatch, 'seq_cls')
    infer_config = InferConfig(prm_model='Qwen/Qwen2.5-Math-PRM-7B', num_return_sequences=1,
                               output_format='all')
    with pytest.raises(ValueError, match="mode='ray'"):
        run_infer(
            ModelConfig(task_type='causal_lm'), TemplateConfig(), DatasetConfig(), infer_config,
            GenerationConfig(), backend='transformers', distributed_config=None)


# --- the full generative-judge chain, end to end, no GPU --------------------------------


@pytest.fixture(scope='module')
def template():
    """The REAL dev template over the cached tokenizer (``load_model=False``), so the scripted sampler
    encodes through production code. Skips (not fails) on a box without the tokenizer cached."""
    from swift.dev.builders.template import build_template
    from swift.dev.config import TemplateConfig
    from swift.model import get_model_processor
    try:
        _, proc = get_model_processor(MODEL, load_model=False)
    except Exception as exc:  # noqa: BLE001 -- an unfetchable tokenizer is not a reward-channel defect
        pytest.skip(f'{MODEL}: tokenizer not fetchable ({type(exc).__name__})')
    return build_template(TemplateConfig(template='qwen2_5', max_length=2048), proc)


class _ScriptedJudgeSampler:
    """One sampler playing both roles a ``generative_reuse`` judge needs.

    For a generation call it returns ``num_samples`` candidate sequences; for a judge call (recognized by
    the judge template's ``=== Candidate response ===`` marker, which embeds the completion being scored)
    it returns a deterministic ``Reward: <score>`` verdict for that completion. Building both through the
    real template's tokenizer keeps ``samples_from_responses`` / ``sampled_texts`` on their production
    shapes. Stateless, so the concurrent group sampling cannot race it.
    """

    def __init__(self, template, answers, verdicts):
        self.template = template
        self.answers = list(answers)
        self.verdicts = dict(verdicts)  # completion text -> score

    def _messages(self, traj):
        return (traj.get('messages') if isinstance(traj, dict) else getattr(traj, 'messages', None)) or []

    def sample(self, trajectories, params=None, **kwargs):
        responses = []
        for traj in trajectories:
            messages = self._messages(traj)
            content = messages[-1].get('content', '') if messages else ''
            if _JUDGE_MARKER in content:
                verdict = next((f'Reward: {score}' for text, score in self.verdicts.items() if text in content),
                               'Reward: 0')
                seq = SampledSequence(stop_reason='stop', tokens=[0], decoded=verdict)
                responses.append(SampleResponse(sequences=[seq], prompt_token_ids=[1]))
            else:
                count = getattr(params, 'num_samples', None) or len(self.answers)
                prompt_ids = self.template.tokenizer.encode('prompt', add_special_tokens=False) or [1]
                seqs = []
                for i in range(count):
                    text = self.answers[i % len(self.answers)]
                    tokens = self.template.tokenizer.encode(text, add_special_tokens=False)
                    seqs.append(SampledSequence(stop_reason='stop', tokens=tokens, decoded=text))
                responses.append(SampleResponse(sequences=seqs, prompt_token_ids=prompt_ids))
        return responses

    def shutdown(self):
        pass

    def close(self):
        pass


def test_generative_judge_scores_candidates_through_run_infer(template, monkeypatch, tmp_path):
    """The end-to-end model-reward chain on one card with no download: ``--orm_model`` equal to the
    sampler's model resolves to ``generative_reuse``, so the judge scores THROUGH the (scripted) sampler,
    its ``Reward: <score>`` verdicts are parsed, weighted by ``orm_channel_weight`` (1.0, unnormalized) and
    emitted per candidate. Asserts the exact scores, proving resolve -> build -> judge -> channels -> emit
    are wired, not merely that a ``scores`` key appeared."""
    import swift.dev.builders as builders
    from swift.dev.config import (DatasetConfig, GenerationConfig, InferConfig, ModelConfig, RLHFConfig,
                                  TemplateConfig)
    from swift.dev.recipe.run_infer import run_infer

    _patch_task_type(monkeypatch, 'causal_lm')  # the ORM model is a causal_lm -> a generative judge
    answers = ['good answer', 'bad answer']
    verdicts = {'good answer': 0.9, 'bad answer': 0.2}
    monkeypatch.setattr(builders, 'load_prompt_rows',
                        lambda *a, **k: [{'messages': [{
                            'role': 'user',
                            'content': 'Say something.'
                        }]}])
    monkeypatch.setattr(
        builders, 'build_sampler',
        lambda model_config, *, template=None, **kwargs: _ScriptedJudgeSampler(template, answers, verdicts))

    rows = run_infer(
        ModelConfig(model=MODEL, task_type='causal_lm'),
        TemplateConfig(template='qwen2_5', max_length=2048),
        DatasetConfig(),
        # orm_model == the sampler's model -> generative_reuse (no ray, no second engine).
        InferConfig(orm_model=MODEL, num_return_sequences=2, batch_size=1, output_format='all'),
        GenerationConfig(max_new_tokens=16, temperature=0.8),
        rlhf_config=RLHFConfig(),
        backend='transformers',
        distributed_config=None,
        split_dataset_ratio=0.0,
        output_path=str(tmp_path / 'judged.jsonl'))

    assert len(rows) == 1
    row = rows[0]
    assert row['responses'] == answers  # the two candidates the scripted sampler generated
    # The judge really scored each candidate: good -> 0.9, bad -> 0.2, in candidate order. (``weight_rewards``
    # rounds through float32, so compare approximately.)
    assert row['scores'] == pytest.approx([0.9, 0.2])
