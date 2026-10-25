# Copyright (c) ModelScope Contributors. All rights reserved.
"""S7 rows 11 / 12 (L1): the distillation assembly's dispatch surface -- signal forms, teacher
group planning, the ``_build_teachers`` guards, and the self-teacher derivation.

Cheap tier (no Ray / vLLM / GPU, mirroring ``test_recipes.py``): these are the pure decision
functions the cluster build consults BEFORE any actor exists, so a stub policy sentinel and a
capturing ``build_ref_model`` seam are the faithful inputs -- the e2e tier
(``test_algorithms_e2e.py`` gkd/opsd/mopd + ``test_distill_teacher_e2e.py``) proves the resolved
shapes really train.

Row 11 pins the three signal forms (``DISTILL_SIGNALS``) and everything keyed off them: which
DeviceGroups get planned (``distill_group_requests``) and which teacher shape gets built. Row 12
pins the teacher resolution itself: the adapter-disabled base (``return [model], disable_lora``),
the dynamic self teacher (OPSD's default), the separate frozen teacher (build_ref_model kwargs),
MOPD's K teachers (pack vs spread, weights), every guard that must refuse an ambiguous combo, and
``_derive_rlhf_teacher`` -- the config-time rule that turns ``teacher_model == [the student]`` into
the disable-adapter form.
"""
import pytest

from swift.dev.config import ModelConfig, RLConfig, TunerConfig
from swift.dev.recipe.distill.assembly import DISTILL_SIGNALS, _build_teachers, distill_group_requests

# --- row 11: signal forms ---------------------------------------------------------------------


def test_distill_signals_map_exactly_the_three_distill_types():
    """Each distill rl_type speaks exactly one teacher signal form; no other type is a distiller.

    Hand oracle: gkd scores full-vocab teacher LOGITS (generalized JSD over the vocabulary),
    opsd scores response-only teacher LOGPS on a privileged prompt, mopd fuses K per-teacher
    MULTI_LOGPS channels. A silent remap (e.g. gkd->logps) would build the wrong worker and the
    wrong teacher shape downstream.
    """
    assert DISTILL_SIGNALS == {'gkd': 'logits', 'opsd': 'logps', 'mopd': 'multi_logps'}


@pytest.mark.parametrize('rl_type,extra,expected', [
    # separate frozen teacher: ONE dedicated group carrying the parallel spec through
    ('gkd', {'teacher_model': ['t'], 'teacher_parallel_spec': 'tp2'}, [('teacher', 'tp2')]),
    ('opsd', {'teacher_model': ['t']}, [('teacher', None)]),
    # self forms plan NO group (the policy actor is the teacher)
    ('opsd', {}, []),
    ('gkd', {'_teacher_use_disable_adapter': True}, []),
    # MOPD pack: K teachers share ONE group; spread: K named groups
    ('mopd', {'teacher_model': ['a', 'b']}, [('teacher', None)]),
    ('mopd', {
        'teacher_model': ['a', 'b', 'c'], 'auxiliary_placement': 'spread'
    }, [('teacher_0', None), ('teacher_1', None), ('teacher_2', None)]),
])
def test_distill_group_requests_plans_groups_per_teacher_shape(rl_type, extra, expected):
    assert distill_group_requests(RLConfig(rl_type=rl_type, **extra)) == expected


def _stub_teachers(monkeypatch):
    """Capture build_ref_model calls; return a per-call sentinel instead of a Ray actor."""
    import swift.dev.builders

    calls = []

    def fake_build_ref_model(**kwargs):
        calls.append(kwargs)
        return f"ref[{kwargs['model_id']}#{kwargs['instance_id'] if 'instance_id' in kwargs else ''}]"

    monkeypatch.setattr(swift.dev.builders, 'build_ref_model', fake_build_ref_model)
    return calls


def _build(rl_config, model='POLICY', tuner='lora'):
    return _build_teachers(
        model=model,
        model_config=ModelConfig(model='student'),
        template=None,
        rl_config=rl_config,
        tuner_config=(TunerConfig(tuner=tuner) if tuner else None),
        distributed_config=None,
        signal=DISTILL_SIGNALS[rl_config.rl_type],
        auxiliary_meshes=None,
    )


@pytest.mark.parametrize('rl_type,extra,expected_options', [
    # adapter-disabled base: the teacher IS the policy actor, forwarded with the adapter off
    ('gkd', {'_teacher_use_disable_adapter': True}, {'disable_lora': True}),
    # OPSD default (no teacher_model): dynamic self-distillation on the policy, adapters ON
    ('opsd', {}, {}),
])
def test_build_teachers_self_forms_reuse_the_policy_actor(rl_type, extra, expected_options, monkeypatch):
    calls = _stub_teachers(monkeypatch)
    teachers, options = _build(RLConfig(rl_type=rl_type, **extra))
    assert teachers == ['POLICY'], 'a self teacher must BE the policy actor, not a new build'
    assert options == expected_options
    assert not calls, 'no separate teacher actor may be built for a self form'


def test_build_teachers_separate_frozen_teacher_passes_spec_and_adapters(monkeypatch):
    """The separate-teacher branch builds ONE frozen ref model on the shared 'teacher' group,
    threading the parallel spec, adapters, type/revision and deepspeed knobs through verbatim."""
    calls = _stub_teachers(monkeypatch)
    cfg = RLConfig(
        rl_type='gkd',
        teacher_model=['teacher-id'],
        teacher_adapters=['ad1'],
        teacher_parallel_spec='tp2',
        teacher_model_type='qwen2',
        teacher_model_revision='v1',
    )
    teachers, options = _build(cfg)
    assert len(calls) == 1 and teachers == [f"ref[teacher-id#]"]
    call = calls[0]
    assert call['name'] == 'teacher' and call['model_id'] == 'teacher-id'
    assert call['adapters'] == ('ad1', ) and call['parallel_spec'] == 'tp2'
    assert call['model_type'] == 'qwen2' and call['model_revision'] == 'v1'
    assert call['policy_model'] == 'POLICY'
    assert options == {'offload': False}


def test_build_teachers_gkd_offload_moves_teacher_to_cpu(monkeypatch):
    """offload_teacher_model (GKD-only) parks the frozen teacher on CPU between forwards."""
    import swift.dev.builders

    offloaded = []

    class _Ref:

        def offload_to_cpu(self):
            offloaded.append(self)

    monkeypatch.setattr(swift.dev.builders, 'build_ref_model', lambda **kw: _Ref())
    teachers, options = _build(RLConfig(rl_type='gkd', teacher_model=['t'], offload_teacher_model=True))
    assert options == {'offload': True}
    assert offloaded == teachers, 'the built teacher must be the one offloaded'


def test_build_teachers_mopd_builds_k_weighted_teachers(monkeypatch):
    """MOPD builds K distinct ref models (per-teacher instance_id) and threads weights+placements:
    pack shares ONE group (serial scoring), so every placement is 'teacher'."""
    calls = _stub_teachers(monkeypatch)
    cfg = RLConfig(rl_type='mopd', teacher_model=['a', 'b'], teacher_weights=[0.6, 0.4])
    teachers, options = _build(cfg)
    assert teachers == ['ref[a#teacher0]', 'ref[b#teacher1]']
    assert [c['instance_id'] for c in calls] == ['teacher0', 'teacher1']
    assert options == {'placements': ['teacher', 'teacher'], 'teacher_weights': [0.6, 0.4]}

    calls.clear()
    cfg = RLConfig(rl_type='mopd', teacher_model=['a', 'b'], auxiliary_placement='spread')
    teachers, options = _build(cfg)
    assert [c['name'] for c in calls] == ['teacher_0', 'teacher_1']
    assert options['placements'] == ['teacher_0', 'teacher_1']
    assert options['teacher_weights'] is None


# --- row 12: the guards (ambiguous / infeasible teacher combos must refuse) -------------------


@pytest.mark.parametrize('rl_type,extra,tuner,match', [
    ('mopd', {'teacher_model': ['s'], '_teacher_use_disable_adapter': True}, 'lora', 'does not apply here'),
    ('mopd', {'teacher_model': ['s'], 'teacher_adapters': ['a']}, 'lora', 'distinct full model'),
    ('mopd', {'teacher_model': ['s'], 'offload_teacher_model': True}, 'lora', 'not supported for MOPD'),
    ('mopd', {}, 'lora', 'at least one teacher'),
    ('gkd', {'offload_teacher_model': True, '_teacher_use_disable_adapter': True}, 'lora', 'distinct frozen teacher'),
    ('opsd', {'teacher_model': ['t'], 'offload_teacher_model': True}, 'lora', 'GKD-only'),
    ('gkd', {'_teacher_use_disable_adapter': True}, None, 'requires a LoRA student'),
    ('gkd', {}, 'lora', 'GKD needs a teacher'),
])
def test_build_teachers_refuses_ambiguous_combos(rl_type, extra, tuner, match, monkeypatch):
    calls = _stub_teachers(monkeypatch)
    with pytest.raises(ValueError, match=match):
        _build(RLConfig(rl_type=rl_type, **extra), tuner=tuner)
    assert not calls, 'the guard must fire BEFORE any teacher actor is built'


def test_build_teachers_mopd_rejects_weights_count_mismatch(monkeypatch):
    _stub_teachers(monkeypatch)
    with pytest.raises(ValueError, match='2 entries but 3 teachers'):
        _build(RLConfig(rl_type='mopd', teacher_model=['a', 'b', 'c'], teacher_weights=[0.5, 0.5]))


def test_single_teacher_id_refuses_a_list_for_single_teacher_algos():
    """GKD/OPSD take exactly one teacher; a K-list must point at MOPD instead of silently using [0]."""
    from swift.dev.recipe.utils import single_teacher_id
    assert single_teacher_id(None, algo='distill') is None
    assert single_teacher_id(['only'], algo='distill') == 'only'
    with pytest.raises(ValueError, match='mopd'):
        single_teacher_id(['a', 'b'], algo='distill')


# --- row 12: the config-time self-teacher derivation -------------------------------------------


def _derive(model='student-id', tuner='lora', **rl_extra):
    from swift.dev.config.process import _derive_rlhf_teacher
    model_config = ModelConfig(model=model)
    tuner_config = TunerConfig(tuner=tuner) if tuner else None
    rl_config = RLConfig(rl_type=rl_extra.pop('rl_type', 'gkd'), **rl_extra)
    _derive_rlhf_teacher(model_config, tuner_config, rl_config)
    return rl_config


def test_derive_teacher_self_reference_becomes_disable_adapter():
    """teacher_model == [the student] + LoRA + no adapters IS the adapter-disabled base: the
    derivation flips the private flag and CLEARS teacher_model so no second copy gets planned."""
    cfg = _derive(teacher_model=['student-id'])
    assert cfg._teacher_use_disable_adapter is True
    assert cfg.teacher_model is None


def test_derive_teacher_distinct_or_full_param_keeps_the_separate_teacher():
    # a genuinely different teacher stays separate
    cfg = _derive(teacher_model=['other-id'])
    assert (cfg._teacher_use_disable_adapter, cfg.teacher_model) == (False, ['other-id'])
    # full-param student has no adapter to disable: the same-id teacher stays an explicit frozen copy
    cfg = _derive(tuner=None, teacher_model=['student-id'])
    assert (cfg._teacher_use_disable_adapter, cfg.teacher_model) == (False, ['student-id'])
    # an adaptered teacher is a distinct weight set, not the base
    cfg = _derive(teacher_model=['student-id'], teacher_adapters=['ad'])
    assert (cfg._teacher_use_disable_adapter, cfg.teacher_model) == (False, ['student-id'])


def test_derive_teacher_coerces_scalar_and_normalizes_adapters():
    """A bare-string teacher_model/teacher_adapters (argv shape) is lifted to the list form."""
    cfg = _derive(teacher_model='other-id', teacher_adapters='ad')
    assert cfg.teacher_model == ['other-id'] and cfg.teacher_adapters == ['ad']
    # the string form of the SELF reference derives identically to the list form
    cfg = _derive(teacher_model='student-id')
    assert cfg._teacher_use_disable_adapter is True and cfg.teacher_model is None
