"""Guards for the full-parameter freeze / trainable wiring (freeze_parameters + trainable_parameters).

Three seams are covered, each without loading weights or touching a GPU:

  - twinkle's ``_freeze_then_activate`` -- the pure freeze/activate core: cumulative-element-count
    ratio, name-prefix, regex, and the load-bearing "trainable_* wins over freeze_*" ordering.
  - dev's ``apply_full_param_freeze`` -- the forwarding + no-op gate that hands TrainConfig's five
    knobs to the model's remote ``freeze_parameters`` seam (and stays silent when none is set, so an
    ordinary full run pays no remote round-trip).
  - ``validate._check_freeze_ratio_pp`` -- the config-only rejection of ``freeze_parameters_ratio``
    under Megatron PP>1, where a per-rank element-count fraction selects a different slice per stage.

The remote seam itself (``TrainableModel.freeze_parameters``, ``dispatch='all'``) is a thin
unwrap-and-delegate whose logic is exactly ``_freeze_then_activate``; exercising the Ray decorator
plumbing needs a live cluster, so it is covered by the megatron e2e suite instead of here.
"""
import pytest
import torch


def _params(*specs):
    """A ``[(name, param)]`` list of fresh all-trainable CPU parameters with the given element counts.

    A freshly built model is all-trainable (requires_grad=True) except what its loader froze, so
    starting every param at True mirrors the state ``freeze_parameters`` runs against.
    """
    return [(name, torch.nn.Parameter(torch.zeros(numel))) for name, numel in specs]


def _trainable(named):
    return {name: param.requires_grad for name, param in named}


# --- twinkle pure core: _freeze_then_activate -------------------------------------------------------


def test_no_knobs_leaves_everything_trainable():
    from twinkle.model.base import _freeze_then_activate

    named = _params(('a.weight', 10), ('b.weight', 10))
    _freeze_then_activate(named)
    assert _trainable(named) == {'a.weight': True, 'b.weight': True}


def test_freeze_names_matches_by_prefix():
    from twinkle.model.base import _freeze_then_activate

    named = _params(('model.layers.0.mlp.weight', 4), ('model.layers.0.attn.weight', 4), ('lm_head.weight', 4))
    _freeze_then_activate(named, freeze_names=('model.layers.0.mlp', 'lm_head'))
    assert _trainable(named) == {
        'model.layers.0.mlp.weight': False,
        'model.layers.0.attn.weight': True,
        'lm_head.weight': False,
    }


def test_freeze_regex_matches_by_search():
    from twinkle.model.base import _freeze_then_activate

    named = _params(('model.layers.0.mlp.weight', 4), ('model.layers.1.mlp.weight', 4), ('model.norm.weight', 4))
    _freeze_then_activate(named, freeze_regex=r'layers\.\d+\.mlp')
    assert _trainable(named) == {
        'model.layers.0.mlp.weight': False,
        'model.layers.1.mlp.weight': False,
        'model.norm.weight': True,
    }


def test_freeze_ratio_freezes_leading_fraction_by_element_count():
    from twinkle.model.base import _freeze_then_activate

    # Sizes [10, 20, 30, 40] -> cumsum [10, 30, 60, 100], total 100. ratio 0.5 => freeze 50 elements
    # => bisect_right(cumsum, 50) == 2 => the first two params (10 + 20 = 30 elements) are frozen.
    named = _params(('p0', 10), ('p1', 20), ('p2', 30), ('p3', 40))
    _freeze_then_activate(named, freeze_ratio=0.5)
    assert _trainable(named) == {'p0': False, 'p1': False, 'p2': True, 'p3': True}


def test_freeze_ratio_one_freezes_all():
    from twinkle.model.base import _freeze_then_activate

    named = _params(('p0', 10), ('p1', 20))
    _freeze_then_activate(named, freeze_ratio=1.0)
    assert _trainable(named) == {'p0': False, 'p1': False}


def test_trainable_wins_over_freeze():
    from twinkle.model.base import _freeze_then_activate

    # Freeze everything by regex, then re-enable one prefix and one regex match: trainable_* is applied
    # AFTER freeze_*, so the re-enabled params end up trainable even though a freeze rule matched them.
    named = _params(('model.layers.0.mlp.weight', 4), ('model.embed.weight', 4), ('model.norm.weight', 4))
    _freeze_then_activate(
        named,
        freeze_regex=r'.*',
        trainable_names=('model.embed', ),
        trainable_regex=r'norm',
    )
    assert _trainable(named) == {
        'model.layers.0.mlp.weight': False,
        'model.embed.weight': True,
        'model.norm.weight': True,
    }


def test_invalid_freeze_regex_raises():
    from twinkle.model.base import _freeze_then_activate

    named = _params(('p0', 4))
    # A deliberately unbalanced group. dev fails loudly rather than legacy's warn-and-skip, so a typo
    # cannot silently freeze nothing.
    with pytest.raises(Exception):
        _freeze_then_activate(named, freeze_regex='(unclosed')


# --- dev forwarding: apply_full_param_freeze --------------------------------------------------------


class _RecordingModel:

    def __init__(self):
        self.calls = []

    def freeze_parameters(self, **kwargs):
        self.calls.append(kwargs)


def _apply(**cfg_kwargs):
    from swift.dev.builders import apply_full_param_freeze
    from swift.dev.config import TrainConfig

    model = _RecordingModel()
    apply_full_param_freeze(model, TrainConfig(**cfg_kwargs))
    return model.calls


def test_apply_is_a_noop_when_no_knob_set():
    # An ordinary full run sets none of the five, so no remote round-trip is issued at all.
    assert _apply() == []


def test_apply_is_a_noop_for_none_train_config():
    from swift.dev.builders import apply_full_param_freeze

    model = _RecordingModel()
    apply_full_param_freeze(model, None)
    assert model.calls == []


def test_apply_forwards_every_knob():
    calls = _apply(
        freeze_parameters=['a', 'b'],
        freeze_parameters_regex=r'mlp',
        freeze_parameters_ratio=0.25,
        trainable_parameters=['c'],
        trainable_parameters_regex=r'norm',
    )
    assert calls == [{
        'freeze_ratio': 0.25,
        'freeze_names': ['a', 'b'],
        'freeze_regex': r'mlp',
        'trainable_names': ['c'],
        'trainable_regex': r'norm',
    }]


def test_apply_forwards_ratio_only_with_empty_defaults():
    calls = _apply(freeze_parameters_ratio=0.5)
    assert calls == [{
        'freeze_ratio': 0.5,
        'freeze_names': [],
        'freeze_regex': None,
        'trainable_names': [],
        'trainable_regex': None,
    }]


# --- config guard: _check_freeze_ratio_pp -----------------------------------------------------------


def _check_ratio_pp(*, ratio, pp, is_megatron, tuner):
    from swift.dev.config import DistributedConfig, TrainConfig, TunerConfig
    from swift.dev.config.validate import _check_freeze_ratio_pp

    train_config = TrainConfig(freeze_parameters_ratio=ratio)
    distributed_config = DistributedConfig(pipeline_model_parallel_size=pp)
    tuner_config = None if tuner == 'full' else TunerConfig(tuner=tuner)
    _check_freeze_ratio_pp(train_config, distributed_config, is_megatron, tuner_config)


def test_ratio_with_megatron_pp_raises():
    with pytest.raises(ValueError, match='freeze_parameters_ratio'):
        _check_ratio_pp(ratio=0.3, pp=2, is_megatron=True, tuner='full')


def test_ratio_with_megatron_pp1_is_allowed():
    _check_ratio_pp(ratio=0.3, pp=1, is_megatron=True, tuner='full')


def test_ratio_with_megatron_pp_under_lora_is_allowed():
    # The ratio is only consumed on the full-parameter path; an adapter run ignores it, so there is
    # nothing broken to reject.
    _check_ratio_pp(ratio=0.3, pp=4, is_megatron=True, tuner='lora')


def test_ratio_without_megatron_is_allowed():
    _check_ratio_pp(ratio=0.3, pp=2, is_megatron=False, tuner='full')


def test_no_ratio_with_megatron_pp_is_allowed():
    _check_ratio_pp(ratio=0.0, pp=4, is_megatron=True, tuner='full')
