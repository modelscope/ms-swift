# Copyright (c) ModelScope Contributors. All rights reserved.
"""S13-27: the MegatronConfig CLI surface reaches mcore under its own name (v4 test_megatron_args).

Every field of ``MegatronConfig`` is a CLI knob (``--apply_rope_fusion true`` & co); ``_megatron_model_
kwargs`` forwards each non-None field into the model kwargs the bridge builds its TransformerConfig
from, renaming only where mcore's name differs (``te_rng_tracker`` -> ``use_te_rng_tracker``). A knob
that stopped arriving here is dropped silently -- the bridge falls back to its own default and the
user's flag reads as accepted while changing nothing, which is exactly what this table pins out.
"""
import dataclasses

import pytest

from swift.dev.builders.model import (
    _MEGATRON_FIELD_ALIASES,
    _NON_MODEL_MEGATRON_FIELDS,
    _megatron_model_kwargs,
)
from swift.dev.config import MegatronConfig

#: One non-default value per field, in the field's own type (bools flip, the rest pick a value).
_NON_DEFAULT = {
    'cross_entropy_fusion_impl': 'flattened',
    'dsa_indexer_loss_coeff': 0.01,
    'dsa_indexer_use_sparse_loss': True,
    'mhc_recompute_layer_num': 2,
    'csa_dense_mode': True,
    'manual_gc_steps': 7,
    'mlp_padding_free': True,
    'sequence_packing_scheduler': True,
}


def _flip(field):
    """A non-default value for one field: flip bools, look the rest up, ints step off the default."""
    if field.name in _NON_DEFAULT:
        return _NON_DEFAULT[field.name]
    default = field.default if field.default is not dataclasses.MISSING else field.default_factory()
    if isinstance(default, bool):
        return not default
    if isinstance(default, int):
        return default + 1
    if default is None:
        return True  # an Optional knob switches on
    raise AssertionError(f'no non-default value pinned for field {field.name!r} (default {default!r})')


#: Knobs consumed OUTSIDE the model kwargs (loop-side GC flags, init flags, the extra-kwargs dict);
#: each has its own consumer below, so the table test skips them.
_LOOP_SIDE_FIELDS = _NON_MODEL_MEGATRON_FIELDS | {'apply_query_key_layer_scaling'}


@pytest.mark.parametrize('field', [f for f in dataclasses.fields(MegatronConfig)
                                   if f.name not in _LOOP_SIDE_FIELDS], ids=lambda f: f.name)
def test_every_megatron_config_knob_reaches_the_bridge_under_its_own_name(field):
    """Set ONE knob off its default; the kwargs the bridge receives carries exactly that change
    under the mcore-side name (the alias table where the two surfaces disagree)."""
    config = MegatronConfig(**{field.name: _flip(field)})
    kwargs = _megatron_model_kwargs(config, None)
    target = _MEGATRON_FIELD_ALIASES.get(field.name, field.name)
    assert kwargs[target] == getattr(config, field.name), (
        f'--{field.name} did not reach the bridge kwargs as {target!r}')


def test_loop_side_knobs_stay_out_of_the_model_kwargs():
    """manual_gc* / skip_megatron_init / data_parallel_random_init are consumed by the training
    loop or the loader, not the TransformerConfig -- landing them in the model kwargs would be a
    TypeError at bridge build time (no such ModelConfig field)."""
    kwargs = _megatron_model_kwargs(
        MegatronConfig(manual_gc=True, manual_gc_eval=True, manual_gc_steps=5,
                       skip_megatron_init=True, data_parallel_random_init=True), None)
    assert not (_NON_MODEL_MEGATRON_FIELDS & set(kwargs))


def test_megatron_extra_kwargs_flatten_into_the_model_kwargs():
    """The escape hatch: arbitrary bridge knobs ride megatron_extra_kwargs verbatim (and always as
    strict -- the user typed them, so a bridge default must never swallow them)."""
    kwargs = _megatron_model_kwargs(MegatronConfig(megatron_extra_kwargs={'custom_bridge_flag': 3}), None)
    assert kwargs.get('custom_bridge_flag') == 3
    assert 'megatron_extra_kwargs' not in kwargs
    assert 'custom_bridge_flag' in kwargs['_strict_model_kwargs']
    # a conflict with a same-named, explicitly set config field is refused, not silently overwritten
    # (a None config field has no value to conflict with -- the extra kwarg just fills it)
    with pytest.raises(ValueError, match='conflicts'):
        # the extra key collides with the ALIASED name the config field landed under
        _megatron_model_kwargs(MegatronConfig(te_rng_tracker=True,
                                              megatron_extra_kwargs={'use_te_rng_tracker': False}), None)


def test_the_alias_table_is_exhaustive_for_renamed_knobs():
    """The only renamed knob is te_rng_tracker; anything else in the alias table is a surprise that
    this test should force a human to look at (and probably update the docstring above)."""
    assert _MEGATRON_FIELD_ALIASES == {'te_rng_tracker': 'use_te_rng_tracker'}


def test_default_config_contributes_no_kwargs():
    """An untouched MegatronConfig forwards exactly its non-None non-loop fields, nothing more: a
    spurious key would shadow a bridge default the user never asked to change."""
    kwargs = _megatron_model_kwargs(MegatronConfig(), None)
    expected = {f.name for f in dataclasses.fields(MegatronConfig)
                if f.name not in _NON_MODEL_MEGATRON_FIELDS
                and getattr(MegatronConfig(), f.name) is not None}
    expected = {_MEGATRON_FIELD_ALIASES.get(name, name) for name in expected}
    assert set(kwargs) == expected
