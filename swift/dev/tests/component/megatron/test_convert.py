# Copyright (c) ModelScope Contributors. All rights reserved.
"""Fast tests for the ``run_convert`` recipe (HF <-> Megatron/mcore weight conversion).

``run_convert`` is the dev counterpart of legacy ``swift export --to_mcore / --to_hf``. The weight work
it delegates to ``swift.megatron`` needs a real GPU + mcore-bridge and is covered by the ``@slow``
round-trip in this file; here the recipe's OWN contract is pinned offline -- the direction guards, the
three-way dispatch (which of ``_convert_hf2mcore`` / ``_convert_mcore`` runs for a given ConvertConfig),
and the ``output_dir`` resolution order. These are pure control-flow decisions made before any megatron
import, so they are asserted by stubbing the two private workers and observing which one is reached.

The CLI surface that drives this recipe (``--to_mcore`` / ``--to_hf`` -> ``run_convert``) is pinned in
``component/config/test_cli_entrypoints.py``.
"""
import pytest

from swift.dev.config import CheckpointConfig, ConvertConfig, ModelConfig
from swift.dev.recipe.convert import run_convert


def _model(model='m'):
    return ModelConfig(model=model)


# --- direction guards -----------------------------------------------------------------


def test_both_directions_are_rejected():
    """``to_mcore`` and ``to_hf`` name opposite directions; setting both is a command error, refused
    before any weight work rather than silently picking one."""
    with pytest.raises(ValueError, match='mutually exclusive'):
        run_convert(_model(), ConvertConfig(to_mcore=True, to_hf=True))


def test_neither_direction_is_rejected():
    """With neither flag there is nothing to do -- the recipe says so instead of no-op'ing."""
    with pytest.raises(ValueError, match='Set either'):
        run_convert(_model(), ConvertConfig())


def test_model_is_required():
    """Both directions need the HF model/config to define the target architecture, so a missing
    ``model`` is refused up front with the reason, not discovered deep inside the bridge."""
    with pytest.raises(ValueError, match='model is required'):
        run_convert(ModelConfig(model=None), ConvertConfig(to_mcore=True))


# --- three-way dispatch ---------------------------------------------------------------


@pytest.fixture
def spy_workers(monkeypatch):
    """Stub both private conversion workers, recording which one ``run_convert`` routes to.

    The real workers import ``swift.megatron`` and load weights; the routing decision under test is
    made entirely from ConvertConfig fields before that, so stubbing them isolates the branch logic.
    """
    calls = {}

    def _mcore(*args, **kwargs):
        calls['which'] = 'mcore'
        calls['output_dir'] = kwargs.get('output_dir')
        return 'MCORE'

    def _hf2mcore(*args, **kwargs):
        calls['which'] = 'hf2mcore'
        calls['output_dir'] = kwargs.get('output_dir')
        return 'HF2MCORE'

    monkeypatch.setattr('swift.dev.recipe.convert._convert_mcore', _mcore)
    monkeypatch.setattr('swift.dev.recipe.convert._convert_hf2mcore', _hf2mcore)
    return calls


def test_to_hf_routes_to_convert_mcore(spy_workers):
    """``to_hf`` always reads an mcore checkpoint, so it goes through ``_convert_mcore``."""
    assert run_convert(_model(), ConvertConfig(to_hf=True)) == 'MCORE'
    assert spy_workers['which'] == 'mcore'


def test_to_mcore_with_mcore_model_routes_to_convert_mcore(spy_workers):
    """``to_mcore`` WITH an ``mcore_model`` source is a reshard (mcore -> mcore), not an HF load, so it
    also goes through ``_convert_mcore`` -- the mcore_model is what flips the direction."""
    assert run_convert(_model(), ConvertConfig(to_mcore=True, mcore_model='/src/mcore')) == 'MCORE'
    assert spy_workers['which'] == 'mcore'


def test_to_mcore_with_mcore_adapter_routes_to_convert_mcore(spy_workers):
    """An ``mcore_adapter`` (mcore-format LoRA) is merged into the base before saving, which is the
    ``_convert_mcore`` path even though the direction flag is ``to_mcore``."""
    assert run_convert(_model(), ConvertConfig(to_mcore=True, mcore_adapter='/src/adapter')) == 'MCORE'
    assert spy_workers['which'] == 'mcore'


def test_to_mcore_without_source_routes_to_hf2mcore(spy_workers):
    """The plain HF -> mcore case: ``to_mcore`` with no mcore source loads the HF weights."""
    assert run_convert(_model(), ConvertConfig(to_mcore=True)) == 'HF2MCORE'
    assert spy_workers['which'] == 'hf2mcore'


# --- output_dir resolution ------------------------------------------------------------


def test_explicit_output_dir_wins(spy_workers):
    """An explicit ``output_dir`` argument overrides the checkpoint config."""
    run_convert(
        _model(), ConvertConfig(to_mcore=True), checkpoint_config=CheckpointConfig(output_dir='/from/ckpt'),
        output_dir='/explicit')
    assert spy_workers['output_dir'] == '/explicit'


def test_checkpoint_output_dir_is_the_fallback(spy_workers):
    """With no explicit override, ``checkpoint_config.output_dir`` is used."""
    run_convert(_model(), ConvertConfig(to_mcore=True), checkpoint_config=CheckpointConfig(output_dir='/from/ckpt'))
    assert spy_workers['output_dir'] == '/from/ckpt'


def test_output_dir_defaults_to_output(spy_workers):
    """With neither an override nor a checkpoint config, the recipe falls back to ``'output'``."""
    run_convert(_model(), ConvertConfig(to_mcore=True))
    assert spy_workers['output_dir'] == 'output'
