"""Fast unit tests for the legacy-args -> dev-Config mapping (swift.dev.cli.sft.args_to_configs).

These assert the mapping *contract* without loading any model/dataset (a SimpleNamespace
stands in for a parsed SftArguments), so they run in the cheap suite:
  - name-based copy fills every Config field from the same-named legacy arg,
  - a None legacy value leaves the Config's OWN default in place (defaults live in the Config),
  - the one representation-mismatch field (torch_dtype object -> str) is normalized,
  - list-typed fields pass straight through (no _as_list coercion needed),
  - tuner_type dispatches full -> None, lora -> TunerConfig, else -> NotImplementedError.
"""
from __future__ import annotations

import pytest
from types import SimpleNamespace

from swift.dev.cli.sft import args_to_configs
from swift.dev.config import CheckpointConfig, DatasetConfig, ModelConfig, TemplateConfig, TrainConfig, TunerConfig


def _args(**overrides):
    """A legacy-args stand-in.

    A real SftArguments always defines every field, so the stub sets the two fields that
    args_to_configs reads by explicit name (torch_dtype, tuner_type) -- absent-attribute
    fall-through is only exercised by _fill_from_args' dynamic hasattr loop. Other keys under
    test are added per case; unset ones let each Config's own default stand (single source).
    """
    base = dict(model='m', dataset=['d1', 'd2'], tuner_type='full', torch_dtype=None)
    base.update(overrides)
    return SimpleNamespace(**base)


def test_none_legacy_value_keeps_config_default():
    """A None legacy field must NOT overwrite the Config default (single source of defaults)."""
    _, template, _, train, _, checkpoint, _ = args_to_configs(
        _args(padding_side=None, warmup_ratio=None, optim=None, output_dir=None))
    assert template.padding_side == TemplateConfig().padding_side == 'right'
    assert train.warmup_ratio == TrainConfig().warmup_ratio == 0.0
    assert train.optim == TrainConfig().optim == 'adamw_torch_fused'
    assert checkpoint.output_dir == CheckpointConfig().output_dir == 'output'


def test_explicit_legacy_value_overrides_default():
    model, template, _, train, _, _, _ = args_to_configs(_args(padding_side='left', warmup_ratio=0.1, max_length=256))
    assert template.padding_side == 'left'
    assert train.warmup_ratio == 0.1
    assert template.max_length == 256


def test_torch_dtype_object_is_normalized_to_str():
    """The one representation mismatch: legacy holds torch.bfloat16, ModelConfig wants a str."""
    import torch
    model, *_ = args_to_configs(_args(torch_dtype=torch.bfloat16))
    assert model.torch_dtype == 'bfloat16'
    model2, *_ = args_to_configs(_args(torch_dtype='float16'))
    assert model2.torch_dtype == 'float16'


def test_list_fields_pass_through_without_coercion():
    """List-typed args map straight onto List Config fields (the old _as_list was redundant)."""
    _, _, dataset, *_ = args_to_configs(_args(dataset=['a', 'b'], val_dataset=['v']))
    assert dataset.dataset == ['a', 'b']
    assert dataset.val_dataset == ['v']


def test_tuner_full_yields_none():
    *_, tuner = args_to_configs(_args(tuner_type='full'))
    assert tuner is None


def test_tuner_lora_yields_tunerconfig():
    *_, tuner = args_to_configs(_args(tuner_type='lora', lora_rank=16, lora_alpha=64, target_modules=['q_proj']))
    assert isinstance(tuner, TunerConfig)
    assert tuner.tuner_type == 'lora'
    assert tuner.lora_rank == 16
    assert tuner.lora_alpha == 64
    assert tuner.target_modules == ['q_proj']


def test_tuner_lora_defaults_when_unset():
    """Unset lora fields fall back to TunerConfig defaults (not re-specified in the CLI)."""
    *_, tuner = args_to_configs(_args(tuner_type='lora'))
    assert tuner.lora_rank == TunerConfig().lora_rank
    assert tuner.target_modules == TunerConfig().target_modules == ['all-linear']


def test_int_fields_coerce_float_from_hf_parser():
    """HfArgumentParser yields float for HF-inherited float fields (e.g. `--save_steps 500` -> 500.0).
    _fill_from_args must coerce these into the int-typed Config fields (save_steps / eval_steps)."""
    _, _, _, train, _, checkpoint, _ = args_to_configs(_args(save_steps=500.0, eval_steps=100.0))
    assert checkpoint.save_steps == 500 and isinstance(checkpoint.save_steps, int)
    assert train.eval_steps == 100 and isinstance(train.eval_steps, int)


def test_unsupported_tuner_type_raises():
    with pytest.raises(NotImplementedError):
        args_to_configs(_args(tuner_type='vera'))


def test_megatron_arguments_are_refused_instead_of_half_mapped():
    """This mapping is HF-surface only; Megatron args must not fall through it.

    MegatronSftArguments is a separate hierarchy that renames the same hyperparameters (lr,
    train_iters, micro_batch_size, lr_decay_style, ...), so the name-based copy would leave 34 of
    TrainConfig's 58 fields at dev defaults, and that surface has no `backend`, so the run would
    quietly train on transformers with the wrong lr/steps/batch size.
    """
    with pytest.raises(NotImplementedError, match='Megatron'):
        args_to_configs(_args(train_iters=100, lr=1e-5, micro_batch_size=2))


def test_optimizer_plugin_selector_is_refused():
    """`--optimizer` (legacy's plugin selector) has no dev equivalent, so it must not parse silently.

    It lives on legacy SftArguments and TrainConfig has no matching field, so the name-based copy
    would simply skip it -- the one case where "no Config field" means "dropped on the floor" rather
    than "Config default stands". The unset case must stay clean: every Megatron/HF run passes here.
    """
    args_to_configs(_args())  # optimizer absent
    args_to_configs(_args(optimizer=None))  # present but unset
    with pytest.raises(NotImplementedError, match='optimizer'):
        args_to_configs(_args(optimizer='muon'))


def test_returns_correct_config_types():
    model, template, dataset, train, dist, checkpoint, _ = args_to_configs(_args())
    assert isinstance(model, ModelConfig)
    assert isinstance(template, TemplateConfig)
    assert isinstance(dataset, DatasetConfig)
    assert isinstance(train, TrainConfig)
    assert isinstance(checkpoint, CheckpointConfig)
