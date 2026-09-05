"""FP4/FP8 low-precision training: the renames, and the ways they silently train nothing.

Why this exists. Megatron describes quantized parameters on two config objects it never
cross-checks. ``TransformerConfig.fp4_param`` decides the parameters are BUILT as NVFP4;
``DistributedDataParallelConfig.fp4_param_gather`` decides the distributed optimizer writes its FP32
master shards back INTO them. Set only the first and
``DistributedOptimizer._copy_main_params_to_model_params`` takes neither branch -- it dispatches on
the DDP flags alone -- and then its generic fallback skips every NVFP4 parameter by design
("NVFP4 params are quantized in the above quantize_nvfp4_param_shard function"). The master weights
advance every step while the model parameters never move. No error, no warning.

The same hole exists for FP8 (``_is_distopt_quantized_param`` is true for a Float8Tensor regardless
of the DDP flag), so the fix covers both: twinkle DERIVES the DDP flag from the model config rather
than trusting two sources of truth to agree.

The second trap is the optimizer. That re-quantize step lives only on DistributedOptimizer, so
``fp4_param`` under any other optimizer is the same silent no-op -- checked twice, in
validate_configs (early, on the driver) and in twinkle (authoritative, on the worker).

The third trap is quieter still and specific to FP8: legacy Megatron-SWIFT overrides megatron's own
amax defaults (1024 / 'max' against 1 / 'most_recent'). Those change the numerics, so dev has to
carry legacy's values AND forward them explicitly -- inheriting megatron's would make one config
train two different ways depending on which entry point launched it. Pinned below.

Nothing here touches a GPU: NVFP4 GEMMs need Blackwell, so the numerics cannot be tested on
pre-Blackwell hardware and every test below is a pure function of the configs.
"""
from __future__ import annotations

import pytest
from types import SimpleNamespace


# === The config contract (pure function of the Configs, no I/O) ===


def _configs(*, use_distributed_optimizer=True, backend='megatron', **model_overrides):
    from swift.dev.config import DatasetConfig, DistributedConfig, ModelConfig, TemplateConfig, TrainConfig
    return {
        'model_config': ModelConfig(model='dummy', **model_overrides),
        'template_config': TemplateConfig(template='qwen2_5'),
        'dataset_config': DatasetConfig(dataset=['dummy']),
        'train_config': TrainConfig(),
        'distributed_config': DistributedConfig(
            backend=backend, nproc_per_node=1, use_distributed_optimizer=use_distributed_optimizer),
    }


def _validate(**kwargs):
    from swift.dev.config import validate_configs
    validate_configs(**_configs(**kwargs))


def test_fp4_off_by_default_is_accepted():
    _validate()


def test_fp4_compute_without_fp4_parameters_is_accepted():
    """The conservative configuration: FP4 GEMMs, parameters still in their normal dtype."""
    _validate(fp4_format='e2m1')


def test_fp4_with_fp4_parameters_is_accepted():
    _validate(fp4_format='e2m1', fp4_param_gather=True)


def test_fp4_param_gather_without_a_format_is_rejected():
    """Without fp4_format no FP4 context is ever entered, so the knob would quietly do nothing."""
    with pytest.raises(ValueError, match='fp4_format'):
        _validate(fp4_param_gather=True)


def test_fp4_on_the_transformers_backend_is_rejected():
    with pytest.raises(ValueError, match='megatron'):
        _validate(backend='hf', fp4_format='e2m1')


def test_fp4_parameters_without_the_distributed_optimizer_are_rejected():
    """The re-quantize step exists only on DistributedOptimizer; elsewhere the params never update."""
    with pytest.raises(ValueError, match='use_distributed_optimizer'):
        _validate(fp4_format='e2m1', fp4_param_gather=True, use_distributed_optimizer=False)


def test_fp4_compute_alone_does_not_need_the_distributed_optimizer():
    """Only *parameters* in FP4 depend on the optimizer; FP4 GEMMs on bf16 params do not."""
    _validate(fp4_format='e2m1', use_distributed_optimizer=False)


# === The builder mapping (dev's legacy-CLI names -> megatron's names) ===


def _fp4_kwargs(**model_overrides):
    from swift.dev.builders.model import _apply_fp4_kwargs
    from swift.dev.config import ModelConfig
    kwargs: dict = {}
    _apply_fp4_kwargs(kwargs, ModelConfig(model='dummy', **model_overrides))
    return kwargs


def test_no_fp4_kwarg_leaks_when_fp4_is_off():
    """An FP4-free run must reach the bridge with exactly the kwargs it had before FP4 existed.

    ``fp4_recipe`` is the one that matters here: it has a non-None default on both sides, so
    forwarding it unconditionally would be indistinguishable from an explicit request.
    """
    assert _fp4_kwargs() == {}
    assert _fp4_kwargs(fp4_param_gather=True) == {}


def test_the_format_is_forwarded_under_megatrons_name():
    """dev calls it fp4_format (the legacy CLI flag); megatron's TransformerConfig calls it fp4."""
    assert _fp4_kwargs(fp4_format='e2m1') == {'fp4': 'e2m1', 'fp4_recipe': 'nvfp4'}


def test_param_gather_is_forwarded_as_fp4_param_only():
    """The DDP flag is deliberately NOT forwarded: twinkle derives it from fp4_param.

    Sending both would recreate the two-sources-of-truth split whose disagreement is the silent
    no-op this whole feature exists to prevent.
    """
    kwargs = _fp4_kwargs(fp4_format='e2m1', fp4_param_gather=True)
    assert kwargs['fp4_param'] is True
    assert 'fp4_param_gather' not in kwargs


def test_every_forwarded_fp4_kwarg_is_a_real_megatron_field():
    """Guard the rename: a typo here is a TypeError at model-build time, on a GPU, minutes in.

    All three land on megatron's ``TransformerConfig`` rather than on mcore-bridge's own subclass --
    mcore-bridge inherits the FP4 surface instead of redeclaring it, which is why the bridge needed
    no config change for this feature.

    Read statically rather than by importing: ``import mcore_bridge`` pulls in megatron.core, which
    needs a working CUDA/TE stack, so an import-based check would skip on the machines where the
    rest of this file runs.
    """
    from swift.dev.tests.test_mtp import _dataclass_fields_from_source, _megatron_transformer_config_source
    megatron_fields = _dataclass_fields_from_source(_megatron_transformer_config_source(), 'TransformerConfig')
    if megatron_fields is None:
        pytest.skip('megatron-core source not available for a static field check')

    forwarded = set(_fp4_kwargs(fp4_format='e2m1', fp4_param_gather=True))
    assert forwarded == {'fp4', 'fp4_recipe', 'fp4_param'}
    assert forwarded <= megatron_fields, f'not declared by megatron TransformerConfig: {forwarded - megatron_fields}'


def test_dev_field_names_match_the_legacy_cli_surface():
    """args_to_configs copies by name, so a divergence here silently drops the legacy --fp4/--fp8 flags."""
    import dataclasses
    from swift.dev.config import ModelConfig
    from swift.megatron.arguments.megatron_args import MegatronArguments

    legacy = {f.name for f in dataclasses.fields(MegatronArguments)}
    dev = {f.name for f in dataclasses.fields(ModelConfig) if f.name.startswith(('fp4', 'fp8'))}
    assert dev == {
        'fp4_format', 'fp4_recipe', 'fp4_param_gather', 'fp8_format', 'fp8_recipe', 'fp8_param_gather',
        'fp8_amax_history_len', 'fp8_amax_compute_algo'
    }
    assert dev <= legacy, f'not on the legacy Megatron arg surface: {dev - legacy}'


def test_dev_matches_legacy_on_the_amax_defaults_not_megatron():
    """These two change the numerics, and legacy overrides megatron rather than inheriting it.

    Taking megatron's defaults would make the same nominal config train differently under the dev
    entry point than under the legacy one -- a divergence nothing else would report.
    """
    import dataclasses
    from swift.dev.config import ModelConfig
    from swift.megatron.arguments.megatron_args import MegatronArguments

    legacy = {f.name: f.default for f in dataclasses.fields(MegatronArguments)}
    dev = {f.name: f.default for f in dataclasses.fields(ModelConfig)}
    for name in ('fp8_amax_history_len', 'fp8_amax_compute_algo', 'fp8_recipe'):
        assert dev[name] == legacy[name], f'{name}: dev {dev[name]!r} != legacy {legacy[name]!r}'
    # And the guard that makes the above meaningful: legacy's values are NOT megatron's.
    megatron_defaults = {'fp8_amax_history_len': 1, 'fp8_amax_compute_algo': 'most_recent'}
    for name, upstream in megatron_defaults.items():
        assert dev[name] != upstream, (f'{name} now equals megatron\'s default {upstream!r}; if upstream changed '
                                       'to match legacy, drop this assertion and the explicit forwarding with it')


# === twinkle: deriving the DDP flag from the model config ===


def _finalize(*, fp4_param=False, fp8_param=False, ddp_config=None, use_distributed_optimizer=True):
    """Run MegatronStrategy._finalize_quantized_param_config against a stub.

    Unbound on a SimpleNamespace because the method reads only three attributes and a real strategy
    would need torch.distributed, a DeviceMesh and a downloaded model to construct.
    """
    from twinkle.model.megatron.strategy.megatron import MegatronStrategy
    stub = SimpleNamespace(
        config=SimpleNamespace(fp4_param=fp4_param, fp8_param=fp8_param),
        ddp_config=dict(ddp_config or {}),
        use_distributed_optimizer=use_distributed_optimizer,
    )
    MegatronStrategy._finalize_quantized_param_config(stub)
    return stub.ddp_config


def test_fp4_param_derives_the_ddp_param_gather_flag():
    assert _finalize(fp4_param=True)['fp4_param_gather'] is True


def test_fp8_param_derives_its_flag_too():
    """Same hole, same fix: _is_distopt_quantized_param is true for a Float8Tensor either way."""
    assert _finalize(fp8_param=True)['fp8_param_gather'] is True


def test_nothing_is_derived_when_no_quantized_parameters_are_requested():
    """A bf16 run's ddp_config must be untouched, including on a model config that predates FP4."""
    assert _finalize() == {}
    assert _finalize(ddp_config={'overlap_grad_reduce': False}) == {'overlap_grad_reduce': False}


def test_a_config_without_the_fp4_field_reads_as_off():
    """An older megatron-core has no fp4_param at all; that must not raise."""
    from twinkle.model.megatron.strategy.megatron import MegatronStrategy
    stub = SimpleNamespace(config=SimpleNamespace(), ddp_config={}, use_distributed_optimizer=True)
    MegatronStrategy._finalize_quantized_param_config(stub)
    assert stub.ddp_config == {}


def test_an_explicit_false_ddp_flag_is_a_contradiction_not_an_override():
    """Honouring it would mean honouring a request to train nothing."""
    with pytest.raises(ValueError, match='fp4_param_gather'):
        _finalize(fp4_param=True, ddp_config={'fp4_param_gather': False})


def test_an_explicit_true_ddp_flag_is_left_alone():
    assert _finalize(fp4_param=True, ddp_config={'fp4_param_gather': True})['fp4_param_gather'] is True


def test_quantized_parameters_require_the_distributed_optimizer():
    with pytest.raises(ValueError, match='use_distributed_optimizer'):
        _finalize(fp4_param=True, use_distributed_optimizer=False)


def test_the_optimizer_check_does_not_fire_for_an_unquantized_run():
    assert _finalize(use_distributed_optimizer=False) == {}


# === FP8: the same shape, plus the amax knobs and the mutual exclusion ===


def test_fp8_compute_without_fp8_parameters_is_accepted():
    _validate(fp8_format='hybrid')


def test_fp8_with_fp8_parameters_is_accepted():
    _validate(fp8_format='e4m3', fp8_param_gather=True)


def test_fp8_param_gather_without_a_format_is_rejected():
    with pytest.raises(ValueError, match='fp8_format'):
        _validate(fp8_param_gather=True)


def test_fp8_on_the_transformers_backend_is_rejected():
    with pytest.raises(ValueError, match='megatron'):
        _validate(backend='hf', fp8_format='hybrid')


def test_fp8_parameters_without_the_distributed_optimizer_are_rejected():
    with pytest.raises(ValueError, match='use_distributed_optimizer'):
        _validate(fp8_format='hybrid', fp8_param_gather=True, use_distributed_optimizer=False)


def test_the_amax_knobs_are_not_dependency_checked():
    """Their defaults are non-None, so a dependency check on them would reject every ordinary run.

    This is the ``mtp_loss_scaling_factor`` lesson applied ahead of time: on a field whose default is
    not None, "did the user set this?" has no answer, and guessing costs the whole CLI surface.
    """
    _validate(fp8_amax_history_len=32, fp8_amax_compute_algo='most_recent')


def test_fp4_and_fp8_together_are_rejected():
    """megatron applies one quantization recipe per layer; its TransformerConfig raises on this too.

    Checked here as well so it fails on the driver, before a model is built on every rank.
    """
    with pytest.raises(ValueError, match='mutually exclusive'):
        _validate(fp4_format='e2m1', fp8_format='hybrid')


def _fp8_kwargs(**model_overrides):
    from swift.dev.builders.model import _apply_fp8_kwargs
    from swift.dev.config import ModelConfig
    kwargs: dict = {}
    _apply_fp8_kwargs(kwargs, ModelConfig(model='dummy', **model_overrides))
    return kwargs


def test_no_fp8_kwarg_leaks_when_fp8_is_off():
    assert _fp8_kwargs() == {}
    assert _fp8_kwargs(fp8_param_gather=True, fp8_amax_history_len=32) == {}


def test_fp8_forwards_the_format_recipe_and_amax_knobs():
    """The amax knobs are forwarded EXPLICITLY: dev's defaults are legacy's, not megatron's."""
    assert _fp8_kwargs(fp8_format='hybrid') == {
        'fp8': 'hybrid',
        'fp8_recipe': 'delayed',
        'fp8_amax_history_len': 1024,
        'fp8_amax_compute_algo': 'max',
    }


def test_fp8_param_gather_is_forwarded_as_fp8_param_only():
    kwargs = _fp8_kwargs(fp8_format='e4m3', fp8_param_gather=True)
    assert kwargs['fp8_param'] is True
    assert 'fp8_param_gather' not in kwargs


def test_every_forwarded_fp8_kwarg_is_a_real_megatron_field():
    from swift.dev.tests.test_mtp import _dataclass_fields_from_source, _megatron_transformer_config_source
    megatron_fields = _dataclass_fields_from_source(_megatron_transformer_config_source(), 'TransformerConfig')
    if megatron_fields is None:
        pytest.skip('megatron-core source not available for a static field check')

    forwarded = set(_fp8_kwargs(fp8_format='hybrid', fp8_param_gather=True))
    assert forwarded == {'fp8', 'fp8_recipe', 'fp8_amax_history_len', 'fp8_amax_compute_algo', 'fp8_param'}
    assert forwarded <= megatron_fields, f'not declared by megatron TransformerConfig: {forwarded - megatron_fields}'
