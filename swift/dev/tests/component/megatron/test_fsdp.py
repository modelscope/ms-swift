"""Megatron-FSDP: the wrapper twinkle used to ignore, and the three things that must agree with it.

Why this exists. megatron ships two sharded data-parallel implementations, and Megatron-FSDP is
selected by a key on ``DistributedDataParallelConfig`` (``use_megatron_fsdp``) rather than by
constructing a different config object. twinkle passes its ddp_config dict straight into that
dataclass, so before this change the flag was ACCEPTED by the config and then ignored: the wrapper
class was hardcoded to DistributedDataParallel. A run configured for FSDP replicated its parameters
instead of sharding them, with no error and no warning -- the same shape of failure as the FP4
param-gather hole (see test_low_precision.py).

Choosing the wrapper is not enough on its own. Three other things are decided elsewhere and have to
agree with it, which is why they now all live on MegatronStrategy:

1. CUDA_DEVICE_MAX_CONNECTIONS. DDP with tensor parallelism wants ``1`` so kernels are issued in call
   order and a collective really overlaps the GEMM after it; FSDP wants more than 1 so its concurrent
   parameter all-gathers are not serialized. twinkle used to hardcode ``1`` in three places. The value
   is latched by the CUDA driver when the context is created, so it must be written before
   ``set_device()`` -- hence a classmethod called by the model rather than work done in the strategy's
   own ``__init__``, which runs after the device is set. megatron is explicit that this is a
   performance knob and not a correctness one, so getting it wrong is slow rather than broken.
2. ``start_param_sync``. DDP has it, Megatron-FSDP does not (FSDP drives its own all-gathers from
   module hooks). ``finish_param_config`` reached for it unconditionally, which is an AttributeError
   under FSDP as soon as the overlap knobs are both on.
3. The checkpoint sharding type. Megatron-FSDP stores optimizer state as DTensors and megatron only
   accepts ``fsdp_dtensor`` for it; twinkle hardcoded ``dp_reshardable``.

Nothing here touches a GPU or initializes a process group: every test is a pure function of the
configs, of megatron's class surface, or of the environment dict.
"""
from __future__ import annotations

import os

import pytest
from types import SimpleNamespace

# === The config contract (pure function of the Configs, no I/O) ===


def _configs(*, backend='megatron', **distributed_overrides):
    from swift.dev.config import DatasetConfig, DistributedConfig, ModelConfig, TemplateConfig, TrainConfig
    return {
        'model_config': ModelConfig(model='dummy'),
        'template_config': TemplateConfig(template='qwen2_5'),
        'dataset_config': DatasetConfig(dataset=['dummy']),
        'train_config': TrainConfig(),
        'distributed_config': DistributedConfig(backend=backend, nproc_per_node=1, **distributed_overrides),
    }


def _validate(**kwargs):
    from swift.dev.config import validate_configs
    validate_configs(**_configs(**kwargs))


def test_fsdp_off_by_default_is_accepted():
    _validate()


def test_fsdp_with_the_distributed_optimizer_is_accepted():
    _validate(use_megatron_fsdp=True)


def test_fsdp_without_the_distributed_optimizer_is_rejected():
    """FSDP shards the parameters; only the distributed optimizer holds matching master shards."""
    with pytest.raises(ValueError, match='use_distributed_optimizer'):
        _validate(use_megatron_fsdp=True, use_distributed_optimizer=False)


def test_fsdp_on_the_transformers_backend_is_rejected():
    """The transformers path has its own FSDP (DistributedConfig.fsdp); this flag would be ignored."""
    with pytest.raises(ValueError, match='transformers'):
        _validate(backend='hf', use_megatron_fsdp=True)


def test_fsdp_with_context_parallelism_is_rejected():
    with pytest.raises(ValueError, match='context_parallel_size'):
        _validate(use_megatron_fsdp=True, context_parallel_size=2)


def test_context_parallelism_without_fsdp_is_still_accepted():
    """The CP check must be gated on the FSDP flag -- CP alone is an ordinary supported run."""
    _validate(context_parallel_size=2)


def test_the_transformers_fsdp_knob_is_untouched_by_this_check():
    """DistributedConfig.fsdp and use_megatron_fsdp are different knobs for different backends.

    Named similarly enough that conflating them is the obvious mistake; this pins that setting the
    transformers one does not trip the megatron check.
    """
    _validate(backend='hf', fsdp='full_shard')


# === Builder forwarding: dev config -> the ddp_config dict twinkle reads ===


def _fsdp_kwargs(**distributed_overrides):
    from swift.dev.builders.model import _apply_fsdp_kwargs
    from swift.dev.config import DistributedConfig
    kwargs: dict = {}
    _apply_fsdp_kwargs(kwargs, DistributedConfig(backend='megatron', nproc_per_node=1, **distributed_overrides))
    return kwargs


def test_no_ddp_config_leaks_when_fsdp_is_off():
    """A DDP run must reach twinkle exactly as it did before FSDP existed: no ddp_config at all."""
    assert _fsdp_kwargs() == {}
    assert _fsdp_kwargs(use_megatron_fsdp=False) == {}


def test_fsdp_is_forwarded_as_a_ddp_config_key():
    """It has to travel INSIDE ddp_config: that dict is what twinkle reads to pick the wrapper class,
    and what megatron's DistributedDataParallelConfig is then built from. A top-level kwarg of the
    same name would reach neither reader."""
    assert _fsdp_kwargs(use_megatron_fsdp=True) == {'ddp_config': {'use_megatron_fsdp': True}}


def test_the_megatron_builder_calls_the_fsdp_forwarder():
    """Guards against the helper existing but never being wired in -- which would look exactly like
    the silent misconfiguration this whole feature is about."""
    import ast
    import inspect
    from swift.dev.builders import model as builders_model

    tree = ast.parse(inspect.getsource(builders_model))
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == '_build_megatron_model':
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                    called.add(sub.func.id)
    assert '_apply_fsdp_kwargs' in called, f'_build_megatron_model does not call it; calls: {sorted(called)}'


def test_dev_field_name_matches_the_legacy_cli_surface():
    """args_to_configs copies by name, so a divergence silently drops legacy's --use_megatron_fsdp."""
    import dataclasses
    from swift.dev.config import DistributedConfig
    from swift.megatron.arguments.megatron_args import MegatronArguments

    legacy = {f.name for f in dataclasses.fields(MegatronArguments)}
    assert 'use_megatron_fsdp' in {f.name for f in dataclasses.fields(DistributedConfig)}
    assert 'use_megatron_fsdp' in legacy, 'not on the legacy Megatron arg surface'


def test_the_flag_is_a_real_megatron_ddp_config_field():
    """twinkle expands the dict into DistributedDataParallelConfig(**ddp_config), so a wrong name
    would be a TypeError at wrap time rather than anything this suite could catch."""
    megatron_core = pytest.importorskip('megatron.core')
    import dataclasses
    from megatron.core.distributed import DistributedDataParallelConfig

    assert megatron_core is not None
    fields = {f.name for f in dataclasses.fields(DistributedDataParallelConfig)}
    assert 'use_megatron_fsdp' in fields
    # Torch FSDP2 is NOT selected this way -- it needs its own config subclass -- which is part of
    # why only Megatron-FSDP is exposed. If this ever becomes a field, revisit that decision.
    assert 'use_torch_fsdp2' not in fields


# === twinkle: the wrapper choice and the three things that must follow it ===


def _strategy(ddp_config=None, use_distributed_optimizer=True, device_mesh=None):
    """A real MegatronStrategy with __init__ skipped (it needs a live process group)."""
    strategy_mod = pytest.importorskip('twinkle.model.megatron.strategy.megatron')
    obj = object.__new__(strategy_mod.MegatronStrategy)
    obj.ddp_config = ddp_config or {}
    obj.use_distributed_optimizer = use_distributed_optimizer
    obj.device_mesh = device_mesh
    return obj


def _strategy_cls():
    strategy_mod = pytest.importorskip('twinkle.model.megatron.strategy.megatron')
    return strategy_mod.MegatronStrategy


@pytest.mark.parametrize(
    'ddp_config, expected',
    [
        ({}, False),
        (None, False),
        ({'use_megatron_fsdp': False}, False),
        ({'use_megatron_fsdp': True}, True),
        # megatron's own deprecated alias, which its arguments.py still derives from the new name.
        ({'use_custom_fsdp': True}, True),
    ],
)
def test_uses_fsdp_reads_both_spellings(ddp_config, expected):
    assert _strategy_cls().uses_fsdp(ddp_config) is expected


def test_ddp_gets_one_device_connection():
    cls = _strategy_cls()
    os.environ.pop('CUDA_DEVICE_MAX_CONNECTIONS', None)
    cls.apply_process_env({})
    assert os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] == '1'


def test_fsdp_raises_the_device_connection_count():
    """Both from unset and from the '1' twinkle's own Ray worker bootstrap leaves behind."""
    cls = _strategy_cls()
    for start in (None, '1'):
        os.environ.pop('CUDA_DEVICE_MAX_CONNECTIONS', None)
        if start is not None:
            os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = start
        cls.apply_process_env({'use_megatron_fsdp': True})
        assert int(os.environ['CUDA_DEVICE_MAX_CONNECTIONS']) > 1


def test_an_explicit_device_connection_count_is_preserved_under_fsdp():
    """Anything other than '1' is someone's deliberate choice about their own topology."""
    cls = _strategy_cls()
    os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '8'
    try:
        cls.apply_process_env({'use_megatron_fsdp': True})
        assert os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] == '8'
    finally:
        os.environ.pop('CUDA_DEVICE_MAX_CONNECTIONS', None)


def test_ddp_sharding_type_is_unchanged():
    """Pins the pre-FSDP value: moving this decision onto the strategy must not alter DDP runs."""
    assert _strategy().get_sharded_sd_metadata()['distrib_optim_sharding_type'] == 'dp_reshardable'


def test_fsdp_switches_the_sharding_type_to_dtensor():
    metadata = _strategy({'use_megatron_fsdp': True}).get_sharded_sd_metadata()
    assert metadata['distrib_optim_sharding_type'] == 'fsdp_dtensor'
    # The rest of the metadata is not FSDP-specific and must survive the switch.
    assert metadata['singleton_local_shards'] is False
    assert metadata['chained_optim_avoid_prefix'] is True


def test_strategy_rejects_fsdp_without_the_distributed_optimizer():
    """The authoritative copy of the check: it runs where the model is built, so it also covers
    cookbook users who construct MegatronModel directly and never touch dev's config layer."""
    with pytest.raises(ValueError, match='use_distributed_optimizer'):
        _strategy({'use_megatron_fsdp': True}, use_distributed_optimizer=False)._check_fsdp()


def test_strategy_rejects_fsdp_with_context_parallelism():
    with pytest.raises(ValueError, match='context parallelism'):
        _strategy({'use_megatron_fsdp': True}, device_mesh=SimpleNamespace(cp_world_size=2))._check_fsdp()


def test_strategy_accepts_context_parallelism_without_fsdp():
    _strategy(device_mesh=SimpleNamespace(cp_world_size=2))._check_fsdp()


def test_strategy_check_tolerates_a_mesh_without_a_cp_attribute():
    """device_mesh is Optional and its attributes vary by construction path; the check must not be
    the thing that turns a missing attribute into a crash."""
    _strategy({'use_megatron_fsdp': True}, device_mesh=SimpleNamespace())._check_fsdp()
    _strategy({'use_megatron_fsdp': True}, device_mesh=None)._check_fsdp()


# === megatron's class surface: what the guards above are guarding against ===


def test_fsdp_provides_every_wrapper_method_twinkle_calls():
    pytest.importorskip('megatron.core')
    from megatron.core.distributed import FullyShardedDataParallel as FSDP

    for name in ('broadcast_params', 'no_sync', 'start_grad_sync', 'finish_grad_sync', 'zero_grad_buffer',
                 'sharded_state_dict'):
        assert hasattr(FSDP, name), f'Megatron-FSDP is missing {name}, which twinkle calls unconditionally'


def test_fsdp_lacks_start_param_sync_which_is_why_it_is_guarded():
    """The reason finish_param_config checks before wiring config.param_sync_func.

    If megatron ever adds it, the guard becomes dead code and can go -- this test is what will say so.
    """
    pytest.importorskip('megatron.core')
    from megatron.core.distributed import DistributedDataParallel as DDP
    from megatron.core.distributed import FullyShardedDataParallel as FSDP

    assert hasattr(DDP, 'start_param_sync')
    assert not hasattr(FSDP, 'start_param_sync')


def test_fsdp_is_not_a_ddp_subclass():
    """Which is why unwrap_model has to name it explicitly instead of relying on the DDP isinstance."""
    pytest.importorskip('megatron.core')
    from megatron.core.distributed import DistributedDataParallel as DDP
    from megatron.core.distributed import FullyShardedDataParallel as FSDP

    assert not issubclass(FSDP, DDP)


def test_strategy_unwrap_names_the_fsdp_class():
    """A wrapper that unwrap_model does not recognize is returned as-is, and callers then reach for
    module attributes on the wrapper rather than the model."""
    import inspect
    strategy_mod = pytest.importorskip('twinkle.model.megatron.strategy.megatron')

    source = inspect.getsource(strategy_mod.MegatronStrategy.unwrap_model)
    assert 'FullyShardedDataParallel' in source
