"""build_model: ModelConfig + DistributedConfig -> twinkle-native TransformersModel / MegatronModel."""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from swift.dev.configs import DistributedConfig, ModelConfig, TrainConfig, TunerConfig
    from swift.dev.model import TrainableModel


def build_model(model_config: ModelConfig,
                distributed_config: DistributedConfig,
                train_config: Optional[TrainConfig] = None,
                tuner_config: Optional[TunerConfig] = None) -> TrainableModel:
    """ModelConfig + DistributedConfig -> twinkle-native Model (no loss/optim yet).

    Thin mapping (no Registry/Factory): model_config fields -> twinkle __init__ kwargs.
    DistributedConfig.backend=='megatron' builds a MegatronModel (via the selected bridge
    backend); otherwise a TransformersModel. twinkle self-builds weights on the correct rank.
    """
    if is_megatron_backend(distributed_config):
        return _build_megatron_model(model_config, distributed_config)
    return _build_transformers_model(model_config, distributed_config, train_config, tuner_config)


def is_megatron_backend(distributed_config: DistributedConfig) -> bool:
    backend = distributed_config.backend
    if backend in (None, 'hf'):
        return False
    if backend == 'megatron':
        return True
    raise ValueError(f"DistributedConfig.backend must be one of {{'megatron', 'hf'}}, got {backend!r}.")


def _build_transformers_model(model_config: ModelConfig,
                              distributed_config: DistributedConfig,
                              train_config: Optional[TrainConfig] = None,
                              tuner_config: Optional[TunerConfig] = None) -> TrainableModel:
    import torch

    from swift.dev.model import TransformersModel

    if not model_config.model:
        raise ValueError('ModelConfig.model (path/id) is required')

    kwargs: dict = {'model_id': model_config.model}
    # dtype: ModelConfig.torch_dtype is a string ('bfloat16'); forward to from_pretrained.
    if model_config.torch_dtype:
        dt = getattr(torch, model_config.torch_dtype, None)
        if dt is not None:
            kwargs['dtype'] = dt
    if model_config.attn_impl:
        kwargs['attn_implementation'] = model_config.attn_impl
    if model_config.model_revision:
        kwargs['revision'] = model_config.model_revision

    # strategy: deepspeed/fsdp config selects the twinkle strategy (default: accelerate).
    from swift.dev.naming import resolve_strategy
    strategy = 'accelerate'
    if distributed_config.deepspeed:
        strategy = 'deepspeed'
        raise NotImplementedError('DeepSpeed is not supported yet')
    elif distributed_config.fsdp:
        strategy = 'native_fsdp'
    kwargs['strategy'] = resolve_strategy(strategy)
    kwargs['mixed_precision'] = 'bf16'

    # DDP find_unused_parameters: mirror HF Trainer's three-way derivation
    find_unused = distributed_config.ddp_find_unused_parameters
    if find_unused is None:
        is_peft = tuner_config is not None
        if is_peft:
            find_unused = True
        else:
            gc_on = train_config.gradient_checkpointing if train_config is not None else False
            find_unused = not gc_on
    kwargs['ddp_config'] = {'find_unused_parameters': bool(find_unused)}

    # No device_mesh passed on purpose: twinkle's local mode assigns its default one (pure data
    # parallel over WORLD_SIZE, infra/__init__.py:538-541), which is exactly the transformers layout --
    # this backend has no TP/PP/CP (validate_configs rejects those sizes here). It is NOT optional
    # bookkeeping though: a None mesh silently changes the training objective to an avg-of-avg, so it
    # is load-bearing that run_sft calls twinkle.initialize first (see _initialize_twinkle, and the
    # 2-GPU aggregation test that pins the result).
    return TransformersModel(**kwargs)


def _resolve_bridge_backend(name: str):
    """bridge_backend name (DistributedConfig default: 'mcore-bridge') -> a BridgeBackend instance."""
    from swift.dev.model.megatron.bridge import MCoreBridgeBackend, MegatronBridgeBackend
    key = name.lower()
    if key == 'mcore-bridge':
        return MCoreBridgeBackend()
    if key == 'megatron-bridge':
        return MegatronBridgeBackend()
    raise NotImplementedError(f"Unknown bridge_backend {name!r}. Known: 'mcore-bridge', 'megatron-bridge'.")


def build_device_mesh(distributed_config: DistributedConfig):
    """DistributedConfig -> the Megatron DeviceMesh (parallel layout).

    A pure function of the config: it reads only the declared sizes, never torch.distributed or
    Megatron's mpu, so it can be called before anything is initialized.
    """
    from twinkle import DeviceMesh

    tp = distributed_config.tensor_model_parallel_size
    pp = distributed_config.pipeline_model_parallel_size
    cp = distributed_config.context_parallel_size
    ep = distributed_config.expert_model_parallel_size

    world_size = distributed_config.nproc_per_node
    if world_size is None:
        raise ValueError('DistributedConfig.nproc_per_node is required for the Megatron backend (it sets the '
                         'DeviceMesh world size). Pass it explicitly -- there is no default, since a wrong '
                         'world size silently builds the wrong data-parallel layout.')
    model_parallel = tp * pp * cp
    if world_size % model_parallel != 0:
        raise ValueError(f'nproc_per_node={world_size} is not divisible by tp*pp*cp={model_parallel} '
                         f'(tp={tp}, pp={pp}, cp={cp}).')
    dp_size = world_size // model_parallel

    mesh_kwargs = dict(world_size=world_size, dp_size=dp_size)
    if tp > 1:
        mesh_kwargs['tp_size'] = tp
    if pp > 1:
        mesh_kwargs['pp_size'] = pp
    if cp > 1:
        mesh_kwargs['cp_size'] = cp
    if ep > 1:
        mesh_kwargs['ep_size'] = ep
    # Megatron TP sequence-parallelism rides on the DeviceMesh (twinkle reads it via
    # strategy.sequence_parallel -> device_mesh.sequence_parallel). Only meaningful with tp > 1.
    if distributed_config.sequence_parallel:
        mesh_kwargs['sequence_parallel'] = True
    return DeviceMesh.from_sizes(**mesh_kwargs)


def _build_megatron_model(model_config: ModelConfig, distributed_config: DistributedConfig) -> TrainableModel:
    """Build a MegatronModel via the selected bridge backend.

    twinkle must already be initialized in Ray mode (run_sft does this) so the 'model' DeviceGroup
    exists. The world size is DistributedConfig.nproc_per_node (== the DeviceGroup size); dp_size is
    derived from it and the model-parallel sizes. Driver-side dist.get_world_size() is NOT used --
    in Ray mode the driver is not part of the model process group (its world size is 1).
    """
    from swift.dev.model.megatron.model import MegatronModel

    if not model_config.model:
        raise ValueError('ModelConfig.model (path/id) is required')

    device_mesh = build_device_mesh(distributed_config)

    mixed_precision = 'bf16'
    if model_config.torch_dtype == 'float32':
        mixed_precision = 'no'
    elif model_config.torch_dtype == 'float16':
        mixed_precision = 'fp16'

    # A few high-frequency Megatron knobs flow straight into MegatronModel.__init__. Forward
    # only when set (None -> twinkle's own default), so the bit-exact SFT baseline is unchanged
    # unless the user opts in. use_distributed_optimizer has a real default (True) so pass it.
    extra_kwargs: dict = {'use_distributed_optimizer': distributed_config.use_distributed_optimizer}
    # name is a dynamic attribute -> getattr is required here (not defensive over-protection).
    for name in ('recompute_granularity', 'recompute_method', 'recompute_num_layers'):
        value = getattr(distributed_config, name)
        if value is not None:
            extra_kwargs[name] = value

    if extra_kwargs.get('recompute_granularity') == 'selective':
        extra_kwargs['recompute_num_layers'] = None
        extra_kwargs['recompute_method'] = None

    # Attention kernel. Always forwarded (unlike the recompute knobs above) because the meaningful
    # default is legacy's 'flash', not mcore's AttnBackend.auto -- under auto TE picks per shape and
    # selects the FUSED cuDNN kernel for a Qwen2.5 bf16 causal THD forward, so leaving it unset makes
    # dev and legacy run different attention kernels on identical config. resolve_* returns the enum
    # (mcore compares by identity), and both bridge backends now read this one value instead of the
    # megatron-bridge path hardcoding its own.
    from swift.dev.naming import resolve_megatron_attn_backend
    extra_kwargs['attention_backend'] = resolve_megatron_attn_backend(model_config.attn_impl)
    # A flash_N / flash_attention_N value also pins the FA VERSION, which is enforced by mutating
    # transformer_engine module globals -- a per-process side effect, so it CANNOT be applied here:
    # in Ray mode build_model runs on the driver, which is not where the model is built. The raw
    # string is forwarded so DevMegatronStrategy (worker-side) can apply the pin itself.
    extra_kwargs['attn_impl'] = model_config.attn_impl

    backend = _resolve_bridge_backend(distributed_config.bridge_backend)
    # In Ray mode the model lives in a remote DeviceGroup named 'model'; in local (torchrun) mode
    # each rank builds the model in-process, so there is no remote group to target.
    model_kwargs = dict(
        model_id=model_config.model,
        device_mesh=device_mesh,
        mixed_precision=mixed_precision,
        backend=backend,
        **extra_kwargs)
    if distributed_config.mode != 'local':
        model_kwargs['remote_group'] = 'model'
    return MegatronModel(**model_kwargs)
