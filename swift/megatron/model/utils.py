# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import fields
from mcore_bridge import ModelConfig
from mcore_bridge import get_mcore_model as _get_mcore_model
from mcore_bridge import hf_to_mcore_config
from transformers.utils import is_torch_npu_available
from typing import Any, Generator, Optional, Tuple

from swift.utils import get_logger

logger = get_logger()


def _check_attention_backend(args, config):
    """Validate attention backend compatibility with configuration."""
    attention_backend = config.attention_backend.name
    if attention_backend == 'flash' and config.softmax_type == 'learnable':
        raise ValueError(f'Attention backend "{attention_backend}" does not support learnable softmax_type.')


def _check_padding_free(args, config):
    """Validate and adjust padding_free setting based on configuration constraints."""
    if not args.padding_free:
        return

    attention_backend = config.attention_backend.name
    message = None

    if config.experimental_attention_variant == 'dsa':
        message = 'DSA is not supported in padding-free mode'
    elif attention_backend == 'unfused':
        message = f'Attention backend "{attention_backend}" is not supported in padding-free mode'

    if message:
        logger.warning(f'{message}. Setting args.padding_free to False.')
        args.padding_free = False


def get_mcore_model_config(args, hf_config):
    kwargs = hf_to_mcore_config(hf_config)
    kwargs['mcore_model_type'] = args.megatron_model_meta.model_type
    kwargs['hf_config'] = hf_config
    for f in fields(ModelConfig):
        key, value = f.name, getattr(args, f.name, None)
        if value is None or isinstance(value, (list, tuple)) and len(value) == 0:
            continue
        kwargs[key] = value

    if args.task_type == 'seq_cls':
        args.problem_type = args.problem_type or getattr(hf_config, 'problem_type', None)
        logger.info(f'args.problem_type: {args.problem_type}')

    kwargs['params_dtype'] = args.torch_dtype
    kwargs['num_layers_in_first_pipeline_stage'] = args.decoder_first_pipeline_num_layers
    kwargs['num_layers_in_last_pipeline_stage'] = args.decoder_last_pipeline_num_layers
    kwargs['fp4_param'] = args.fp4_param_gather
    kwargs['fp8_param'] = args.fp8_param_gather
    swiglu = kwargs.get('swiglu', True)
    add_bias_linear = kwargs.get('add_bias_linear', False)
    num_moe_experts = kwargs.get('num_moe_experts', None)
    position_embedding_type = kwargs.get('position_embedding_type', 'rope')
    if position_embedding_type != 'rope':
        kwargs['apply_rope_fusion'] = False
    if not swiglu and not add_bias_linear:
        kwargs['bias_activation_fusion'] = False
    if add_bias_linear and num_moe_experts and args.moe_grouped_gemm:
        kwargs['bias_dropout_fusion'] = False
    if num_moe_experts is None:
        kwargs['expert_model_parallel_size'] = 1
        kwargs['expert_tensor_parallel_size'] = 1

    if args.router_replay_mode != 'disabled':
        kwargs['moe_enable_routing_replay'] = True
    if args.megatron_extra_kwargs:
        kwargs.update(args.megatron_extra_kwargs)
    config = ModelConfig(**kwargs)
    if is_torch_npu_available() and getattr(args, 'attention_backend', 'flash') != 'local':
        setattr(config, 'use_flash_attn', True)
    _check_attention_backend(args, config)
    _check_padding_free(args, config)
    return config


class MegatronBridgeBackend:
    """Adapter for NVIDIA ``megatron.bridge.AutoBridge``.

    Limitations:
    - LoRA / PEFT loading is not yet supported.
    - MLLM is not yet supported
    - FP8 export is not yet supported.
    """

    def __init__(self, auto_bridge: Any, hf_config: Optional[Any] = None):
        self._bridge = auto_bridge
        self._hf_config = hf_config

    @classmethod
    def from_hf_config(cls, hf_config) -> 'MegatronBridgeBackend':
        from megatron.bridge.models.conversion.auto_bridge import AutoBridge
        return cls(AutoBridge.from_hf_config(hf_config), hf_config)

    def load_weights(self, models, hf_model_dir, peft_format=False, adapter_name='default', converter=None):
        if peft_format:
            raise NotImplementedError('LoRA loading via megatron-bridge backend is not yet supported. '
                                      'Please use bridge_backend="mcore-bridge" for LoRA training.')
        if converter is not None:
            logger.warning('converter is not supported by megatron-bridge backend, ignoring.')
        self._bridge.load_hf_weights(models, hf_path=hf_model_dir)

    def export_weights(self,
                       models,
                       target_device=None,
                       only_master_rank=False,
                       peft_format=False,
                       adapter_name='default',
                       converter=None,
                       tqdm_desc='Exporting: ',
                       disable_tqdm=True,
                       _is_saving=False) -> Generator[Tuple[str, 'torch.Tensor'], None, None]:
        if peft_format:
            raise NotImplementedError('LoRA export via megatron-bridge backend is not yet supported. '
                                      'Please use bridge_backend="mcore-bridge" for LoRA training.')
        cpu = (target_device == 'cpu')
        for weight_tuple in self._bridge.export_hf_weights(models, cpu=cpu):
            key = weight_tuple.param_name
            tensor = weight_tuple.weight
            if converter is not None and tensor is not None:
                kv = converter(key, tensor)
                if kv is None:
                    continue
                key, tensor = kv
            yield key, tensor

    def save_weights(self,
                     models,
                     output_dir,
                     peft_format=False,
                     max_shard_size='5GB',
                     args=None,
                     processor=None) -> None:
        if peft_format:
            raise NotImplementedError('LoRA saving via megatron-bridge backend is not yet supported. '
                                      'Please use bridge_backend="mcore-bridge" for LoRA training.')

        # 1. Save weights via megatron-bridge (safetensors format)
        self._bridge.save_hf_weights(models, path=output_dir)

        # 2. Save HF config and tokenizer on rank 0.
        # We use the original HF config (not the one reconstructed by megatron-bridge)
        # because the bridge's config-only path may drop fields like num_attention_heads.
        import torch.distributed as dist
        is_master = (not dist.is_initialized()) or dist.get_rank() == 0
        if is_master and args is not None and self._hf_config is not None:
            from copy import deepcopy

            from swift.model import save_checkpoint
            from swift.utils import HfConfigFactory
            hf_config = deepcopy(self._hf_config)
            llm_config = HfConfigFactory.get_text_config(hf_config)

            # MTP: write back num_nextn_predict_layers
            mtp_num_layers = getattr(args, 'mtp_num_layers', None)
            if mtp_num_layers:
                for key in ['num_nextn_predict_layers', 'mtp_num_hidden_layers']:
                    if hasattr(llm_config, key):
                        setattr(llm_config, key, mtp_num_layers)
                        break
                else:
                    llm_config.num_nextn_predict_layers = mtp_num_layers

            HfConfigFactory.del_config_attr(hf_config, 'quantization_config')

            # FP8: write back quantization_config
            expert_dtype = None
            fp8_format = getattr(args, 'fp8_format', None)
            fp8_recipe = getattr(args, 'fp8_recipe', 'delayed')
            fp8_param = getattr(args, 'fp8_param_gather', False)
            if fp8_format is not None and fp8_recipe == 'blockwise' and fp8_param:
                from transformers.utils.quantization_config import FineGrainedFP8Config
                hf_config.quantization_config = FineGrainedFP8Config()
                expert_dtype = 'fp8'
            if getattr(args, 'model_type', None) == 'deepseek_v4':
                HfConfigFactory.set_config_attr(hf_config, 'expert_dtype', expert_dtype)

            hf_config.save_pretrained(output_dir)
            if processor is not None:
                additional_saved_files = getattr(getattr(processor, 'model_meta', None), 'additional_saved_files', None)
                save_checkpoint(
                    None,
                    processor,
                    output_dir,
                    model_dirs=[args.model_dir],
                    additional_saved_files=additional_saved_files)
        if dist.is_initialized():
            dist.barrier()


def get_mcore_model(args, hf_config):
    bridge_backend = args.bridge_backend
    if bridge_backend == 'megatron-bridge':
        return _get_megatron_bridge_model(args, hf_config)
    config = get_mcore_model_config(args, hf_config)
    models = _get_mcore_model(config)

    return models


def _get_megatron_bridge_model(args, hf_config):
    import dataclasses

    backend = MegatronBridgeBackend.from_hf_config(hf_config)
    auto_bridge = backend._bridge

    # Validate model support via AutoBridge.supports()
    from megatron.bridge.models.conversion.auto_bridge import AutoBridge
    if not AutoBridge.supports(hf_config):
        raise ValueError(f'Model {getattr(hf_config, "model_type", "unknown")} is not supported by '
                         f'megatron-bridge. Please use bridge_backend="mcore-bridge" or check '
                         f'AutoBridge.list_supported_models() for supported architectures.')

    # --- Step 1: Get provider (GPTModelProvider, which extends TransformerConfig) ---
    provider = auto_bridge.to_megatron_provider(load_weights=False)

    # --- Step 2: Build overrides from args ---
    # Auto-match: iterate over provider's dataclass fields and pick up matching args fields.
    # This mirrors mcore-bridge's get_mcore_model_config which does:
    #   for f in fields(ModelConfig): kwargs[f.name] = getattr(args, f.name, None)
    overrides = {}
    provider_fields = {f.name for f in dataclasses.fields(provider)}
    for field_name in provider_fields:
        value = getattr(args, field_name, None)
        if value is None or (isinstance(value, (list, tuple)) and len(value) == 0):
            continue
        overrides[field_name] = value

    # Explicit field name mappings (args name → provider field name)
    explicit_mappings = {
        'decoder_first_pipeline_num_layers': 'num_layers_in_first_pipeline_stage',
        'decoder_last_pipeline_num_layers': 'num_layers_in_last_pipeline_stage',
    }
    for args_key, provider_key in explicit_mappings.items():
        value = getattr(args, args_key, None)
        if value is not None and provider_key in provider_fields:
            overrides[provider_key] = value

    # dtype
    dtype = getattr(args, 'torch_dtype', None)

    # MoE: if no experts, force EP/ETP to 1
    # num_moe_experts comes from HF config (parsed by AutoBridge into provider),
    # not from args — so check provider too.
    num_moe_experts = overrides.get('num_moe_experts') or getattr(provider, 'num_moe_experts', None)
    if num_moe_experts is None:
        overrides['expert_model_parallel_size'] = 1
        overrides['expert_tensor_parallel_size'] = 1

    # Router replay
    if getattr(args, 'router_replay_mode', 'disabled') != 'disabled':
        if 'moe_enable_routing_replay' in provider_fields:
            overrides['moe_enable_routing_replay'] = True

    # megatron_extra_kwargs (user-specified raw overrides)
    if getattr(args, 'megatron_extra_kwargs', None):
        overrides.update(args.megatron_extra_kwargs)

    # padding_free requires variable_seq_lengths=True so that RotaryEmbedding
    # generates freqs matching the actual packed sequence length (cu_seqlens[-1])
    # instead of the fixed seq_length. Without this, mcore-bridge's patcher
    # use_batched_rope check fails and falls back to the original
    # _apply_rotary_pos_emb_thd which calls torch.split on a padded tensor.
    if getattr(args, 'padding_free', False) and 'variable_seq_lengths' in provider_fields:
        overrides['variable_seq_lengths'] = True

    # --- Step 3: Apply overrides and finalize ---
    provider.apply_overrides_and_finalize(dtype=dtype, overrides=overrides)

    # --- Step 4: Create raw models (no DDP/Float16 wrapping) ---
    # swift's wrap_model handles DDP/Float16 wrapping with the correct DDP config from args.
    models = provider.provide_distributed_model(
        wrap_with_ddp=False,
        mixed_precision_wrapper=None,
        use_cpu_initialization=getattr(args, 'use_cpu_initialization', False),
    )
    if not isinstance(models, list):
        models = [models]

    # --- Step 5: Attach backend to model.config.bridge ---
    for model in models:
        model.config.bridge = backend

    logger.info('Created Megatron model via megatron-bridge backend')
    return models
