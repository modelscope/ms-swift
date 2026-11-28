# Copyright (c) ModelScope Contributors. All rights reserved.
"""HF rank-zero loading and native model construction for the Swift backend."""
import importlib
import json
import torch
import torch.distributed as dist
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch


def initialize_native_lora(adapter_model: torch.nn.Module, adapters: list[str]) -> None:
    """Initialize or reload materialized native adapters after base checkpoint loading.

    PEFT injection into a meta backbone cannot retain initialized tensor values.
    Only adapter tensors are touched here; native base/Host weights stay intact.
    """
    from peft.tuners.lora.layer import LoraLayer
    from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

    adapter_name = next(iter(adapter_model.peft_config))
    if adapters:
        weights = load_peft_weights(adapters[0], device='cpu')
        result = set_peft_model_state_dict(adapter_model, weights, adapter_name=adapter_name)
        missing = [key for key in result.missing_keys if '.lora_' in key]
        if missing or result.unexpected_keys:
            raise ValueError(f'Incomplete native LoRA checkpoint: {missing}, {result.unexpected_keys}')
    else:
        config = adapter_model.peft_config[adapter_name]
        for module in adapter_model.modules():
            if isinstance(module, LoraLayer):
                module.reset_lora_parameters(adapter_name, init_lora_weights=config.init_lora_weights)


@contextmanager
def _checked_checkpoint_read():
    """A partially loaded checkpoint must fail before rank-0 broadcasting."""
    from transformers import PreTrainedModel

    original = PreTrainedModel.from_pretrained.__func__

    def load(cls, *args, **kwargs):
        kwargs['output_loading_info'] = True
        model, info = original(cls, *args, **kwargs)
        failures = {key: info[key] for key in ('missing_keys', 'mismatched_keys', 'error_msgs') if info.get(key)}
        if failures:
            raise ValueError(f'Incomplete pretrained checkpoint: {failures}')
        return model

    with patch.object(PreTrainedModel, 'from_pretrained', classmethod(load)):
        yield


@contextmanager
def _fp8_checkpoint_conversion(dtype=None):
    """Repair anchored single-source expert conversion in Transformers 5.9.

    HF's quantizer only adds dequantization to sources ending in ``.weight``;
    the V4 down-projection source has already become ``.weight$``. Retain HF's
    conversion operations and let the quantizer add the final anchors itself.
    The override exists only during this rank-0 checkpoint read.
    """
    from transformers.core_model_loading import WeightConverter
    from transformers.integrations.finegrained_fp8 import Fp8Dequantize
    from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer

    original = FineGrainedFP8HfQuantizer.update_weight_conversions
    original_dequantize = Fp8Dequantize._dequantize_one

    def dequantize(self, quantized, scales):
        result = original_dequantize(self, quantized, scales)
        return result.to(dtype) if dtype is not None else result

    def update(self, conversions, *args, **kwargs):
        fixed = []
        for conversion in conversions:
            if isinstance(conversion, WeightConverter) and (
                    any(source.endswith('.weight$') for source in conversion.source_patterns)
                    or any(target.endswith('$') for target in conversion._original_target_patterns)):
                replacement = WeightConverter(
                    source_patterns=[source.removesuffix('$') for source in conversion.source_patterns],
                    target_patterns=[target.removesuffix('$') for target in conversion._original_target_patterns],
                    operations=conversion.operations,
                )
                # scope_prefix was introduced after Transformers 5.8.
                if hasattr(conversion, 'scope_prefix'):
                    replacement.scope_prefix = conversion.scope_prefix
                conversion = replacement
            fixed.append(conversion)
        return original(self, fixed, *args, **kwargs)

    with patch.object(FineGrainedFP8HfQuantizer, 'update_weight_conversions', update), \
            patch.object(Fp8Dequantize, '_dequantize_one', dequantize):
        yield


def load_rank0_model(args):
    """Read weights once globally; propagate loading errors before mesh setup."""
    source_model = None
    payload = [None]
    if dist.get_rank() == 0:
        try:
            if args.resume_from_checkpoint:
                from torch.distributed.checkpoint import FileSystemReader
                metadata = FileSystemReader(args.resume_from_checkpoint).read_metadata()
                prefix = 'app.model.model.'  # app state -> FSDPTurbo wrapper -> HF model
                dtypes = {
                    name.removeprefix(prefix): value.properties.dtype
                    for name, value in metadata.state_dict_metadata.items() if name.startswith(prefix)
                }
                if not dtypes:
                    raise ValueError('Checkpoint has no FSDPTurbo model tensor metadata.')
                payload[0] = (dtypes, None)
            else:
                source_model = _read_pretrained_model(args)
                payload[0] = ({name: param.dtype for name, param in source_model.named_parameters()}, None)
        except Exception as error:
            payload[0] = (None, f'{type(error).__name__}: {error}')
    dist.broadcast_object_list(payload, src=0)
    dtypes, error = payload[0]
    if error is not None:
        raise RuntimeError(f'FSDPTurbo rank-0 checkpoint loading failed: {error}')
    return source_model, dtypes


def _read_pretrained_model(args):
    kwargs = {'device_map': 'cpu'}
    if args.model_info.quant_method == 'fp8':
        from transformers import FineGrainedFP8Config
        kwargs['quantization_config'] = FineGrainedFP8Config(dequantize=True)
        with _checked_checkpoint_read(), _fp8_checkpoint_conversion(args.torch_dtype):
            source_model, _ = args.get_model_processor(**kwargs)
        for name, param in source_model.named_parameters():
            if not param.is_floating_point() or param.element_size() < 2:
                raise ValueError(f'Checkpoint parameter {name} was not dequantized: {param.dtype}')
    else:
        with _checked_checkpoint_read():
            source_model, _ = args.get_model_processor(**kwargs)
    return source_model


def _native_factory(args):
    from functools import partial

    from .model_specs import get_model_spec

    model_type = getattr(args, 'model_type', None)
    if model_type is None:
        return None
    path = get_model_spec(model_type).native_model_factory
    if path is None:
        return None
    if args.torch_dtype != torch.bfloat16:
        raise ValueError('Native DeepSeek-V4.1 pretrained training requires BF16.')
    module_name, _, name = path.rpartition('.')
    module = importlib.import_module(module_name)
    factory = getattr(module, name)
    from transformers import PreTrainedTokenizerFast

    model_kwargs = getattr(args, 'model_kwargs', None) or {}
    model_args = module.build_deepseek_v41_model_args(
        config_path=model_kwargs.get('config_path'), model_name_or_path=args.model)
    from .data import _native_cp_alignment

    alignment = _native_cp_alignment(model_args, getattr(args, 'cp_size', 1))
    cache_length = (args.max_length + alignment - 1) // alignment * alignment
    model_args = replace(
        model_args,
        max_seq_len=cache_length,
        max_batch_size=max(
            args.training_args.per_device_train_batch_size,
            getattr(args.training_args, 'per_device_eval_batch_size', 1)
            if getattr(args.training_args, 'do_eval', False) else 1))
    options = {key: value for key, value in model_kwargs.items() if key != 'config_path'}
    # Keep architecture and tokenizer selection owned by the checked loader;
    # expose the native builder's memory and operator options without silently
    # discarding Swift --model_kwargs (notably chunk_loss at long sequence).
    import inspect
    allowed = set(
        inspect.signature(factory).parameters) - {'model_args', 'tokenizer', 'config_path', 'model_name_or_path'}
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f'Unsupported native V4.1 model_kwargs: {sorted(unknown)}')
    return partial(
        factory, model_args=model_args, tokenizer=PreTrainedTokenizerFast.from_pretrained(Path(args.model)), **options)


def create_meta_model(args):
    native = _native_factory(args)
    if native is not None:
        from transformers import PretrainedConfig

        model = native()
        config = json.loads((Path(args.model) / 'config.json').read_text())
        model.config = PretrainedConfig()
        model.config.update(config)
        model.config.dtype = args.torch_dtype
        for name in ('text_config', 'vision_config'):
            if config.get(name):
                nested = PretrainedConfig()
                nested.update(config[name])
                nested.dtype = args.torch_dtype
                setattr(model.config, name, nested)
        model.name_or_path = args.model
        model.model.head.weight.data = model.model.head.weight.data.to(torch.bfloat16)
        return model
    model, _ = args.get_model_processor(device_map='cpu', return_dummy_model=True)
    return model


def materialize_resume_model(model, *, device):
    """Allocate only local shards; DCP restores them after optimizer creation."""
    buffers = [(module, name, buffer) for module in model.modules()
               for name, buffer in module.named_buffers(recurse=False) if name in module._non_persistent_buffers_set]
    model.to_empty(device=device)
    for module, name, buffer in buffers:
        setattr(module, name, buffer.to(device))


def materialize_sharded_model(model, source_model, *, offload=False):
    """Broadcast each tensor into rank-local shards using PyTorch DCP.

    Meta construction preserves non-persistent buffers (for example rotary
    frequencies). All parameters and persistent buffers come from HF's loaded
    state, including its expert-layout and checkpoint dequantization mappings.
    """
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    state_dict = source_model.state_dict() if source_model is not None else {}
    set_model_state_dict(
        model,
        state_dict,
        options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, cpu_offload=offload),
    )
    missing = [name for name, tensor in list(model.named_parameters()) + list(model.named_buffers()) if tensor.is_meta]
    if missing:
        raise RuntimeError(f'FSDPTurbo checkpoint left meta tensors: {missing}')
