"""run_convert: HF <-> Megatron(mcore) weight conversion as a dev recipe.

dev counterpart of legacy ``swift export --to_mcore / --to_hf`` (swift/megatron/convert.py::
convert_hf2mcore / convert_mcore2hf).

Split of responsibility, and why it is drawn here:

  - the WEIGHT work stays in legacy ``swift.megatron`` (``get_mcore_model``, the bridge's
    load_weights/save_weights, ``save_mcore_checkpoint``/``load_mcore_checkpoint``,
    ``test_convert_precision``). That code is the format contract -- it decides the on-disk sharding,
    the tensor-name mapping and the dist-checkpoint layout. Reimplementing it against dev's
    MegatronModel would produce a SECOND writer that has to stay bit-compatible with the first one
    forever, so this recipe deliberately calls the existing one.
  - the CONFIG work happens here: dev's atomic Configs are translated into the single
    ``MegatronArguments`` object those functions expect.

Note this recipe does NOT go through ``swift.dev.builders.build_model``. dev's MegatronModel is a
TRAINING surface (device mesh, optimizer, mixed precision) and has no mcore-checkpoint read/write
entry point; conversion needs a plain cpu-initialized mcore model instead, which is what
``get_mcore_model`` returns.
"""
from __future__ import annotations

import logging
import math
import os
import shutil
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

if TYPE_CHECKING:
    from swift.dev.config import (CheckpointConfig, ConvertConfig, DistributedConfig, ModelConfig, TemplateConfig)

logger = logging.getLogger(__name__)

# Forced onto MegatronArguments for every conversion. These are not user knobs: conversion is a
# single-process CPU-side format migration, so optimizer/RNG state is meaningless (no_*_optim/rng),
# weights must be materialized on CPU rather than sharded onto devices (use_cpu_initialization), and
# the attention/recompute settings are pinned to the numerically plainest path so a precision test
# compares weights rather than kernel differences.
CONVERT_KWARGS: Dict[str, Any] = {
    'use_cpu_initialization': True,
    'no_save_optim': True,
    'no_save_rng': True,
    'no_load_optim': True,
    'no_load_rng': True,
    'finetune': True,
    'attention_backend': 'unfused',
    'padding_free': False,
    'recompute_granularity': 'none',  # deepseek-v4
}

# Parallelism sizes are read off DistributedConfig; the mcore checkpoint layout depends on them, so
# they must reach MegatronArguments even though conversion does no training.
_PARALLEL_FIELDS = (
    'tensor_model_parallel_size',
    'pipeline_model_parallel_size',
    'context_parallel_size',
    'expert_model_parallel_size',
    'sequence_parallel',
    'bridge_backend',
)


def run_convert(
    model_config: ModelConfig,
    convert_config: ConvertConfig,
    *,
    template_config: Optional[TemplateConfig] = None,
    distributed_config: Optional[DistributedConfig] = None,
    checkpoint_config: Optional[CheckpointConfig] = None,
    output_dir: Optional[str] = None,
) -> str:
    """Convert between HF and mcore formats; returns the output directory.

    Direction is taken from ``convert_config``:
      - ``to_mcore=True`` with no ``mcore_model``  -> HF  -> mcore
      - ``to_hf=True``                             -> mcore -> HF
      - ``to_mcore=True`` with ``mcore_model``     -> mcore -> mcore (reshard, and/or merge
        ``mcore_adapter``)

    ``output_dir`` overrides ``checkpoint_config.output_dir``.
    """
    if convert_config.to_mcore and convert_config.to_hf:
        raise ValueError('ConvertConfig.to_mcore and to_hf are mutually exclusive; pick one direction.')
    if not convert_config.to_mcore and not convert_config.to_hf:
        raise ValueError('Set either ConvertConfig.to_mcore or ConvertConfig.to_hf.')
    if not model_config.model:
        raise ValueError('ModelConfig.model is required: the HF model/config is what defines the '
                         'target architecture for both conversion directions.')

    resolved_output = output_dir or (checkpoint_config.output_dir if checkpoint_config else None) or 'output'
    if convert_config.to_hf or convert_config.mcore_model or convert_config.mcore_adapter:
        return _convert_mcore(
            model_config,
            convert_config,
            template_config=template_config,
            distributed_config=distributed_config,
            output_dir=resolved_output)
    return _convert_hf2mcore(
        model_config,
        convert_config,
        template_config=template_config,
        distributed_config=distributed_config,
        output_dir=resolved_output)


def _convert_hf2mcore(
    model_config: ModelConfig,
    convert_config: ConvertConfig,
    *,
    template_config: Optional[TemplateConfig],
    distributed_config: Optional[DistributedConfig],
    output_dir: str,
) -> str:
    """HF -> mcore: load the HF weights, hand them to the bridge, write a dist checkpoint."""
    from swift.megatron.model import get_mcore_model
    from swift.megatron.utils import patch_torch_dist_shard, save_mcore_checkpoint, test_convert_precision

    hf_model, template, processor = _load_hf_model_template(
        model_config, template_config, load_model=True, patch_offload=not convert_config.test_convert_precision)
    _set_thread_count(convert_config, hf_model, model_config)
    patch_torch_dist_shard(convert_config.thread_count)

    megatron_args = _build_megatron_args(
        model_config, convert_config, distributed_config, processor=processor, output_dir=output_dir)
    mg_model = get_mcore_model(megatron_args, processor.model_info.config)[0]
    logger.info('Megatron model created successfully.')

    bridge = mg_model.config.bridge
    bridge.load_weights([mg_model], processor.model_info.model_dir)
    logger.info('Successfully transferred HF model weights to MG model.')

    # SWIFT_TEST_CONVERT_PRECISION is the legacy escape hatch used by the align tests: verify the
    # conversion without paying for the (large) checkpoint write.
    if not _skip_save():
        _save_args(megatron_args, output_dir)
        logger.info('Saving the model...')
        save_mcore_checkpoint(megatron_args, [mg_model])
    # Placed last: the test runs forward passes, which can perturb the numerics being measured.
    if convert_config.test_convert_precision:
        test_convert_precision(
            megatron_args, hf_model, mg_model, template, test_convert_dtype=_dtype(convert_config.test_convert_dtype))
    return output_dir


def _convert_mcore(
    model_config: ModelConfig,
    convert_config: ConvertConfig,
    *,
    template_config: Optional[TemplateConfig],
    distributed_config: Optional[DistributedConfig],
    output_dir: str,
) -> str:
    """mcore -> HF, or mcore -> mcore (reshard / merge an mcore LoRA)."""
    from swift.megatron.arguments import MegatronArguments
    from swift.megatron.model import get_mcore_model
    from swift.megatron.utils import (load_mcore_checkpoint, patch_torch_dist_shard, prepare_mcore_model,
                                      save_mcore_checkpoint, test_convert_precision)

    # No HF weights are needed to READ an mcore checkpoint -- only the processor/config for the target
    # architecture -- so the model is not loaded here.
    _, template, processor = _load_hf_model_template(model_config, template_config, load_model=False)

    # The source checkpoint's own args.json carries the tuner/task settings it was written with
    # (tuner_type, task_type, num_labels, bridge_backend). They must win over defaults, otherwise a
    # LoRA checkpoint would be rebuilt as a full model and fail to load.
    extra_config = MegatronArguments.load_args_config(convert_config.mcore_adapter or convert_config.mcore_model)
    extra_config['mcore_adapter'] = convert_config.mcore_adapter
    if convert_config.mcore_model is not None:
        extra_config['mcore_model'] = convert_config.mcore_model

    megatron_args = _build_megatron_args(
        model_config,
        convert_config,
        distributed_config,
        processor=processor,
        # Only the mcore->mcore direction writes a dist checkpoint into output_dir; for ->HF the
        # bridge writes it and MegatronArguments.output_dir would otherwise create a stray dir.
        output_dir=output_dir if convert_config.to_mcore else None,
        extra=extra_config)

    mg_model = get_mcore_model(megatron_args, processor.model_info.config)[0]
    if megatron_args.mcore_model is None:
        raise ValueError('Please specify `ConvertConfig.mcore_model`.')
    load_mcore_checkpoint(megatron_args, [mg_model], load_arg='mcore_model')
    if megatron_args.mcore_adapter is not None:
        peft_model = prepare_mcore_model(megatron_args, mg_model)
        load_mcore_checkpoint(megatron_args, [mg_model], load_arg='mcore_adapter')
        logger.info('Merge LoRA...')
        mg_model = peft_model.merge_and_unload()
    logger.info('Megatron model created successfully.')

    if convert_config.to_hf:
        from swift.dev.utils import is_master

        bridge = mg_model.config.bridge
        logger.info('Converting weights and saving the model...')
        bridge.save_weights([mg_model], output_dir, args=megatron_args, processor=processor)
        if is_master():
            # Prefer the source checkpoint's args.json so the HF export records how the weights were
            # actually trained, not how this conversion was invoked.
            src = convert_config.mcore_adapter or convert_config.mcore_model
            src_args = os.path.join(src, 'args.json') if src else None
            if src_args and os.path.exists(src_args):
                shutil.copy(src_args, os.path.join(output_dir, 'args.json'))
            else:
                _save_args(megatron_args, output_dir)
        if convert_config.test_convert_precision:
            hf_model, template, _ = _load_hf_model_template(
                model_config, template_config, load_model=True, model=output_dir)
            test_convert_precision(
                megatron_args,
                hf_model,
                mg_model,
                template,
                test_convert_dtype=_dtype(convert_config.test_convert_dtype))
    else:
        _set_thread_count(convert_config, mg_model, model_config)
        patch_torch_dist_shard(convert_config.thread_count)
        _save_args(megatron_args, output_dir)
        logger.info('Saving the model...')
        save_mcore_checkpoint(megatron_args, [mg_model])
    return output_dir


def _build_megatron_args(
    model_config: ModelConfig,
    convert_config: ConvertConfig,
    distributed_config: Optional[DistributedConfig],
    *,
    processor,
    output_dir: Optional[str],
    extra: Optional[Dict[str, Any]] = None,
):
    """dev Configs -> the single MegatronArguments object the legacy converters consume.

    MegatronArguments.__post_init__ resolves model_info / model_dir / megatron_model_meta and
    initializes torch.distributed itself, so nothing else has to be pre-populated.
    """
    from swift.megatron.arguments import MegatronArguments

    kwargs = dict(CONVERT_KWARGS)
    if processor.model_info.is_moe_model:
        # Grouped GEMM is the layout the mcore MoE checkpoint is written in; without it the expert
        # weights would be laid out per-expert and not load back.
        kwargs['moe_grouped_gemm'] = True
    if distributed_config is not None:
        for name in _PARALLEL_FIELDS:
            value = getattr(distributed_config, name, None)
            if value is not None:
                kwargs[name] = value
    if extra:
        kwargs.update(extra)

    return MegatronArguments(
        model=model_config.model,
        model_type=model_config.model_type,
        **kwargs,
        output_dir=output_dir,
        # MegatronArguments expects a real torch.dtype here (BaseArguments._init_mixed_precision
        # matches on torch.bfloat16/float16/float32), not dev's string form.
        torch_dtype=_dtype(model_config.torch_dtype) or processor.model_info.torch_dtype)


def _load_hf_model_template(
    model_config: ModelConfig,
    template_config: Optional[TemplateConfig],
    *,
    load_model: bool,
    model: Optional[str] = None,
    patch_offload: bool = False,
) -> Tuple[Any, Any, Any]:
    """Load the HF model (optionally) + template, via dev's own builders.

    legacy calls ``prepare_model_template(args)``, which only works off an ExportArguments instance
    (it dispatches to args.get_model_processor/get_template). dev has the equivalent pair already, so
    those are used instead of synthesizing a fake args object.
    """
    from swift.dev.builders import build_template
    from swift.dev.config import TemplateConfig
    from swift.model import get_model_processor

    kwargs: Dict[str, Any] = {}
    if model_config.torch_dtype:
        kwargs['torch_dtype'] = _dtype(model_config.torch_dtype)
    if model_config.model_type:
        kwargs['model_type'] = model_config.model_type
    if load_model and patch_offload:
        # Keeps the HF weights on CPU/meta where possible so both models can coexist while the
        # precision test runs.
        kwargs['patch_offload'] = True
    hf_model, processor = get_model_processor(model or model_config.model, load_model=load_model, **kwargs)
    template = build_template(template_config or TemplateConfig(), processor)
    if hf_model is not None and getattr(template, 'use_model', False):
        template.model = hf_model
    return hf_model, template, processor


def _set_thread_count(convert_config: ConvertConfig, model, model_config: ModelConfig) -> None:
    """Derive the torch-dist shard-writer thread count from the checkpoint size (legacy: ~1 per 10GB).

    Mutating convert_config here keeps a single source of truth for the value, which both the
    patch and any later log line read.
    """
    if convert_config.thread_count is not None:
        return
    import torch

    from swift.utils import get_n_params_grads

    dtype = _dtype(model_config.torch_dtype) or torch.float32
    checkpoint_size = sum(get_n_params_grads(model)[0]) * torch.finfo(dtype).bits // 8e9
    convert_config.thread_count = max(math.ceil(checkpoint_size / 10), 2)  # 10GB


def _save_args(megatron_args, output_dir: str) -> None:
    """Write args.json next to the weights.

    ``save_args`` lives on legacy's BaseArguments, not on MegatronArguments (which is what this recipe
    builds), so the same file is written directly rather than reaching for a method that is not there.
    """
    from swift.dev.utils import is_master

    if not is_master():
        return
    import json

    from swift.dev.utils import check_json_format

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'args.json')
    logger.info(f'The converted args will be saved in: {path}')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(check_json_format(megatron_args.__dict__), f, ensure_ascii=False, indent=2)


def _dtype(value):
    """dev's string torch_dtype -> the real torch.dtype legacy Arguments/`torch.finfo` require."""
    if value is None:
        return None
    import torch

    if isinstance(value, torch.dtype):
        return value
    return getattr(torch, value)


def _skip_save() -> bool:
    from transformers.utils import strtobool

    return bool(strtobool(os.getenv('SWIFT_TEST_CONVERT_PRECISION', '0')))
