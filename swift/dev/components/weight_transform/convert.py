# Copyright (c) ModelScope Contributors. All rights reserved.
"""Convert between HF and Megatron/mcore checkpoint formats.

Of the three model -> model transforms this is the only one that does NOT stream safetensors itself: it
drives the megatron bridge, whose ``load_weights``/``save_weights`` already migrate tensor by tensor, and
whose sharding is fixed by the torchrun TP/PP/EP model-parallel sizes (orthogonal to the keyspace
data-parallelism ``StreamingWeightTransform`` uses). It therefore extends ``BaseWeightTransform`` and keeps
the load/save calls inside ``run`` rather than the streaming skeleton.
"""
from __future__ import annotations

import dataclasses
import math
import os
import shutil
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from swift.dev.utils.logger import get_logger

from .base import BaseWeightTransform

if TYPE_CHECKING:
    from swift.dev.config import (
        CheckpointConfig,
        ConvertConfig,
        DistributedConfig,
        MegatronConfig,
        ModelConfig,
        MoEConfig,
        TemplateConfig,
    )

logger = get_logger()


class ConvertComponent(BaseWeightTransform):
    """Migrate weights between HF safetensors and the mcore ``torch_dist`` checkpoint.

    Direction is taken from ``convert_config``:
      - ``to_mcore=True`` with no ``mcore_model``  -> HF  -> mcore
      - ``to_hf=True``                             -> mcore -> HF
      - ``to_mcore=True`` with ``mcore_model``     -> mcore -> mcore (reshard, and/or merge
        ``mcore_adapter``)

    The two directions are separate workers (``_convert_hf2mcore`` / ``_convert_mcore``); ``run`` only
    picks between them, so the routing decision stays readable and the weight work stays out of it.
    """

    # Forced onto MegatronArguments for every conversion. These are not user knobs: conversion is a
    # single-process CPU-side format migration, so optimizer/RNG state is meaningless (no_*_optim/rng),
    # weights must be materialized on CPU rather than sharded onto devices (use_cpu_initialization), and
    # the attention/recompute settings are pinned to the numerically plainest path so a precision test
    # compares weights rather than kernel differences.
    _CONVERT_KWARGS: Dict[str, Any] = {
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

    def __init__(
        self,
        model_config: ModelConfig,
        convert_config: ConvertConfig,
        *,
        template_config: Optional[TemplateConfig] = None,
        distributed_config: Optional[DistributedConfig] = None,
        checkpoint_config: Optional[CheckpointConfig] = None,
        megatron_config: Optional[MegatronConfig] = None,
        moe_config: Optional[MoEConfig] = None,
        output_dir: Optional[str] = None,
    ):
        super().__init__(model_config, output_dir=output_dir, checkpoint_config=checkpoint_config)
        self.convert_config = convert_config
        self.template_config = template_config
        self.distributed_config = distributed_config
        self.megatron_config = megatron_config
        self.moe_config = moe_config

    def validate(self) -> None:
        """Refuse an ambiguous or under-specified direction before any megatron import."""
        if self.convert_config.to_mcore and self.convert_config.to_hf:
            raise ValueError('ConvertConfig.to_mcore and to_hf are mutually exclusive; pick one direction.')
        if not self.convert_config.to_mcore and not self.convert_config.to_hf:
            raise ValueError('Set either ConvertConfig.to_mcore or ConvertConfig.to_hf.')
        if not self.model_config.model:
            raise ValueError('ModelConfig.model is required: the HF model/config is what defines the '
                             'target architecture for both conversion directions.')

    def run(self, output_dir: str) -> str:
        """Route to the mcore-reading worker when a mcore source is involved, else HF -> mcore."""
        if self.convert_config.to_hf or self.convert_config.mcore_model or self.convert_config.mcore_adapter:
            return self._convert_mcore(output_dir)
        return self._convert_hf2mcore(output_dir)

    def _convert_hf2mcore(self, output_dir: str) -> str:
        """HF -> mcore: load the HF weights, hand them to the bridge, write a dist checkpoint."""
        from swift.megatron.model import get_mcore_model
        from swift.megatron.utils import patch_torch_dist_shard, save_mcore_checkpoint, test_convert_precision
        hf_model, template, processor = self._load_hf_model_template(
            load_model=True, patch_offload=not self.convert_config.test_convert_precision)
        self._set_thread_count(hf_model)
        patch_torch_dist_shard(self.convert_config.thread_count)

        megatron_args = self._build_megatron_args(processor, output_dir)
        mg_model = get_mcore_model(megatron_args, processor.model_info.config)[0]
        logger.info('Megatron model created successfully.')

        bridge = mg_model.config.bridge
        bridge.load_weights([mg_model], processor.model_info.model_dir)
        logger.info('Successfully transferred HF model weights to MG model.')

        # SWIFT_TEST_CONVERT_PRECISION is the legacy escape hatch used by the align tests: verify the
        # conversion without paying for the (large) checkpoint write.
        if not self._skip_save():
            self._save_args(megatron_args, output_dir)
            logger.info('Saving the model...')
            save_mcore_checkpoint(megatron_args, [mg_model])
        # Placed last: the test runs forward passes, which can perturb the numerics being measured.
        if self.convert_config.test_convert_precision:
            test_convert_precision(
                megatron_args,
                hf_model,
                mg_model,
                template,
                test_convert_dtype=self._dtype(self.convert_config.test_convert_dtype))
        return output_dir

    def _convert_mcore(self, output_dir: str) -> str:
        """mcore -> HF, or mcore -> mcore (reshard / merge an mcore LoRA)."""
        from swift.megatron.arguments import MegatronArguments
        from swift.megatron.model import get_mcore_model
        from swift.megatron.utils import (
            load_mcore_checkpoint,
            patch_torch_dist_shard,
            prepare_mcore_model,
            save_mcore_checkpoint,
            test_convert_precision,
        )

        # No HF weights are needed to READ an mcore checkpoint -- only the processor/config for the target
        # architecture -- so the model is not loaded here.
        _, template, processor = self._load_hf_model_template(load_model=False)

        # The source checkpoint's own args.json carries the tuner/task settings it was written with
        # (tuner_type, task_type, num_labels, bridge_backend). They must win over defaults, otherwise a
        # LoRA checkpoint would be rebuilt as a full model and fail to load.
        extra_config = MegatronArguments.load_args_config(
            self.convert_config.mcore_adapter or self.convert_config.mcore_model)
        extra_config['mcore_adapter'] = self.convert_config.mcore_adapter
        if self.convert_config.mcore_model is not None:
            extra_config['mcore_model'] = self.convert_config.mcore_model

        megatron_args = self._build_megatron_args(
            processor,
            # Only the mcore->mcore direction writes a dist checkpoint into output_dir; for ->HF the
            # bridge writes it and MegatronArguments.output_dir would otherwise create a stray dir.
            output_dir if self.convert_config.to_mcore else None,
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

        if self.convert_config.to_hf:
            from swift.dev.utils import is_master

            bridge = mg_model.config.bridge
            logger.info('Converting weights and saving the model...')
            bridge.save_weights(
                [mg_model],
                output_dir,
                max_shard_size=(self.checkpoint_config.max_shard_size if self.checkpoint_config else '5GB'),
                args=megatron_args,
                processor=processor)
            if is_master():
                # Prefer the source checkpoint's args.json so the HF export records how the weights were
                # actually trained, not how this conversion was invoked.
                src = self.convert_config.mcore_adapter or self.convert_config.mcore_model
                src_args = os.path.join(src, 'args.json') if src else None
                if src_args and os.path.exists(src_args):
                    shutil.copy(src_args, os.path.join(output_dir, 'args.json'))
                else:
                    self._save_args(megatron_args, output_dir)
            if self.convert_config.test_convert_precision:
                # Pin the model_type AND template resolved from the source model (the load_model=False
                # call above): output_dir is a freshly-written dir with no hub name to match, so an
                # unpinned re-load falls back to ambiguous config architectures / candidate templates.
                # legacy gets both for free off its cached args.model_type / args.template.
                resolved_model_type = processor.model_info.model_type
                resolved_template = template.template_meta.template_type
                hf_model, template, _ = self._load_hf_model_template(
                    load_model=True,
                    model=output_dir,
                    model_type=resolved_model_type,
                    template_type=resolved_template)
                test_convert_precision(
                    megatron_args,
                    hf_model,
                    mg_model,
                    template,
                    test_convert_dtype=self._dtype(self.convert_config.test_convert_dtype))
        else:
            self._set_thread_count(mg_model)
            patch_torch_dist_shard(self.convert_config.thread_count)
            self._save_args(megatron_args, output_dir)
            logger.info('Saving the model...')
            save_mcore_checkpoint(megatron_args, [mg_model])
        return output_dir

    def _build_megatron_args(self, processor, output_dir: Optional[str], extra: Optional[Dict[str, Any]] = None):
        """dev Configs -> the single MegatronArguments object the legacy converters consume.

        MegatronArguments.__post_init__ resolves model_info / model_dir / megatron_model_meta and
        initializes torch.distributed itself, so nothing else has to be pre-populated.
        """
        from swift.megatron.arguments import MegatronArguments

        kwargs = dict(self._CONVERT_KWARGS)
        if processor.model_info.is_moe_model:
            # Grouped GEMM is the layout the mcore MoE checkpoint is written in; without it the expert
            # weights would be laid out per-expert and not load back.
            kwargs['moe_grouped_gemm'] = True
        if self.distributed_config is not None:
            for name in self._PARALLEL_FIELDS:
                value = getattr(self.distributed_config, name, None)
                if value is not None:
                    kwargs[name] = value
        accepted = {field.name for field in dataclasses.fields(MegatronArguments)}
        for config in (self.megatron_config, self.moe_config):
            if config is None:
                continue
            explicit = getattr(config, '_explicit_fields', None)
            for field in dataclasses.fields(config):
                value = getattr(config, field.name)
                changed = field.name in explicit if explicit is not None else value != field.default
                if not changed:
                    continue
                if field.name == 'megatron_extra_kwargs':
                    extra_kwargs = value or {}
                    unknown = set(extra_kwargs) - accepted
                    if unknown:
                        raise ValueError(f'Unsupported Megatron conversion options: {sorted(unknown)}.')
                    kwargs.update(extra_kwargs)
                elif field.name in accepted:
                    kwargs[field.name] = value
                else:
                    raise ValueError(f'{type(config).__name__}.{field.name} is not supported by Megatron conversion. '
                                     'Remove it or upgrade the installed Megatron implementation.')
        if extra:
            kwargs.update(extra)

        return MegatronArguments(
            model=self.model_config.model,
            model_type=self.model_config.model_type,
            **kwargs,
            output_dir=output_dir,
            # MegatronArguments expects a real torch.dtype here (BaseArguments._init_mixed_precision
            # matches on torch.bfloat16/float16/float32), not dev's string form.
            torch_dtype=self._dtype(self.model_config.torch_dtype) or processor.model_info.torch_dtype)

    def _load_hf_model_template(
        self,
        *,
        load_model: bool,
        model: Optional[str] = None,
        model_type: Optional[str] = None,
        template_type: Optional[str] = None,
        patch_offload: bool = False,
    ) -> Tuple[Any, Any, Any]:
        """Load the HF model (optionally) + template, via dev's own builders.

        legacy calls ``prepare_model_template(args)``, which only works off an ExportArguments instance
        (it dispatches to args.get_model_processor/get_template). dev has the equivalent pair already, so
        those are used instead of synthesizing a fake args object.

        ``model_type``/``template_type`` pin the types resolved from the source model for a load whose
        ``model`` is a freshly-written directory (the to-HF precision re-load). swift name-matches a hub id
        to a registered model_type/template, but an arbitrary output dir has no name to match and falls
        back to config architectures / candidate templates, which is ambiguous for some families (qwen2 vs
        qwen2_gte; qwen2_5 vs qwq vs deepseek_r1). legacy is immune because BaseArguments caches
        args.model_type and args.template on first resolution; dev's atomic Configs are never mutated, so
        both resolved values are threaded in explicitly.
        """
        from copy import copy

        from swift.dev.builders import build_template, load_model_processor
        from swift.dev.config import TemplateConfig

        model_config = self.model_config
        load_config = model_config
        override_model = model is not None and model != model_config.model
        override_type = model_type is not None and model_config.model_type is None
        if override_model or override_type:
            load_config = copy(model_config)
            if override_model:
                load_config.model = model
            if override_type:
                load_config.model_type = model_type
        build_config = self.template_config or TemplateConfig()
        if template_type is not None and build_config.template is None:
            build_config = copy(build_config)
            build_config.template = template_type
        kwargs: Dict[str, Any] = {}
        if load_model and patch_offload:
            # patch_offload turns accelerate's align-device hook into a no-op (swift/model/patcher.py).
            # HF->mcore conversion only has the bridge READ the HF weights tensor by tensor and never runs
            # an HF forward, so the hook that shuttles offloaded blocks onto the GPU per forward is pure
            # overhead here. The precision test is the exception -- it DOES run an HF forward to compare
            # against mcore -- which is why the caller passes patch_offload=not test_convert_precision.
            kwargs['patch_offload'] = True
        hf_model, processor = load_model_processor(load_config, load_model=load_model, **kwargs)
        template = build_template(build_config, processor)
        if hf_model is not None and getattr(template, 'use_model', False):
            template.model = hf_model
        return hf_model, template, processor

    def _set_thread_count(self, model) -> None:
        """Derive the torch-dist shard-writer thread count from the checkpoint size (legacy: ~1 per 10GB).

        Mutating convert_config here keeps a single source of truth for the value, which both the
        patch and any later log line read.
        """
        if self.convert_config.thread_count is not None:
            return
        import torch

        from swift.dev.utils import get_n_params_grads

        dtype = self._dtype(self.model_config.torch_dtype) or torch.float32
        checkpoint_size = sum(get_n_params_grads(model)[0]) * torch.finfo(dtype).bits // 8e9
        self.convert_config.thread_count = max(math.ceil(checkpoint_size / 10), 2)  # 10GB

    @staticmethod
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

    @staticmethod
    def _dtype(value):
        """dev's string torch_dtype -> the real torch.dtype legacy Arguments/`torch.finfo` require."""
        if value is None:
            return None
        import torch

        if isinstance(value, torch.dtype):
            return value
        return getattr(torch, value)

    @staticmethod
    def _skip_save() -> bool:
        from transformers.utils import strtobool

        return bool(strtobool(os.getenv('SWIFT_TEST_CONVERT_PRECISION', '0')))
