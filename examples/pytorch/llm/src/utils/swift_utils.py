# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

from torch.nn import Module

from swift import (AdapterConfig, LoRAConfig, ResTuningConfig, Swift,
                   SwiftConfig, SwiftTuners, get_logger)
from .model import MODEL_MAPPING
from .utils import find_all_linear_for_lora

logger = get_logger()


def prepare_model(model: Module, args) -> Module:
    swift_config: Dict[str, SwiftConfig] = dict()
    for sft_type in [_type.strip() for _type in args.sft_type.split(',')]:
        if sft_type.lower() == SwiftTuners.LORA.lower():
            if 'ALL' in args.lora_target_modules:
                assert len(args.lora_target_modules) == 1
                args.lora_target_modules = find_all_linear_for_lora(
                    model, args.quantization_bit, args.model_type)
                logger.info(
                    f'Setting lora_target_modules: {args.lora_target_modules}')

            lora_config = LoRAConfig(
                r=args.lora_rank,
                target_modules=args.lora_target_modules,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout_p)
            logger.info(f'lora_config: {lora_config}')
            swift_config['lora'] = lora_config
        elif sft_type.lower() == SwiftTuners.ADAPTER.lower():
            adapter_config = AdapterConfig(
                dim=model.config.hidden_size,
                target_modules=MODEL_MAPPING[args.model_type].get(
                    'adapter_TM', ['mlp']),
                method_name='forward',
                hidden_pos=0,
                adapter_length=args.adapter_length,
            )
            logger.info(f'adapter_config: {adapter_config}')
            swift_config['adapter'] = adapter_config
        elif sft_type.lower() == SwiftTuners.RESTUNING.lower():
            restuner_config = ResTuningConfig(
                dims=model.config.hidden_size,
                **MODEL_MAPPING[args.model_type]['restuner_TM'])
            logger.info(f'restuner_config: {restuner_config}')
            swift_config['restuner'] = restuner_config
    return Swift.prepare_model(model, swift_config)
