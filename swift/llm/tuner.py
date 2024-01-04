# Copyright (c) Alibaba, Inc. and its affiliates.

from swift.tuners import (LongLoRAConfig, LongLoRAModelType, LoraConfig,
                          LoRAConfig, NEFTuneConfig, Swift)
from swift.utils import freeze_model_parameters, get_logger
from .utils import SftArguments, find_all_linear_for_lora

logger = get_logger()


def prepare_model(model, args: SftArguments):
    # Preparing LoRA
    if args.sft_type in ('lora', 'qalora', 'longlora'):
        if args.resume_from_checkpoint is None:
            if 'ALL' in args.lora_target_modules:
                assert len(args.lora_target_modules) == 1
                args.lora_target_modules = find_all_linear_for_lora(
                    model, args.quantization_bit, args.model_type)
                logger.info(
                    f'Setting lora_target_modules: {args.lora_target_modules}')
            if args.sft_type == 'lora':
                lora_kwargs = {}
                if args.tuner_backend == 'swift':
                    lora_config_cls = LoRAConfig
                elif args.tuner_backend == 'peft':
                    lora_config_cls = LoraConfig
                    lora_kwargs['task_type'] = 'CAUSAL_LM'
                lora_config = lora_config_cls(
                    r=args.lora_rank,
                    target_modules=args.lora_target_modules,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout_p,
                    lora_dtype=args.lora_dtype,
                    **lora_kwargs)
                model = Swift.prepare_model(model, lora_config)
                logger.info(f'lora_config: {lora_config}')
            elif args.sft_type == 'longlora':
                assert args.tuner_backend != 'peft', (
                    'peft does not support longlora. You need to set `--tuner_backend swift`.'
                )
                assert LongLoRAModelType.LLAMA in args.model_type
                longlora_config = LongLoRAConfig(
                    r=args.lora_rank,
                    target_modules=args.lora_target_modules,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout_p,
                    lora_dtype=args.lora_dtype,
                    model_type=LongLoRAModelType.LLAMA,
                    use_flash_attn=args.use_flash_attn)
                model = Swift.prepare_model(model, longlora_config)
                logger.info(f'longlora_config: {longlora_config}')
            elif args.sft_type == 'qalora':
                assert getattr(
                    model, 'quantization_method',
                    None) == 'gptq', 'qalora must be used with auto_gptq'
                lora_kwargs = {}
                lora_config = LoRAConfig(
                    r=args.lora_rank,
                    target_modules=args.lora_target_modules,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout_p,
                    lora_dtype=args.lora_dtype,
                    use_qa_lora=True,
                    **lora_kwargs)
                model = Swift.prepare_model(model, lora_config)
                logger.info(f'lora_config: {lora_config}')
        else:
            model = Swift.from_pretrained(
                model, args.resume_from_checkpoint, is_trainable=True)
    elif args.sft_type == 'full':
        if args.freeze_parameters > 0:
            freeze_model_parameters(model, args.freeze_parameters)
    else:
        raise ValueError(f'args.sft_type: {args.sft_type}')

    if args.neftune_alpha > 0.001:
        neftune_config = NEFTuneConfig(noise_alpha=args.neftune_alpha)
        model = Swift.prepare_model(model, neftune_config)
        logger.info(f'neftune_config: {neftune_config}')
    return model
