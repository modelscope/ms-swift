# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import types

import numpy as np
import torch
import transformers
from packaging import version

from swift.trainers import TrainerCallback
from swift.tuners import (AdaLoraConfig, AdapterConfig, BOFTConfig, IA3Config, LLaMAProConfig, LongLoRAModelType,
                          LoraConfig, LoRAConfig, NEFTuneConfig, ReftConfig, Swift, VeraConfig)
from swift.utils import activate_model_parameters, freeze_model_parameters, get_logger, use_torchacc
from swift.utils.module_mapping import MODEL_KEYS_MAPPING
from .utils import SftArguments, find_all_linears, find_embedding, find_ln, get_model_with_value_head, is_adapter
from .utils.callbacks import DynamicLayerActivationCallback, TrainerAdapterCallback

logger = get_logger()


def handle_target_modules(model, args: SftArguments) -> None:
    if args.sft_type == 'ia3':
        assert len(args.ia3_feedforward_modules) > 0, ('Setting ia3_target_modules to `ALL` '
                                                       'need to pass MLP linear names to `ia3_feedforward_modules`')
    target_modules = args.target_modules
    if args.lora_use_embedding:
        target_modules.remove('EMBEDDING')
        target_modules += find_embedding(model)
    if args.lora_use_all:
        target_modules.remove('ALL')
        target_modules += find_all_linears(model, args.quantization_bit, args.model_type, args.quant_method)
    args.target_modules = target_modules
    if not args.target_regex:
        logger.info(f'target_modules: {args.target_modules}')


def handle_same_dim_target_modules(model: torch.nn.Module, config: VeraConfig):
    target_modules = config.target_modules
    modules_dict = {
        name: module.weight.shape
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and any([t in name for t in target_modules])
    }  # only Linear for now
    if len(set(modules_dict.values())) > 1:
        v = [t for t in target_modules if 'v' in t]
        if not v:
            raise ValueError('Please manually pass in `vera_target_modules`, do not use `DEFAULT` or `ALL`,'
                             'because Vera need all target linears to be the same size.')
        v = v[0]
        shape = [shape for name, shape in modules_dict.items() if v in name][0]
        names = [_name for _name, _shape in modules_dict.items() if _shape == shape]
        config.target_modules = [t for t in target_modules if any([t in name for name in names])]
    return config


def handle_modules_to_save(model, args: SftArguments) -> None:
    modules_to_save = args.modules_to_save
    if args.lora_m2s_use_embedding:
        modules_to_save += find_embedding(model)
    if args.lora_m2s_use_ln:
        modules_to_save += find_ln(model)
    args.modules_to_save = modules_to_save
    logger.info(f'modules_to_save: {args.modules_to_save}')


def apply_liger(model_type: str):
    from liger_kernel.transformers import (apply_liger_kernel_to_llama, apply_liger_kernel_to_mistral,
                                           apply_liger_kernel_to_mixtral, apply_liger_kernel_to_gemma,
                                           apply_liger_kernel_to_qwen2)
    if 'llama3' in model_type:
        apply_liger_kernel_to_llama()
    elif 'mistral' in model_type:
        apply_liger_kernel_to_mistral()
    elif 'mixtral' in model_type:
        apply_liger_kernel_to_mixtral()
    elif 'gemma' in model_type:
        apply_liger_kernel_to_gemma()
    elif 'qwen2' in model_type:
        apply_liger_kernel_to_qwen2()
    else:
        raise ValueError(f'Unsupported liger model_type: {model_type}')


def prepare_model(model, args: SftArguments):
    if args.use_liger:
        # Apply liger
        apply_liger(args.model_type)

    # This model_type is used to map the model structure
    model_type = args.model_type or args.model_id_or_path
    for key in MODEL_KEYS_MAPPING.keys():
        if key in model_type.lower():
            model_type = key
            break

    # Preparing LoRA
    if is_adapter(args.sft_type):
        if args.resume_from_checkpoint is None:
            handle_target_modules(model, args)
            handle_modules_to_save(model, args)
            if args.init_lora_weights and isinstance(args.init_lora_weights,
                                                     str) and args.init_lora_weights.lower() in ('true', 'false'):
                args.init_lora_weights = args.init_lora_weights.lower() in ('true', 'True')
            if args.target_regex:
                logger.info(f'Value of target_modules: `{args.target_modules}` will have no effect '
                            f'because target_regex value: `{args.target_regex}` exists.')
            lora_kwargs = {
                'r': args.lora_rank,
                'target_modules': args.target_regex or args.target_modules,
                'lora_alpha': args.lora_alpha,
                'lora_dropout': args.lora_dropout,
                'bias': args.lora_bias_trainable,
                'modules_to_save': args.modules_to_save,
                'use_rslora': args.use_rslora,
                'use_dora': args.use_dora,
                'lorap_lr_ratio': args.lora_lr_ratio,
                'init_lora_weights': args.init_lora_weights,
            }

            if args.sft_type in ('lora', 'longlora'):
                # Fix the name of the layer in xcomposer that contains Plora.
                if any(['lora_' in n for n, p in model.named_parameters()]):
                    model.requires_grad_(False)
                if args.lora_dtype == 'AUTO':
                    args.lora_dtype = None
                if args.tuner_backend == 'swift':
                    lora_config = LoRAConfig(lora_dtype=args.lora_dtype, **lora_kwargs)
                    model = Swift.prepare_model(model, lora_config)
                    logger.info(f'lora_config: {lora_config}')
                elif args.tuner_backend == 'peft':
                    lora_config = LoraConfig(task_type='CAUSAL_LM', lora_dtype=args.lora_dtype, **lora_kwargs)
                    model = Swift.prepare_model(model, lora_config)
                    logger.info(f'lora_config: {lora_config}')
                elif args.tuner_backend == 'unsloth':
                    from unsloth import FastLanguageModel
                    assert args.sft_type == 'lora', 'Unsloth does not support LongLoRA'
                    lora_kwargs.pop('lorap_lr_ratio')
                    model = FastLanguageModel.get_peft_model(
                        model,
                        use_gradient_checkpointing=True,
                        max_seq_length=args.max_length,
                        **lora_kwargs,
                    )
                    logger.info(f'unsloth_config: {lora_kwargs}')
                if args.sft_type == 'longlora':
                    assert LongLoRAModelType.LLAMA in args.model_type
                    assert version.parse(transformers.__version__) >= version.parse('4.39.3')
                    from swift.tuners.longlora.llama import replace_llama_attn
                    replace_llama_attn(model)
                    model.config.group_size_ratio = 0.25
            elif args.sft_type == 'adalora':
                lora_kwargs.pop('lorap_lr_ratio', None)
                lora_kwargs['rank_pattern'] = None
                adalora_config = AdaLoraConfig(
                    task_type='CAUSAL_LM',
                    **lora_kwargs,
                    target_r=args.adalora_target_r,
                    init_r=args.adalora_init_r,
                    tinit=args.adalora_tinit,
                    tfinal=args.adalora_tfinal,
                    deltaT=args.adalora_deltaT,
                    beta1=args.adalora_beta1,
                    beta2=args.adalora_beta2,
                    orth_reg_weight=args.adalora_orth_reg_weight,
                )
                model = Swift.prepare_model(model, adalora_config)
                logger.info(f'adalora_config: {adalora_config}')
            elif args.sft_type == 'ia3':
                ia3_config = IA3Config(
                    task_type='CAUSAL_LM',
                    target_modules=args.target_modules,
                    feedforward_modules=args.ia3_feedforward_modules or [],
                    modules_to_save=args.modules_to_save,
                )
                model = Swift.prepare_model(model, ia3_config)
                logger.info(f'ia3_config: {ia3_config}')
            elif args.sft_type == 'llamapro':
                llamapro_config = LLaMAProConfig(
                    model_type=model_type,
                    num_new_blocks=args.llamapro_num_new_blocks,
                    num_groups=args.llamapro_num_groups)
                model = Swift.prepare_model(model, llamapro_config)
                logger.info(f'llamapro_config: {llamapro_config}')
            elif args.sft_type == 'adapter':
                assert model_type in MODEL_KEYS_MAPPING
                mlp_key = MODEL_KEYS_MAPPING[model_type].mlp
                mlp_key = mlp_key.split('.{}.')[1]
                adapter_config = AdapterConfig(
                    dim=model.config.hidden_size,
                    target_modules=[mlp_key],
                    hidden_pos=0,
                    adapter_length=args.adapter_length,
                    act_layer=args.adapter_act)
                model = Swift.prepare_model(model, adapter_config)
                logger.info(f'adapter_config: {adapter_config}')
            elif args.sft_type == 'vera':
                vera_config = VeraConfig(
                    r=args.vera_rank,
                    target_modules=args.target_modules,
                    projection_prng_key=args.vera_projection_prng_key,
                    vera_dropout=args.vera_dropout,
                    d_initial=args.vera_d_initial,
                    modules_to_save=args.modules_to_save,
                )
                vera_config = handle_same_dim_target_modules(model, vera_config)
                model = Swift.prepare_model(model, vera_config)
                logger.info(f'vera_config: {vera_config}')
            elif args.sft_type == 'boft':
                boft_config = BOFTConfig(
                    boft_block_size=args.boft_block_size,
                    boft_block_num=args.boft_block_num,
                    boft_n_butterfly_factor=args.boft_n_butterfly_factor,
                    target_modules=args.target_modules,
                    boft_dropout=args.boft_dropout,
                    modules_to_save=args.modules_to_save,
                )
                model = Swift.prepare_model(model, boft_config)
                logger.info(f'boft_config: {boft_config}')
            elif args.sft_type == 'fourierft':
                from peft import FourierFTConfig
                fourier_config = FourierFTConfig(
                    target_modules=args.target_modules,
                    modules_to_save=args.modules_to_save,
                    n_frequency=args.fourier_n_frequency,
                    scaling=args.fourier_scaling,
                )
                model = Swift.prepare_model(model, fourier_config)
                logger.info(f'fourier_config: {fourier_config}')
            elif args.sft_type == 'reft':
                reft_config = ReftConfig(
                    model_type=model_type,
                    layer_key=args.reft_layer_key,
                    r=args.reft_rank,
                    layers=args.reft_layers,
                    intervention_type=args.reft_intervention_type,
                    args=args.reft_args,
                )
                logger.info(f'reft config: {reft_config}')
                model = Swift.prepare_model(model, {'reft': reft_config})
        else:
            if use_torchacc():
                model = Swift.from_pretrained(
                    model, args.resume_from_checkpoint, adapter_name='default', is_trainable=True)
            else:
                model = Swift.from_pretrained(model, args.resume_from_checkpoint, is_trainable=True)
        # fix bug: Attempting to unscale FP16 gradients.
        #   peft: https://github.com/huggingface/peft/issues/1249
        #   modules_to_save + fp16
        is_logging = False
        for p in model.parameters():
            if p.requires_grad and p.dtype == torch.float16:
                if not is_logging:
                    logger.info('Convert trainable parameters from fp16 to fp32.')
                    is_logging = True
                p.data = p.data.to(dtype=torch.float32)
    elif args.sft_type == 'full':
        model.train()
        model.requires_grad_(True)

        freeze_model_parameters(model, args.freeze_parameters_ratio, args.freeze_parameters)
        if len(args.additional_trainable_parameters) > 0:
            activate_model_parameters(model, args.additional_trainable_parameters)
        if use_torchacc() and args.resume_from_checkpoint is not None:
            import safetensors
            weights_file = os.path.join(args.resume_from_checkpoint, 'pytorch_model.bin')
            safe_weights_file = os.path.join(args.resume_from_checkpoint, 'model.safetensors')
            if os.path.isfile(weights_file) or os.path.isfile(safe_weights_file):
                if args.save_safetensors and os.path.isfile(safe_weights_file):
                    state_dict = safetensors.torch.load_file(safe_weights_file, device='cpu')
                else:
                    state_dict = torch.load(weights_file, map_location='cpu')
                model.load_state_dict(state_dict, False)
                del state_dict
            else:
                from transformers.modeling_utils import load_sharded_checkpoint
                # We load the sharded checkpoint
                load_result = load_sharded_checkpoint(
                    model, args.resume_from_checkpoint, strict=False, prefer_safe=args.save_safetensors)
                if len(load_result.missing_keys) != 0:
                    if model._keys_to_ignore_on_save is not None and set(load_result.missing_keys) == set(
                            model._keys_to_ignore_on_save):
                        model.tie_weights()
                    else:
                        logger.warning(
                            f'There were missing keys in the checkpoint model loaded: {load_result.missing_keys}.')
                if len(load_result.unexpected_keys) != 0:
                    logger.warning(
                        f'There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}.')
    else:
        raise ValueError(f'args.sft_type: {args.sft_type}')

    if args.sequence_parallel_size > 1:
        from swift.trainers.xtuner import dispatch_module_xtuner
        dispatch_module_xtuner(model)
    if args.neftune_backend == 'swift' and args.neftune_noise_alpha not in {None, 0.}:
        neftune_config = NEFTuneConfig(noise_alpha=args.neftune_noise_alpha)
        model = Swift.prepare_model(model, {'neftune': neftune_config})
        logger.info(f'neftune_config: {neftune_config}')

    if args.use_galore:
        from swift.trainers.optimizers.galore import GaLoreConfig
        if args.galore_target_modules is None:
            args.galore_target_modules = find_all_linears(model, 0, args.model_type, args.quant_method)
        if args.galore_with_embedding:
            args.galore_target_modules += find_embedding(model)
        args.training_args.galore_config = GaLoreConfig(
            target_modules=args.galore_target_modules,
            rank=args.galore_rank,
            update_proj_gap=args.galore_update_proj_gap,
            galore_scale=args.galore_scale,
            proj_type=args.galore_proj_type,
            optim_per_parameter=args.galore_optim_per_parameter,
            quantize=args.galore_quantization,
            proj_quant=args.galore_proj_quant,
            proj_bits=args.galore_proj_bits,
            proj_group_size=args.galore_proj_group_size,
            cos_threshold=args.galore_cos_threshold,
            gamma_proj=args.galore_gamma_proj,
            queue_size=args.galore_queue_size,
        )

    callbacks = []
    if args.lisa_activated_layers > 0:
        assert args.sft_type == 'full', 'LISA only supports full parameter training.'
        lisa_callback = DynamicLayerActivationCallback(
            n_layers=args.lisa_activated_layers,  # Number of layers to activate
            step_interval=args.lisa_step_interval,  # Step interval to update active layers
            model=model)
        lisa_callback.switch_active_layers()  # Make trainable parameters printing a correct value
        callbacks.append(lisa_callback)

    # add value head for reward model
    if args.train_type == 'rm':
        model = get_model_with_value_head(model)

    if is_adapter(args.sft_type) and args.tuner_backend == 'swift':
        callbacks.append(TrainerAdapterCallback(args))
    return model, callbacks
