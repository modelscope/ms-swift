# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import types

import numpy as np
import torch
import transformers
from packaging import version

from swift.torchacc_utils import consolidate_checkpoint
from swift.trainers import TrainerCallback
from swift.tuners import (AdaLoraConfig, AdapterConfig, BOFTConfig, IA3Config, LongLoRAModelType, LoraConfig,
                          LoRAConfig, NEFTuneConfig, Swift, VeraConfig)
from swift.tuners.llamapro import LLaMAProConfig
from swift.tuners.module_mapping import MODEL_KEYS_MAPPING
from swift.utils import activate_model_parameters, freeze_model_parameters, get_logger, use_torchacc
from .utils import SftArguments, find_all_linears, find_embedding, find_ln, is_adapter

logger = get_logger()


def handle_target_modules(model, args: SftArguments) -> None:
    if args.sft_type == 'ia3':
        target_modules = args.ia3_target_modules
        assert len(args.ia3_feedforward_modules) > 0, ('Setting ia3_target_modules to `ALL` '
                                                       'need to pass MLP linear names to `ia3_feedforward_modules`')
    elif args.sft_type == 'vera':
        target_modules = args.vera_target_modules
    elif args.sft_type == 'boft':
        target_modules = args.boft_target_modules
    else:
        target_modules = args.lora_target_modules
    if args.lora_use_embedding:
        target_modules += find_embedding(model)
    if args.lora_use_all:
        target_modules += find_all_linears(model, args.quantization_bit, args.model_type, args.quant_method)
    if args.sft_type == 'ia3':
        args.ia3_target_modules = target_modules
        logger.info(f'ia3_target_modules: {args.ia3_target_modules}')
    elif args.sft_type == 'vera':
        args.vera_target_modules = target_modules
        logger.info(f'vera_target_modules: {args.ia3_target_modules}')
    elif args.sft_type == 'boft':
        args.boft_target_modules = target_modules
        logger.info(f'boft_target_modules: {args.boft_target_modules}')
    else:
        args.lora_target_modules = target_modules
        logger.info(f'lora_target_modules: {args.lora_target_modules}')


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
    if args.sft_type == 'ia3':
        modules_to_save = args.ia3_modules_to_save
    elif args.sft_type == 'vera':
        modules_to_save = args.vera_modules_to_save
    elif args.sft_type == 'boft':
        modules_to_save = args.boft_modules_to_save
    else:
        modules_to_save = args.lora_modules_to_save
    if args.lora_m2s_use_embedding:
        modules_to_save += find_embedding(model)
    if args.lora_m2s_use_ln:
        modules_to_save += find_ln(model)

    if args.sft_type == 'ia3':
        args.ia3_modules_to_save = modules_to_save
        logger.info(f'ia3_modules_to_save: {args.ia3_modules_to_save}')
    elif args.sft_type == 'vera':
        args.vera_modules_to_save = modules_to_save
        logger.info(f'vera_modules_to_save: {args.vera_modules_to_save}')
    elif args.sft_type == 'boft':
        args.boft_modules_to_save = modules_to_save
        logger.info(f'boft_modules_to_save: {args.boft_modules_to_save}')
    else:
        args.lora_modules_to_save = modules_to_save
        logger.info(f'lora_modules_to_save: {args.lora_modules_to_save}')


def prepare_model(model, args: SftArguments):
    # Preparing LoRA
    if is_adapter(args.sft_type):
        if args.resume_from_checkpoint is None:
            handle_target_modules(model, args)
            handle_modules_to_save(model, args)
            if args.init_lora_weights and args.init_lora_weights.lower() in ('true', 'false'):
                args.init_lora_weights = args.init_lora_weights.lower() in ('true', 'True')
            lora_kwargs = {
                'r': args.lora_rank,
                'target_modules': args.lora_target_modules,
                'lora_alpha': args.lora_alpha,
                'lora_dropout': args.lora_dropout_p,
                'bias': args.lora_bias_trainable,
                'modules_to_save': args.lora_modules_to_save,
                'use_rslora': args.use_rslora,
                'use_dora': args.use_dora,
                'lorap_lr_ratio': args.lora_lr_ratio,
                'init_lora_weights': args.init_lora_weights,
            }
            if args.sft_type in ('lora', 'longlora'):
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
                    target_modules=args.ia3_target_modules,
                    feedforward_modules=args.ia3_feedforward_modules or [],
                    modules_to_save=args.ia3_modules_to_save,
                )
                model = Swift.prepare_model(model, ia3_config)
                logger.info(f'ia3_config: {ia3_config}')
            elif args.sft_type == 'llamapro':
                model_type = args.model_type or args.model_id_or_path
                for key in MODEL_KEYS_MAPPING.keys():
                    if key in model_type.lower():
                        model_type = key
                        break

                llamapro_config = LLaMAProConfig(
                    model_type=model_type,
                    num_new_blocks=args.llamapro_num_new_blocks,
                    num_groups=args.llamapro_num_groups)
                model = Swift.prepare_model(model, llamapro_config)
                logger.info(f'llamapro_config: {llamapro_config}')
            elif args.sft_type == 'adapter':
                model_type = args.model_type or args.model_id_or_path
                for key in MODEL_KEYS_MAPPING.keys():
                    if key in model_type.lower():
                        model_type = key
                        break

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
                    target_modules=args.vera_target_modules,
                    projection_prng_key=args.vera_projection_prng_key,
                    vera_dropout=args.vera_dropout,
                    d_initial=args.vera_d_initial,
                    modules_to_save=args.vera_modules_to_save,
                )
                vera_config = handle_same_dim_target_modules(model, vera_config)
                model = Swift.prepare_model(model, vera_config)
                logger.info(f'vera_config: {vera_config}')
            elif args.sft_type == 'boft':
                boft_config = BOFTConfig(
                    boft_block_size=args.boft_block_size,
                    boft_block_num=args.boft_block_num,
                    boft_n_butterfly_factor=args.boft_n_butterfly_factor,
                    target_modules=args.boft_target_modules,
                    boft_dropout=args.boft_dropout,
                    modules_to_save=args.boft_modules_to_save,
                )
                model = Swift.prepare_model(model, boft_config)
                logger.info(f'boft_config: {boft_config}')
        else:
            if use_torchacc():
                if args.fsdp_num > 1:
                    consolidate_checkpoint(args.resume_from_checkpoint, 'adapter_model')
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

        if args.freeze_parameters > 0:
            freeze_model_parameters(model, args.freeze_parameters)
        if len(args.additional_trainable_parameters) > 0:
            activate_model_parameters(model, args.additional_trainable_parameters)
        if use_torchacc() and args.resume_from_checkpoint is not None:
            if args.fsdp_num > 1:
                consolidate_checkpoint(args.resume_from_checkpoint, 'model')
            weights_file = os.path.join(args.resume_from_checkpoint, 'model.bin')
            state_dict = torch.load(weights_file, map_location='cpu')
            model.load_state_dict(state_dict, False)
            # release memory
            del state_dict
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
        )

    callbacks = []
    if args.lisa_activated_layers > 0:
        assert args.sft_type == 'full', 'LISA only supports full parameter training.'

        class DynamicLayerActivationCallback(TrainerCallback):

            def __init__(self, n_layers: int, step_interval: int, model: torch.nn.Module):
                super().__init__()
                self.n_layers = n_layers
                self.step_interval = step_interval
                self.model = model
                layers_name = None
                layers = None
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.ModuleList):
                        layers_name = name
                        layers = module
                        break
                assert layers_name is not None
                self.layers_attribute = layers_name
                self.total_layers = len(layers)

                # Freeze all layers upon initialization
                self.freeze_all_layers()
                self.active_layers_indices = []

            def freeze_all_layers(self):
                layers = self.model.get_submodule(self.layers_attribute)
                for layer in layers:
                    for param in layer.parameters():
                        param.requires_grad = False

            def on_step_begin(self, args, state, control, **kwargs):
                # Check if it's time to switch active layers, including at step 0
                if state.global_step % self.step_interval == 0 or state.global_step == 1:
                    self.switch_active_layers()

            def switch_active_layers(self):
                # First, disable gradients for all layers
                self.freeze_all_layers()

                # Randomly select n_layers to activate
                layers = self.model.get_submodule(self.layers_attribute)
                self.active_layers_indices = np.random.choice(range(self.total_layers), self.n_layers, replace=False)
                # Enable gradients only for the selected layers
                for idx in self.active_layers_indices:
                    for param in layers[idx].parameters():
                        param.requires_grad = True

        lisa_callback = DynamicLayerActivationCallback(
            n_layers=args.lisa_activated_layers,  # Number of layers to activate
            step_interval=args.lisa_step_interval,  # Step interval to update active layers
            model=model)
        lisa_callback.switch_active_layers()  # Make trainable parameters printing a correct value
        callbacks.append(lisa_callback)

    class TrainerAdapterCallback(TrainerCallback):

        def __init__(self):
            self.global_step = 0

        # offload original_modules to cpu, to save memory
        def on_train_begin(self, _args, state, control, **kwargs):
            if hasattr(model, 'set_active_adapters'):
                model.set_active_adapters(model.adapters.keys(), offload='cpu')
            if args.sft_type == 'adalora':
                model.peft_config['default'].total_step = state.max_steps

                def zero_grad(_self, *args, **kwargs):
                    _self.update_and_allocate(self.global_step + 1)
                    _self._zero_grad(*args, **kwargs)

                model._zero_grad = model.zero_grad
                model.zero_grad = types.MethodType(zero_grad, model)

        def on_step_end(self, _args, state, control, **kwargs):
            if args.sft_type == 'adalora':
                self.global_step = state.global_step

    if is_adapter(args.sft_type) and args.tuner_backend == 'swift':
        callbacks.append(TrainerAdapterCallback())
    return model, callbacks
