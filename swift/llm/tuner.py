# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import types
from typing import List

import json
import numpy as np
import torch
import transformers
from packaging import version
from transformers import TrainerCallback

from swift.llm import SftArguments, get_model_arch
from swift.plugin import Tuner, extra_tuners
from swift.tuners import (AdaLoraConfig, AdapterConfig, BOFTConfig, IA3Config, LLaMAProConfig, LongLoRAModelType,
                          LoraConfig, LoRAConfig, ReftConfig, Swift, VeraConfig)
from swift.utils import activate_model_parameters, freeze_model_parameters, get_logger, use_torchacc

logger = get_logger()


def handle_vera_target_modules(model: torch.nn.Module, config: VeraConfig):
    """This function is only useful on the vera tuner"""
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


class TrainerAdapterCallback(TrainerCallback):

    def __init__(self, args):
        self.global_step = 0
        self.args = args

    # offload original_modules to cpu, to save memory
    def on_train_begin(self, _args, state, control, **kwargs):
        model = kwargs['model']
        if hasattr(model, 'set_active_adapters'):
            model.set_active_adapters(model.adapters.keys(), offload='cpu')
        if self.args.train_type == 'adalora':
            model.peft_config['default'].total_step = state.max_steps

            def zero_grad(_self, *args, **kwargs):
                _self.update_and_allocate(self.global_step + 1)
                _self._zero_grad(*args, **kwargs)

            model._zero_grad = model.zero_grad
            model.zero_grad = types.MethodType(zero_grad, model)

    def on_step_end(self, _args, state, control, **kwargs):
        if self.args.train_type == 'adalora':
            self.global_step = state.global_step


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


def prepare_tuner(model, args: SftArguments):
    if args.use_liger:
        # Apply liger
        apply_liger(args.model_type)

    if args.is_adapter:
        model.requires_grad_(False)
        if args.resume_from_checkpoint is None:
            target_modules = args.handle_target_modules(model)
            lora_kwargs = {
                'r': args.lora_rank,
                'target_modules': target_modules,
                'lora_alpha': args.lora_alpha,
                'lora_dropout': args.lora_dropout,
                'bias': args.lora_bias,
                'modules_to_save': args.modules_to_save,
                'use_rslora': args.use_rslora,
                'use_dora': args.use_dora,
                'lorap_lr_ratio': args.lorap_lr_ratio,
                'init_lora_weights': args.init_lora_weights,
            }

            if args.train_type in ('lora', 'longlora'):
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
                    assert args.train_type == 'lora', 'Unsloth does not support LongLoRA'
                    lora_kwargs.pop('lorap_lr_ratio')
                    model = FastLanguageModel.get_peft_model(
                        model,
                        use_gradient_checkpointing=True,
                        max_seq_length=args.max_length,
                        **lora_kwargs,
                    )
                    logger.info(f'unsloth_config: {lora_kwargs}')
                if args.train_type == 'longlora':
                    assert LongLoRAModelType.LLAMA in args.model_type
                    assert version.parse(transformers.__version__) >= version.parse('4.39.3')
                    from swift.tuners.longlora.llama import replace_llama_attn
                    replace_llama_attn(model)
                    model.config.group_size_ratio = 0.25
            elif args.train_type == 'adalora':
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
            elif args.train_type == 'llamapro':
                llamapro_config = LLaMAProConfig(
                    model_type=model.model_meta.model_arch,
                    num_new_blocks=args.llamapro_num_new_blocks,
                    num_groups=args.llamapro_num_groups)
                model = Swift.prepare_model(model, llamapro_config)
                logger.info(f'llamapro_config: {llamapro_config}')
            elif args.train_type == 'adapter':
                model_arch = get_model_arch(model.model_meta.model_arch)
                mlp_key = model_arch.mlp
                mlp_key = mlp_key.split('.{}.')[1]
                adapter_config = AdapterConfig(
                    dim=model.config.hidden_size,
                    target_modules=[mlp_key],
                    hidden_pos=0,
                    adapter_length=args.adapter_length,
                    act_layer=args.adapter_act)
                model = Swift.prepare_model(model, adapter_config)
                logger.info(f'adapter_config: {adapter_config}')
            elif args.train_type == 'vera':
                vera_config = VeraConfig(
                    r=args.vera_rank,
                    target_modules=args.target_modules,
                    projection_prng_key=args.vera_projection_prng_key,
                    vera_dropout=args.vera_dropout,
                    d_initial=args.vera_d_initial,
                    modules_to_save=args.modules_to_save,
                )
                vera_config = handle_vera_target_modules(model, vera_config)
                model = Swift.prepare_model(model, vera_config)
                logger.info(f'vera_config: {vera_config}')
            elif args.train_type == 'boft':
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
            elif args.train_type == 'fourierft':
                from peft import FourierFTConfig
                fourier_config = FourierFTConfig(
                    target_modules=args.target_modules,
                    modules_to_save=args.modules_to_save,
                    n_frequency=args.fourier_n_frequency,
                    scaling=args.fourier_scaling,
                )
                model = Swift.prepare_model(model, fourier_config)
                logger.info(f'fourier_config: {fourier_config}')
            elif args.train_type == 'reft':
                reft_config = ReftConfig(
                    model_type=model.model_meta.model_arch,
                    layer_key=args.reft_layer_key,
                    r=args.reft_rank,
                    layers=args.reft_layers,
                    intervention_type=args.reft_intervention_type,
                    args=args.reft_args,
                )
                logger.info(f'reft config: {reft_config}')
                model = Swift.prepare_model(model, {'reft': reft_config})
        else:
            if getattr(model, 'is_tuner_plugin', False):
                with open(os.path.join(args.resume_from_checkpoint, 'sft_args.json'), 'r') as f:
                    content = json.load(f)

                tuner: Tuner = extra_tuners[content['sft_type']]
                model = tuner.from_pretrained(model, args.resume_from_checkpoint)
            elif use_torchacc():
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
    elif args.train_type in extra_tuners:
        tuner: Tuner = extra_tuners[args.train_type]
        model = tuner.prepare_model(model, args)
        model.is_tuner_plugin = True
    elif args.train_type == 'full':
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
        raise ValueError(f'args.train_type: {args.train_type}')

    if args.sequence_parallel_size > 1:
        from swift.trainers.xtuner import dispatch_module_xtuner
        dispatch_module_xtuner(model)

    return model
