# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect
import os
from typing import List, Union

import torch
import torch.nn as nn
import transformers
from packaging import version
from transformers import TrainingArguments

from swift.llm import TrainArguments, deep_getattr, get_model_arch
from swift.plugin import Tuner, extra_tuners
from swift.tuners import Swift
from swift.utils import (activate_parameters, find_all_linears, find_embedding, find_norm, freeze_parameters,
                         get_logger, use_torchacc)

logger = get_logger()


def apply_liger(model_type: str):
    from liger_kernel.transformers import (apply_liger_kernel_to_llama, apply_liger_kernel_to_mistral,
                                           apply_liger_kernel_to_mixtral, apply_liger_kernel_to_gemma,
                                           apply_liger_kernel_to_qwen2, apply_liger_kernel_to_qwen3,
                                           apply_liger_kernel_to_qwen2_vl, apply_liger_kernel_to_qwen2_5_vl,
                                           apply_liger_kernel_to_phi3, apply_liger_kernel_to_mllama)
    from swift.llm import ModelType
    if model_type in (ModelType.llama, ModelType.llama3, ModelType.llama3_1, ModelType.llama3_2):
        apply_liger_kernel_to_llama()
    elif model_type in (ModelType.mistral):
        apply_liger_kernel_to_mistral()
    elif model_type in (ModelType.mixtral):
        apply_liger_kernel_to_mixtral()
    elif model_type in (ModelType.gemma, ModelType.gemma2):
        apply_liger_kernel_to_gemma()
    elif model_type in (ModelType.qwen2, ModelType.qwen2_5):
        apply_liger_kernel_to_qwen2()
    elif model_type in (ModelType.qwen3):
        apply_liger_kernel_to_qwen3()
    elif model_type in (ModelType.phi3):
        apply_liger_kernel_to_phi3()
    elif model_type in (ModelType.llama3_2_vision):
        apply_liger_kernel_to_mllama()
    elif model_type in (ModelType.qwen2_vl):
        apply_liger_kernel_to_qwen2_vl()
    elif model_type in (ModelType.qwen2_5_vl):
        apply_liger_kernel_to_qwen2_5_vl()
    else:
        raise ValueError(f'Unsupported liger model_type: {model_type}')


def get_multimodal_target_regex(
    model,
    *,
    freeze_llm: bool = False,
    freeze_vit: bool = True,
    freeze_aligner: bool = True,
    include_embedding: bool = False,
) -> str:
    model_arch = get_model_arch(model.model_meta.model_arch)
    modules = []
    if not freeze_llm:
        modules += model_arch.language_model
    if not freeze_vit:
        modules += model_arch.vision_tower
    if not freeze_aligner:
        modules += model_arch.aligner
    assert len(modules) > 0, f'modules: {modules}'

    extra_layers = []
    if include_embedding:
        extra_layers.append(nn.Embedding)
    res = []
    for module in modules:
        rejected_modules = []
        if not freeze_vit:
            for aligner in model_arch.aligner:
                if aligner.startswith(f'{module}.'):
                    rejected_modules.append(aligner)

        sub_module = deep_getattr(model, module)
        target_modules = find_all_linears(sub_module, model_arch, extra_layers)
        target_modules = [tm for tm in target_modules if tm]
        target_pattern = rf'.*\.({"|".join(target_modules)})' if target_modules else ''
        rejected_pattern = rf'(?!({"|".join(rejected_modules)}))' if rejected_modules else ''
        res.append(rf'{rejected_pattern}{module}{target_pattern}')

    return rf'^({"|".join(res)})$'


def get_target_modules(args, model) -> Union[str, List[str]]:
    """Replace all-linear to actual modules"""
    model_meta = model.model_meta
    if isinstance(args.target_modules, str):
        return args.target_modules
    target_modules = args.target_modules.copy()
    if 'all-linear' in target_modules:
        if model_meta.is_multimodal:
            return get_multimodal_target_regex(
                model,
                freeze_llm=args.freeze_llm,
                freeze_vit=args.freeze_vit,
                freeze_aligner=args.freeze_aligner,
                include_embedding='all-embedding' in target_modules)
        else:
            target_modules.remove('all-linear')
            target_modules += find_all_linears(model)
    if 'all-embedding' in target_modules:
        target_modules.remove('all-embedding')
        target_modules += find_embedding(model)
    return target_modules


def get_modules_to_save(args, model, task_type=None):
    modules_to_save = args.modules_to_save.copy()
    if 'all-embedding' in args.modules_to_save:
        modules_to_save.remove('all-embedding')
        modules_to_save += find_embedding(model)
    if 'all-norm' in args.modules_to_save:
        modules_to_save.remove('all-norm')
        modules_to_save += find_norm(model)
    if task_type and task_type.lower() == 'seq_cls':  # reward_model
        modules_to_save.append('v_head')
    return modules_to_save


def get_vera_target_modules(model, config):
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
            raise ValueError('Please manually pass in `vera_target_modules`, do not use `all-linear`,'
                             'because Vera need all target linears to be the same size.')
        v = v[0]
        shape = [shape for name, shape in modules_dict.items() if v in name][0]
        names = [_name for _name, _shape in modules_dict.items() if _shape == shape]
        config.target_modules = [t for t in target_modules if any([t in name for name in names])]
    return config


def prepare_adapter(args: TrainArguments, model, *, template=None, train_dataset=None, task_type=None):
    from swift.tuners import (AdaLoraConfig, AdapterConfig, BOFTConfig, LLaMAProConfig, LongLoRAModelType, LoraConfig,
                              LoRAConfig, ReftConfig, Swift, VeraConfig)
    task_type = (task_type or args.task_type).upper()
    target_modules = get_target_modules(args, model)
    modules_to_save = get_modules_to_save(args, model, task_type)
    lora_kwargs = {
        'r': args.lora_rank,
        'target_modules': target_modules,
        'lora_alpha': args.lora_alpha,
        'lora_dropout': args.lora_dropout,
        'bias': args.lora_bias,
        'modules_to_save': modules_to_save,
        'use_rslora': args.use_rslora,
        'use_dora': args.use_dora,
        'lorap_lr_ratio': args.lorap_lr_ratio,
        'init_lora_weights': args.init_weights,
    }
    if args.train_type in ('lora', 'longlora'):
        if args.use_swift_lora:
            lora_config = LoRAConfig(lora_dtype=args.lora_dtype, **lora_kwargs)
            model = Swift.prepare_model(model, lora_config)
            logger.info(f'lora_config: {lora_config}')
        elif args.tuner_backend == 'peft':
            if task_type == 'EMBEDDING':
                task_type = None
            elif task_type == 'RERANKER':
                task_type = 'SEQ_CLS'
            elif task_type == 'GENERATIVE_RERANKER':
                task_type = 'CAUSAL_LM'
            lora_config = LoraConfig(task_type=task_type, lora_dtype=args.lora_dtype, **lora_kwargs)
            if args.init_weights == 'lora-ga':
                try:
                    import lora_ga
                except ImportError as e:
                    error_message = """
                    Since 'LoRA-GA' is not implemented by PEFT, you will need to install it directly from GitHub.
                    Command: 'pip install git+https://github.com/lxline/LoRA-GA.git'.
                    """
                    logger.info(error_message)
                    raise RuntimeError(error_message) from e
                model = lora_ga.entrypoint.get_lora_ga_model(
                    model=model,
                    data_collator=template.data_collator,
                    dataset=train_dataset,
                    batch_size=args.lora_ga_batch_size,
                    num_iters=args.lora_ga_iters,
                    max_length=args.lora_ga_max_length,
                    direction=args.lora_ga_direction,
                    dtype=args.lora_dtype,
                    scale=args.lora_ga_scale,
                    stable_gamma=args.lora_ga_stable_gamma,
                )
            else:
                model = Swift.prepare_model(model, lora_config)
            logger.info(f'lora_config: {lora_config}')
        elif args.tuner_backend == 'unsloth':
            if args.resume_from_checkpoint is None:
                if args.model_meta.is_multimodal:
                    from unsloth import FastVisionModel as UnslothModel
                else:
                    from unsloth import FastLanguageModel as UnslothModel
                assert args.train_type == 'lora', 'Unsloth does not support LongLoRA'
                lora_kwargs.pop('lorap_lr_ratio')
                model = UnslothModel.get_peft_model(
                    model,
                    use_gradient_checkpointing='unsloth',
                    max_seq_length=args.max_length or 2048,  # 2048 is the default value of unsloth
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
        from swift.plugin.optimizer import calculate_max_steps
        adalora_config = AdaLoraConfig(
            task_type=task_type,
            **lora_kwargs,
            target_r=args.adalora_target_r,
            init_r=args.adalora_init_r,
            tinit=args.adalora_tinit,
            tfinal=args.adalora_tfinal,
            deltaT=args.adalora_deltaT,
            beta1=args.adalora_beta1,
            beta2=args.adalora_beta2,
            orth_reg_weight=args.adalora_orth_reg_weight,
            total_step=calculate_max_steps(args.training_args, train_dataset),
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
            target_modules=target_modules,
            projection_prng_key=args.vera_projection_prng_key,
            vera_dropout=args.vera_dropout,
            d_initial=args.vera_d_initial,
            modules_to_save=args.modules_to_save,
        )
        vera_config = get_vera_target_modules(model, vera_config)
        model = Swift.prepare_model(model, vera_config)
        logger.info(f'vera_config: {vera_config}')
    elif args.train_type == 'boft':
        boft_config = BOFTConfig(
            boft_block_size=args.boft_block_size,
            boft_block_num=args.boft_block_num,
            boft_n_butterfly_factor=args.boft_n_butterfly_factor,
            target_modules=target_modules,
            boft_dropout=args.boft_dropout,
            modules_to_save=args.modules_to_save,
        )
        model = Swift.prepare_model(model, boft_config)
        logger.info(f'boft_config: {boft_config}')
    elif args.train_type == 'fourierft':
        from peft import FourierFTConfig
        fourier_config = FourierFTConfig(
            target_modules=target_modules,
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
    elif args.train_type == 'bone':
        # Version loosing
        from peft import BoneConfig
        bone_config = BoneConfig(
            target_modules=target_modules,
            r=args.reft_rank,
            init_weights=args.init_weights,
        )
        logger.info(f'bone config: {bone_config}')
        model = Swift.prepare_model(model, bone_config)
    return model


def torchacc_resume_from_checkpoint(args, model):
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
                logger.warning(f'There were missing keys in the checkpoint model loaded: {load_result.missing_keys}.')
        if len(load_result.unexpected_keys) != 0:
            logger.warning(f'There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}.')


class TunerMixin:

    @classmethod
    def prepare_model(cls, args, model, *, template=None, train_dataset=None, task_type=None):
        if args.use_liger_kernel and 'use_liger_kernel' not in inspect.signature(TrainingArguments).parameters:
            # Apply liger
            apply_liger(args.model_type)

        if args.is_adapter:
            if args.tuner_backend != 'unsloth' and args.train_type not in extra_tuners:
                # Fix the name of the layer in xcomposer that contains Plora.
                # Unsloth prepares and loads lora outside this function when
                # resume_from_checkpoint, so do not disable grad here
                model.requires_grad_(False)
            if args.resume_from_checkpoint:
                if args.train_type in extra_tuners:
                    tuner: Tuner = extra_tuners[args.train_type]
                else:
                    tuner = Swift
                kwargs = {}
                if use_torchacc():
                    kwargs = {'adapter_name': 'default'}
                model = tuner.from_pretrained(model, args.resume_from_checkpoint, is_trainable=True, **kwargs)
            else:
                if args.train_type in extra_tuners:
                    tuner: Tuner = extra_tuners[args.train_type]
                    model = tuner.prepare_model(args, model)
                else:
                    model = prepare_adapter(
                        args, model, template=template, train_dataset=train_dataset, task_type=task_type)
            # fix bug: Attempting to unscale FP16 gradients.
            #   peft: https://github.com/huggingface/peft/issues/1249
            for p in model.parameters():
                if p.requires_grad and p.dtype == torch.float16:
                    logger.info_once('Convert trainable parameters from fp16 to fp32.')
                    p.data = p.data.to(dtype=torch.float32)
        elif args.train_type == 'full':
            model.train()
            model.requires_grad_(True)

            freeze_parameters(model, args.freeze_parameters_ratio, args.freeze_parameters, args.freeze_parameters_regex)
            if len(args.trainable_parameters) > 0 or args.trainable_parameters_regex is not None:
                activate_parameters(model, args.trainable_parameters, args.trainable_parameters_regex)
            if use_torchacc() and args.resume_from_checkpoint:
                torchacc_resume_from_checkpoint(args, model)
        else:
            raise ValueError(f'args.train_type: {args.train_type}')

        if args.resume_only_model:
            args.training_args.resume_from_checkpoint = None
        if args.use_galore:
            from swift.trainers.optimizers.galore import GaLoreConfig
            if args.galore_target_modules is None:
                args.galore_target_modules = find_all_linears(model)
            if args.galore_with_embedding:
                args.galore_target_modules += find_embedding(model)
            args.galore_config = GaLoreConfig(
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
            args.training_args.galore_config = args.galore_config

        if args.sequence_parallel_size > 1:
            from swift.trainers.sequence_parallel import sequence_parallel
            sequence_parallel.prepare_model(model, template.tokenizer)

        return model
