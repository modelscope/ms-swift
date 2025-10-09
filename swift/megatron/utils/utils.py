# Copyright (c) Alibaba, Inc. and its affiliates.
from contextlib import contextmanager
from copy import deepcopy
from typing import Optional, Tuple

import torch.distributed as dist
from megatron.core import mpu
from megatron.core.extensions.transformer_engine import TEGroupedLinear, TELayerNormColumnParallelLinear, TELinear
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint, sharded_state_dict_default
from megatron.training import checkpointing, get_args
from peft.utils.other import ModulesToSaveWrapper
from torch import nn

from swift.utils import (activate_parameters, deep_getattr, find_layers, freeze_parameters, get_logger,
                         get_model_parameter_info)

logger = get_logger()


def find_all_linears(model):

    def _cond(name, module):
        if isinstance(module, (TELinear, TELayerNormColumnParallelLinear, TEGroupedLinear, nn.Linear)):
            return True
        return False

    return find_layers(model, _cond)


def find_router(model):
    return find_layers(model, lambda name, module: isinstance(module, TopKRouter))


def find_embedding(model):
    return find_layers(model, lambda name, module: isinstance(module, LanguageModelEmbedding))


def get_multimodal_target_regex(
    args,
    model,
    *,
    freeze_llm: bool = False,
    freeze_vit: bool = True,
    freeze_aligner: bool = True,
) -> str:
    modules = []
    visual_cls = args.megatron_model_meta.visual_cls
    vision_tower = [f'visual.{vit}' for vit in visual_cls._vision_tower]
    aligner = [f'visual.{aligner}' for aligner in visual_cls._aligner]
    if not freeze_llm:
        modules.append('language_model')
    if not freeze_vit:
        modules += vision_tower
    if not freeze_aligner:
        modules += aligner
    assert len(modules) > 0, f'modules: {modules}'

    res = []
    for module in modules:
        rejected_modules = []
        if not freeze_vit:
            for _aligner in aligner:
                if _aligner.startswith(f'{module}.'):
                    rejected_modules.append(_aligner)

        sub_module = deep_getattr(model, module)
        if sub_module is None:
            continue
        target_modules = find_all_linears(sub_module)
        if not target_modules:
            continue
        target_modules = [tm for tm in target_modules if tm]
        target_pattern = rf'.*\.({"|".join(target_modules)})' if target_modules else ''
        rejected_pattern = rf'(?!({"|".join(rejected_modules)}))' if rejected_modules else ''
        res.append(rf'{rejected_pattern}{module}{target_pattern}')

    return rf'^({"|".join(res)})$'


def get_target_modules(args, model):
    if isinstance(args.target_modules, str):
        return args.target_modules
    target_modules = args.target_modules.copy()
    if 'all-linear' in target_modules:
        if args.model_meta.is_multimodal:
            return get_multimodal_target_regex(
                args,
                model,
                freeze_llm=args.freeze_llm,
                freeze_vit=args.freeze_vit,
                freeze_aligner=args.freeze_aligner,
            )
        else:
            target_modules.remove('all-linear')
            target_modules += find_all_linears(model)
    if 'all-embedding' in target_modules:
        target_modules.remove('all-embedding')
        target_modules += find_embedding(model)
    if 'all-router' in target_modules:
        target_modules.remove('all-router')
        target_modules += find_router(model)
    return target_modules


def get_modules_to_save(args, model):
    modules_to_save = args.modules_to_save.copy()
    if 'all-embedding' in args.modules_to_save:
        modules_to_save.remove('all-embedding')
        modules_to_save += find_embedding(model)
    if args.task_type == 'seq_cls':
        modules_to_save.append('output_layer')
    return modules_to_save


def set_linear_is_expert(model):
    for n, module in model.named_modules():
        if '.local_experts.' in n and isinstance(module, (TELinear, TELayerNormColumnParallelLinear)) or isinstance(
                module, TEGroupedLinear):
            module.is_expert = True


@contextmanager
def _patch_deepcopy():
    import copy
    _origin_deepcopy = copy.deepcopy

    def new_deepcopy(x, *args, **kwargs):
        if getattr(x, 'tp_group', None) is not None:
            origin_tp_group = x.tp_group
            x.tp_group = None
            res = _origin_deepcopy(x, *args, **kwargs)
            x.tp_group = origin_tp_group
            res.tp_group = origin_tp_group
            return res
        else:
            return _origin_deepcopy(x, *args, **kwargs)

    copy.deepcopy = new_deepcopy
    try:
        yield
    finally:
        copy.deepcopy = _origin_deepcopy


def prepare_adapter(model):
    from swift.tuners import LoraConfig, Swift
    args = get_args()
    set_linear_is_expert(model)
    target_modules = get_target_modules(args, model)
    modules_to_save = get_modules_to_save(args, model)
    lora_kwargs = {
        'r': args.lora_rank,
        'target_modules': target_modules,
        'lora_alpha': args.lora_alpha,
        'lora_dropout': args.lora_dropout,
        'bias': args.lora_bias,
        'modules_to_save': modules_to_save,
        'use_rslora': args.use_rslora,
    }
    lora_config = LoraConfig(task_type='CAUSAL_LM', lora_dtype=args.lora_dtype, **lora_kwargs)
    logger.info(f'lora_config: {lora_config}')
    with _patch_deepcopy():
        model = Swift.prepare_model(model, lora_config)
    if args.ref_adapter_load:
        lora_config = deepcopy(lora_config)
        lora_config.inference_mode = True
        with _patch_deepcopy():
            model.add_adapter('ref_adapter', lora_config)
        model.base_model._cast_adapter_dtype(adapter_name='ref_adapter', autocast_adapter_dtype=True)
    return model


def prepare_mcore_model(model):
    args = get_args()
    if args.train_type == 'full':
        freeze_parameters(model, args.freeze_parameters_ratio, args.freeze_parameters, args.freeze_parameters_regex)
        if args.trainable_parameters or args.trainable_parameters_regex:
            activate_parameters(model, args.trainable_parameters, args.trainable_parameters_regex)
    elif args.train_type == 'lora':
        model.prepare_inputs_for_generation = None  # fix error
        model = prepare_adapter(model)
    logger.info(f'model: {model}')
    logger.info_if(
        f'[rank{dist.get_rank()}] model_parameter_info: {get_model_parameter_info(model)}',
        cond=mpu.get_data_parallel_rank() == 0)
    return model


@contextmanager
def adapter_state_dict_context():
    args = get_args()
    if args.train_type == 'full':
        yield
        return
    _origin_generate_state_dict = checkpointing.generate_state_dict

    def generate_state_dict(args, model, *_args, **kwargs):
        state_dict = _origin_generate_state_dict(args, model, *_args, **kwargs)
        new_state_dict = {}
        state_dict_model = state_dict['model']
        for n, p in model[0].named_parameters():
            if not p.requires_grad:
                continue
            if n in state_dict_model:
                new_state_dict[n] = state_dict_model[n]
            key = n.replace('.weight', '._extra_state')
            if key.endswith('._extra_state0'):
                key = key.replace('._extra_state0', '._extra_state')
            if key in state_dict_model:
                new_state_dict[key] = state_dict_model[key]
        state_dict['model'] = new_state_dict
        return state_dict

    checkpointing.generate_state_dict = generate_state_dict
    try:
        yield
    finally:
        checkpointing.generate_state_dict = _origin_generate_state_dict


def tuners_sharded_state_dict(
        module,
        prefix: str = '',
        sharded_offsets: Tuple[Tuple[int, int, int]] = (),
        metadata: Optional[dict] = None,
):
    sharded_state_dict = {}
    # Save parameters
    module._save_to_state_dict(sharded_state_dict, '', keep_vars=True)
    sharded_state_dict = make_sharded_tensors_for_checkpoint(
        sharded_state_dict, prefix, sharded_offsets=sharded_offsets)
    # Recurse into submodules
    for name, module in module.named_children():
        if 'Dict' in module.__class__.__name__:
            modules = module.named_children()
        else:
            modules = [(None, module)]
        for n, m in modules:
            _prefix = f'{prefix}{name}.' if n is None else f'{prefix}{name}.{n}.'
            sharded_state_dict.update(sharded_state_dict_default(m, _prefix, sharded_offsets, metadata))
    return sharded_state_dict


def copy_original_module_weight(model):
    for module in model.modules():
        if isinstance(module, ModulesToSaveWrapper):
            original_module = module.original_module
            default_module = module.modules_to_save['default']
            original_module.load_state_dict(default_module.state_dict())


def copy_ref_adapter_weight(model, ref_adapter_name: str):
    from swift.megatron.tuners import LoraParallelLinear
    for module in model.modules():
        if isinstance(module, LoraParallelLinear):
            for key in ['lora_A', 'lora_B']:
                sub_module = getattr(module, key)
                if 'default' in sub_module and ref_adapter_name in sub_module:
                    sub_module[ref_adapter_name].load_state_dict(sub_module['default'].state_dict())
            for key in ['lora_embedding_A', 'lora_embedding_B']:
                sub_module = getattr(module, key)
                if 'default' in sub_module and ref_adapter_name in sub_module:
                    sub_module[ref_adapter_name].data.copy_(sub_module['default'])
        elif isinstance(module, ModulesToSaveWrapper):
            sub_module = module.modules_to_save
            if 'default' in sub_module and ref_adapter_name in sub_module:
                sub_module[ref_adapter_name].load_state_dict(sub_module['default'].state_dict())
