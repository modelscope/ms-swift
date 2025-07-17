# Copyright (c) Alibaba, Inc. and its affiliates.
from contextlib import contextmanager
from typing import Optional, Tuple

import torch.distributed as dist
from megatron.core import mpu
from megatron.core.extensions.transformer_engine import TEGroupedLinear, TELayerNormColumnParallelLinear, TELinear
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint, sharded_state_dict_default
from megatron.training import checkpointing, get_args

from swift.utils import activate_parameters, find_layers, freeze_parameters, get_logger, get_model_parameter_info

logger = get_logger()


def find_all_linears(model):

    def _cond(name, module):
        if isinstance(module, (TELinear, TELayerNormColumnParallelLinear, TEGroupedLinear)):
            return True
        return False

    return find_layers(model, _cond)


def find_embedding(model):
    return find_layers(model, lambda name, module: isinstance(module, LanguageModelEmbedding))


def get_target_modules(args, model):
    if isinstance(args.target_modules, str):
        return args.target_modules
    target_modules = args.target_modules.copy()
    if 'all-linear' in target_modules:
        target_modules.remove('all-linear')
        target_modules += find_all_linears(model)
    if 'all-embedding' in target_modules:
        target_modules.remove('all-embedding')
        target_modules += find_embedding(model)
    return target_modules


def get_modules_to_save(args, model):
    modules_to_save = args.modules_to_save.copy()
    if 'all-embedding' in args.modules_to_save:
        modules_to_save.remove('all-embedding')
        modules_to_save += find_embedding(model)
    return modules_to_save


def set_linear_is_expert(model):
    for n, module in model.named_modules():
        if '.local_experts.' in n and isinstance(module, (TELinear, TELayerNormColumnParallelLinear)) or isinstance(
                module, TEGroupedLinear):
            module.is_expert = True


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
    return Swift.prepare_model(model, lora_config)


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
        if 'ModulesToSaveWrapper' in module.__class__.__name__ and hasattr(module, 'original_module'):
            original_module = module.original_module
            modules_to_save = module.modules_to_save
            if 'default' in modules_to_save:
                original_module.load_state_dict(modules_to_save['default'].state_dict())
