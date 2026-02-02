# Copyright (c) ModelScope Contributors. All rights reserved.
from contextlib import contextmanager
from typing import Optional, Tuple

import megatron.core
import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.core.extensions.transformer_engine import TEGroupedLinear, TELayerNormColumnParallelLinear, TELinear
from megatron.core.inference.communication_utils import recv_from_prev_pipeline_rank_, send_to_next_pipeline_rank
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint, sharded_state_dict_default
from megatron.training import checkpointing, get_args
from packaging import version
from peft.tuners.lora import Linear as LoraLinear
from peft.utils.other import ModulesToSaveWrapper
from torch import nn

from swift.tuners import LoraConfig, Swift
from swift.utils import (activate_parameters, deep_getattr, find_layers, freeze_parameters, get_logger,
                         get_model_parameter_info)

mcore_013 = version.parse(megatron.core.__version__) >= version.parse('0.13.0rc0')

logger = get_logger()


def find_all_linears(model, extra_layers=None):

    def _cond(name, module):
        if (extra_layers and isinstance(module, tuple(extra_layers))) or name != 'output_layer' and isinstance(
                module, (TELinear, TELayerNormColumnParallelLinear, TEGroupedLinear, nn.Linear)):
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
    include_embedding: bool = False,
    include_router: bool = False,
) -> str:
    from ..model import get_megatron_model_meta
    megatron_model_meta = get_megatron_model_meta(args.hf_model_type)
    modules = []
    visual_cls = megatron_model_meta.visual_cls
    vision_tower = [f'visual.{vit}' for vit in visual_cls._vision_tower]
    aligner = [f'visual.{aligner}' for aligner in visual_cls._aligner]
    if not freeze_llm:
        modules.append('language_model')
    if not freeze_vit:
        modules += vision_tower
    if not freeze_aligner:
        modules += aligner
    assert len(modules) > 0, f'modules: {modules}'
    extra_layers = []
    if include_embedding:
        extra_layers.append(LanguageModelEmbedding)
    if include_router:
        extra_layers.append(TopKRouter)

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
        target_modules = find_all_linears(sub_module, extra_layers)
        if not target_modules:
            continue
        target_modules = [tm for tm in target_modules if tm]
        target_pattern = rf'.*\.({"|".join(target_modules)})' if target_modules else ''
        rejected_pattern = rf'(?!({"|".join(rejected_modules)}))' if rejected_modules else ''
        res.append(rf'{rejected_pattern}{module}(?=\.){target_pattern}')

    return rf'^({"|".join(res)})$'


def get_target_modules(args, model):
    if isinstance(args.target_modules, str):
        return args.target_modules
    target_modules = args.target_modules.copy()
    if 'all-linear' in target_modules:
        if args.is_multimodal:
            return get_multimodal_target_regex(
                args,
                model,
                freeze_llm=args.freeze_llm,
                freeze_vit=args.freeze_vit,
                freeze_aligner=args.freeze_aligner,
                include_embedding='all-embedding' in target_modules,
                include_router='all-router' in target_modules,
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
    if args.task_type == 'seq_cls':
        args.modules_to_save.append('output_layer')
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
    from swift.megatron.tuners import LoraParallelLinear
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
    if args.ref_adapter_load or args.ref_adapters:
        model.add_adapter('ref_adapter', lora_config)
        model.base_model._cast_adapter_dtype(adapter_name='ref_adapter', autocast_adapter_dtype=True)
        for n, p in model.named_parameters():
            if '.ref_adapter.' in n:
                p.requires_grad = False
    # setting average_gradients_across_tp_domain
    for m in model.modules():
        if isinstance(m, LoraLinear):
            # just check
            assert args.is_multimodal or args.hf_model_type == 'qwen3_next'
            assert not isinstance(m, LoraParallelLinear)
            for p in m.parameters():
                if p.requires_grad:
                    p.average_gradients_across_tp_domain = True
    return model


def prepare_mcore_model(model):
    args = get_args()
    if args.tuner_type == 'full':
        freeze_parameters(model, args.freeze_parameters_ratio, args.freeze_parameters, args.freeze_parameters_regex)
        if args.trainable_parameters or args.trainable_parameters_regex:
            activate_parameters(model, args.trainable_parameters, args.trainable_parameters_regex)
    elif args.tuner_type == 'lora':
        model.prepare_inputs_for_generation = None  # fix error
        model = prepare_adapter(model)
    logger.info(f'model: {model}')
    logger.info_if(
        f'[rank{dist.get_rank()}] model_parameter_info: {get_model_parameter_info(model)}',
        cond=mpu.get_data_parallel_rank() == 0)
    return model


@contextmanager
def adapter_state_dict_context(is_peft_format: bool = True):
    if not is_peft_format:
        yield
        return
    _origin_generate_state_dict = checkpointing.generate_state_dict

    def generate_state_dict(args, model, *_args, **kwargs):
        state_dict = _origin_generate_state_dict(args, model, *_args, **kwargs)
        if 'model' not in state_dict:
            return state_dict
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
    if hasattr(model, 'language_model'):
        model = model.language_model
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


def forward_step_helper(model, inputs, dtype=None):
    args = get_args()
    if mpu.is_pipeline_first_stage():
        micro_batch_size = 1  # use qkv_format 'thd'
        if not args.padding_free:
            micro_batch_size = args.micro_batch_size
        seq_length = inputs['position_ids'].shape[-1]
        if args.sequence_parallel:
            seq_length //= mpu.get_tensor_model_parallel_world_size()
        recv_shape_buffer = torch.tensor([seq_length, micro_batch_size, args.hidden_size],
                                         device=torch.cuda.current_device(),
                                         dtype=torch.int64)
    else:
        recv_shape_buffer = torch.empty((3, ), device=torch.cuda.current_device(), dtype=torch.int64)
        recv_from_prev_pipeline_rank_(recv_shape_buffer)
    if not mpu.is_pipeline_last_stage():
        send_to_next_pipeline_rank(recv_shape_buffer)
    shape = recv_shape_buffer.tolist()

    if not mpu.is_pipeline_first_stage():
        dtype = dtype or args.params_dtype
        recv_buffer = torch.empty(shape, device=torch.cuda.current_device(), dtype=dtype)
        recv_from_prev_pipeline_rank_(recv_buffer)
        model.set_input_tensor(recv_buffer)
    output_tensor = model(**inputs)
    if not mpu.is_pipeline_last_stage():
        send_to_next_pipeline_rank(output_tensor)
        output_tensor = None

    return output_tensor


def get_padding_to(args):
    padding_to = None
    if args.tensor_model_parallel_size > 1 and args.sequence_parallel:
        padding_to = args.tensor_model_parallel_size
    if args.context_parallel_size > 1:
        padding_to = (padding_to or 1) * args.context_parallel_size
    origin_padding_to = padding_to
    fp8_format = getattr(args, 'fp8_format', None) or getattr(args, 'fp8', None)
    if args.fp8_recipe == 'blockwise':
        padding_to = (padding_to or 1) * 128
    elif fp8_format is not None:
        padding_to = max((padding_to or 1) * 8, 16)
    if args.attention_backend == 'fused':
        padding_to = max(padding_to, ((origin_padding_to) or 1) * 64)
    return padding_to


def get_local_layer_specs(config, layer_specs, vp_stage=None):
    kwargs = {'vp_stage': vp_stage} if mcore_013 else {}
    num_layers_to_build = get_num_layers_to_build(config, **kwargs)

    if getattr(config, 'pipeline_model_parallel_layout', None) is not None:
        from megatron.core.transformer.enums import LayerType
        local_layer_specs = [
            layer_specs[layer_id] for layer_id in config.pipeline_model_parallel_layout.get_layer_id_list(
                layer_type=LayerType.decoder, **kwargs)
        ]
    else:
        offset = get_transformer_layer_offset(config, **kwargs)
        local_layer_specs = layer_specs[offset:offset + num_layers_to_build]
    return local_layer_specs


class MegatronTrainerState:
    """
    A lightweight trainer state class for Megatron training, providing compatibility
    with transformers TrainerState interface.

    This class allows reward functions to access training progress information
    (current step and total steps) in the same way as they would with
    transformers Trainer.

    Attributes:
        global_step (int): The current training step (number of update steps completed).
        max_steps (int): The total number of training steps.
    """

    def __init__(self, global_step: int = 0, max_steps: int = 0):
        """
        Initialize MegatronTrainerState.

        Args:
            global_step: The current training step. Defaults to 0.
            max_steps: The total number of training steps. Defaults to 0.
        """
        self.global_step = global_step
        self.max_steps = max_steps

    def update(self, global_step: Optional[int] = None, max_steps: Optional[int] = None):
        """
        Update the trainer state.

        Args:
            global_step: The current training step. If None, keeps the current value.
            max_steps: The total number of training steps. If None, keeps the current value.
        """
        if global_step is not None:
            self.global_step = global_step
        if max_steps is not None:
            self.max_steps = max_steps

    def __repr__(self) -> str:
        return f'MegatronTrainerState(global_step={self.global_step}, max_steps={self.max_steps})'
