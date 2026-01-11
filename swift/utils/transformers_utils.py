# Copyright (c) Alibaba, Inc. and its affiliates.
import re
from bisect import bisect_right
from contextlib import contextmanager, nullcontext
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.trainer_utils import set_seed

from .logger import get_logger
from .utils import deep_getattr

logger = get_logger()


def get_n_params_grads(model) -> Tuple[List[int], List[int]]:
    n_params, n_grads = [], []
    for p in model.parameters():
        if is_deepspeed_zero3_enabled():
            import deepspeed
            context = deepspeed.zero.GatheredParameters(p)
        else:
            context = nullcontext()
        with context:
            n_params.append(p.numel())
            n_grads.append(p.numel() if p.requires_grad else 0)
    return n_params, n_grads


def get_model_parameter_info(model: nn.Module, name: Optional[str] = None) -> str:
    n_params, n_grads = get_n_params_grads(model)
    n_params = sum(n_params)
    n_grads = sum(n_grads)
    n_buffers = sum(p.numel() for p in model.buffers())

    if name is None:
        name = model.__class__.__name__

    n_params /= 1e6
    n_grads /= 1e6
    n_buffers /= 1e6
    s = (f'{name}: '
         f'{n_params:.4f}M Params ({n_grads:.4f}M Trainable '
         f'[{100 * n_grads / n_params:.4f}%]), '
         f'{n_buffers:.4f}M Buffers.')
    return s


def find_sub_module(module: torch.nn.Module, module_name: str) -> List[torch.nn.Module]:
    _modules = list()
    for name, sub_module in module.named_modules():
        if not name:
            continue
        if name.endswith(module_name):
            _modules.append(sub_module)
    return _modules


def show_layers(model: nn.Module, max_lines: Optional[int] = 20) -> None:
    named_p = list(model.named_parameters())
    for i, (n, p) in enumerate(named_p):
        if max_lines is not None and i >= max_lines:
            logger.info('...')
            break
        logger.info(f'[{n}]: requires_grad={p.requires_grad}, dtype={p.dtype}, device={p.device}')


def freeze_parameters(model: nn.Module,
                      freeze_parameters_ratio: float,
                      freeze_parameters: List[str],
                      freeze_parameters_regex: Optional[str] = None) -> None:
    if freeze_parameters_ratio > 0:
        n_parameters = get_n_params_grads(model)[0]
        n_parameters = np.array(n_parameters, dtype=np.int64)
        n_freeze_parameters = int(np.sum(n_parameters) * freeze_parameters_ratio)
        n_parameters_cs = np.cumsum(n_parameters)
        idx = bisect_right(n_parameters_cs, n_freeze_parameters)
        for _, p in zip(range(idx), model.parameters()):
            p.requires_grad = False

    if freeze_parameters:
        for n, p in model.named_parameters():
            for freeze_p in freeze_parameters:
                if n.startswith(freeze_p):
                    p.requires_grad = False

    if freeze_parameters_regex is not None:
        try:
            pattern = re.compile(freeze_parameters_regex)
        except re.error as e:
            logger.warning(f"Invalid freeze_parameters_regex '{freeze_parameters_regex}': {e}")
            return

        for n, p in model.named_parameters():
            if pattern.search(n):
                p.requires_grad = False


def activate_parameters(model: nn.Module,
                        additional_trainable_parameters: List[str],
                        trainable_parameters_regex: Optional[str] = None) -> None:
    has_activate = False
    if len(additional_trainable_parameters) > 0:
        for n, p in model.named_parameters():
            for additional_tp in additional_trainable_parameters:
                if n.startswith(additional_tp):
                    p.requires_grad = True
                    has_activate = True
        if not has_activate:
            logger.warning('len(additional_trainable_parameters) > 0 but no parameters are activated. '
                           f'additional_trainable_parameters: {additional_trainable_parameters}')

    has_activate = False
    if trainable_parameters_regex is not None:
        try:
            pattern = re.compile(trainable_parameters_regex)
        except re.error as e:
            logger.warning(f"Invalid trainable_parameters_regex '{trainable_parameters_regex}': {e}")
            return

        for n, p in model.named_parameters():
            if pattern.search(n):
                p.requires_grad = True
                has_activate = True

        if not has_activate:
            logger.warning('trainable_parameters_regex is provided but no parameters are activated. '
                           f'trainable_parameters_regex: {trainable_parameters_regex}')


def find_layers(
    model: nn.Module,
    cond: Callable[[str, nn.Module], bool],
    sub_module: Optional[str] = None,
    min_name_len: Optional[int] = None,
) -> List[str]:
    # The content of target_module_names cannot exist in inner_nodes.
    sub_module_str = sub_module
    if sub_module is None:
        sub_module = model
    else:
        sub_module = deep_getattr(model, sub_module)
    inner_nodes = set()
    for name, module in model.named_modules():
        name = re.sub(r'\d+\.', '{}.', name)
        if not cond(name, module):
            inner_nodes.add(name)
    target_module_names = set()
    for name, module in sub_module.named_modules():
        if sub_module_str:
            name = f'{sub_module_str}.{name}' if name else sub_module_str
        if cond(name, module):
            module_name_list = name.split('.')
            module_name = module_name_list.pop()
            i = 1
            for inner_node in inner_nodes:
                while module_name_list and inner_node.endswith(re.sub(
                        r'\d+\.', '{}.', module_name)) or min_name_len and i < min_name_len:
                    module_name = f'{module_name_list.pop()}.{module_name}'
                    i += 1
            target_module_names.add(module_name)
    return list(target_module_names)


def find_norm(model: nn.Module) -> List[str]:
    # find_layer_norm
    return find_layers(
        model,
        lambda name, module: isinstance(module, torch.nn.LayerNorm) or 'rmsnorm' in module.__class__.__name__.lower())


def find_embedding(model: nn.Module) -> List[str]:
    return find_layers(model, lambda name, module: isinstance(module, torch.nn.Embedding))


def find_all_linears(model, model_arch=None, extra_layers=None, sub_module=None):
    if model_arch is None:
        model_arch = model.model_meta.model_arch
    # lm_head
    if model_arch and model_arch.lm_head:
        output = model_arch.lm_head
        idx = output.rfind('.')
        lm_head_name = output[idx + 1:]
    else:
        lm_head_name = 'lm_head'
    # 'score', 'classifier': classification model
    # 'v_head': reward model
    ignore_layers = [lm_head_name, 'score', 'v_head', 'classifier'] + ['lora_A', 'lora_B', 'base_layer']
    ignore_linear_cls = [
        'glulinear'  # phi4-mm
    ]

    def _cond(name, module):
        module_name = module.__class__.__name__.lower()
        if (extra_layers and isinstance(module, tuple(extra_layers)) or
            ('linear' in module_name and all(linear_cls not in module_name
                                             for linear_cls in ignore_linear_cls))) and all(layer not in name
                                                                                            for layer in ignore_layers):
            return True
        return False

    return find_layers(model, _cond, sub_module=sub_module)


def get_cu_seqlens_from_position_ids(position_ids: torch.LongTensor):
    position_ids = position_ids[0]
    seq_start_indices = torch.where(position_ids == 0)[0]
    seq_end_indices = torch.cat([seq_start_indices[1:], torch.tensor([len(position_ids)], device=position_ids.device)])
    seq_lengths = seq_end_indices - seq_start_indices
    cu_seqlens = torch.cumsum(torch.cat([torch.tensor([0], device=position_ids.device), seq_lengths]), dim=0)
    return cu_seqlens


def get_position_ids_from_cu_seqlens(cu_seqlens: torch.LongTensor):
    seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    position_ids = torch.cat([torch.arange(seq_len, device=cu_seqlens.device) for seq_len in seq_lengths], dim=0)
    return position_ids.unsqueeze(0)


def seed_worker(worker_id: int, num_workers: int, rank: int):
    """
    Helper function to set worker seed during Dataloader initialization.
    """
    init_seed = torch.initial_seed() % 2**32
    worker_seed = num_workers * rank + init_seed
    set_seed(worker_seed)


@contextmanager
def unwrap_model_for_generation(
    model,
    accelerator,
    gather_deepspeed3_params=True,
    gather_parameters: List[nn.Parameter] = None,
):
    unwrapped_model = accelerator.unwrap_model(model)
    if accelerator.state.deepspeed_plugin is not None and accelerator.state.deepspeed_plugin.zero_stage == 3:
        if not gather_deepspeed3_params:
            yield accelerator.unwrap_model(model)
        else:
            import deepspeed
            parameters = [
                parameter for name, parameter in model.named_parameters()
                if not gather_parameters or name in gather_parameters
            ]
            with deepspeed.zero.GatheredParameters(parameters):
                from trl.models.utils import remove_hooks, add_hooks
                remove_hooks(model)
                yield accelerator.unwrap_model(model)
                add_hooks(model)
    else:
        yield unwrapped_model


@contextmanager
def disable_deepspeed_zero3():
    import transformers.integrations.deepspeed as ds_module
    orig_weak_ref = ds_module._hf_deepspeed_config_weak_ref
    ds_module._hf_deepspeed_config_weak_ref = None
    try:
        yield
    finally:
        ds_module._hf_deepspeed_config_weak_ref = orig_weak_ref


def get_modules_to_not_convert(model):
    if not hasattr(model, 'model_meta') or not hasattr(model, 'model_info'):
        return
    model_arch = model.model_meta.model_arch
    prefix_list = []
    suffix_list = []
    if model.model_info.is_moe_model:
        suffix_list += ['mlp.gate', 'mlp.shared_expert_gate']
    if model_arch is not None:
        for key in ['vision_tower', 'aligner']:
            value = getattr(model_arch, key, None)
            if value:
                prefix_list += value
    suffix_list.append('lm_head')
    res = []
    for n, m in model.named_modules():
        if 'linear' in m.__class__.__name__.lower() and (any(n.endswith(suffix) for suffix in suffix_list)
                                                         or any(n.startswith(prefix) for prefix in prefix_list)):
            res.append(n)
    return res if res else None


def get_packed_seq_params(position_ids: torch.Tensor):
    assert position_ids.shape[0] == 1, f'position_ids.shape: {position_ids.shape}'
    position_ids_f = position_ids.flatten()
    indices_q = torch.arange(position_ids_f.shape[0], device=position_ids_f.device, dtype=torch.int32)

    cu_seqlens = torch.cat([
        indices_q[position_ids_f == 0],
        torch.tensor(position_ids_f.shape, device=position_ids_f.device, dtype=torch.int32),
    ])

    max_length = cu_seqlens.diff().max()  # position_ids_f.max() + 1
    return {
        'cu_seq_lens_q': cu_seqlens,
        'cu_seq_lens_k': cu_seqlens,
        'max_length_q': max_length,
        'max_length_k': max_length,
    }
