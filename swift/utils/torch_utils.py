# Copyright (c) Alibaba, Inc. and its affiliates.

import hashlib
import os
import pickle
import re
import time
import uuid
from bisect import bisect_right
from contextlib import contextmanager, nullcontext
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from datasets.utils.filelock import FileLock
from modelscope.hub.utils.utils import get_cache_dir
from transformers.integrations import is_deepspeed_zero3_enabled

from .env import get_dist_setting
from .logger import get_logger

logger = get_logger()


def _find_local_mac() -> str:
    mac = uuid.getnode()
    mac_address = ':'.join(('%012x' % mac)[i:i + 2] for i in range(0, 12, 2))
    return mac_address


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


def freeze_parameters(model: nn.Module, freeze_parameters_ratio: float, freeze_parameters: List[str]) -> None:
    if freeze_parameters_ratio > 0:
        n_parameters = get_n_params_grads(model)[0]
        n_parameters = np.array(n_parameters, dtype=np.int64)
        n_freeze_parameters = int(np.sum(n_parameters) * freeze_parameters_ratio)
        n_parameters_cs = np.cumsum(n_parameters)
        idx = bisect_right(n_parameters_cs, n_freeze_parameters)
        for _, p in zip(range(idx), model.parameters()):
            p.requires_grad = False

    if len(freeze_parameters) > 0:
        for n, p in model.named_parameters():
            for freeze_p in freeze_parameters:
                if n.startswith(freeze_p):
                    p.requires_grad = False


def activate_parameters(model: nn.Module, additional_trainable_parameters: List[str]) -> None:
    if len(additional_trainable_parameters) == 0:
        return
    has_activate = False
    for n, p in model.named_parameters():
        for additional_tp in additional_trainable_parameters:
            if n.startswith(additional_tp):
                p.requires_grad = True
                has_activate = True
    if not has_activate:
        logger.warning('len(additional_trainable_parameters) > 0 but no parameters are activated. '
                       f'additional_trainable_parameters: {additional_trainable_parameters}')


def time_synchronize() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()  # second


def _get_max_memory(device_ids: List[int]) -> Dict[Union[int, str], int]:
    """add feat in accelerate to support MP + DDP"""
    import psutil
    # Make sure CUDA is initialized on each GPU to have the right memory info.
    for i in device_ids:
        _ = torch.tensor([0], device=i)

    device_ids_set = set(device_ids)
    max_memory = {}
    for i in range(torch.cuda.device_count()):
        max_memory[i] = 0
        if i in device_ids_set:
            max_memory[i] = torch.cuda.mem_get_info(i)[0]
    max_memory['cpu'] = psutil.virtual_memory().available
    return max_memory


def _sync_max_memory(max_memory: Dict[Union[int, str], int]) -> Dict[Union[int, str], int]:
    """Make sure that the model structure of MP(device_map) is the same, when using DDP."""
    max_memory_list = [v for k, v in max_memory.items() if (v > 0 and k != 'cpu')]
    _, local_rank, world_size, _ = get_dist_setting()
    src_tensor = torch.tensor(max_memory_list).to(local_rank)
    tgt_tensor_list = [torch.zeros_like(src_tensor) for _ in range(world_size)]
    dist.all_gather(tgt_tensor_list, src_tensor)
    tgt_tensor = torch.stack(tgt_tensor_list, dim=0)
    new_max_memory_iter = iter(tgt_tensor.min(dim=0)[0].tolist())
    new_max_memory = {}
    for k, v in max_memory.items():
        new_max_memory[k] = v
        if v > 0 and k != 'cpu':
            new_max_memory[k] = next(new_max_memory_iter)
    return new_max_memory


def _find_layers(model: nn.Module, cond: Callable[[str, nn.Module], bool]) -> List[str]:
    # The content of target_module_names cannot exist in inner_nodes.
    inner_nodes = set()
    for name, module in model.named_modules():
        name = re.sub(r'\d+\.', '{}.', name)
        if not cond(name, module):
            inner_nodes.add(name)
    target_module_names = set()
    for name, module in model.named_modules():
        if cond(name, module):
            module_name_list = name.split('.')
            module_name = module_name_list.pop()
            for inner_node in inner_nodes:
                while module_name_list and inner_node.endswith(re.sub(r'\d+\.', '{}.', module_name)):
                    module_name = f'{module_name_list.pop()}.{module_name}'
            target_module_names.add(module_name)
    return list(target_module_names)


def find_norm(model: nn.Module) -> List[str]:
    # find_layer_norm
    return _find_layers(
        model,
        lambda name, module: isinstance(module, torch.nn.LayerNorm) or 'rmsnorm' in module.__class__.__name__.lower())


def find_embedding(model: nn.Module) -> List[str]:
    return _find_layers(model, lambda name, module: isinstance(module, torch.nn.Embedding))


def find_all_linears(model: nn.Module) -> List[str]:
    from swift.llm import get_model_arch
    model_info = model.model_info
    model_arch = get_model_arch(model.model_meta.model_arch)
    if model_arch and model_arch.lm_head:
        output = model_arch.lm_head
        idx = output.rfind('.')
        lm_head_name = output[idx + 1:]
    else:
        lm_head_name = 'lm_head'

    quant_method = model_info.quant_method
    quant_bits = model_info.quant_bits
    if quant_method == 'bnb':
        from bitsandbytes.nn import Linear4bit, Linear8bitLt
        if quant_bits == 4:
            linear_cls = [Linear4bit]
        elif quant_bits == 8:
            linear_cls = [Linear8bitLt]
    elif quant_method == 'hqq':
        from hqq.core.quantize import HQQLinear
        linear_cls = [HQQLinear]
    elif quant_method == 'eetq':
        from eetq import EetqLinear
        linear_cls = [EetqLinear]
    elif quant_method == 'gptq':
        from peft.utils import get_auto_gptq_quant_linear, get_quantization_config
        gptq_quantization_config = get_quantization_config(model, 'gptq')
        AutoGPTQQuantLinear = get_auto_gptq_quant_linear(gptq_quantization_config)
        linear_cls = [AutoGPTQQuantLinear]
    elif quant_method == 'awq':
        from awq.modules.linear import WQLinear_GEMM
        linear_cls = [WQLinear_GEMM]
    elif quant_method == 'aqlm':
        from aqlm import QuantizedLinear
        linear_cls = [QuantizedLinear]
    else:
        linear_cls = [nn.Linear]

    # 'score': classification model
    # 'v_head': reward model
    ignore_layers = [lm_head_name, 'score', 'v_head']
    return _find_layers(
        model, lambda name, module: isinstance(module, tuple(linear_cls)) and all(layer not in name
                                                                                  for layer in ignore_layers))


@contextmanager
def safe_ddp_context(hash_id: str):
    lock_dir = os.path.join(get_cache_dir(), 'lockers')
    os.makedirs(lock_dir, exist_ok=True)
    file_path = hashlib.sha256(hash_id.encode('utf-8')).hexdigest() + '.lock'
    file_path = os.path.join(lock_dir, file_path)
    with FileLock(file_path):
        yield


class Serializer:

    @staticmethod
    def to_tensor(obj):
        res = pickle.dumps(obj)
        res = np.array([len(res)], dtype=np.int64).tobytes() + res
        res = np.frombuffer(res, dtype=np.uint8).copy()
        res = torch.from_numpy(res)
        return res

    @staticmethod
    def from_tensor(obj):
        if isinstance(obj, torch.Tensor):
            obj = obj.cpu().numpy()
        res = obj.tobytes()
        buffer_size = np.frombuffer(res[:8], dtype=np.int64)[0]
        res = res[8:]
        return pickle.loads(res[:buffer_size])
