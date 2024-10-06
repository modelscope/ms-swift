# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import socket
import time
import uuid
from bisect import bisect_right
from contextlib import contextmanager, nullcontext
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import Linear, Module
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.utils import is_torch_npu_available

from .env import get_dist_setting, is_dist, is_dist_ta, is_local_master
from .logger import get_logger

logger = get_logger()


def is_on_same_device(model: torch.nn.Module) -> bool:
    device_set = set(map(lambda p: p.device, model.parameters()))
    return len(device_set) == 1


def _find_free_port() -> str:
    # Copied from https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/launch.py # noqa: E501
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


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


def get_model_info(model: Module, name: Optional[str] = None) -> str:
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


def show_layers(model: Module, max_lines: Optional[int] = 20) -> None:
    named_p = list(model.named_parameters())
    for i, (n, p) in enumerate(named_p):
        if max_lines is not None and i >= max_lines:
            logger.info('...')
            break
        logger.info(f'[{n}]: requires_grad={p.requires_grad}, dtype={p.dtype}, device={p.device}')


def freeze_model_parameters(model: Module, freeze_parameters_ratio: float, freeze_parameters: List[str]) -> None:
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


def activate_model_parameters(model: Module, additional_trainable_parameters: List[str]) -> None:
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


def broadcast_string(string: Optional[str], buffer_size: int = 1024) -> str:
    """String broadcasting in case of DDP
    string: main rank: str
        other rank: None or str(not use)
    return: all rank: str
    """
    assert dist.is_initialized()
    rank, local_rank, _, _ = get_dist_setting()
    device = f'npu:{local_rank}' if is_torch_npu_available() else f'cuda:{local_rank}'
    assert rank >= 0
    if rank == 0:
        assert string is not None
        tensor = torch.tensor(
            [ord(c) for c in string] + [0] * (buffer_size - len(string)), dtype=torch.int64, device=device)
    else:
        tensor = torch.zeros(buffer_size, dtype=torch.int64, device=device)
    dist.broadcast(tensor, 0)
    first_zero = (tensor == 0).nonzero()[0].item()
    res = tensor.tolist()[:first_zero]
    return ''.join([chr(x) for x in res])


def time_synchronize() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()  # second


def _get_max_memory(device_ids: List[int]) -> Dict[Union[int, str], int]:
    """add feat in accelerate to support DDP + MP"""
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


def _find_layers(model: Module, module_cls: type) -> List[str]:
    module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, module_cls):
            module_name = '.'.join(name.split('.')[-2:])
            module_names.add(module_name)
    return list(module_names)


def find_embedding(model: Module) -> List[str]:
    return _find_layers(model, torch.nn.Embedding)


def find_all_linears(model: Module, quantization_bit: int, model_type: str, quant_method: str) -> List[str]:
    """ref: https://github.com/artidoro/qlora"""
    head_module_name = 'lm_head'
    from swift.llm import MODEL_KEYS_MAPPING
    if model_type in MODEL_KEYS_MAPPING:
        output = MODEL_KEYS_MAPPING[model_type].output
        idx = output.rfind('.')
        head_module_name = output[idx + 1:]
    if quant_method == 'bnb':
        if quantization_bit == 4:
            from bitsandbytes.nn import Linear4bit
            linear_cls = [Linear4bit]
        elif quantization_bit == 8:
            from bitsandbytes.nn import Linear8bitLt
            linear_cls = [Linear8bitLt]
    elif quant_method == 'hqq':
        from hqq.core.quantize import HQQLinear
        linear_cls = [HQQLinear]
    elif quant_method == 'eetq':
        from eetq import EetqLinear
        linear_cls = [EetqLinear]
    else:
        linear_cls = [Linear]
    if 'int4' in model_type or 'int8' in model_type:
        from peft.utils import get_auto_gptq_quant_linear, get_quantization_config
        gptq_quantization_config = get_quantization_config(model, 'gptq')
        AutoGPTQQuantLinear = get_auto_gptq_quant_linear(gptq_quantization_config)
        if AutoGPTQQuantLinear is None:
            from bitsandbytes.nn import Linear4bit
            linear_cls = [Linear4bit]
        else:
            linear_cls = [AutoGPTQQuantLinear]
    if 'awq' in model_type:
        from awq.modules.linear import WQLinear_GEMM
        linear_cls.append(WQLinear_GEMM)
    if 'aqlm' in model_type:
        from aqlm import QuantizedLinear
        linear_cls.append(QuantizedLinear)

    # The content of target_module_names cannot exist in inner_nodes.
    # O(n^2logn), n represents the number of nodes, n<1000.
    inner_nodes = set()
    for name, module in model.named_modules():
        if not isinstance(module, tuple(linear_cls)):
            inner_nodes.add(name)
    target_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, tuple(linear_cls)) and head_module_name not in name:
            module_name_list = name.split('.')
            module_name = module_name_list.pop()
            for inner_node in inner_nodes:
                while inner_node.endswith(module_name):
                    module_name = f'{module_name_list.pop()}.{module_name}'
            target_module_names.add(module_name)
    return list(target_module_names)


@contextmanager
def safe_ddp_context():
    if (is_dist() or is_dist_ta()) and not is_local_master() and dist.is_initialized():
        dist.barrier()
    yield
    if (is_dist() or is_dist_ta()) and is_local_master() and dist.is_initialized():
        dist.barrier()
    if (is_dist() or is_dist_ta()) and dist.is_initialized():  # sync
        dist.barrier()
