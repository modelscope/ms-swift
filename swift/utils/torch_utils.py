# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import socket
import time
import uuid
from bisect import bisect_right
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import Module
from transformers.utils import is_torch_npu_available, strtobool

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


def get_model_info(model: Module, name: Optional[str] = None) -> str:
    if name is None:
        name = model.__class__.__name__

    n_params = sum(p.numel() for p in model.parameters())
    n_grads = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_buffers = sum(p.numel() for p in model.buffers())

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


def get_dist_setting() -> Tuple[int, int, int, int]:
    """return rank, local_rank, world_size, local_world_size"""
    rank = int(os.getenv('RANK', -1))
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    local_world_size = int(os.getenv('LOCAL_WORLD_SIZE', 1))
    return rank, local_rank, world_size, local_world_size


def is_local_master():
    local_rank = get_dist_setting()[1]
    return local_rank in {-1, 0}


def is_master():
    rank = get_dist_setting()[0]
    return rank in {-1, 0}


def use_torchacc() -> bool:
    return strtobool(os.getenv('USE_TORCHACC', '0'))


def torchacc_trim_graph():
    return strtobool(os.getenv('TORCHACC_TRIM_GRAPH', '0'))


def is_dist():
    """Determine if the training is distributed"""
    if use_torchacc():
        return False
    rank, local_rank, _, _ = get_dist_setting()
    return rank >= 0 and local_rank >= 0


def is_mp() -> bool:
    if use_torchacc():
        return False
    n_gpu = torch.cuda.device_count()
    local_world_size = get_dist_setting()[3]
    assert n_gpu % local_world_size == 0, f'n_gpu: {n_gpu}, local_world_size: {local_world_size}'
    if n_gpu // local_world_size >= 2:
        return True
    return False


def is_ddp_plus_mp() -> bool:
    if not is_dist():
        return False
    if not is_mp():
        return False
    logger.info('Using DDP + MP(device_map)')
    return True


def show_layers(model: Module, max_lines: Optional[int] = 20) -> None:
    named_p = list(model.named_parameters())
    for i, (n, p) in enumerate(named_p):
        if max_lines is not None and i >= max_lines:
            logger.info('...')
            break
        logger.info(f'[{n}]: requires_grad={p.requires_grad}, dtype={p.dtype}, device={p.device}')


def freeze_model_parameters(model: Module, freeze_parameters: float) -> None:
    n_parameters = np.array([p.numel() for p in model.parameters()], dtype=np.int64)
    n_freeze_parameters = int(np.sum(n_parameters) * freeze_parameters)
    n_parameters_cs = np.cumsum(n_parameters)
    idx = bisect_right(n_parameters_cs, n_freeze_parameters)
    for _, p in zip(range(idx), model.parameters()):
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
    if use_torchacc():
        device = 'xla'
    assert rank >= 0
    if rank == 0:
        assert string is not None
        tensor = torch.tensor(
            [ord(c) for c in string] + [0] * (buffer_size - len(string)), dtype=torch.int64, device=device)
    else:
        tensor = torch.zeros(buffer_size, dtype=torch.int64, device=device)
    dist.broadcast(tensor, 0)
    if use_torchacc():
        tensor = tensor.to('cpu')
    first_zero = (tensor == 0).nonzero()[0].item()
    res = tensor.tolist()[:first_zero]
    return ''.join([chr(x) for x in res])


def time_synchronize() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()  # second
