# Copyright (c) Alibaba, Inc. and its affiliates.

import datetime as dt
import os
import random
import re
import socket
from typing import List, Optional, Tuple, Type, TypeVar

import numpy as np
import torch
import torch.distributed as dist
from numpy.random import RandomState
from torch.nn import Module
from transformers import HfArgumentParser

from .logger import get_logger, is_master

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


def _get_version(work_dir: str) -> int:
    if os.path.isdir(work_dir):
        fnames = os.listdir(work_dir)
    else:
        fnames = []
    v_list = [-1]
    for fname in fnames:
        m = re.match(r'v(\d+)', fname)
        if m is None:
            continue
        v = m.group(1)
        v_list.append(int(v))
    return max(v_list) + 1


def add_version_to_work_dir(work_dir: str) -> str:
    """add version"""
    work_dir = os.path.abspath(work_dir)
    version = _get_version(work_dir)
    time = dt.datetime.now().strftime('%Y%m%d-%H%M%S')

    work_dir = os.path.join(work_dir, f'v{version}-{time}')
    logger.info(f'work_dir: {work_dir}')
    return work_dir


def seed_everything(seed: Optional[int] = None, gpu_dtm: bool = False) -> int:
    if seed is None:
        seed_max = np.iinfo(np.int32).max
        seed = random.randint(0, seed_max)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f'Global seed set to {seed}')
    if gpu_dtm:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f'Setting deterministic: {True}, benchmark: {False}')
    return seed


def print_model_info(model: Module, name: Optional[str] = None) -> None:
    if name is None:
        name = model.__class__.__name__

    n_params = sum(p.numel() for p in model.parameters())
    n_grads = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_buffers = sum(p.numel() for p in model.buffers())

    n_params /= 1e6
    n_grads /= 1e6
    n_buffers /= 1e6
    s = [
        f'{name}: ', f'{n_params:.4f}M Params ({n_grads:.4f}M Trainable), ',
        f'{n_buffers:.4f}M Buffers, ',
        f'Trainable percentage: {100 * n_grads / n_params:.2f}%'
    ]
    s += '.'
    logger.info(''.join(s))


def find_sub_module(module: torch.nn.Module, module_name: str) -> List[torch.nn.Module]:
    _modules = list()
    for name, sub_module in module.named_modules():
        if not name:
            continue
        if module_name == name or getattr(sub_module, 'adapter_name', None) == module_name:
            _modules.append(sub_module)
        else:
            _modules.extend(find_sub_module(sub_module, module_name))
    return _modules


def get_seed(random_state: RandomState) -> int:
    seed_max = np.iinfo(np.int32).max
    seed = random_state.randint(0, seed_max)
    return seed


_T = TypeVar('_T')


def parse_args(class_type: Type[_T],
               argv: Optional[List[str]] = None) -> Tuple[_T, List[str]]:
    parser = HfArgumentParser([class_type])
    args, remaining_args = parser.parse_args_into_dataclasses(
        argv, return_remaining_strings=True)
    logger.info(f'args: {args}')
    return args, remaining_args


def get_dist_setting() -> Tuple[int, int, int, int]:
    """return rank, local_rank, world_size"""
    rank = int(os.getenv('RANK', -1))
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    local_world_size = int(os.getenv('LOCAL_WORLD_SIZE', 1))
    return rank, local_rank, world_size, local_world_size


def is_local_master():
    local_rank = get_dist_setting()[1]
    return local_rank in {-1, 0}


def is_dist():
    """Determine if the training is distributed"""
    rank, local_rank, _, _ = get_dist_setting()
    return rank >= 0 and local_rank >= 0


def show_layers(model: Module, max_lines: Optional[int] = 20) -> None:
    named_p = list(model.named_parameters())
    for i, (n, p) in enumerate(named_p):
        if max_lines is not None and i >= max_lines:
            logger.info('...')
            break
        logger.info(
            f'[{n}]: requires_grad={p.requires_grad}, dtype={p.dtype}, device={p.device}'
        )


def broadcast_string(string: Optional[str], buffer_size: int = 100) -> str:
    """String broadcasting in case of DDP
    string: main rank: str
        other rank: None
    return: all rank: str
    """
    assert dist.is_initialized()
    rank, local_rank, _, _ = get_dist_setting()
    assert rank >= 0
    if rank == 0:
        assert string is not None
        tensor = torch.tensor(
            [ord(c) for c in string] + [0] * (buffer_size - len(string)),
            dtype=torch.int64,
            device=local_rank)
    else:
        tensor = torch.zeros(buffer_size, dtype=torch.int64, device=local_rank)
    dist.broadcast(tensor, 0)
    first_zero = (tensor == 0).nonzero()[0].item()
    res = tensor.tolist()[:first_zero]
    return ''.join([chr(x) for x in res])
