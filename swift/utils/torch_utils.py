# Copyright (c) ModelScope Contributors. All rights reserved.
import gc
import hashlib
import os
import pickle
import time
import uuid
from contextlib import contextmanager
from datetime import timedelta
from typing import Any, Mapping, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets.utils.filelock import FileLock
from modelscope.hub.utils.utils import get_cache_dir
from transformers.utils import is_torch_cuda_available, is_torch_mps_available, is_torch_npu_available

from swift.utils import is_mp
from .env import get_dist_setting, get_node_setting, is_dist, is_local_master, is_master
from .logger import get_logger

logger = get_logger()


def _find_local_mac() -> str:
    mac = uuid.getnode()
    mac_address = ':'.join(('%012x' % mac)[i:i + 2] for i in range(0, 12, 2))
    return mac_address


def time_synchronize() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()  # second


_DISABLE_USE_BARRIER = False


@contextmanager
def disable_safe_ddp_context_use_barrier():
    global _DISABLE_USE_BARRIER
    _DISABLE_USE_BARRIER = True
    try:
        yield
    finally:
        _DISABLE_USE_BARRIER = False


@contextmanager
def safe_ddp_context(hash_id: Optional[str], use_barrier: bool = True):
    if _DISABLE_USE_BARRIER:
        use_barrier = False
    if use_barrier and dist.is_initialized():
        if is_dist():
            if not is_master():
                dist.barrier()
            if not is_local_master():
                # Compatible with multi-machine scenarios,
                # where each machine uses different storage hardware.
                dist.barrier()
        yield
        if is_dist():
            if is_master():
                dist.barrier()
            if is_local_master():
                dist.barrier()
    elif hash_id is not None:
        lock_dir = os.path.join(get_cache_dir(), 'lockers')
        os.makedirs(lock_dir, exist_ok=True)
        file_path = hashlib.sha256(hash_id.encode('utf-8')).hexdigest() + '.lock'
        file_path = os.path.join(lock_dir, file_path)
        with FileLock(file_path):
            yield
    else:
        yield


def get_device(local_rank: Optional[Union[str, int]] = None) -> str:
    if local_rank is None:
        local_rank = max(0, get_dist_setting()[1])
    local_rank = str(local_rank)
    if is_torch_npu_available():
        device = 'npu:{}'.format(local_rank)
    elif is_torch_mps_available():
        device = 'mps:{}'.format(local_rank)
    elif is_torch_cuda_available():
        device = 'cuda:{}'.format(local_rank)
    else:
        device = 'cpu'

    return device


def get_current_device():
    if is_torch_npu_available():
        current_device = torch.npu.current_device()
    elif is_torch_cuda_available():
        current_device = torch.cuda.current_device()
    elif is_torch_mps_available():
        current_device = 'mps'
    else:
        current_device = 'cpu'
    return current_device


def get_torch_device():
    if is_torch_cuda_available():
        return torch.cuda
    elif is_torch_npu_available():
        return torch.npu
    elif is_torch_mps_available():
        return torch.mps
    else:
        return torch.cpu


def set_device(local_rank: Optional[Union[str, int]] = None):
    if local_rank is None:
        local_rank = max(0, get_dist_setting()[1])
    if is_torch_npu_available():
        torch.npu.set_device(local_rank)
    elif is_torch_cuda_available():
        torch.cuda.set_device(local_rank)


def get_device_count() -> int:
    if is_torch_npu_available():
        return torch.npu.device_count()
    elif is_torch_cuda_available():
        return torch.cuda.device_count()
    else:
        return 0


def empty_cache():
    if is_torch_npu_available():
        torch.npu.empty_cache()
    elif is_torch_mps_available():
        torch.mps.empty_cache()
    elif is_torch_cuda_available():
        torch.cuda.empty_cache()


def gc_collect() -> None:
    gc.collect()
    empty_cache()


def get_last_valid_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Get the last valid (non-padding) token position indices for each sample.

    This function correctly handles sequences with different padding directions (left/right/none)
    within the same batch by computing the last valid index for each sequence individually.

    Args:
        attention_mask: Attention mask [batch_size, seq_len] where 1=valid, 0=padding

    Returns:
        torch.Tensor: Indices of last valid positions [batch_size]

    Examples:
        >>> # Right padding
        >>> attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
        >>> get_last_valid_indices(attention_mask)
        tensor([2, 3])

        >>> # Left padding
        >>> attention_mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])
        >>> get_last_valid_indices(attention_mask)
        tensor([4, 4])
    """
    seq_len = attention_mask.shape[1]

    # Flip the mask horizontally to bring the last elements to the front.
    # `argmax` will then find the index of the first '1', which corresponds to the last valid token.
    last_valid_indices = torch.fliplr(attention_mask).argmax(dim=1)

    # Convert the index from the right-to-left frame to the original left-to-right frame.
    indices = seq_len - 1 - last_valid_indices

    return indices


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


def set_default_ddp_config():
    # It runs normally with Python as well.
    rank, local_rank, _, _ = get_dist_setting()
    if rank == -1 or local_rank == -1:
        os.environ['NPROC_PER_NODE'] = '1'
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['LOCAL_WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')


def init_process_group(backend: Optional[str] = None, timeout: int = 18000000):
    if dist.is_initialized():
        return
    set_device()
    if backend is None:
        if is_torch_npu_available():
            backend = 'hccl'
        elif torch.cuda.is_available():
            backend = 'nccl'
        else:
            backend = 'gloo'
    timeout = timedelta(seconds=timeout)
    dist.init_process_group(backend=backend, timeout=timeout)


def check_shared_disk(error, cache_dir: Optional[str] = None):
    nnodes = get_node_setting()[1]
    if nnodes <= 1:
        return True
    assert dist.is_initialized()
    if cache_dir is None:
        cache_dir = os.path.join(get_cache_dir(), 'tmp')
    os.makedirs(cache_dir, exist_ok=True)
    tmp_path = os.path.join(cache_dir, 'check_shared_disk.tmp')
    is_shared_disk = True

    try:
        with safe_ddp_context(None, True):
            if is_master():
                with open(tmp_path, 'w'):
                    pass
            if not os.path.exists(tmp_path):
                is_shared_disk = False
        shared_state = [None] * dist.get_world_size()
        dist.all_gather_object(shared_state, is_shared_disk)
    finally:
        if is_master() and os.path.exists(tmp_path):
            os.remove(tmp_path)
    if not all(shared_state):
        raise error


def to_float_dtype(data: Any, dtype: torch.dtype) -> Any:
    """Change the float inputs to a dtype"""
    if isinstance(data, Mapping):
        return type(data)({k: to_float_dtype(v, dtype) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(to_float_dtype(v, dtype) for v in data)
    elif isinstance(data, torch.Tensor) and torch.is_floating_point(data):
        return data.to(dtype=dtype)
    else:
        return data


def to_device(data: Any, device: Union[str, torch.device, int], non_blocking: bool = False) -> Any:
    """Move inputs to a device"""
    if isinstance(data, Mapping):
        return type(data)({k: to_device(v, device, non_blocking) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(to_device(v, device, non_blocking) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device=device, non_blocking=non_blocking)
    else:
        return data


def get_generative_reranker_logits(lm_head_weight, tokenizer, hidden_states):
    positive_token = os.environ.get('GENERATIVE_RERANKER_POSITIVE_TOKEN', 'yes')
    negative_token = os.environ.get('GENERATIVE_RERANKER_NEGATIVE_TOKEN', 'no')
    positive_token_id = tokenizer.convert_tokens_to_ids(positive_token)
    negative_token_id = tokenizer.convert_tokens_to_ids(negative_token)
    weight = lm_head_weight[[positive_token_id, negative_token_id]]
    logits = F.linear(hidden_states, weight)
    return logits[..., 0:1] - logits[..., 1:2]


def get_max_reserved_memory() -> float:
    devices = list(range(get_device_count())) if is_mp() else [None]
    try:
        mems = [get_torch_device().max_memory_reserved(device=device) for device in devices]
    except AttributeError:
        return 0  # fix mps
    return sum(mems) / 1024**3
