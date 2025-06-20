# Copyright (c) Alibaba, Inc. and its affiliates.
import gc
import hashlib
import os
import pickle
import re
import time
import uuid
from bisect import bisect_right
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from datasets.utils.filelock import FileLock
from modelscope.hub.utils.utils import get_cache_dir
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.trainer_utils import set_seed
from transformers.utils import is_torch_cuda_available, is_torch_mps_available, is_torch_npu_available

from .env import get_dist_setting, get_node_setting, is_dist, is_dist_ta, is_local_master, is_master
from .logger import get_logger
from .utils import deep_getattr

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

    if len(freeze_parameters) > 0:
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
    for i in range(get_device_count()):
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
        from swift.llm import get_model_arch
        model_arch = get_model_arch(model.model_meta.model_arch)
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


@contextmanager
def safe_ddp_context(hash_id: Optional[str], use_barrier: bool = False):
    if use_barrier and dist.is_initialized():
        if is_dist() or is_dist_ta():
            if not is_master():
                dist.barrier()
            if not is_local_master():
                # Compatible with multi-machine scenarios,
                # where each machine uses different storage hardware.
                dist.barrier()
        yield
        if is_dist() or is_dist_ta():
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


def seed_worker(worker_id: int, num_workers: int, rank: int):
    """
    Helper function to set worker seed during Dataloader initialization.
    """
    init_seed = torch.initial_seed() % 2**32
    worker_seed = num_workers * rank + init_seed
    set_seed(worker_seed)


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
