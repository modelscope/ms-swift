# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/transformers.
import logging
import os
import shutil
from functools import wraps
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

import requests
import torch
import torch.distributed as dist
from accelerate.utils.modeling import (get_balanced_memory,
                                       infer_auto_device_map)
from datasets import Dataset as HfDataset
from modelscope import MsDataset
from modelscope.utils.config_ds import MS_CACHE_HOME
from modelscope.utils.logger import get_logger as get_ms_logger
from torch import device as Device
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
from transformers import trainer

from swift.hub import ModelScopeConfig
from swift.utils import (get_dist_setting, get_logger, is_ddp_plus_mp, is_dist,
                         is_local_master, is_master, parse_args)

logger = get_logger()
ms_logger = get_ms_logger()

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def download_files(url: str, local_path: str, cookies) -> None:
    resp = requests.get(url, cookies=cookies, stream=True)
    with open(local_path, 'wb') as f:
        for data in tqdm(resp.iter_lines()):
            f.write(data)


def download_dataset(model_id: str,
                     files: List[str],
                     force_download: bool = False) -> str:
    url = f'http://www.modelscope.cn/api/v1/datasets/{model_id}/repo?Revision=master&FilePath={{fpath}}'
    cache_dir = os.path.join(MS_CACHE_HOME, 'datasets', model_id, 'master')
    local_dir = os.path.join(cache_dir, 'raw')
    tmp_dir = os.path.join(cache_dir, 'tmp')
    os.makedirs(local_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    cookies = ModelScopeConfig.get_cookies()
    with TemporaryDirectory(dir=tmp_dir) as temp_dir:
        for remote_fpath in files:
            url = url.format(fpath=remote_fpath)
            temp_fpath = os.path.join(temp_dir, remote_fpath)
            local_fpath = os.path.join(local_dir, remote_fpath)
            if not force_download and os.path.exists(local_fpath):
                continue
            download_files(url, temp_fpath, cookies)
            shutil.copy2(temp_fpath, local_fpath)

    return local_dir


_old_msdataset_load = MsDataset.load


@wraps(_old_msdataset_load)
def _msdataset_ddp_load(*args, **kwargs):
    if is_dist() and not is_local_master():
        dist.barrier()
    dataset = _old_msdataset_load(*args, **kwargs)
    if is_dist() and is_local_master():
        dist.barrier()

    if is_dist():
        dist.barrier()
    return dataset


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


def _sync_max_memory(
        max_memory: Dict[Union[int, str], int]) -> Dict[Union[int, str], int]:
    """Make sure that the model structure of MP(device_map) is the same, when using DDP."""
    max_memory_list = [
        v for k, v in max_memory.items() if (v > 0 and k != 'cpu')
    ]
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


@wraps(infer_auto_device_map)
def _infer_auto_device_map_patch(
        model: Module,
        max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
        **kwargs) -> Dict[str, Union[int, str, Device]]:
    """The auxiliary function for supports DDP+MP. Monkey Patching.
    add feat in accelerate to support DDP + MP"""
    verbose = kwargs.pop('verbose', False)
    n_gpu = torch.cuda.device_count()
    _, local_rank, _, local_world_size = get_dist_setting()
    device_ids = list(range(local_rank, n_gpu, local_world_size))
    max_memory = _get_max_memory(device_ids)
    max_memory = _sync_max_memory(max_memory)
    max_memory = get_balanced_memory(
        model, max_memory, low_zero=False, **kwargs)
    max_memory = {k: v for k, v in max_memory.items() if v > 0}
    return infer_auto_device_map(model, max_memory, verbose=verbose, **kwargs)


def dataset_map(
    dataset: HfDataset, preprocess_func: Callable[[Dict[str, Any]],
                                                  Dict[str,
                                                       Optional[List[int]]]]
) -> HfDataset:
    # faster than dataset.map
    input_ids = []
    labels = []
    for d in tqdm(dataset):
        d = preprocess_func(d)
        if d['input_ids'] is None:
            continue
        input_ids.append(d['input_ids'])
        labels.append(d['labels'])
    return HfDataset.from_dict({'input_ids': input_ids, 'labels': labels})


logger_format = logging.Formatter('[%(levelname)s:%(name)s] %(message)s')

logger.handlers[0].setFormatter(logger_format)
ms_logger.handlers[0].setFormatter(logger_format)
if is_master():
    logger.setLevel(logging.INFO)
    ms_logger.setLevel(logging.INFO)
else:
    logger.setLevel(logging.ERROR)
    ms_logger.setLevel(logging.ERROR)

_TArgsClass = TypeVar('_TArgsClass')
_T = TypeVar('_T')


def get_main(
        args_class: Type[_TArgsClass],
        llm_x: Callable[[_TArgsClass],
                        _T]) -> Callable[[Optional[List[str]]], _T]:

    def x_main(argv: Optional[List[str]] = None) -> _T:
        args, remaining_argv = parse_args(args_class, argv)
        args.init_argument()
        if len(remaining_argv) > 0:
            if args.ignore_args_error:
                logger.warning(f'remaining_argv: {remaining_argv}')
            else:
                raise ValueError(f'remaining_argv: {remaining_argv}')
        return llm_x(args)

    return x_main


# monkey patching
MsDataset.load = _msdataset_ddp_load
if is_ddp_plus_mp():
    import transformers
    import accelerate
    _old_ddp_init = DDP.__init__
    accelerate.accelerator.torch.nn.parallel.DistributedDataParallel.__init__ = (
        lambda self, model, device_ids, output_device, *args, **kwargs:
        _old_ddp_init(self, model, *args, **kwargs))
    transformers.modeling_utils.get_balanced_memory = lambda *args, **kwargs: None
    transformers.modeling_utils.infer_auto_device_map = _infer_auto_device_map_patch
    _old_accelerator_init = trainer.Accelerator.__init__
    trainer.Accelerator.__init__ = (
        lambda self, device_placement=False, *args, **kwargs:
        _old_accelerator_init(
            self, device_placement=device_placement, *args, **kwargs))
    trainer.Accelerator.verify_device_map = lambda *args, **kwargs: False
