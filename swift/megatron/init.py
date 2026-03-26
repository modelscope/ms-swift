# Copyright (c) ModelScope Contributors. All rights reserved.
import concurrent.futures
import importlib.metadata
import inspect
import logging
import os
import peft
import subprocess
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from copy import copy
from functools import partial
from packaging import version
from tqdm import tqdm
from transformers.utils import is_torch_npu_available
from typing import List, Optional, Tuple

from swift.utils import get_logger, is_flash_attn_3_available, split_list

logger = get_logger()


def _patch__batched_p2p_ops():
    from megatron.core.pipeline_parallel import p2p_communication

    _batched_p2p_ops_origin = p2p_communication._batched_p2p_ops

    def _batched_p2p_ops(**kwargs):
        kwargs['group'] = None
        return _batched_p2p_ops_origin(**kwargs)

    p2p_communication._batched_p2p_ops = _batched_p2p_ops


def _patch_torch_FileSystemReader():
    from torch.distributed.checkpoint.filesystem import FileSystemReader
    from torch.futures import Future
    _origin_read_data = FileSystemReader.read_data
    _origin__slice_file = FileSystemReader._slice_file
    READER_MAX_WORKERS = int(os.environ.get('MCORE_READER_MAX_WORKERS', '16'))

    @contextmanager
    def _patch__slice_file(prog_bar):

        def _slice_file(self, *args, **kwargs):
            prog_bar.update()
            return _origin__slice_file(self, *args, **kwargs)

        FileSystemReader._slice_file = _slice_file
        try:
            yield
        finally:
            FileSystemReader._slice_file = _origin__slice_file

    def read_data(self, plan, planner):

        def _worker(plan_shard):
            _origin_read_data(self, plan_shard, planner)

        prog_bar = tqdm(total=len(plan.items), dynamic_ncols=True, desc='Loading: ')
        plan_shards = split_list(plan.items, READER_MAX_WORKERS, contiguous=False)
        with _patch__slice_file(prog_bar):
            with concurrent.futures.ThreadPoolExecutor(max_workers=READER_MAX_WORKERS) as pool:
                futures = []
                for i in range(READER_MAX_WORKERS):
                    plan_shard = copy(plan)
                    plan_shard.items = plan_shards[i]
                    futures.append(pool.submit(_worker, plan_shard))
                concurrent.futures.wait(futures)
        prog_bar.close()
        fut: Future = Future()
        fut.set_result(None)
        return fut

    FileSystemReader.read_data = read_data


def _patch_validate_non_overlapping_shards_metadata():
    # too slow
    from torch.distributed._shard.sharded_tensor import api
    from torch.distributed._shard.sharding_spec import api as api2
    from torch.distributed.checkpoint import default_planner

    def validate_non_overlapping_shards_metadata(*args, **kwargs):
        pass

    api.validate_non_overlapping_shards_metadata = validate_non_overlapping_shards_metadata
    api2.validate_non_overlapping_shards_metadata = validate_non_overlapping_shards_metadata

    def _validate_global_plan(*args, **kwargs):
        return True

    default_planner._validate_global_plan = _validate_global_plan


def _patch__write_item():
    import megatron.core
    if version.parse(megatron.core.__version__) >= version.parse('0.13.0rc0'):
        return
    # mcore 0.12
    from megatron.core.dist_checkpointing.strategies import filesystem_async

    _origin__write_item = filesystem_async._write_item
    if 'serialization_format' in inspect.signature(_origin__write_item).parameters:
        from torch.distributed.checkpoint.filesystem import SerializationFormat

        def _write_item(self, *args, **kwargs):
            if 'serialization_format' not in kwargs:
                kwargs['serialization_format'] = SerializationFormat.TORCH_SAVE
            return _origin__write_item(self, *args, **kwargs)

        filesystem_async._write_item = _write_item


def _patch_unified_memory():
    if is_torch_npu_available():
        return

    mcore_015 = version.parse(importlib.metadata.version('megatron-core')) >= version.parse('0.15.0rc0')
    if not mcore_015:
        return
    from torch.utils import cpp_extension
    load_inline = cpp_extension.load_inline

    def _new_load_inline(*args, **kwargs):
        name = kwargs.get('name')
        if name == 'managed_alloc_runtime':
            raise RuntimeError
        return load_inline(*args, **kwargs)

    # not create unified memory mempool
    cpp_extension.load_inline = _new_load_inline
    try:
        from megatron.core.inference import unified_memory
    except Exception:
        pass
    finally:
        cpp_extension.load_inline = load_inline


def init_megatron_env():
    os.environ.pop('VLLM_USE_MODELSCOPE', None)
    logging_level = logging.root.level
    _patch_unified_memory()
    _patch__batched_p2p_ops()
    _patch__write_item()
    logging.root.setLevel(logging_level)  # revert logger level
    try:
        _patch_torch_FileSystemReader()
    except Exception:
        logger.warning('Failed to patch FileSystemReader.')
    try:
        _patch_validate_non_overlapping_shards_metadata()
    except Exception:
        logger.warning('Patch validate_non_overlapping_shards_metadata failed.')
        pass

    import megatron.core
    logger.info(f'megatron.core.__version__: {megatron.core.__version__}')
