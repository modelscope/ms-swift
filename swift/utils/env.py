# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from transformers.utils import strtobool

from .logger import get_logger

logger = get_logger()


def use_hf_hub():
    return strtobool(os.environ.get('USE_HF', '0'))


def is_deepspeed_enabled():
    return strtobool(os.environ.get('ACCELERATE_USE_DEEPSPEED', '0'))


def use_torchacc() -> bool:
    return strtobool(os.getenv('USE_TORCHACC', '0'))


def get_dist_setting() -> Tuple[int, int, int, int]:
    """return rank, local_rank, world_size, local_world_size"""
    rank = int(os.getenv('RANK', -1))
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    world_size = int(os.getenv('WORLD_SIZE') or os.getenv('_PATCH_WORLD_SIZE') or 1)
    # compat deepspeed launch
    local_world_size = int(os.getenv('LOCAL_WORLD_SIZE', None) or os.getenv('LOCAL_SIZE', 1))
    return rank, local_rank, world_size, local_world_size


def get_node_setting():
    node_rank = int(os.getenv('NODE_RANK', 0))
    nnodes = int(os.getenv('NNODES', 1))
    return node_rank, nnodes


def is_local_master():
    local_rank = get_dist_setting()[1]
    return local_rank in {-1, 0}


def is_master():
    rank = get_dist_setting()[0]
    return rank in {-1, 0}


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
    if strtobool(os.environ.get('USE_FAST_INFERENCE', 'false')):
        return False
    from swift.utils import get_device_count
    n_gpu = get_device_count()
    local_world_size = get_dist_setting()[3]
    assert n_gpu % local_world_size == 0, f'n_gpu: {n_gpu}, local_world_size: {local_world_size}'
    if n_gpu // local_world_size >= 2:
        return True
    return False


def is_mp_ddp() -> bool:
    _, _, world_size, _ = get_dist_setting()
    if is_dist() and is_mp() and world_size > 1:
        logger.info_once('Using MP(device_map) + DDP')
        return True
    return False


def is_dist_ta() -> bool:
    """Determine if the TorchAcc training is distributed"""
    _, _, world_size, _ = get_dist_setting()
    if use_torchacc() and world_size > 1:
        if not dist.is_initialized():
            import torchacc as ta
            # Initialize in advance
            dist.init_process_group(backend=ta.dist.BACKEND_NAME)
        return True
    else:
        return False


def is_pai_training_job() -> bool:
    return 'PAI_TRAINING_JOB_ID' in os.environ


def get_pai_tensorboard_dir() -> Optional[str]:
    return os.environ.get('PAI_OUTPUT_TENSORBOARD')
