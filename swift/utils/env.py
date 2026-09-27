# Copyright (c) ModelScope Contributors. All rights reserved.
import os
from transformers.integrations import deepspeed_config
from transformers.utils import strtobool
from typing import Optional, Tuple

from .logger import get_logger

logger = get_logger()


def use_hf_hub():
    return strtobool(os.environ.get('USE_HF', '0'))


def get_hf_endpoint():
    hf_endpoint = os.environ.get('HF_ENDPOINT', 'https://huggingface.co/')
    if hf_endpoint.endswith('/'):
        hf_endpoint = hf_endpoint[:-1]
    return hf_endpoint


def is_deepspeed_enabled():
    return deepspeed_config() is not None


def get_int_env(name: str, default: int) -> int:
    """Read an integer env var, falling back to `default` when it is unset, blank or unparsable.

    A launcher script or a Dockerfile that exports `RANK=` before it knows the value leaves the
    variable present but empty, and `int('')` raises from inside these readers.
    """
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(f'Invalid `{name}`: `{value}`, falling back to `{default}`.')
        return default


def get_dist_setting() -> Tuple[int, int, int, int]:
    """return rank, local_rank, world_size, local_world_size"""
    rank = get_int_env('RANK', -1)
    local_rank = get_int_env('LOCAL_RANK', -1)
    world_size = int(os.getenv('WORLD_SIZE') or os.getenv('_PATCH_WORLD_SIZE') or 1)
    # compat deepspeed launch
    local_world_size = int(os.getenv('LOCAL_WORLD_SIZE', None) or os.getenv('LOCAL_SIZE', 1))
    return rank, local_rank, world_size, local_world_size


def get_node_setting():
    node_rank = get_int_env('NODE_RANK', 0)
    nnodes = get_int_env('NNODES', 1)
    return node_rank, nnodes


def is_local_master():
    local_rank = get_dist_setting()[1]
    return local_rank in {-1, 0}


def is_master():
    rank = get_dist_setting()[0]
    return rank in {-1, 0}


def is_last_rank():
    rank, _, world_size, _ = get_dist_setting()
    return rank in {-1, world_size - 1}


def is_dist():
    """Determine if the training is distributed"""
    rank, local_rank, _, _ = get_dist_setting()
    return rank >= 0 and local_rank >= 0


def is_mp() -> bool:

    from swift.utils import get_device_count
    n_gpu = get_device_count()
    local_world_size = get_dist_setting()[3]
    if os.environ.get('SWIFT_SINGLE_DEVICE_MODE', '0') != '1':
        assert n_gpu % local_world_size == 0, f'n_gpu: {n_gpu}, local_world_size: {local_world_size}'
        if n_gpu // local_world_size >= 2:
            return True
        return False
    else:
        return False


def is_mp_ddp() -> bool:
    _, _, world_size, _ = get_dist_setting()
    disable_mp_ddp = strtobool(os.environ.get('DISABLE_MP_DDP', '0'))
    if not disable_mp_ddp and is_dist() and is_mp() and world_size > 1:
        logger.info_once('Using MP(device_map) + DDP')
        return True
    return False


def select_device(device_ids='0'):
    os.environ['CUDA_VISIBLE_DEVICES'] = device_ids
    os.environ['ASCEND_RT_VISIBLE_DEVICES'] = device_ids


def is_pai_training_job() -> bool:
    return 'PAI_TRAINING_JOB_ID' in os.environ


def get_pai_tensorboard_dir() -> Optional[str]:
    return os.environ.get('PAI_OUTPUT_TENSORBOARD')
